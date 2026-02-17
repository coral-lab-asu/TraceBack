import os
import json
import pandas as pd
import re
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

from LLM import Call_OpenAI, Call_Gemini, Call_DeepSeek, Call_HF, Call_Novita
from relevant_col_gen import get_relevant_cols
from subqueries import get_subqueries
from query_attribution import attribution


def _get_tqdm(disabled: bool):
    if disabled:
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        return None
    return tqdm


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                print(f"[Warn] Skipping malformed JSON line {line_no}")
                continue
            if isinstance(data, dict):
                yield line_no, data


def col_filter_table(cols, df):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Include only selected columns + row_index
    filtered_df = df[["row_id"] + cols]
    return filtered_df


def make_unique(headers):
    counts = {}
    unique = []
    for h in headers:
        counts[h] = counts.get(h, 0) + 1
        if counts[h] == 1:
            unique.append(h)
        else:
            unique.append(f"{h}_{counts[h]}")
    return unique


def convert_to_df(table_rows):
    max_cols = max(len(row) for row in table_rows)
    fill_value = ""
    padded_rows = [row + [fill_value] * (max_cols - len(row)) for row in table_rows]
    raw_header = padded_rows[0]

    header = make_unique(raw_header)
    padded_rows = [header] + padded_rows[1:]

    row_numbers = list(range(len(padded_rows)))
    df = pd.DataFrame(padded_rows, columns=header)
    df.insert(0, "row_id", row_numbers)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def map_to_original_cols(ans, column_headers, relevant_cols):
    col_dict = {}
    for column_index in range(len(column_headers)):
        col_dict[column_headers[column_index]] = column_index

    mapped = []
    for cell in ans:
        if not cell:
            continue
        if not isinstance(cell, (list, tuple)) or len(cell) < 2:
            continue
        row_idx, col_idx = int(cell[0]), int(cell[1])
        # Skip malformed column indices
        if col_idx < 0 or col_idx >= len(relevant_cols):
            continue
        header_name = relevant_cols[col_idx]
        if header_name not in col_dict:
            continue
        orig_col_idx = col_dict[header_name]
        mapped.append([row_idx, orig_col_idx])
    return mapped


def parse_attribution(text, column_headers, relevant_cols):
    pattern = re.compile(r"\[(.*?)\]$")
    final_answer = []

    for line in text.splitlines():
        line = line.strip()
        stack = []
        for ch in line:
            if ch in ["(", "[", ","] or ch.isnumeric():
                stack.append(ch)
            elif ch in [")", "]"]:
                res = ""
                while stack:
                    x = stack.pop()
                    if x in ["(", "["]:
                        break
                    res += x
                try:
                    tup = eval("(" + res[::-1] + ")")
                    # Only keep proper (row, col) 2-tuples
                    if isinstance(tup, tuple) and len(tup) == 2:
                        final_answer.append(tup)
                except Exception:
                    continue

    return map_to_original_cols(final_answer, column_headers, relevant_cols)


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def build_llm(args):
    backend = str(args.backend).lower().strip()
    if backend == "openai":
        return Call_OpenAI(args.model)
    if backend == "gemini":
        return Call_Gemini(args.model)
    if backend == "deepseek":
        return Call_DeepSeek(args.model)
    if backend == "novita":
        return Call_Novita(
            args.model,
            max_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
        )
    if backend == "hf":
        cache_dir = args.cache_dir.strip() or None
        return Call_HF(
            args.model,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown backend: {args.backend}")


def run():
    parser = argparse.ArgumentParser(description="Run TraceBack workflow on ToTTo.")
    parser.add_argument("--backend", default="openai", choices=["openai", "gemini", "deepseek", "novita", "hf"])
    parser.add_argument("--model", default="gpt-4o", help="Model name (OpenAI/Gemini/DeepSeek) or HF model id")
    parser.add_argument("--limit", type=int, default=0, help="Max examples to run (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSON (skip processed ids)")
    parser.add_argument("--refresh-debug", action="store_true", help="With --resume: recompute entries missing traceback_debug")
    parser.add_argument("--output", default="", help="Output JSON path (default: derived from model)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="HF only: max new tokens per call")
    parser.add_argument("--temperature", type=float, default=0.0, help="HF only: sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="HF only: nucleus sampling top_p")
    parser.add_argument("--cache-dir", default="", help="HF only: cache dir for model/tokenizer downloads")
    parser.add_argument("--no-row-filtering", action="store_true", help="Disable Step 2 (row filtering)")
    parser.add_argument("--no-nli-filtering", action="store_true", help="Disable NLI filtering for sub-queries")
    parser.add_argument("--nli-threshold", type=float, default=0.5, help="NLI entailment threshold for sub-query filtering (lower keeps more)")
    parser.add_argument("--no-mysql", action="store_true", help="Allow SQLite fallback instead of requiring MySQL")
    parser.add_argument("--mysql-table-name", default="", help="MySQL table name prefix (default: per-run unique)")
    args = parser.parse_args()

    all_results = []
    root_dir = Path(__file__).resolve().parent.parent
    file_path = root_dir / "Datasets" / "Totto" / "totto_processed.jsonl"
    model_slug = sanitize_model_id(args.model)
    out_dir = "totto_results"
    os.makedirs(out_dir, exist_ok=True)
    output_file_path = Path(args.output) if args.output else (Path(out_dir) / f"totto_answer_attribution_{model_slug}_traceback.json")
    # Ensure repo root is importable so we can use the workflow runner.
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
    from traceback_workflow import TraceBackWorkflowRunner

    model = build_llm(args)
    runner = TraceBackWorkflowRunner(
        call_llm=model.call,
        nli_threshold=float(args.nli_threshold),
        enable_row_filtering=not bool(args.no_row_filtering),
        enable_nli_filtering=not bool(args.no_nli_filtering),
        require_mysql=not bool(args.no_mysql),
        mysql_table_name=(args.mysql_table_name.strip() or f"traceback_tmp_totto_{os.getpid()}"),
    )

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"totto_{model_slug}_traceback.log"
    logging.basicConfig(filename=str(log_file), level=logging.INFO, format="%(asctime)s - %(message)s")

    processed_ids = set()
    index_by_id = {}
    if args.resume and output_file_path.exists():
        try:
            existing = json.loads(output_file_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                all_results = existing
                for idx, rec in enumerate(all_results):
                    eid = rec.get("example_id")
                    if eid is not None:
                        index_by_id[eid] = idx
                        if args.refresh_debug and not rec.get("traceback_debug"):
                            continue
                        processed_ids.add(eid)
                print(f"[Resume] Loaded {len(all_results)} existing records from {output_file_path}")
        except Exception as e:
            print(f"[Warn] Could not load existing output for resume: {e}")

    # Pre-compute progress total (remaining examples)
    todo_total = 0
    for _line_no, data in _iter_jsonl(file_path):
        example_id = data.get("example_id")
        if processed_ids and example_id in processed_ids:
            continue
        table_array = data.get("table_array")
        highlighted_cells = data.get("highlighted_cells")
        if highlighted_cells is None:
            continue
        if not table_array or not isinstance(table_array, list) or not table_array[0]:
            continue
        todo_total += 1

    effective_total = todo_total
    if args.limit and int(args.limit) > 0:
        effective_total = min(effective_total, int(args.limit))

    tqdm = _get_tqdm(bool(args.no_progress))
    pbar = tqdm(total=effective_total, desc=f"ToTTo | {model_slug}", unit="ex", dynamic_ncols=True) if (tqdm and effective_total) else None

    processed = 0
    for line_no, data in _iter_jsonl(file_path):
        if args.limit and int(args.limit) > 0 and processed >= int(args.limit):
            break

        example_id = data.get("example_id")
        if processed_ids and example_id in processed_ids:
            continue
        question = data.get("question")
        answer = data.get("answer")
        table_array = data.get("table_array")
        if not table_array:
            continue
        highlighted_cells = data.get("highlighted_cells")
        if highlighted_cells is None:
            continue
        table_page_title = data.get("table_page_title")
        table_title = table_page_title or f"Totto-{example_id}"

        logging.info(table_array)
        logging.info(question)
        logging.info("-" * 50)
        logging.info(answer)
        logging.info(highlighted_cells)

        try:
            out = runner.run_example(
                table=table_array,
                question=question,
                answer=answer,
                table_title=table_title,
            )
            corrected = out.get("result_cells") or []
            traceback_debug = out.get("debug") or {}
            relevant_cols = (traceback_debug or {}).get("relevant_cols") or []
        except Exception as e:
            print(f"[Warn] TraceBack workflow failed for example {example_id}: {e}")
            corrected = []
            relevant_cols = []
            traceback_debug = {"error": str(e)}

        rec = {
            "example_id": example_id,
            "result": corrected,
            "relevant_column_headers": relevant_cols,
            "highlighted_cell_ids": highlighted_cells,
            "traceback_debug": traceback_debug,
        }
        if example_id in index_by_id:
            all_results[index_by_id[example_id]] = rec
        else:
            index_by_id[example_id] = len(all_results)
            all_results.append(rec)

        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with output_file_path.open("w", encoding="utf-8") as f_out:
            json.dump(all_results, f_out, ensure_ascii=False, indent=4)

        processed += 1
        if pbar is not None:
            pbar.update(1)
        else:
            print(f"[{processed}/{effective_total or '?'}] Done example_id={example_id} (line {line_no})")

    if pbar is not None:
        pbar.close()


if __name__ == "__main__":
    run()
