import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parent.parent


DATASETS: Dict[str, Dict[str, Any]] = {
    "aitqa": {
        "data_path": ROOT_DIR / "Datasets" / "AITQA" / "aitqa_processed.jsonl",
        "pred_default": ROOT_DIR / "aitqa_answer_attribution_gpt-4o_new.json",
        "id_key_data": "id",
        # Historical naming in outputs: "feta_id" is actually the AITQA "id" string.
        "id_key_pred": "feta_id",
        "table_key": "table",
        "question_key": "question",
        "answer_key": "answers",
        "gold_cells_key": "highlighted_cells",
        "title_fn": lambda obj: f"AITQA-{obj.get('id')}",
    },
    "fetaqa": {
        "data_path": ROOT_DIR / "Datasets" / "FetaQA" / "fetaQA_dev_processed.jsonl",
        "pred_default": ROOT_DIR / "fetaqa_results" / "fetaqa_answer_attribution_gpt-4o_withrows_1.json",
        "id_key_data": "feta_id",
        "id_key_pred": "feta_id",
        "table_key": "table_array",
        "question_key": "question",
        "answer_key": "answer",
        "gold_cells_key": "highlighted_cells",
        "title_fn": lambda obj: f"{obj.get('table_page_title', '')} - {obj.get('table_section_title', '')}".strip(" -"),
    },
    "totto": {
        "data_path": ROOT_DIR / "Datasets" / "Totto" / "totto_processed.jsonl",
        "pred_default": ROOT_DIR / "totto_results" / "totto_answer_attribution_gpt-4o_traceback.json",
        "id_key_data": "example_id",
        "id_key_pred": "example_id",
        "table_key": "table_array",
        "question_key": "question",
        "answer_key": "answer",
        "gold_cells_key": "highlighted_cells",
        "title_fn": lambda obj: str(obj.get("table_page_title", "")).strip(),
    },
}


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_or_empty(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
                continue
            except Exception:
                pass
            # Some lines may have trailing junk; try raw_decode.
            try:
                obj, _idx = decoder.raw_decode(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def _to_answer_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [str(x) for x in value if x is not None]
        return "; ".join(parts)
    return str(value)


def _normalize_cells(value: Any) -> List[Tuple[int, int]]:
    if value in (None, "null"):
        return []
    if not isinstance(value, list):
        return []
    # Handle accidental 3D nesting (list of list of lists).
    if value and isinstance(value[0], list) and value and isinstance(value[0][0], list):
        value = value[0]
    out: List[Tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            r = int(item[0])
            c = int(item[1])
        except Exception:
            continue
        out.append((r, c))
    return list(set(out))


def _is_citebench_pred_list(preds: Any) -> bool:
    if not isinstance(preds, list) or not preds:
        return False
    for rec in preds:
        if not isinstance(rec, dict):
            continue
        # CITEBENCH baselines write a unified list of records with these keys.
        if all(k in rec for k in ("dataset", "qid", "example_id", "result")):
            return True
        break
    return False


def _coerce_citebench_preds(preds: List[Dict[str, Any]], *, dataset: str, id_key_pred: str) -> List[Dict[str, Any]]:
    """Convert CITEBENCH baseline outputs into the dataset-specific id schema expected by this script.

    - AITQA: uses `qid` like "q-0" -> stored under `feta_id` (historical key in this repo).
    - FetaQA: uses `qid` like "feta-2275" -> stored under `feta_id` as int.
    - ToTTo: uses `qid` like "totto-7391450717765563190" -> stored under `example_id` as int.
    """

    target = str(dataset).strip().lower()
    out: List[Dict[str, Any]] = []
    for rec in preds:
        if not isinstance(rec, dict):
            continue
        ds = str(rec.get("dataset", "")).strip().lower()
        if ds == "feta":
            ds = "fetaqa"
        if ds != target:
            continue
        qid = rec.get("qid")
        if qid is None:
            continue
        result = rec.get("result") or []

        if target == "aitqa":
            out.append({id_key_pred: str(qid), "result": result})
            continue

        if target == "fetaqa":
            m = re.match(r"^feta-(\d+)$", str(qid).strip(), flags=re.IGNORECASE)
            if not m:
                continue
            out.append({id_key_pred: int(m.group(1)), "result": result})
            continue

        if target == "totto":
            m = re.match(r"^totto-(\d+)$", str(qid).strip(), flags=re.IGNORECASE)
            if not m:
                continue
            out.append({id_key_pred: int(m.group(1)), "result": result})
            continue

    return out


def _sanitize_tag(tag: str) -> str:
    tag = str(tag or "").strip()
    if not tag:
        return ""
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", tag)
    tag = re.sub(r"_+", "_", tag).strip("_")
    return tag[:80]


def _parse_numbered_facts(text: str) -> List[str]:
    facts: List[str] = []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*\d+\.\s*(.+?)\s*$", line)
        if not m:
            continue
        facts.append(m.group(1).strip())
    return facts


def _format_numbered(items: Sequence[str]) -> str:
    return "\n".join([f"{i + 1}. {t}" for i, t in enumerate(items)])


def _pad_row(row: List[Any], ncols: int) -> List[str]:
    if len(row) >= ncols:
        return [str(x) for x in row[:ncols]]
    return [str(x) for x in row] + [""] * (ncols - len(row))


def build_attributed_table_block(
    table: List[List[Any]],
    *,
    cells_table_indices: Sequence[Tuple[int, int]],
) -> Tuple[List[str], str]:
    """
    Returns (headers, block_str).
    - table is a list of rows; row 0 is headers.
    - cells_table_indices are (row_idx, col_idx) into table (header row is 0).
    """
    if not table or not table[0]:
        return [], ""
    headers = [str(h) for h in table[0]]
    ncols = len(headers)
    selected = set((int(r), int(c)) for r, c in cells_table_indices)
    row_ids = sorted({r for r, _c in selected if 0 < r < len(table)})

    lines: List[str] = []
    lines.append("Table:")
    lines.append("Column Names : " + ", ".join(headers))
    for r in row_ids:
        base = _pad_row(table[r], ncols)
        masked = [base[c] if (r, c) in selected else "-" for c in range(ncols)]
        lines.append(f"Row-{r} : {json.dumps(masked, ensure_ascii=False)}")
    return headers, "\n".join(lines)


def parse_alignment_counts(text: str) -> Optional[Dict[str, int]]:
    if not text:
        return None
    patterns = {
        "total_set1": r"Total\s+Number\s+of\s+facts\s+in\s+Set-1\s*:\s*(\d+)",
        "total_set2": r"Total\s+number\s+of\s+facts\s+in\s+Set-2\s*:\s*(\d+)",
        "covered": r"Number\s+of\s+atomic\s+facts\s+in\s+1\s+covered\s+by\s+2\s*:\s*(\d+)",
        "unnecessary": r"Number\s+of\s+atomic\s+facts\s+in\s+2\s+that\s+are\s+unnecessary\s*:\s*(\d+)",
    }
    out: Dict[str, int] = {}
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            return None
        out[key] = int(m.group(1))
    return out


def compute_fairscore_from_counts(counts: Iterable[Dict[str, int]]) -> Dict[str, float]:
    covered_sum = 0
    total_set1_sum = 0
    total_set2_sum = 0
    unnecessary_sum = 0
    n = 0
    for rec in counts:
        covered_sum += int(rec.get("covered", 0))
        total_set1_sum += int(rec.get("total_set1", 0))
        total_set2_sum += int(rec.get("total_set2", 0))
        unnecessary_sum += int(rec.get("unnecessary", 0))
        n += 1
    recall = (covered_sum / total_set1_sum) if total_set1_sum else 0.0
    precision = ((total_set2_sum - unnecessary_sum) / total_set2_sum) if total_set2_sum else 0.0
    return {
        "n": int(n),
        "fairscore_precision": float(precision),
        "fairscore_recall": float(recall),
        "covered_sum": int(covered_sum),
        "total_set1_sum": int(total_set1_sum),
        "total_set2_sum": int(total_set2_sum),
        "unnecessary_sum": int(unnecessary_sum),
    }


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.2f}"

def _fmt_delta_pct(pred: Optional[float], actual: Optional[float]) -> str:
    if pred is None or actual is None:
        return "—"
    return f"{(pred - actual) * 100:.2f}"


def render_md_table(rows: Sequence[Dict[str, Any]]) -> str:
    # Match the paper-style table: compare FAIRScore (Pred) vs label-based (Actual) metrics.
    header = ["Dataset", "N", "Pred P", "Actual P", "ΔP", "Pred R", "Actual R", "ΔR"]
    lines: List[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---", "---:"] + ["---:"] * (len(header) - 2)) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(int(row.get("n", 0))),
                    str(row.get("pred_p", "—")),
                    str(row.get("actual_p", "—")),
                    str(row.get("delta_p", "—")),
                    str(row.get("pred_r", "—")),
                    str(row.get("actual_r", "—")),
                    str(row.get("delta_r", "—")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"

def compute_actual_cell_metrics(
    *,
    example_ids: Iterable[str],
    preds_by_id: Dict[str, List[Tuple[int, int]]],
    gold_by_id: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, Any]:
    tp = 0
    pred_total = 0
    gt_total = 0
    n = 0
    for ex_id in example_ids:
        pred = _normalize_cells(preds_by_id.get(ex_id) or [])
        gt = _normalize_cells(gold_by_id.get(ex_id) or [])
        if not pred or not gt:
            continue
        pred_set = set(pred)
        gt_set = set(gt)
        tp += len(pred_set & gt_set)
        pred_total += len(pred_set)
        gt_total += len(gt_set)
        n += 1
    return {
        "n": int(n),
        "cell_precision": (tp / pred_total) if pred_total else 0.0,
        "cell_recall": (tp / gt_total) if gt_total else 0.0,
        "tp": int(tp),
        "pred_total": int(pred_total),
        "gt_total": int(gt_total),
    }


def make_llm_caller(backend: str, model_name: str) -> Callable[[str], str]:
    backend = (backend or "openai").lower().strip()
    from LLM import Call_OpenAI, Call_Gemini, Call_DeepSeek  # type: ignore

    if backend == "openai":
        return Call_OpenAI(model_name).call
    if backend == "gemini":
        return Call_Gemini(model_name).call
    if backend == "deepseek":
        return Call_DeepSeek(model_name).call
    raise ValueError(f"Unknown backend: {backend}")


def _ensure_answer_facts(
    *,
    llm: Callable[[str], str],
    prompts: Dict[str, str],
    example_id: str,
    answer: str,
    cache: Dict[str, Any],
    cache_path: Path,
    lock: threading.Lock,
    sleep_s: float,
    fail_fast: bool,
) -> Optional[List[str]]:
    with lock:
        rec = cache.get(example_id)
    if rec is None:
        msg = f"{prompts['answer']}\n\nSentence : {answer}\n\n<Output>:\n"
        try:
            resp = llm(msg) or ""
        except Exception as e:
            if fail_fast:
                raise
            with lock:
                cache[example_id] = {"error": str(e)}
                _save_json(cache_path, cache)
            return None
        with lock:
            # Another worker may have written it while we were calling the LLM.
            if example_id not in cache:
                cache[example_id] = {"raw": resp, "facts": _parse_numbered_facts(resp)}
                _save_json(cache_path, cache)
        if sleep_s:
            time.sleep(sleep_s)

    with lock:
        rec = cache.get(example_id)
    if isinstance(rec, dict) and rec.get("error"):
        return None
    facts = rec.get("facts") if isinstance(rec, dict) else None
    return facts if isinstance(facts, list) else []


def _ensure_cell_facts(
    *,
    llm: Callable[[str], str],
    prompts: Dict[str, str],
    example_id: str,
    question: str,
    table: List[List[Any]],
    cells_table: List[Tuple[int, int]],
    cache: Dict[str, Any],
    cache_path: Path,
    lock: threading.Lock,
    sleep_s: float,
    fail_fast: bool,
) -> Optional[List[str]]:
    if not cells_table:
        return None
    with lock:
        rec = cache.get(example_id)
    if rec is None:
        _headers, table_block = build_attributed_table_block(table, cells_table_indices=cells_table)
        msg = (
            f"{prompts['cells']}\n\nInput:\n{table_block}\n\nQuestion: {question}\n\n"
            f"Attributed Cells: {json.dumps(cells_table, ensure_ascii=False)}\n\n<Output>\n"
        )
        try:
            resp = llm(msg) or ""
        except Exception as e:
            if fail_fast:
                raise
            with lock:
                cache[example_id] = {"error": str(e)}
                _save_json(cache_path, cache)
            return None
        with lock:
            if example_id not in cache:
                cache[example_id] = {
                    "raw": resp,
                    "facts": _parse_numbered_facts(resp),
                    "cells": cells_table,
                    "table_block": table_block,
                    "question": question,
                }
                _save_json(cache_path, cache)
        if sleep_s:
            time.sleep(sleep_s)
    else:
        # Backfill the table block for older cache entries (no extra LLM call).
        if isinstance(rec, dict) and "table_block" not in rec:
            used_cells = rec.get("cells")
            if not isinstance(used_cells, list):
                used_cells = cells_table
            _headers, table_block = build_attributed_table_block(table, cells_table_indices=used_cells)
            with lock:
                rec2 = cache.get(example_id)
                if isinstance(rec2, dict) and "table_block" not in rec2:
                    rec2.setdefault("cells", used_cells)
                    rec2["table_block"] = table_block
                    rec2.setdefault("question", question)
                    _save_json(cache_path, cache)

    with lock:
        rec = cache.get(example_id)
    if isinstance(rec, dict) and rec.get("error"):
        return None
    facts = rec.get("facts") if isinstance(rec, dict) else None
    return facts if isinstance(facts, list) else []


def _ensure_alignment(
    *,
    llm: Callable[[str], str],
    prompts: Dict[str, str],
    dataset_name: str,
    example_id: str,
    question: str,
    table_title: str,
    headers: List[str],
    set1_facts: List[str],
    set2_facts: List[str],
    cache: Dict[str, Any],
    cache_path: Path,
    lock: threading.Lock,
    sleep_s: float,
    max_retries: int,
    fail_fast: bool,
) -> Optional[Dict[str, int]]:
    if not set1_facts or not set2_facts:
        return None
    with lock:
        rec = cache.get(example_id)
    if rec is not None:
        if isinstance(rec, dict) and isinstance(rec.get("counts"), dict):
            return rec["counts"]
        return None

    set1_txt = _format_numbered(set1_facts)
    set2_txt = _format_numbered(set2_facts)

    base_msg = (
        f"{prompts['align']}\n\n"
        f"Input :\nSet-1\n{set1_txt}\n\nSet-2\n{set2_txt}\n\n"
        f"Question : {question}\nTable Title: {table_title}\nColumn Headers: {json.dumps(headers, ensure_ascii=False)}\n\n"
        "<Output>:\n"
    )

    last_resp = ""
    parsed: Optional[Dict[str, int]] = None
    for attempt in range(max_retries + 1):
        msg = base_msg
        if attempt > 0:
            msg = (
                base_msg
                + "\nIMPORTANT: Return ONLY these 4 lines with integer values:\n"
                + "Total Number of facts in Set-1 : <int>\n"
                + "Total number of facts in Set-2 : <int>\n"
                + "Number of atomic facts in 1 covered by 2: <int>\n"
                + "Number of atomic facts in 2 that are unnecessary: <int>\n"
            )
        try:
            last_resp = llm(msg) or ""
        except Exception as e:
            if attempt >= max_retries:
                if fail_fast:
                    raise
                with lock:
                    cache[example_id] = {"error": str(e)}
                    _save_json(cache_path, cache)
                return None
            if sleep_s:
                time.sleep(sleep_s)
            continue

        parsed = parse_alignment_counts(last_resp)
        if parsed is not None:
            break
        if sleep_s:
            time.sleep(sleep_s)

    if parsed is None:
        if fail_fast:
            raise RuntimeError(f"Failed to parse alignment output for {dataset_name}:{example_id}\n{last_resp}")
        with lock:
            cache[example_id] = {"error": "parse_failed", "raw": last_resp}
            _save_json(cache_path, cache)
        return None

    # Use our own fact counts for totals for stability (but keep raw output too).
    parsed = dict(parsed)
    parsed["total_set1"] = len(set1_facts)
    parsed["total_set2"] = len(set2_facts)

    with lock:
        cache[example_id] = {"counts": parsed, "raw": last_resp}
        _save_json(cache_path, cache)
    if sleep_s:
        time.sleep(sleep_s)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FAIRScore (reference-less) evaluation for TraceBack outputs on AITQA/FetaQA/ToTTo."
    )
    parser.add_argument("--datasets", default="aitqa,fetaqa,totto", help="Comma-separated subset: aitqa,fetaqa,totto")
    parser.add_argument("--backend", default="openai", choices=["openai", "gemini", "deepseek"], help="LLM backend")
    parser.add_argument("--model", default="gpt-4o", help="Model name for the selected backend")
    parser.add_argument("--cells", default="pred", choices=["pred", "gold", "both"], help="Which cell set to score")
    parser.add_argument(
        "--pred-row-offset",
        type=int,
        default=1,
        help="Offset added to predicted row indices to match table row ids (default: 1)",
    )
    parser.add_argument("--outdir", default=str(ROOT_DIR / "results" / "fairscore"), help="Cache directory")
    parser.add_argument(
        "--summary-md",
        default=str(ROOT_DIR / "results" / "eval" / "metrics_fairscore.md"),
        help="Summary Markdown output path",
    )
    parser.add_argument(
        "--summary-json",
        default=str(ROOT_DIR / "results" / "eval" / "metrics_fairscore.json"),
        help="Summary JSON output path",
    )

    parser.add_argument(
        "--preds",
        default="",
        help=(
            "Unified prediction JSON path used for all datasets (e.g., CITEBENCH baseline outputs). "
            "If empty, uses per-dataset --aitqa-preds/--fetaqa-preds/--totto-preds."
        ),
    )
    parser.add_argument(
        "--pred-tag",
        default="",
        help=(
            "Optional tag to namespace pred-dependent caches (pred_cell_facts/pred_alignment). "
            "Recommended when comparing multiple pred files with the same backend+model."
        ),
    )
    parser.add_argument("--aitqa-preds", default=str(DATASETS["aitqa"]["pred_default"]), help="AITQA prediction JSON path")
    parser.add_argument(
        "--fetaqa-preds", default=str(DATASETS["fetaqa"]["pred_default"]), help="FetaQA prediction JSON path"
    )
    parser.add_argument("--totto-preds", default=str(DATASETS["totto"]["pred_default"]), help="ToTTo prediction JSON path")

    parser.add_argument("--limit", type=int, default=0, help="Max examples per dataset (0 = no limit)")
    parser.add_argument("--start", type=int, default=0, help="Start index in the prediction list")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between LLM calls")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries when alignment output can't be parsed")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker threads (default: 1)")
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=1,
        help="Max concurrent LLM requests across all workers (default: 1)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Ignore caches and recompute")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error")
    args = parser.parse_args()

    datasets = [d.strip().lower() for d in str(args.datasets).split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"Unknown dataset: {d}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = {
        "answer": _load_text(ROOT_DIR / "Prompts" / "atomic_facts_from_answer.txt"),
        "cells": _load_text(ROOT_DIR / "Prompts" / "atomic_facts_from_predicted_cells.txt"),
        "align": _load_text(ROOT_DIR / "Prompts" / "alignment.txt"),
    }

    # Thread-safe LLM wrapper with bounded parallelism + per-thread clients.
    max_inflight = max(1, int(getattr(args, "max_inflight", 1)))
    semaphore = threading.BoundedSemaphore(value=max_inflight)
    thread_local = threading.local()

    def llm(prompt: str) -> str:
        with semaphore:
            if not hasattr(thread_local, "client"):
                thread_local.client = make_llm_caller(str(args.backend), str(args.model))
            return thread_local.client(prompt)

    summary_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {
        "backend": args.backend,
        "model": args.model,
        "cells": args.cells,
        "pred_row_offset": int(args.pred_row_offset),
        "datasets": {},
    }

    for dataset in datasets:
        cfg = DATASETS[dataset]
        preds_override = str(getattr(args, "preds", "") or "").strip()
        pred_path = Path(preds_override) if preds_override else Path(getattr(args, f"{dataset}_preds"))
        if not pred_path.is_absolute():
            pred_path = (ROOT_DIR / pred_path).resolve()
        data_path = Path(cfg["data_path"])

        if not data_path.exists():
            raise SystemExit(f"Missing dataset file: {data_path}")
        if not pred_path.exists():
            raise SystemExit(f"Missing prediction file: {pred_path}")

        run_id = f"{dataset}__{args.backend}__{args.model}".replace("/", "__")
        ds_out = outdir / run_id
        ds_out.mkdir(parents=True, exist_ok=True)

        pred_tag = _sanitize_tag(str(getattr(args, "pred_tag", "") or ""))
        if not pred_tag and preds_override:
            pred_tag = _sanitize_tag(pred_path.stem)
        pred_tag_suffix = f"__{pred_tag}" if pred_tag else ""

        cache_paths = {
            "answer_facts": ds_out / "answer_facts.json",
            "pred_cell_facts": ds_out / f"pred_cell_facts{pred_tag_suffix}.json",
            "gold_cell_facts": ds_out / "gold_cell_facts.json",
            "pred_alignment": ds_out / f"pred_alignment{pred_tag_suffix}.json",
            "gold_alignment": ds_out / "gold_alignment.json",
        }
        caches = {
            "answer_facts": {} if args.overwrite else _load_json_or_empty(cache_paths["answer_facts"]),
            "pred_cell_facts": {} if args.overwrite else _load_json_or_empty(cache_paths["pred_cell_facts"]),
            "gold_cell_facts": {} if args.overwrite else _load_json_or_empty(cache_paths["gold_cell_facts"]),
            "pred_alignment": {} if args.overwrite else _load_json_or_empty(cache_paths["pred_alignment"]),
            "gold_alignment": {} if args.overwrite else _load_json_or_empty(cache_paths["gold_alignment"]),
        }

        preds = _load_json(pred_path)
        if not isinstance(preds, list):
            raise SystemExit(f"Expected a JSON list in {pred_path}")

        id_key_pred = str(cfg["id_key_pred"])
        if _is_citebench_pred_list(preds):
            preds = _coerce_citebench_preds(preds, dataset=dataset, id_key_pred=id_key_pred)
            if not preds:
                raise SystemExit(f"No {dataset} records found in CITEBENCH predictions: {pred_path}")
            print(f"[Info] Coerced CITEBENCH predictions for {dataset}: {len(preds)} records")

        # Dataset examples indexed by id (string)
        id_key_data = str(cfg["id_key_data"])
        ex_by_id: Dict[str, Dict[str, Any]] = {}
        for ex in _iter_jsonl(data_path):
            if id_key_data not in ex:
                continue
            ex_by_id[str(ex[id_key_data])] = ex

        limit = int(args.limit) if int(args.limit) > 0 else None
        start = max(0, int(args.start))

        cache_locks = {k: threading.Lock() for k in cache_paths.keys()}

        # Preselect records to process (honor --start/--limit over the prediction list indices).
        selected: List[Dict[str, Any]] = []
        for idx, rec in enumerate(preds):
            if idx < start:
                continue
            if limit is not None and len(selected) >= limit:
                break
            if not isinstance(rec, dict) or id_key_pred not in rec:
                continue
            ex_id = str(rec[id_key_pred])
            if ex_id not in ex_by_id:
                continue
            selected.append(rec)

        def _process_one(rec: Dict[str, Any]) -> None:
            example_id = str(rec[id_key_pred])
            ex = ex_by_id.get(example_id)
            if not ex:
                return

            table = ex.get(cfg["table_key"]) or []
            if not isinstance(table, list) or not table or not isinstance(table[0], list):
                return
            headers = [str(h) for h in table[0]]

            question = str(ex.get(cfg["question_key"], ""))
            answer = _to_answer_str(ex.get(cfg["answer_key"]))
            table_title = str(cfg["title_fn"](ex))

            set1_facts = _ensure_answer_facts(
                llm=llm,
                prompts=prompts,
                example_id=example_id,
                answer=answer,
                cache=caches["answer_facts"],
                cache_path=cache_paths["answer_facts"],
                lock=cache_locks["answer_facts"],
                sleep_s=float(args.sleep),
                fail_fast=bool(args.fail_fast),
            )
            if not set1_facts:
                return

            # Pred cells: stored as 0-based over data rows; convert to table indices (header row is 0).
            pred_cells = _normalize_cells(rec.get("result") or [])
            pred_cells_table = [(r + int(args.pred_row_offset), c) for r, c in pred_cells]

            if args.cells in ("pred", "both"):
                set2_facts = _ensure_cell_facts(
                    llm=llm,
                    prompts=prompts,
                    example_id=example_id,
                    question=question,
                    table=table,
                    cells_table=pred_cells_table,
                    cache=caches["pred_cell_facts"],
                    cache_path=cache_paths["pred_cell_facts"],
                    lock=cache_locks["pred_cell_facts"],
                    sleep_s=float(args.sleep),
                    fail_fast=bool(args.fail_fast),
                )
                if set2_facts:
                    _ensure_alignment(
                        llm=llm,
                        prompts=prompts,
                        dataset_name=dataset,
                        example_id=example_id,
                        question=question,
                        table_title=table_title,
                        headers=headers,
                        set1_facts=set1_facts,
                        set2_facts=set2_facts,
                        cache=caches["pred_alignment"],
                        cache_path=cache_paths["pred_alignment"],
                        lock=cache_locks["pred_alignment"],
                        sleep_s=float(args.sleep),
                        max_retries=int(args.max_retries),
                        fail_fast=bool(args.fail_fast),
                    )

            if args.cells in ("gold", "both"):
                gold_cells_table = _normalize_cells(ex.get(cfg["gold_cells_key"]))
                set2_facts = _ensure_cell_facts(
                    llm=llm,
                    prompts=prompts,
                    example_id=example_id,
                    question=question,
                    table=table,
                    cells_table=gold_cells_table,
                    cache=caches["gold_cell_facts"],
                    cache_path=cache_paths["gold_cell_facts"],
                    lock=cache_locks["gold_cell_facts"],
                    sleep_s=float(args.sleep),
                    fail_fast=bool(args.fail_fast),
                )
                if set2_facts:
                    _ensure_alignment(
                        llm=llm,
                        prompts=prompts,
                        dataset_name=dataset,
                        example_id=example_id,
                        question=question,
                        table_title=table_title,
                        headers=headers,
                        set1_facts=set1_facts,
                        set2_facts=set2_facts,
                        cache=caches["gold_alignment"],
                        cache_path=cache_paths["gold_alignment"],
                        lock=cache_locks["gold_alignment"],
                        sleep_s=float(args.sleep),
                        max_retries=int(args.max_retries),
                        fail_fast=bool(args.fail_fast),
                    )

        workers = max(1, int(getattr(args, "workers", 1)))
        processed = 0
        if workers == 1:
            for rec in selected:
                _process_one(rec)
                processed += 1
                if processed % 10 == 0:
                    print(f"[{dataset}] processed {processed}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_process_one, rec) for rec in selected]
                for _f in as_completed(futures):
                    processed += 1
                    if processed % 10 == 0:
                        print(f"[{dataset}] processed {processed}")

        def _collect_counts(align_cache: Dict[str, Any]) -> List[Dict[str, int]]:
            out: List[Dict[str, int]] = []
            scored_ids: List[str] = []
            for _id, rec in align_cache.items():
                if not isinstance(rec, dict) or not isinstance(rec.get("counts"), dict):
                    continue
                c = rec["counts"]
                try:
                    out.append(
                        {
                            "total_set1": int(c.get("total_set1", 0)),
                            "total_set2": int(c.get("total_set2", 0)),
                            "covered": int(c.get("covered", 0)),
                            "unnecessary": int(c.get("unnecessary", 0)),
                        }
                    )
                    scored_ids.append(str(_id))
                except Exception:
                    continue
            return out, scored_ids

        pred_counts, pred_scored_ids = _collect_counts(caches["pred_alignment"]) if args.cells in ("pred", "both") else ([], [])
        gold_counts, gold_scored_ids = _collect_counts(caches["gold_alignment"]) if args.cells in ("gold", "both") else ([], [])

        pred_metrics = compute_fairscore_from_counts(pred_counts) if pred_counts else None
        gold_metrics = compute_fairscore_from_counts(gold_counts) if gold_counts else None

        # "Actual" is label-based cell-level P/R computed on the same example set as FAIRScore counts.
        preds_by_id: Dict[str, List[Tuple[int, int]]] = {}
        gold_by_id: Dict[str, List[Tuple[int, int]]] = {}
        for rec in preds:
            if not isinstance(rec, dict) or id_key_pred not in rec:
                continue
            ex_id = str(rec[id_key_pred])
            raw_pred_cells = _normalize_cells(rec.get("result") or [])
            preds_by_id[ex_id] = [(r + int(args.pred_row_offset), c) for r, c in raw_pred_cells]
        for ex_id, ex in ex_by_id.items():
            gold_by_id[ex_id] = _normalize_cells(ex.get(cfg["gold_cells_key"]))

        actual = None
        if pred_scored_ids:
            actual = compute_actual_cell_metrics(
                example_ids=pred_scored_ids,
                preds_by_id=preds_by_id,
                gold_by_id=gold_by_id,
            )

        # Use the common N (pred_scored_ids filtered by having pred+gold).
        n = int(actual.get("n", 0)) if actual else int((pred_metrics or {}).get("n", 0) or 0)

        pred_p = pred_metrics.get("fairscore_precision") if pred_metrics else None
        pred_r = pred_metrics.get("fairscore_recall") if pred_metrics else None
        actual_p = actual.get("cell_precision") if actual else None
        actual_r = actual.get("cell_recall") if actual else None

        summary_rows.append(
            {
                "dataset": dataset.upper(),
                "n": n,
                "pred_p": _fmt_pct(pred_p),
                "actual_p": _fmt_pct(actual_p),
                "delta_p": _fmt_delta_pct(pred_p, actual_p),
                "pred_r": _fmt_pct(pred_r),
                "actual_r": _fmt_pct(actual_r),
                "delta_r": _fmt_delta_pct(pred_r, actual_r),
            }
        )
        summary_json["datasets"][dataset] = {
            "pred": pred_metrics,
            "gold": gold_metrics,
            "actual_cell": actual,
            "pred_alignment_cache": str(cache_paths["pred_alignment"]),
            "gold_alignment_cache": str(cache_paths["gold_alignment"]),
            "pred_scored_ids": pred_scored_ids,
            "gold_scored_ids": gold_scored_ids,
        }

    md = render_md_table(summary_rows)
    md_path = Path(args.summary_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    print(md, end="")
    print(f"Wrote {md_path}")

    json_path = Path(args.summary_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
