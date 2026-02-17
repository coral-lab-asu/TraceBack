import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                # skip malformed line
                continue
    return items


def load_citebench(path: Path) -> List[Dict]:
    data = read_json(path)
    assert isinstance(data, list), "CITEBENCH.json must be a JSON array"
    return data


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def build_input_block(item: Dict) -> Dict[str, str]:
    """
    Build the formatted blocks used by the prompt templates.
    CITEBENCH schema keys observed: example_id, dataset, table_id, qid, table, question, answer, highlighted_cells
    """
    table = item.get("table")
    if not table or not isinstance(table, list) or not table[0]:
        raise ValueError("Invalid table format; expected first row as header")

    column_headers = table[0]
    table_rows = table[1:]
    # Single question per item; wrap into numbered list to match templates
    questions_list = f"1. {item.get('question', '').strip()}"
    # Answers may be list or str
    ans = item.get("answer")
    if isinstance(ans, list):
        original_answer = "\n".join(str(a) for a in ans)
    else:
        original_answer = str(ans) if ans is not None else ""

    return {
        "column_headers": json.dumps(column_headers, ensure_ascii=False),
        "table_title": str(item.get("table_id", item.get("qid", ""))),
        "table_rows_without_header": json.dumps(table_rows, ensure_ascii=False),
        "questions_list": questions_list,
        "original_answer": original_answer,
    }


def fill_prompt(prefix: str, blocks: Dict[str, str], cot: bool = False, include_title: bool = False) -> str:
    base = [prefix, "", "Input :"]
    base.append(f"<Relevant-Columns>: {blocks['column_headers']}")
    if include_title:
        base.append(f"<Table Title>: {blocks['table_title']}")
    base.append(f"<Table Rows>: {blocks['table_rows_without_header']}")
    base.append("<Questions>:")
    base.append(blocks["questions_list"])
    base.append("<Original-Answer>:")
    base.append(blocks["original_answer"])
    base.append("")
    if cot:
        base.append("Brief Reasoning (no parentheses):")
        base.append("")
        base.append("Final Answer:")
    else:
        base.append("Output:")
    base.append("")
    return "\n".join(base)


_TUPLE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def parse_cell_tuples(text: str) -> List[List[int]]:
    """Extract (row,col) integer tuples with priority to the last top-level bracketed block.

    - Prefer parsing the last substring bounded by '[' and the following matching ']'.
    - Fallback: scan the whole text.
    - Deduplicate while preserving order.
    """
    # Try to locate the last bracketed block to avoid picking tuples from reasoning
    start, end = -1, -1
    end = text.rfind(']')
    if end != -1:
        start = text.rfind('[', 0, end)
    segment = text[start:end + 1] if (start != -1 and end != -1 and start < end) else text

    seen = set()
    results: List[List[int]] = []
    for m in _TUPLE_RE.finditer(segment):
        r = int(m.group(1))
        c = int(m.group(2))
        key = (r, c)
        if key not in seen:
            seen.add(key)
            results.append([r, c])
    return results


def sanitize_model_id(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _build_openai_client():
    try:
        # Reuse the repo's lightweight .env loader if available.
        from LLM import _load_dotenv_from_root  # type: ignore
        _load_dotenv_from_root()
    except Exception:
        pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set it or add it to .env at repo root)")
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package is required for --backend openai (pip install openai)") from e
    return OpenAI(api_key=api_key)


def _openai_chat(
    client,
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
) -> str:
    delay = 1.0
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "developer", "content": "Follow instructions exactly. Output only what the prompt asks for."},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"OpenAI call failed after {max_retries} retries: {last_err}")


def _truncate(s: str, n: int) -> str:
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + "\n... <truncated>"


def _parse_indices_arg(s: str) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


def debug_print_example(idx: int,
                        item: Dict,
                        blocks: Dict[str, str],
                        prompt: str,
                        text: str,
                        tuples: List[List[int]],
                        tok,
                        max_chars: int):
    print(f"\n===== DEBUG[example {idx}] =====")
    print(f"meta: dataset={item.get('dataset')}, qid={item.get('qid')}, example_id={item.get('example_id')}")
    print(f"headers: {blocks['column_headers']}")
    # show first 2 table rows
    try:
        import json as _json
        rows = _json.loads(blocks['table_rows_without_header'])
        print(f"rows[0:2]: {rows[:2]}")
    except Exception:
        pass
    # prompt stats
    try:
        n_tokens = len(tok(prompt, return_tensors='pt')['input_ids'][0])
    except Exception:
        n_tokens = -1
    print(f"prompt chars={len(prompt)}, tokens={n_tokens}")
    print("--- PROMPT (truncated) ---")
    print(_truncate(prompt, max_chars))
    print("--- RAW OUTPUT (truncated) ---")
    print(_truncate(text, max_chars))
    print(f"parsed tuples: {tuples}")
    print("===== /DEBUG =====\n")


def main():
    # tqdm setup (safe import and print wrapper)
    try:
        from tqdm.auto import tqdm  # type: ignore
        def tprint(msg: str):
            try:
                tqdm.write(msg)
            except Exception:
                print(msg)
    except Exception:
        def tqdm(x=None, **kwargs):
            return x
        def tprint(msg: str):
            print(msg)

    parser = argparse.ArgumentParser(description="Run ICL attribution across models and prompt settings")
    parser.add_argument("--dataset", default="Datasets/CITEBENCH.json")
    parser.add_argument("--backend", default="hf", choices=["hf", "openai"],
                        help="LLM backend for generation (hf=local transformers, openai=API)")
    parser.add_argument("--styles", nargs="*", default=["zero", "zero-cot", "few", "few-cot"],
                        help="Prompt styles: zero, zero-cot, few, few-cot")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model ids. For --backend hf: HF model ids. For --backend openai: OpenAI model names.")
    parser.add_argument("--openai-model", default="gpt-4o",
                        help="Default OpenAI model when --backend openai and --models is not provided")
    parser.add_argument("--openai-temperature", type=float, default=0.0)
    parser.add_argument("--openai-max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=50, help="Max examples to run (0 = all)")
    parser.add_argument("--outdir", default="results/ICL")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--include-title", action="store_true", help="Include <Table Title> in prompts")
    parser.add_argument("--cache-dir", default="", help="Custom cache directory for model/tokenizer downloads")
    parser.add_argument("--verbose", action="store_true", help="Print per-example progress and summaries")
    parser.add_argument("--debug-all", action="store_true", help="Print full debug for all examples")
    parser.add_argument("--debug-indices", default="", help="Comma-separated example indices to debug (e.g. '0,5,10')")
    parser.add_argument("--debug-max-chars", type=int, default=2000, help="Max chars to print for prompt/output in debug")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL (skip already-processed items)")
    args = parser.parse_args()

    if args.models is None:
        if args.backend == "openai":
            args.models = [args.openai_model]
        else:
            args.models = [
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "google/gemma-3-1b-it",
                "google/gemma-3-4b-it",
                "google/gemma-3-12b-it",
            ]

    dataset_path = Path(args.dataset)
    data = load_citebench(dataset_path)
    effective_limit = len(data) if args.limit <= 0 else args.limit
    if args.limit <= 0:
        try:
            # tprint may not exist yet if tqdm import failed above, so fallback to print
            tprint(f"[Info] Using all {len(data)} examples (limit=0)")
        except Exception:
            print(f"[Info] Using all {len(data)} examples (limit=0)")

    prompt_map = {
        "zero": Path("Prompts_2/answer_attr_zero.txt"),
        "zero-cot": Path("Prompts_2/answer_attr_zero_cot.txt"),
        "few": Path("Prompts_2/answer_attr_few.txt"),
        "few-cot": Path("Prompts_2/answer_attr_few_cot.txt"),
    }
    for style in args.styles:
        if style not in prompt_map:
            print(f"[Warn] Unknown style '{style}' — skipping")
            continue
        if not prompt_map[style].exists():
            print(f"[Error] Prompt file missing: {prompt_map[style]}")
            sys.exit(1)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    debug_idx_set = set(_parse_indices_arg(args.debug_indices))

    openai_client = None
    if args.backend == "openai":
        try:
            openai_client = _build_openai_client()
        except Exception as e:
            print(f"[Error] OpenAI backend init failed: {e}")
            sys.exit(1)
    else:
        # Lazy import transformers to allow environments without it to still parse help
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except Exception as e:
            print("[Error] transformers not available. Install with: pip install transformers accelerate torch --upgrade")
            print(e)
            sys.exit(1)

    for model_id in args.models:
        safe_model = sanitize_model_id(model_id)

        tok = None
        gen = None
        model = None
        if args.backend == "hf":
            # Load model/tokenizer/pipeline once per model_id
            tprint(f"\n[Loading] {model_id}")
            try:
                # trust_remote_code is often required for chat templates; harmless for others
                cache_dir = args.cache_dir or None
                token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
                auth_kwargs = {"token": token} if token else {}
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir, **auth_kwargs)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    cache_dir=cache_dir,
                    **auth_kwargs,
                )
                gen = pipeline(
                    task="text-generation",
                    model=model,
                    tokenizer=tok,
                    return_full_text=False,
                )
            except Exception as e:
                tprint(f"[Skip] Could not load model {model_id}: {e}")
                continue
        else:
            tprint(f"\n[OpenAI] Using model={model_id}")

        # Process all styles using the same loaded model
        for style in args.styles:
            prompt_path = prompt_map.get(style)
            if prompt_path is None:
                continue
            prefix = load_text(prompt_path)
            cot = style.endswith("cot")

            out_dir = Path(args.outdir) / safe_model
            ensure_dir(out_dir)
            out_file = out_dir / f"{style}.json"
            out_file_jsonl = out_dir / f"{style}.jsonl"
            # Build processed set if resuming
            processed_keys = set()
            if args.resume and out_file_jsonl.exists():
                existing = read_jsonl(out_file_jsonl)
                for rec in existing:
                    key = (rec.get("dataset"), rec.get("example_id", rec.get("qid")))
                    processed_keys.add(key)
                tprint(f"[Resume] Found {len(processed_keys)} previously processed records in {out_file_jsonl}")
            results: List[Dict] = []

            data_slice = data[: effective_limit]
            iterator = enumerate(tqdm(data_slice, total=len(data_slice), desc=f"{safe_model} | {style}", leave=False))
            # Open JSONL for appends
            f_jsonl = out_file_jsonl.open("a", encoding="utf-8")
            for i, item in iterator:
                key = (item.get("dataset"), item.get("example_id", item.get("qid")))
                if processed_keys and key in processed_keys:
                    if args.verbose:
                        tprint(f"[{i}] skip (resume) dataset={key[0]} example_id={key[1]}")
                    continue
                blocks = build_input_block(item)
                prompt = fill_prompt(prefix, blocks, cot=cot, include_title=args.include_title)
                if args.verbose:
                    q_preview = item.get('question', '').replace('\n', ' ')[:120]
                    tprint(f"[{i}] generating | dataset={item.get('dataset')} qid={item.get('qid')} question='{q_preview}...' ")
                try:
                    if args.backend == "hf":
                        assert gen is not None
                        # If tokenizer has a chat template, prefer it for more stable formatting
                        prompt_text = prompt
                        try:
                            if tok is not None and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
                                messages = [
                                    {"role": "system", "content": "You are a helpful, instruction-following assistant. Output only the final answer line exactly in the specified format."},
                                    {"role": "user", "content": prompt},
                                ]
                                prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        except Exception:
                            prompt_text = prompt
                        out = gen(prompt_text, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0, temperature=args.temperature, top_p=args.top_p)
                        text = out[0]["generated_text"] if isinstance(out, list) else str(out)
                    else:
                        assert openai_client is not None
                        text = _openai_chat(
                            openai_client,
                            model=model_id,
                            prompt=prompt,
                            temperature=args.openai_temperature,
                            max_tokens=args.openai_max_tokens,
                        )
                except Exception as e:
                    text = f"[GENERATION_ERROR] {e}"

                tuples = parse_cell_tuples(text)
                if args.verbose:
                    tprint(f"[{i}] parsed tuples: {tuples}")
                if args.debug_all or (i in debug_idx_set):
                    debug_print_example(i, item, blocks, prompt, text, tuples, tok, args.debug_max_chars)
                rec = {
                    "dataset": item.get("dataset"),
                    "qid": item.get("qid"),
                    "example_id": item.get("example_id"),
                    "result": tuples,
                    "model": model_id,
                    "prompt_style": style,
                    "raw_output": text,
                }
                results.append(rec)
                # incremental write to JSONL
                try:
                    f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_jsonl.flush()
                    try:
                        os.fsync(f_jsonl.fileno())
                    except Exception:
                        pass
                except Exception:
                    pass
            try:
                f_jsonl.close()
            except Exception:
                pass
            # Write aggregated JSON from JSONL (includes previous + current)
            all_records = read_jsonl(out_file_jsonl)
            with out_file.open("w", encoding="utf-8") as f:
                json.dump(all_records, f, ensure_ascii=False, indent=2)
            tprint(f"[Saved] {out_file} ({len(results)} examples)")

        # Cleanup: free GPU/CPU memory before next model
        if args.backend == "hf":
            try:
                import gc
                del gen
                del model
                del tok
                gc.collect()
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass


if __name__ == "__main__":
    main()
