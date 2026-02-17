import argparse
import importlib.machinery as _machinery
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hack: stub out torchvision so that transformers' text models (RoBERTa MNLI)
# don't try to load the real, incompatible torchvision package on the cluster.
# We only need text NLI, so a minimal stub is sufficient.
# ---------------------------------------------------------------------------
if "torchvision" not in sys.modules:
    tv = types.ModuleType("torchvision")
    transforms_mod = types.ModuleType("torchvision.transforms")

    class _InterpolationMode:
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3
        LANCZOS = 1
        BOX = 4
        HAMMING = 5

    transforms_mod.InterpolationMode = _InterpolationMode
    tv.transforms = transforms_mod

    tv.__spec__ = _machinery.ModuleSpec("torchvision", loader=None)
    transforms_mod.__spec__ = _machinery.ModuleSpec("torchvision.transforms", loader=None)

    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms_mod

# Avoid pulling in real torchvision when importing transformers.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from LLM import Call_OpenAI

# For MySQL-based row filtering (Evidence Span Extractor Agent),
# reuse the helpers from Code_with_rows by adding that directory
# to the import path.
import sys as _sys

_ROOT_DIR = Path(__file__).resolve().parent.parent
_sys.path.append(str(_ROOT_DIR / "Code_with_rows"))
from database import DataBase  # type: ignore
from relevant_row_gen import get_relevant_rows  # type: ignore


def load_citebench(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("CITEBENCH.json must be a JSON array")
    return data


def make_unique(headers: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    unique: List[str] = []
    for h in headers:
        counts[h] = counts.get(h, 0) + 1
        if counts[h] == 1:
            unique.append(h)
        else:
            unique.append(f"{h}_{counts[h]}")
    return unique


def convert_to_df(table_rows: List[List[Any]]) -> pd.DataFrame:
    """Convert raw table (including header row) to a DataFrame with row_id."""
    max_cols = max(len(row) for row in table_rows)
    padded_rows = [row + [""] * (max_cols - len(row)) for row in table_rows]
    raw_header = padded_rows[0]
    header = make_unique([str(h) for h in raw_header])
    padded_rows = [header] + padded_rows[1:]
    df = pd.DataFrame(padded_rows, columns=header)
    df.insert(0, "row_id", list(range(len(padded_rows))))
    # Best-effort type coercion
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


def col_filter_df(cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        # If some requested cols are missing, ignore them
        cols = [c for c in cols if c in df.columns]
    if not cols:
        return df
    return df[["row_id"] + cols]


def load_subquery_prompt() -> str:
    path = Path("Prompts/subquery_prompt.txt")
    return path.read_text(encoding="utf-8").strip()


def load_relevant_cols_prompt() -> str:
    path = Path("Prompts/relevant_cols.txt")
    return path.read_text(encoding="utf-8").strip()


def load_answer_attr_prompt() -> str:
    # We reuse the answer attribution prompt but treat it as
    # sub-query attribution (no <Original-Answer> field).
    path = Path("Prompts/answer_attribution_aitqa.txt")
    return path.read_text(encoding="utf-8").strip()


def parse_answer_list(ans: Any) -> str:
    if isinstance(ans, list):
        return " / ".join(str(a) for a in ans)
    return str(ans)


def call_gpt(system_prompt: str, user_prompt: str, model_name: str = "gpt-4o") -> str:
    client = Call_OpenAI(model=model_name)
    prompt = system_prompt + "\n\n" + user_prompt
    return client.call(prompt)


def split_subqueries(raw: str) -> List[str]:
    """Split LLM output into a list of sub-questions."""
    if not raw:
        return []
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    subs: List[str] = []
    for ln in lines:
        # Remove leading numbering like "1." or "- "
        ln = re.sub(r"^[\-\*\d\.\)\s]+", "", ln)
        if len(ln) < 3:
            continue
        subs.append(ln)
    return subs


class NLIEntailmentFilter:
    def __init__(self, model_name: str = "roberta-large-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = ["entailment", "neutral", "contradiction"]

    def filter(self, question: str, subqueries: List[str], threshold: float = 0.5) -> List[str]:
        kept: List[str] = []
        for sq in subqueries:
            inputs = self.tokenizer.encode_plus(
                question,
                sq,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = F.softmax(logits, dim=1)
            ent_prob = probs[0][self.labels.index("entailment")].item()
            if ent_prob >= threshold:
                kept.append(sq)
        return kept or subqueries


_TUPLE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def parse_cell_tuples(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for m in _TUPLE_RE.finditer(text):
        r = int(m.group(1))
        c = int(m.group(2))
        key = (r, c)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def map_to_original_cols(
    cells: List[Tuple[int, int]], column_headers: List[str], relevant_cols: List[str]
) -> List[List[int]]:
    col_dict = {name: idx for idx, name in enumerate(column_headers)}
    mapped: List[List[int]] = []
    for r, c in cells:
        if c < 0 or c >= len(relevant_cols):
            continue
        header_name = relevant_cols[c]
        if header_name not in col_dict:
            continue
        mapped.append([r, col_dict[header_name]])
    return mapped


def run_traceback_on_example(
    item: Dict[str, Any],
    nli_filter: NLIEntailmentFilter,
    model_name: str = "gpt-4o",
) -> Dict[str, Any]:
    dataset = str(item.get("dataset"))
    qid = str(item.get("qid"))
    example_id = int(item.get("example_id"))
    table = item.get("table") or []
    if not table or not table[0]:
        return {
            "dataset": dataset,
            "qid": qid,
            "example_id": example_id,
            "result": [],
        }

    column_headers = [str(h) for h in table[0]]
    table_rows = table[1:]
    question = str(item.get("question", ""))
    answer_str = parse_answer_list(item.get("answer", []))
    table_title = f"{dataset}-{qid}"

    # ---- Step 1: Column Relevance ----
    rel_cols_prompt = load_relevant_cols_prompt()
    rel_user = (
        f"{rel_cols_prompt}\nInput : \n"
        f"Table Title: {table_title}\n"
        f"<Column Names>: {column_headers}\n"
        f"Question: {question}\n"
        f"Answer: {answer_str}\n\nOutput :\n"
    )
    rel_text = call_gpt("", rel_user, model_name=model_name)
    m = re.search(r"(?<=<Relevant Columns>: )(.*)", rel_text, re.IGNORECASE)
    if m:
        cols_expr = m.group(1)
        try:
            relevant_cols = list(eval(cols_expr))
        except Exception:
            relevant_cols = column_headers
    else:
        relevant_cols = column_headers

    # ---- Step 2: Evidence Rows (MySQL-based row filtering when available) ----
    # Convert table to DataFrame with row_id and attempt SQL-based filtering.
    df = convert_to_df(table)

    # Strict MySQL-based Evidence Span Extractor:
    # if the database or SQL step fails, we let the error propagate
    # instead of silently falling back.
    db = DataBase()
    db.upload_table(table_title, df)
    col_df = col_filter_df(relevant_cols, df)
    row_df = get_relevant_rows(db, table_title, relevant_cols, col_df, question, answer_str)
    if row_df is None or row_df.shape[0] == 0:
        # If SQL returns no rows, fall back to column-filtered table (no evidence rows found).
        row_df = col_df

    pruned_row_ids: List[int] = []
    if "row_id" in row_df.columns:
        pruned_row_ids = [int(r) for r in row_df["row_id"].tolist() if int(r) > 0]
    if not pruned_row_ids:
        # No valid data rows after SQL + fallback; treat as empty evidence.
        pruned_rows: List[List[Any]] = []
        return {
            "dataset": dataset,
            "qid": qid,
            "example_id": example_id,
            "result": [],
            "model": "TraceBack__gpt4o_full",
            "prompt_style": "traceback-full",
            "raw_subqueries": [],
        }
    pruned_rows = [table[r] for r in pruned_row_ids]

    # ---- Step 3: Query Decomposition (using pruned table schema + values) ----
    subq_prompt = load_subquery_prompt()
    # We include a compact text version of the pruned table for the agent.
    rows_as_text = "; ".join(
        f"Row {i+1}: " + ", ".join(str(x) for x in row)
        for i, row in enumerate(pruned_rows)
    )
    subq_user = (
        f"{subq_prompt}\nInput :\n"
        f"Table-Schema :\n"
        f"<Column Names>: {column_headers}\n"
        f"<Filtered-Table-Rows>: {rows_as_text}\n"
        f"Question: {question}\n"
        f"Answer: {answer_str}\n\nOutput :\n"
    )
    raw_subq = call_gpt("", subq_user, model_name=model_name)
    subqueries = split_subqueries(raw_subq)
    subqueries = nli_filter.filter(answer_str, subqueries)

    # ---- Step 4 + 5: Sub-Query Attribution + Final Attribution ----
    attr_prompt = load_answer_attr_prompt()
    final_cells: List[List[int]] = []
    seen_cells = set()

    for sq in subqueries:
        table_json = json.dumps(pruned_rows, ensure_ascii=False)
        relevant_cols_json = json.dumps(relevant_cols, ensure_ascii=False)
        attr_user = (
            f"{attr_prompt}\nInput :\n"
            f"Table-Schema :\n"
            f"<Relevant-Columns>: {relevant_cols_json}\n"
            f"Table Title: {table_title}\n"
            f"<Table Rows>: {table_json}\n"
            f"<Questions>:\n{sq}\n\nOutput :\n"
        )
        raw_attr = call_gpt("", attr_user, model_name=model_name)
        tuples = parse_cell_tuples(raw_attr)
        mapped = map_to_original_cols(tuples, column_headers, relevant_cols)
        for r, c in mapped:
            # r is 0-based over pruned_rows; CITEBENCH uses full-table rows including header
            rr = r + 1  # shift because table_rows excludes header row
            key_rc = (rr, c)
            if key_rc not in seen_cells:
                seen_cells.add(key_rc)
                final_cells.append([rr, c])

    return {
        "dataset": dataset,
        "qid": qid,
        "example_id": example_id,
        "result": final_cells,
        "model": "TraceBack__gpt4o_full",
        "prompt_style": "traceback-full",
        "raw_subqueries": subqueries,
    }


def main():
    parser = argparse.ArgumentParser(description="Full TraceBack pipeline over CITEBENCH.json")
    parser.add_argument("--citebench", default="Datasets/CITEBENCH.json")
    parser.add_argument("--outdir", default="results/TraceBack_full")
    parser.add_argument("--limit", type=int, default=0, help="Max examples (0 = all)")
    parser.add_argument("--datasets", nargs="*", default=["aitqa", "feta", "totto"])
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--nli-threshold", type=float, default=0.5)
    args = parser.parse_args()

    cb_items = load_citebench(Path(args.citebench))
    # Filter by dataset if requested
    allowed = set(args.datasets)
    items = [it for it in cb_items if str(it.get("dataset")) in allowed]
    if args.limit > 0:
        items = items[: args.limit]

    print(f"[Info] Running TraceBack over {len(items)} CITEBENCH examples from {args.datasets}")

    out_dir = Path(args.outdir) / "TraceBack__gpt4o_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "traceback-full.json"

    # If there is an existing result file, load it so we can resume.
    results: List[Dict[str, Any]] = []
    seen_keys = set()
    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                results = existing
                for r in existing:
                    key = (str(r.get("dataset")), str(r.get("qid")), int(r.get("example_id", 0)))
                    seen_keys.add(key)
                print(f"[Info] Loaded {len(results)} existing results from {out_file}")
        except Exception as e:
            print(f"[Warn] Could not load existing results ({e}); starting fresh.")

    # Filter out already-processed examples (by dataset, qid, example_id).
    pending: List[Dict[str, Any]] = []
    for it in items:
        key = (str(it.get("dataset")), str(it.get("qid")), int(it.get("example_id", 0)))
        if key not in seen_keys:
            pending.append(it)

    total = len(items)
    start_idx = len(results)
    print(f"[Info] {len(pending)} examples to process (already have {start_idx}).")

    nli_filter = NLIEntailmentFilter()

    for idx, it in enumerate(pending, start=start_idx + 1):
        print(f"[{idx}/{total}] dataset={it.get('dataset')} qid={it.get('qid')}")
        rec = run_traceback_on_example(it, nli_filter, model_name=args.model)
        results.append(rec)
        # Persist after each example so progress is not lost.
        out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Saved] {out_file} ({len(results)} examples)")


if __name__ == "__main__":
    main()
