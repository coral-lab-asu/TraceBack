import json
import ast
import re
import sys
import time
import types
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


_ROOT_DIR = Path(__file__).resolve().parent


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def sanitize_identifier(text: str, prefix: str = "tbl") -> str:
    """
    Make a best-effort SQL identifier from arbitrary text.
    MySQL allows backticks, but the LLM prompt + downstream parsing is simpler
    if we stick to [A-Za-z0-9_].
    """
    raw = re.sub(r"\W+", "_", str(text).strip())
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        raw = prefix
    if raw[0].isdigit():
        raw = prefix + "_" + raw
    # Keep it reasonably short for MySQL identifier limits.
    return raw[:64]


def make_unique(headers: Sequence[Any]) -> List[str]:
    counts: Dict[str, int] = {}
    unique: List[str] = []
    for h in headers:
        name = str(h)
        counts[name] = counts.get(name, 0) + 1
        if counts[name] == 1:
            unique.append(name)
        else:
            unique.append("%s_%d" % (name, counts[name]))
    return unique


def split_subqueries(raw: str) -> List[str]:
    if not raw:
        return []
    lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        # Remove common bullet/numbering prefixes: "1.", "1)", "-", "*"
        ln = re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip()
        if len(ln) < 3:
            continue
        out.append(ln)
    return out


_PAIR_RE = re.compile(r"[\(\[]\s*(\d+)\s*,\s*(\d+)\s*[\)\]]")


def _normalize_numeric_like(value: Any) -> Any:
    """
    Best-effort normalization for SQL row filtering.

    Handles common numeric-like strings such as:
      - "3,904" -> 3904
      - "$5,813" -> 5813
      - "24%" -> 24
      - "1,23" -> 1.23  (comma-decimal)
    """
    if value is None:
        return value
    if isinstance(value, (int, float)):
        return value

    s = str(value).strip()
    if not s:
        return s
    if not re.search(r"\d", s):
        return value

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    # Strip leading currency symbols and trailing percent.
    s = re.sub(r"^[\$\u20ac\u00a3]", "", s).strip()
    if s.endswith("%"):
        s = s[:-1].strip()

    if "," in s and "." in s:
        # Assume comma is a thousands separator when both are present.
        s = s.replace(",", "")
    elif "," in s:
        # Thousands-grouped like 1,234 or 12,345,678
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):
            s = s.replace(",", "")
        # Decimal-comma like 1,23
        elif re.fullmatch(r"\d+,\d+", s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    if re.fullmatch(r"-?\d+", s):
        n = int(s)
        return -n if neg else n
    if re.fullmatch(r"-?\d*\.\d+", s):
        try:
            n = float(s)
        except Exception:
            return value
        return -n if neg else n
    return value


def parse_cell_pairs(text: str) -> List[Tuple[int, int]]:
    seen = set()
    pairs: List[Tuple[int, int]] = []
    for m in _PAIR_RE.finditer(str(text)):
        r = int(m.group(1))
        c = int(m.group(2))
        key = (r, c)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _maybe_enable_torchvision_stub() -> None:
    """
    Some clusters have an incompatible torchvision that breaks transformers import.
    We only need text NLI, so stub torchvision if it's absent.
    """
    if "torchvision" in sys.modules:
        return
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
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms_mod


class OptionalNLIEntailmentFilter:
    """
    Best-effort RoBERTa-MNLI entailment filter.

    If torch/transformers/model weights are unavailable, it becomes a no-op.
    """

    def __init__(self, model_name: str = "roberta-large-mnli"):
        self.available = False
        self.labels = ["entailment", "neutral", "contradiction"]
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

        try:
            _maybe_enable_torchvision_stub()
            import os as _os

            _os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            self.available = True
        except Exception:
            self.available = False

    def filter(self, premise: str, candidates: Sequence[str], threshold: float = 0.5) -> List[str]:
        if not candidates:
            return []
        if not self.available:
            return list(candidates)
        kept: List[str] = []
        for cand in candidates:
            try:
                inputs = self._tokenizer.encode_plus(
                    str(premise),
                    str(cand),
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                )
                with self._torch.no_grad():
                    logits = self._model(**inputs).logits
                    probs = self._torch.nn.functional.softmax(logits, dim=1)
                ent_prob = probs[0][self.labels.index("entailment")].item()
                if float(ent_prob) >= float(threshold):
                    kept.append(str(cand))
            except Exception:
                continue
        return kept or list(candidates)


_SQL_RE = re.compile(r"<SQL>\s*(.*?)\s*</SQL>", re.IGNORECASE | re.DOTALL)
_CREATE_OR_SELECT_RE = re.compile(r"\b(?:CREATE\s+TABLE|SELECT)\b.*?;", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> Optional[str]:
    if not text:
        return None
    m = _SQL_RE.search(text)
    if m:
        sql = m.group(1).strip()
        # Many prompts/examples use "<SQL>: ... </SQL>" (note the colon).
        # Strip common leading markers so the SQL starts with CREATE/SELECT.
        sql = re.sub(r"^\s*:\s*", "", sql)
        sql = re.sub(r"^\s*SQL\s*:\s*", "", sql, flags=re.IGNORECASE)
        return sql or None
    m2 = _CREATE_OR_SELECT_RE.search(text)
    if m2:
        sql = m2.group(0).strip()
        sql = re.sub(r"^\s*:\s*", "", sql)
        sql = re.sub(r"^\s*SQL\s*:\s*", "", sql, flags=re.IGNORECASE)
        return sql or None
    return None


def _extract_select_from_create(sql: str) -> Optional[str]:
    """
    Convert a CREATE TABLE ... AS SELECT ...; style query into the SELECT ...; part.
    This avoids leaving behind per-example tables in the MySQL DB.
    """
    if not sql:
        return None
    m = re.search(r"\bSELECT\b", sql, re.IGNORECASE)
    if not m:
        return None
    return sql[m.start() :].strip()


def _to_str(x: Any) -> str:
    if isinstance(x, list):
        return " / ".join(str(v) for v in x)
    return str(x)


class TraceBackWorkflowRunner:
    def __init__(
        self,
        *,
        call_llm: Callable[[str], str],
        nli_threshold: float = 0.5,
        enable_row_filtering: bool = True,
        enable_nli_filtering: bool = True,
        require_mysql: bool = False,
        mysql_table_name: str = "traceback_tmp",
    ):
        self.call_llm = call_llm
        self.nli_threshold = float(nli_threshold)
        self.enable_row_filtering = bool(enable_row_filtering)
        self.enable_nli_filtering = bool(enable_nli_filtering)
        self.require_mysql = bool(require_mysql)
        self.mysql_table_name = sanitize_identifier(mysql_table_name, prefix="tbl")

        self._prompt_subq = load_text(_ROOT_DIR / "Prompts" / "subquery_prompt.txt")
        self._prompt_relcols = load_text(_ROOT_DIR / "Prompts" / "relevant_cols.txt")
        self._prompt_relrows = load_text(_ROOT_DIR / "Prompts" / "relevant_rows.txt")
        self._prompt_sub_attr = load_text(_ROOT_DIR / "Prompts" / "traceback_subquery_attribution.txt")
        self._prompt_final = load_text(_ROOT_DIR / "Prompts" / "traceback_final_attribution.txt")

        self._nli = OptionalNLIEntailmentFilter() if self.enable_nli_filtering else None

        self._db_ok = False
        self._db = None
        if self.enable_row_filtering:
            try:
                sys.path.append(str(_ROOT_DIR / "src"))
                from database import DataBase  # type: ignore

                self._db = DataBase()
                self._db_ok = True
            except Exception:
                self._db_ok = False
                self._db = None

    def run_example(
        self,
        *,
        table: List[List[Any]],
        question: str,
        answer: Any,
        table_title: str,
    ) -> Dict[str, Any]:
        return run_traceback_workflow_example(
            table=table,
            question=question,
            answer=answer,
            table_title=table_title,
            call_llm=self.call_llm,
            nli_threshold=self.nli_threshold,
            enable_row_filtering=self.enable_row_filtering,
            require_mysql_for_rows=self.require_mysql,
            _prompt_subq=self._prompt_subq,
            _prompt_relcols=self._prompt_relcols,
            _prompt_relrows=self._prompt_relrows,
            _prompt_sub_attr=self._prompt_sub_attr,
            _prompt_final=self._prompt_final,
            _nli=self._nli,
            _db=self._db if self._db_ok else None,
            _mysql_table_name=self.mysql_table_name,
        )


def run_traceback_workflow_example(
    *,
    table: List[List[Any]],
    question: str,
    answer: Any,
    table_title: str,
    call_llm: Callable[[str], str],
    nli_threshold: float = 0.5,
    enable_row_filtering: bool = True,
    require_mysql_for_rows: bool = False,
    _prompt_subq: Optional[str] = None,
    _prompt_relcols: Optional[str] = None,
    _prompt_relrows: Optional[str] = None,
    _prompt_sub_attr: Optional[str] = None,
    _prompt_final: Optional[str] = None,
    _nli: Optional[OptionalNLIEntailmentFilter] = None,
    _db: Any = None,
    _mysql_table_name: str = "traceback_tmp",
) -> Dict[str, Any]:
    """
    Runs the 5-step TraceBack workflow (best-effort) and returns:
      - result_cells: list of [row_idx, col_idx] where row_idx is 0-based over data rows (header excluded)
      - debug: intermediate artifacts for inspection
    """
    if not table or not table[0]:
        return {"result_cells": [], "debug": {"reason": "empty_table"}}

    t0_total = time.perf_counter()
    timings_sec: Dict[str, float] = {}
    steps: Dict[str, Any] = {}

    def _call(step_name: str, prompt: str) -> str:
        t0 = time.perf_counter()
        out = call_llm(prompt) or ""
        timings_sec[step_name] = time.perf_counter() - t0
        return str(out)

    # Prepare headers + data rows.
    raw_headers = table[0]
    headers = make_unique(raw_headers)
    raw_rows = table[1:]
    ncols = len(headers)
    rows: List[List[Any]] = []
    for r in raw_rows:
        rr = list(r) + [""] * (ncols - len(r)) if len(r) < ncols else list(r)[:ncols]
        rows.append(rr)

    answer_str = _to_str(answer)

    # ---- Step 3 (Query Decomposition): sub-questions ----
    subq_prompt = _prompt_subq or load_text(_ROOT_DIR / "Prompts" / "subquery_prompt.txt")
    subq_user = (
        "%s\nInput :\nTable-Schema :\n<Column Names>: %s\nTable Title: %s\nQuestion: %s\nAnswer: %s\n\nOutput :\n"
        % (subq_prompt, json.dumps(headers, ensure_ascii=False), table_title, question, answer_str)
    )
    raw_subq = _call("step3_subqueries", subq_user)
    subqueries = split_subqueries(raw_subq)
    steps["step3_subqueries"] = {"raw": raw_subq, "subqueries": list(subqueries)}

    if _nli is not None and getattr(_nli, "available", False):
        subqueries_before = list(subqueries)
        # Use (question + answer) as a best-effort premise.
        premise = "Question: %s\nAnswer: %s" % (question, answer_str)
        subqueries = _nli.filter(premise, subqueries, threshold=nli_threshold)
        steps.setdefault("step3_subqueries", {})["nli"] = {
            "enabled": True,
            "threshold": float(nli_threshold),
            "before": subqueries_before,
            "after": list(subqueries),
        }
    else:
        steps.setdefault("step3_subqueries", {})["nli"] = {"enabled": False}

    # ---- Step 1 (Column Relevance) ----
    rel_cols_prompt = _prompt_relcols or load_text(_ROOT_DIR / "Prompts" / "relevant_cols.txt")
    rel_user = (
        "%s\nInput : \nTable Title: %s\n<Column Names>: %s\nQuestion: %s\nAnswer: %s\n\nOutput :\n"
        % (
            rel_cols_prompt,
            table_title,
            json.dumps(headers, ensure_ascii=False),
            question,
            answer_str,
        )
    )
    rel_text = _call("step1_relevant_cols", rel_user)
    m = re.search(r"(?<=<Relevant Columns>: )(.*)", rel_text, re.IGNORECASE)
    if m:
        cols_expr = m.group(1).strip()
        try:
            relevant_cols = list(ast.literal_eval(cols_expr))
        except Exception:
            relevant_cols = list(headers)
    else:
        relevant_cols = list(headers)

    # Keep only columns that exist (defensive).
    relevant_cols = [c for c in relevant_cols if c in headers]
    if not relevant_cols:
        relevant_cols = list(headers)
    steps["step1_relevant_cols"] = {"raw": rel_text, "relevant_cols": list(relevant_cols)}

    # ---- Step 2 (Evidence Span Extractor): row filtering via SQL (optional) ----
    row_ids = list(range(1, len(rows) + 1))
    if enable_row_filtering:
        try:
            import pandas as pd  # type: ignore

            safe_title = sanitize_identifier(_mysql_table_name)
            df = pd.DataFrame(rows, columns=headers)
            df.insert(0, "row_id", row_ids)
            # Normalize common numeric-like strings (commas/currency/percent) so the SQL agent
            # is less likely to produce mismatching literals like "3.904" vs "3,904".
            for col in list(df.columns):
                df[col] = df[col].map(_normalize_numeric_like)

            # Show only relevant cols to the LLM to reduce prompt size, but keep row_id.
            cols_for_df = ["row_id"] + [c for c in relevant_cols if c in df.columns]
            df_view = df[cols_for_df]

            rel_rows_prompt = _prompt_relrows or load_text(_ROOT_DIR / "Prompts" / "relevant_rows.txt")
            rows_user = (
                "%s\nInput :\nTable-Schema :\n <Table Title>: %s\n<Column Names>: %s\n"
                "Table: %s\nQuestion: %s\nAnswer: %s\n\nOutput :\n"
                % (
                    rel_rows_prompt,
                    safe_title,
                    json.dumps(relevant_cols, ensure_ascii=False),
                    df_view.to_string(index=False),
                    question,
                    answer_str,
                )
            )
            rows_text = _call("step2_relevant_rows", rows_user)
            sql = extract_sql(rows_text)
            if not sql and require_mysql_for_rows:
                raise RuntimeError("MySQL required but no <SQL> was produced by the row-filter agent")
            steps["step2_relevant_rows"] = {
                "raw": rows_text,
                "sql_extracted": sql,
                "require_mysql": bool(require_mysql_for_rows),
                "db_available": _db is not None,
            }
            if sql:
                q = sql.strip()
                # Prefer running SELECT directly to avoid creating per-example tables.
                if re.match(r"^\s*CREATE\s+TABLE\b", q, flags=re.IGNORECASE):
                    sel = _extract_select_from_create(q)
                    q = sel or q
                if require_mysql_for_rows and not re.match(r"^\s*SELECT\b", q, flags=re.IGNORECASE):
                    raise RuntimeError("MySQL required but extracted query is not SELECT/CREATE")

                filtered = None
                backend = None
                sql_time = None
                sql_rows_returned = None
                sql_columns_returned: List[str] = []
                sql_returned_row_ids: List[int] = []

                # 1) Try MySQL (optional; requires a running server + SQLAlchemy + pymysql)
                if _db is not None:
                    try:
                        backend = "mysql"
                        _db.upload_table(safe_title, df_view)
                        t_sql = time.perf_counter()
                        filtered = _db.run_sql(q)
                        sql_time = time.perf_counter() - t_sql
                    except Exception:
                        filtered = None
                elif require_mysql_for_rows:
                    raise RuntimeError("MySQL required but database connection is unavailable")

                # 2) Fallback to in-process SQLite (no server required)
                if filtered is None and not require_mysql_for_rows:
                    try:
                        backend = "sqlite"
                        con = sqlite3.connect(":memory:")
                        df_view.to_sql(safe_title, con, index=False, if_exists="replace")
                        t_sql = time.perf_counter()
                        # pandas read_sql_query tolerates trailing semicolons in most cases
                        filtered = pd.read_sql_query(q, con)
                        sql_time = time.perf_counter() - t_sql
                        con.close()
                    except Exception:
                        filtered = None
                if filtered is None and require_mysql_for_rows:
                    raise RuntimeError("MySQL required but SQL execution did not return rows")

                if filtered is not None:
                    try:
                        sql_rows_returned = int(len(filtered))
                    except Exception:
                        sql_rows_returned = None
                    try:
                        sql_columns_returned = [str(c) for c in list(getattr(filtered, "columns", []))]
                    except Exception:
                        sql_columns_returned = []
                    if "row_id" in getattr(filtered, "columns", []):
                        keep = []
                        for v in filtered["row_id"].tolist():
                            try:
                                vv = int(v)
                            except Exception:
                                continue
                            sql_returned_row_ids.append(vv)
                            if 1 <= vv <= len(rows) and vv not in keep:
                                keep.append(vv)
                        if keep:
                            row_ids = keep
                steps.setdefault("step2_relevant_rows", {}).update(
                    {
                        "sql_executed": q,
                        "sql_backend": backend,
                        "sql_time_sec": sql_time,
                        "sql_rows_returned": sql_rows_returned,
                        "sql_columns_returned": sql_columns_returned,
                        "sql_returned_row_ids": sql_returned_row_ids,
                        "filtered_row_ids": list(row_ids),
                        "row_filter_applied": bool(sql_returned_row_ids) and (len(row_ids) != len(rows)),
                    }
                )
        except Exception:
            # If strict MySQL is required, propagate the error. Otherwise, fall back to no row filtering.
            if require_mysql_for_rows:
                raise
            pass
    else:
        steps["step2_relevant_rows"] = {"skipped": True}

    # Build pruned table rows (only relevant columns) and keep row_id mapping.
    rel_col_indices = [headers.index(c) for c in relevant_cols if c in headers]
    pruned_rows: List[List[Any]] = []
    for rid in row_ids:
        base = rows[rid - 1]
        pruned_rows.append([base[i] for i in rel_col_indices])

    # ---- Step 4 (Sub-Query Attribution) ----
    sub_attr_prompt = _prompt_sub_attr or load_text(_ROOT_DIR / "Prompts" / "traceback_subquery_attribution.txt")
    sub_attr_user = (
        "%s\n\nInput:\n<Relevant-Columns>: %s\n<Table Rows>: %s\nAnswer: %s\n<Sub-Questions>:\n%s\n"
        % (
            sub_attr_prompt,
            json.dumps(relevant_cols, ensure_ascii=False),
            json.dumps(pruned_rows, ensure_ascii=False),
            answer_str,
            "\n".join("%d. %s" % (i + 1, sq) for i, sq in enumerate(subqueries)),
        )
    )
    sub_attr_text = _call("step4_subquery_attribution", sub_attr_user)
    candidate_pairs = parse_cell_pairs(sub_attr_text)
    steps["step4_subquery_attribution"] = {"raw": sub_attr_text, "candidate_pairs": list(candidate_pairs)}

    # ---- Step 5 (Final Attribution) ----
    final_prompt = _prompt_final or load_text(_ROOT_DIR / "Prompts" / "traceback_final_attribution.txt")
    final_user = (
        "%s\n\nInput:\n<Relevant-Columns>: %s\n<Table Rows>: %s\nQuestion: %s\nAnswer: %s\n"
        "<Sub-Questions>:\n%s\n<Candidate Cells>: %s\n\nOutput:\n"
        % (
            final_prompt,
            json.dumps(relevant_cols, ensure_ascii=False),
            json.dumps(pruned_rows, ensure_ascii=False),
            question,
            answer_str,
            "\n".join("%d. %s" % (i + 1, sq) for i, sq in enumerate(subqueries)),
            json.dumps(candidate_pairs, ensure_ascii=False),
        )
    )
    final_text = _call("step5_final_attribution", final_user)
    final_pairs = parse_cell_pairs(final_text) or candidate_pairs
    steps["step5_final_attribution"] = {"raw": final_text, "final_pairs": list(final_pairs)}

    # Map back to original table indices:
    # - rows: output expects 0-based over data rows (header excluded) => (row_id - 1)
    # - cols: indices in original header order
    result_cells: List[List[int]] = []
    seen_rc = set()
    for r, c in final_pairs:
        if r < 0 or r >= len(row_ids):
            continue
        if c < 0 or c >= len(rel_col_indices):
            continue
        orig_row_id = int(row_ids[r])  # 1..N over full table rows (excluding header)
        pred_row = orig_row_id - 1
        pred_col = int(rel_col_indices[c])
        key = (pred_row, pred_col)
        if key in seen_rc:
            continue
        seen_rc.add(key)
        result_cells.append([pred_row, pred_col])

    return {
        "result_cells": result_cells,
        "debug": {
            "table_title": table_title,
            "mysql_table_name": sanitize_identifier(_mysql_table_name),
            "headers": headers,
            "relevant_cols": relevant_cols,
            "row_ids": row_ids,
            "subqueries": subqueries,
            "candidate_pairs": candidate_pairs,
            "final_pairs": final_pairs,
            # Convenience: final pairs mapped back to the original table coordinates.
            # (0-based over data rows; 0-based over original headers)
            "result_cells_mapped": result_cells,
            "steps": steps,
            "timings_sec": timings_sec,
            "total_time_sec": time.perf_counter() - t0_total,
            "enable_row_filtering": bool(enable_row_filtering),
            "require_mysql_for_rows": bool(require_mysql_for_rows),
            "pruned_shape": {"rows": len(pruned_rows), "cols": len(relevant_cols)},
        },
    }
