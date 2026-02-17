import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METRICS: Sequence[Tuple[str, str]] = (
    ("row_precision", "Row P"),
    ("row_recall", "Row R"),
    ("column_precision", "Col P"),
    ("column_recall", "Col R"),
    ("cell_precision", "Cell P"),
    ("cell_recall", "Cell R"),
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                yield item


def _check_list_depth(lst: Any) -> int:
    if not isinstance(lst, list):
        return 0
    if all(isinstance(elem, list) for elem in lst):
        if all(isinstance(sub_elem, list) for elem in lst for sub_elem in elem):
            return 3
        return 2
    return 1


def _normalize_cells(value: Any) -> List[Tuple[int, int]]:
    if value in (None, "null"):
        return []
    if not isinstance(value, list):
        return []
    if _check_list_depth(value) == 3 and value:
        # Match the existing eval scripts' behavior: take the first slice.
        value = value[0]
    cells: List[Tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            r = int(item[0])
            c = int(item[1])
        except Exception:
            continue
        cells.append((r, c))
    # De-duplicate while keeping it a plain list of tuples.
    return list(set(cells))


def _get_gt_cells(item: Dict[str, Any]) -> List[Tuple[int, int]]:
    # Support both naming conventions used across scripts.
    gt = item.get("highlighted_cell_ids")
    if gt is None:
        gt = item.get("highlighted_cells")
    return _normalize_cells(gt)


def compute_metrics(
    data: Iterable[Dict[str, Any]],
    *,
    pred_row_offset: int = 1,
    skip_empty_predictions: bool = True,
) -> Tuple[int, Dict[str, float]]:
    total_row_tp = total_row_pred = total_row_gt = 0
    total_col_tp = total_col_pred = total_col_gt = 0
    total_cell_tp = total_cell_pred = total_cell_gt = 0
    n = 0

    for item in data:
        pred_cells = _normalize_cells(item.get("result") or [])
        gt_cells = _get_gt_cells(item)
        if not gt_cells:
            continue

        # Shift predicted row indices (GT uses 1-based row ids; predictions are 0-based over data rows).
        pred_cells = [(r + pred_row_offset, c) for r, c in pred_cells]

        if not pred_cells and skip_empty_predictions:
            # Match Code/eval_script.py and Code_with_rows/eval_script.py behavior:
            # skip examples with empty predictions (exclude them from both N and GT totals).
            continue

        pred_rows = set(r for r, _c in pred_cells)
        pred_cols = set(c for _r, c in pred_cells)
        gt_rows = set(r for r, _c in gt_cells)
        gt_cols = set(c for _r, c in gt_cells)

        total_row_gt += len(gt_rows)
        total_col_gt += len(gt_cols)
        gt_cell_set = set(gt_cells)
        total_cell_gt += len(gt_cell_set)

        if pred_cells:
            total_row_tp += len(pred_rows & gt_rows)
            total_row_pred += len(pred_rows)

            total_col_tp += len(pred_cols & gt_cols)
            total_col_pred += len(pred_cols)

            pred_cell_set = set(pred_cells)
            total_cell_tp += len(pred_cell_set & gt_cell_set)
            total_cell_pred += len(pred_cell_set)

        n += 1

    metrics = {
        "row_precision": total_row_tp / total_row_pred if total_row_pred else 0.0,
        "row_recall": total_row_tp / total_row_gt if total_row_gt else 0.0,
        "column_precision": total_col_tp / total_col_pred if total_col_pred else 0.0,
        "column_recall": total_col_tp / total_col_gt if total_col_gt else 0.0,
        "cell_precision": total_cell_tp / total_cell_pred if total_cell_pred else 0.0,
        "cell_recall": total_cell_tp / total_cell_gt if total_cell_gt else 0.0,
    }
    return n, metrics


def _fmt(v: Optional[float], *, percent: bool) -> str:
    if v is None:
        return "—"
    if percent:
        return f"{v * 100:.2f}"
    return f"{v:.3f}"


def render_markdown(
    rows: Sequence[Tuple[str, int, Dict[str, float]]],
    *,
    percent: bool,
) -> str:
    header = ["Dataset", "N"] + [label for _k, label in METRICS]
    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---", "---:"] + ["---:"] * len(METRICS)) + " |")
    for name, n, m in rows:
        vals = [_fmt(m.get(k), percent=percent) for k, _label in METRICS]
        lines.append("| " + " | ".join([name, str(n)] + vals) + " |")
    return "\n".join(lines) + "\n"


def _iter_joined_examples(
    *,
    pred_path: Path,
    gt_path: Path,
    pred_id_key: str,
    gt_id_key: str,
    gt_cells_key: str = "highlighted_cells",
) -> Iterable[Dict[str, Any]]:
    pred_data = _load_json(pred_path)
    if not isinstance(pred_data, list):
        raise SystemExit(f"Expected a JSON list in {pred_path}")

    pred_map: Dict[Any, Dict[str, Any]] = {}
    for item in pred_data:
        if not isinstance(item, dict):
            continue
        pred_id = item.get(pred_id_key)
        if pred_id is None:
            continue
        # Match the repo's eval scripts: if there are duplicates, keep the first.
        pred_map.setdefault(pred_id, item)

    for gt_item in _iter_jsonl(gt_path):
        gt_id = gt_item.get(gt_id_key)
        if gt_id is None:
            continue
        if gt_cells_key not in gt_item:
            continue
        pred_item = pred_map.get(gt_id)
        if pred_item is None:
            continue
        yield {"result": pred_item.get("result"), "highlighted_cells": gt_item.get(gt_cells_key)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate TraceBack outputs for AITQA/FetaQA/ToTTo and emit a Markdown table.")
    parser.add_argument("--aitqa", default="aitqa_answer_attribution_gpt-4o_new.json", help="AITQA prediction JSON path")
    parser.add_argument("--fetaqa", default="fetaqa_results/fetaqa_answer_attribution_gpt-4o_withrows_1.json", help="FetaQA prediction JSON path")
    parser.add_argument("--totto", default="totto_results/totto_answer_attribution_gpt-4o_traceback.json", help="ToTTo prediction JSON path")
    parser.add_argument("--aitqa-gt", default="Datasets/AITQA/aitqa_processed.jsonl", help="AITQA ground-truth JSONL path")
    parser.add_argument("--fetaqa-gt", default="Datasets/FetaQA/fetaQA_dev_processed.jsonl", help="FetaQA ground-truth JSONL path")
    parser.add_argument("--totto-gt", default="Datasets/Totto/totto_processed.jsonl", help="ToTTo ground-truth JSONL path")
    parser.add_argument("--output", default="results/eval/metrics_traceback.md", help="Output Markdown path")
    parser.add_argument("--percent", action="store_true", help="Format metrics as percentages (0-100)")
    parser.add_argument("--pred-row-offset", type=int, default=1, help="Row index offset added to predictions before scoring (default: 1)")
    args = parser.parse_args()

    specs = [
        # Match the per-dataset eval scripts bundled in this repo:
        # - ToTTo includes examples even if prediction is empty (see Code_with_rows/eval_totto.py)
        # - AITQA/FetaQA skip empty predictions (see Code/eval_script.py and Code_with_rows/eval_script.py)
        ("ToTTo", Path(args.totto), Path(args.totto_gt), "example_id", "example_id", False),
        ("FetaQA", Path(args.fetaqa), Path(args.fetaqa_gt), "feta_id", "feta_id", True),
        ("AITQA", Path(args.aitqa), Path(args.aitqa_gt), "feta_id", "id", True),
    ]

    rows: List[Tuple[str, int, Dict[str, float]]] = []
    missing = []
    for name, pred_path, gt_path, pred_id_key, gt_id_key, skip_empty_predictions in specs:
        if not pred_path.exists():
            missing.append(str(pred_path))
            continue
        if not gt_path.exists():
            missing.append(str(gt_path))
            continue
        data_iter = _iter_joined_examples(
            pred_path=pred_path,
            gt_path=gt_path,
            pred_id_key=str(pred_id_key),
            gt_id_key=str(gt_id_key),
        )
        n, m = compute_metrics(
            data_iter,
            pred_row_offset=int(args.pred_row_offset),
            skip_empty_predictions=bool(skip_empty_predictions),
        )
        rows.append((name, n, m))

    if missing:
        raise SystemExit("Missing prediction files: " + ", ".join(missing))

    md = render_markdown(rows, percent=bool(args.percent))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md, end="")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
