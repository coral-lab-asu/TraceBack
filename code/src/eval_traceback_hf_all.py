import argparse
import json
import re
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


def _get_gt_cells(rec: Dict[str, Any]) -> List[Tuple[int, int]]:
    # TraceBack outputs include gold in results as highlighted_cell_ids (preferred).
    gt = rec.get("highlighted_cell_ids")
    if gt is None:
        gt = rec.get("highlighted_cells")
    return _normalize_cells(gt)


def _eval_counts(
    pred_cells: Sequence[Tuple[int, int]],
    gt_cells: Sequence[Tuple[int, int]],
) -> Dict[str, int]:
    pred_rows = {r for r, _c in pred_cells}
    pred_cols = {c for _r, c in pred_cells}
    gt_rows = {r for r, _c in gt_cells}
    gt_cols = {c for _r, c in gt_cells}

    pred_set = set(pred_cells)
    gt_set = set(gt_cells)

    return {
        "row_tp": len(pred_rows & gt_rows),
        "row_pred_total": len(pred_rows),
        "row_gt_total": len(gt_rows),
        "col_tp": len(pred_cols & gt_cols),
        "col_pred_total": len(pred_cols),
        "col_gt_total": len(gt_cols),
        "cell_tp": len(pred_set & gt_set),
        "cell_pred_total": len(pred_set),
        "cell_gt_total": len(gt_set),
    }


def _add_counts(dst: Dict[str, int], inc: Dict[str, int]) -> None:
    for k, v in inc.items():
        dst[k] = dst.get(k, 0) + int(v)


def _finalize_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    row_tp = counts.get("row_tp", 0)
    row_pred = counts.get("row_pred_total", 0)
    row_gt = counts.get("row_gt_total", 0)
    col_tp = counts.get("col_tp", 0)
    col_pred = counts.get("col_pred_total", 0)
    col_gt = counts.get("col_gt_total", 0)
    cell_tp = counts.get("cell_tp", 0)
    cell_pred = counts.get("cell_pred_total", 0)
    cell_gt = counts.get("cell_gt_total", 0)

    return {
        "row_precision": (row_tp / row_pred) if row_pred else 0.0,
        "row_recall": (row_tp / row_gt) if row_gt else 0.0,
        "column_precision": (col_tp / col_pred) if col_pred else 0.0,
        "column_recall": (col_tp / col_gt) if col_gt else 0.0,
        "cell_precision": (cell_tp / cell_pred) if cell_pred else 0.0,
        "cell_recall": (cell_tp / cell_gt) if cell_gt else 0.0,
    }


def compute_metrics_from_pred(
    preds: Iterable[Dict[str, Any]],
    *,
    pred_row_offset: int,
    skip_empty_predictions: bool,
) -> Tuple[int, Dict[str, int], Dict[str, float]]:
    counts: Dict[str, int] = {}
    n = 0
    for rec in preds:
        gt_cells = _get_gt_cells(rec)
        if not gt_cells:
            continue

        pred_cells = _normalize_cells(rec.get("result") or [])
        pred_cells = [(r + int(pred_row_offset), c) for r, c in pred_cells]

        if skip_empty_predictions and not pred_cells:
            continue

        _add_counts(counts, _eval_counts(pred_cells, gt_cells))
        n += 1

    return n, counts, _finalize_metrics(counts)


def _unsanitize_model_dir(name: str) -> str:
    return str(name).replace("__", "/")


def _model_sort_key(name: str) -> Tuple[int, float, str]:
    disp = _unsanitize_model_dir(name).lower()
    if "qwen" in disp:
        brand = 0
    elif "gemma" in disp:
        brand = 1
    else:
        brand = 2
    m = re.findall(r"(\d+(?:\.\d+)?)\s*b", disp)
    size = float(m[-1]) if m else float("inf")
    return (brand, size, disp)


def _run_display(run_dir: str) -> str:
    run = str(run_dir)
    if run.startswith("nli"):
        suf = run[len("nli") :]
        if "p" in suf:
            return f"nli={suf.replace('p', '.', 1)}"
    return run


def _fmt(v: Optional[float], *, percent: bool) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.2f}" if percent else f"{v:.3f}"


def render_markdown(rows: Sequence[Dict[str, Any]], *, percent: bool, show_run: bool, show_n: bool) -> str:
    header = ["Model"]
    if show_run:
        header.append("Run")
    header.append("Dataset")
    if show_n:
        header.append("N")
    header += [label for _k, label in METRICS]
    lines: List[str] = []
    lines.append("| " + " | ".join(header) + " |")
    align = ["---", "---"]
    if show_run:
        align.append("---")
    if show_n:
        align.append("---:")
    align += ["---:"] * len(METRICS)
    lines.append("| " + " | ".join(align) + " |")
    for row in rows:
        metrics = row.get("metrics", {}) or {}
        vals = [_fmt(metrics.get(k), percent=percent) for k, _label in METRICS]
        parts: List[str] = [str(row.get("model", ""))]
        if show_run:
            parts.append(str(row.get("run", "")))
        parts.append(str(row.get("dataset", "")))
        if show_n:
            parts.append(str(int(row.get("n", 0))))
        lines.append(
            "| "
            + " | ".join(
                [
                    *parts,
                    *vals,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_grouped_markdown(
    groups: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
    *,
    percent: bool,
    dataset_order: Sequence[str],
    show_run: bool,
    show_n: bool,
) -> str:
    """Render a grouped HTML table similar to results/eval/metrics_citebench_icl_by_dataset.md."""

    def _brand_bg(name: str) -> str:
        n = name.lower()
        if "qwen" in n:
            return "#e8f4ff"  # light blue
        if "gemma" in n:
            return "#f7e8ff"  # light purple
        return "#fff4e5"  # light orange

    def _brand(name: str) -> str:
        d = name.lower()
        if "qwen" in d:
            return "Qwen"
        if "gemma" in d:
            return "Gemma"
        return "Other"

    def _size_gb(name: str) -> float:
        m = re.findall(r"(\d+(?:\.\d+)?)\s*b", name.lower())
        if m:
            try:
                return float(m[-1])
            except Exception:
                return float("inf")
        return float("inf")

    def _dataset_idx(ds: str) -> int:
        try:
            return list(dataset_order).index(ds.lower())
        except ValueError:
            return len(dataset_order)

    def _run_sort_key(run: str) -> Tuple[float, str]:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", run)
        if m:
            try:
                return (float(m.group(1)), run)
            except Exception:
                return (float("inf"), run)
        return (float("inf"), run)

    # Model ordering: Qwen, Gemma, Other; size ascending within brand.
    by_brand: Dict[str, List[str]] = {"Qwen": [], "Gemma": [], "Other": []}
    for m in groups.keys():
        by_brand[_brand(m)].append(m)
    for k in by_brand:
        by_brand[k].sort(key=lambda x: (_size_gb(x), x.lower()))
    model_order = by_brand["Qwen"] + by_brand["Gemma"] + by_brand["Other"]

    header: List[str] = []
    if show_run:
        header.append("Run")
    header.append("Dataset")
    if show_n:
        header.append("N")
    header += [label for _k, label in METRICS]
    n_cols = len(header)

    lines: List[str] = []
    lines.append("# TraceBack HF Metrics by Dataset (Grouped)\n")
    lines.append('<table style="border-collapse:collapse; width:100%;">')
    lines.append("  <thead>")
    lines.append("    <tr>")
    for h in header:
        lines.append(f'      <th style="border:1px solid #999; padding:4px; text-align:center;">{h}</th>')
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")

    for model in model_order:
        lines.append("    <tr>")
        lines.append(
            f'      <th colspan="{n_cols}" style="padding:8px; text-align:left; border-top:2px solid #000; border-bottom:1px solid #000; background:{_brand_bg(model)}; color:#000;"><strong>{model}</strong></th>'
        )
        lines.append("    </tr>")

        run_map = groups.get(model, {})
        show_run_headers = (not show_run) and (len(run_map) > 1)
        for run in sorted(run_map.keys(), key=_run_sort_key):
            ds_map = run_map.get(run, {})

            if show_run_headers:
                lines.append("    <tr>")
                lines.append(
                    f'      <td colspan="{n_cols}" style="border:1px solid #ddd; padding:6px; font-weight:600; background:#fafafa;">{run}</td>'
                )
                lines.append("    </tr>")

            # Global row first (if present)
            if "global" in ds_map:
                g = ds_map["global"]
                metrics = g.get("metrics", {}) or {}
                lines.append("    <tr>")
                if show_run:
                    lines.append(f'      <td style="border:1px solid #ddd; padding:4px;">{run}</td>')
                lines.append('      <td style="border:1px solid #ddd; padding:4px; font-weight:600; background:#fafafa;">global</td>')
                if show_n:
                    lines.append(
                        f'      <td style="border:1px solid #ddd; padding:4px; text-align:right;">{int(g.get("n", 0))}</td>'
                    )
                for key, _label in METRICS:
                    val = metrics.get(key)
                    txt = _fmt(val, percent=percent)
                    lines.append(f'      <td style="border:1px solid #ddd; padding:4px; text-align:right;">{txt}</td>')
                lines.append("    </tr>")

            # Dataset rows
            ds_keys = [k for k in ds_map.keys() if k != "global"]
            for ds in sorted(ds_keys, key=lambda x: (_dataset_idx(x), x)):
                payload = ds_map.get(ds, {})
                metrics = payload.get("metrics", {}) or {}
                lines.append("    <tr>")
                if show_run:
                    lines.append('      <td style="border:1px solid #ddd; padding:4px;">&nbsp;</td>')
                lines.append(f'      <td style="border:1px solid #ddd; padding:4px;">{ds}</td>')
                if show_n:
                    lines.append(
                        f'      <td style="border:1px solid #ddd; padding:4px; text-align:right;">{int(payload.get("n", 0))}</td>'
                    )
                for key, _label in METRICS:
                    val = metrics.get(key)
                    txt = _fmt(val, percent=percent)
                    lines.append(f'      <td style="border:1px solid #ddd; padding:4px; text-align:right;">{txt}</td>')
                lines.append("    </tr>")

        lines.append(
            f'    <tr><td colspan="{n_cols}" style="border-bottom:2px solid #000; height:2px; padding:0;"></td></tr>'
        )

    lines.append("  </tbody>")
    lines.append("</table>")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TraceBack HF sweep outputs under results/traceback_hf and write a Markdown table (model×dataset)."
    )
    parser.add_argument("--results-root", default="results/traceback_hf", help="Root folder with dataset/model/run/pred.json")
    parser.add_argument("--output", default="results/eval/metrics_traceback_hf.md", help="Output Markdown path")
    parser.add_argument("--output-json", default="", help="Optional: write computed rows to JSON as well")
    parser.add_argument("--percent", action="store_true", help="Format metrics as percentages (0-100)")
    parser.add_argument("--format", default="grouped", choices=["grouped", "flat"], help="Markdown format")
    parser.add_argument("--show-run", action="store_true", help="Include Run column (default: hidden)")
    parser.add_argument("--show-n", action="store_true", help="Include N column (default: hidden)")
    parser.add_argument("--pred-row-offset", type=int, default=1, help="Row index offset added to predictions before scoring (default: 1)")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include empty predictions for ALL datasets (default matches repo scripts: AITQA/FetaQA skip empty, ToTTo include).",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"Results root not found: {results_root}")

    dataset_order = ["aitqa", "fetaqa", "totto"]

    found: List[Tuple[str, str, str, Path]] = []
    for pred_path in sorted(results_root.rglob("pred.json")):
        rel = pred_path.relative_to(results_root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        dataset = parts[-4]
        model_dir = parts[-3]
        run_dir = parts[-2]
        found.append((dataset, model_dir, run_dir, pred_path))

    if not found:
        raise SystemExit(f"No pred.json files found under {results_root}")

    # Compute metrics
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    rows: List[Dict[str, Any]] = []
    for dataset, model_dir, run_dir, pred_path in found:
        data = _load_json(pred_path)
        if not isinstance(data, list):
            continue

        if args.include_empty:
            skip_empty_predictions = False
        else:
            # Match repo scripts: ToTTo includes empty preds; others skip them.
            skip_empty_predictions = (dataset.lower() != "totto")

        n, counts, metrics = compute_metrics_from_pred(
            data,
            pred_row_offset=int(args.pred_row_offset),
            skip_empty_predictions=skip_empty_predictions,
        )
        model_disp = _unsanitize_model_dir(model_dir)
        run_disp = _run_display(run_dir)
        rec = {
            "dataset": dataset,
            "model": model_disp,
            "run": run_disp,
            "n": n,
            "metrics": metrics,
            "raw_counts": counts,
            "pred_path": str(pred_path),
        }
        rows.append(rec)
        grouped.setdefault(model_disp, {}).setdefault(run_disp, {})[dataset] = rec

    # Sort rows by (run, model, dataset)
    def _dataset_idx(ds: str) -> int:
        try:
            return dataset_order.index(ds.lower())
        except ValueError:
            return len(dataset_order)

    rows.sort(
        key=lambda r: (
            str(r.get("run", "")),
            _model_sort_key(str(r.get("model", "")).replace("/", "__")),
            _dataset_idx(str(r.get("dataset", ""))),
            str(r.get("dataset", "")),
        )
    )

    # Add global aggregates per (model, run)
    for model, run_map in grouped.items():
        for run, ds_map in run_map.items():
            global_counts: Dict[str, int] = {}
            global_n = 0
            for ds_rec in ds_map.values():
                global_n += int(ds_rec.get("n", 0))
                _add_counts(global_counts, ds_rec.get("raw_counts", {}) or {})
            ds_map["global"] = {"n": global_n, "raw_counts": global_counts, "metrics": _finalize_metrics(global_counts)}

    if str(args.format).strip().lower() == "flat":
        md = render_markdown(rows, percent=bool(args.percent), show_run=bool(args.show_run), show_n=bool(args.show_n))
    else:
        md = render_grouped_markdown(
            grouped,
            percent=bool(args.percent),
            dataset_order=dataset_order,
            show_run=bool(args.show_run),
            show_n=bool(args.show_n),
        )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(md, end="")
    print(f"Wrote {out_path}")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
