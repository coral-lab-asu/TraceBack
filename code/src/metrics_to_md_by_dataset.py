import argparse
import json
from pathlib import Path


METRIC_ORDER = [
    ("row_precision", "Row P"),
    ("row_recall", "Row R"),
    ("column_precision", "Col P"),
    ("column_recall", "Col R"),
    ("cell_precision", "Cell P"),
    ("cell_recall", "Cell R"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert metrics JSON to grouped HTML table in Markdown (model as group header; style×dataset rows)."
    )
    parser.add_argument("--input", default="results/eval/metrics_citebench.json", help="Path to metrics JSON")
    parser.add_argument("--output", default="results/eval/metrics_citebench_by_dataset.md", help="Path to output Markdown")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    obj = json.loads(in_path.read_text(encoding="utf-8"))
    inseq_mode = "inseq" in in_path.name.lower()

    # groups[model][style] = {"global": {...}, "by_dataset": {...}}
    groups = {}
    all_datasets = set()
    for rec in obj.get("evaluated", []):
        m = str(rec.get("model"))
        s = str(rec.get("style"))
        g = rec.get("global", {}) or {}
        by_ds = rec.get("by_dataset", {}) or {}
        if isinstance(by_ds, dict):
            all_datasets.update(str(k) for k in by_ds.keys())
        groups.setdefault(m, {})[s] = {"global": g, "by_dataset": by_ds}

    # Model ordering (same heuristic as metrics_to_md.py)
    def _disp(name: str) -> str:
        return str(name).replace("__", "/")

    def _brand(name: str) -> str:
        d = _disp(name).lower()
        if "qwen" in d:
            return "Qwen"
        if "gemma" in d:
            return "Gemma"
        return "Other"

    def _size_gb(name: str) -> float:
        import re

        d = _disp(name).lower()
        m = re.findall(r"(\d+(?:\.\d+)?)\s*b", d)
        if m:
            try:
                # Some model ids include multiple *B tokens (e.g., "30B-A3B"); prefer the largest.
                return max(float(x) for x in m)
            except Exception:
                return float("inf")
        return float("inf")

    by_brand = {"Qwen": [], "Gemma": [], "Other": []}
    for m in groups.keys():
        by_brand[_brand(m)].append(m)
    for k in by_brand:
        by_brand[k].sort(key=lambda x: (_size_gb(x), _disp(x)))
    models = by_brand["Qwen"] + by_brand["Gemma"] + by_brand["Other"]

    # Preferred style order (falls back to any extras)
    style_order = ["zero", "zero-cot", "few", "few-cot"]
    # Preferred dataset order
    dataset_order = ["aitqa", "feta", "totto"]
    # Include any extra datasets seen
    dataset_order += sorted([d for d in all_datasets if d not in set(dataset_order)])

    def _brand_bg(name: str) -> str:
        n = name.lower()
        if "qwen" in n:
            return "#e8f4ff"  # light blue
        if "gemma" in n:
            return "#f7e8ff"  # light purple
        return "#fff4e5"  # light orange

    lines = []
    lines.append("# CITEBENCH Metrics by Dataset (Grouped)\n")
    lines.append('<table style="border-collapse:collapse; width:100%;">')
    lines.append("  <thead>")
    header = ["Style", "Dataset"] + [label for _, label in METRIC_ORDER]
    lines.append("    <tr>")
    for h in header:
        lines.append(f'      <th style="border:1px solid #999; padding:4px; text-align:center;">{h}</th>')
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")

    for m in models:
        disp = _disp(m)
        lines.append("    <tr>")
        lines.append(
            f'      <th colspan="{2+len(METRIC_ORDER)}" style="padding:8px; text-align:left; border-top:2px solid #000; border-bottom:1px solid #000; background:{_brand_bg(disp)}; color:#000;"><strong>{disp}</strong></th>'
        )
        lines.append("    </tr>")

        model_styles = groups.get(m, {})
        ordered_styles = [s for s in style_order if s in model_styles]
        ordered_styles += [s for s in model_styles.keys() if s not in ordered_styles]

        for s in ordered_styles:
            payload = model_styles.get(s, {})
            global_metrics = payload.get("global", {}) or {}
            by_ds = payload.get("by_dataset", {}) or {}

            # Global row
            disp_style = "inseq" if (inseq_mode and s == "attention") else s
            global_label = "CITEBENCH" if inseq_mode else "global"
            lines.append("    <tr>")
            lines.append(f'      <td style="border:1px solid #ddd; padding:4px;">{disp_style}</td>')
            lines.append(
                f'      <td style="border:1px solid #ddd; padding:4px; font-weight:600; background:#fafafa;">{global_label}</td>'
            )
            for key, _label in METRIC_ORDER:
                val = global_metrics.get(key)
                txt = f"{float(val):.3f}" if isinstance(val, (int, float)) else "&mdash;"
                lines.append(f'      <td style="border:1px solid #ddd; padding:4px; text-align:right;">{txt}</td>')
            lines.append("    </tr>")

            # Per-dataset rows
            for ds in dataset_order:
                if ds not in by_ds:
                    continue
                g = by_ds.get(ds, {}) or {}
                lines.append("    <tr>")
                lines.append('      <td style="border:1px solid #ddd; padding:4px;">&nbsp;</td>')
                lines.append(f'      <td style="border:1px solid #ddd; padding:4px;">{ds}</td>')
                for key, _label in METRIC_ORDER:
                    val = g.get(key)
                    txt = f"{float(val):.3f}" if isinstance(val, (int, float)) else "&mdash;"
                    lines.append(f'      <td style="border:1px solid #ddd; padding:4px; text-align:right;">{txt}</td>')
                lines.append("    </tr>")

        lines.append(
            f'    <tr><td colspan="{2+len(METRIC_ORDER)}" style="border-bottom:2px solid #000; height:2px; padding:0;"></td></tr>'
        )

    lines.append("  </tbody>")
    lines.append("</table>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
