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


def main():
    parser = argparse.ArgumentParser(description="Convert metrics to grouped HTML table in Markdown (model as group header, styles as rows)")
    parser.add_argument("--input", default="results/eval/metrics_citebench.json", help="Path to metrics JSON")
    parser.add_argument("--output", default="results/eval/metrics_citebench.md", help="Path to output Markdown")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    obj = json.loads(in_path.read_text(encoding="utf-8"))
    inseq_mode = "inseq" in in_path.name.lower()

    # Group by model
    groups = {}
    for rec in obj.get("evaluated", []):
        m = str(rec.get("model"))
        s = str(rec.get("style"))
        g = rec.get("global", {})
        groups.setdefault(m, {})[s] = g

    # Build ordered model list: sort within brand (Qwen/Gemma/Other) by parameter size ascending
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
        # Parse sizes like 1b, 4b, 12b, 27b, 3B, 7B, 72B etc.; take the last occurrence
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

    # Partition by brand
    by_brand = {"Qwen": [], "Gemma": [], "Other": []}
    for m in groups.keys():
        by_brand[_brand(m)].append(m)

    # Sort each brand by size ascending, then by display name
    for k in by_brand:
        by_brand[k].sort(key=lambda x: (_size_gb(x), _disp(x)))

    # Final models order: Qwen, Gemma, then Other
    models = by_brand["Qwen"] + by_brand["Gemma"] + by_brand["Other"]
    # Preferred style order
    style_order = ["zero", "zero-cot", "few", "few-cot"]

    # Build HTML table so we can emulate merged model header + bold border
    lines = []
    lines.append("# CITEBENCH Global Metrics (Grouped)\n")
    lines.append("<table style=\"border-collapse:collapse; width:100%;\">")
    # Head for style/metrics (model printed as group header rows)
    lines.append("  <thead>")
    header = ["Style"] + [label for _, label in METRIC_ORDER]
    lines.append("    <tr>")
    for h in header:
        lines.append(f"      <th style=\"border:1px solid #999; padding:4px; text-align:center;\">{h}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")

    for mi, m in enumerate(models):
        disp = m.replace("__", "/")
        # Brand-specific header background for readability
        def _brand_bg(name: str) -> str:
            n = name.lower()
            if "qwen" in n:
                return "#e8f4ff"  # light blue
            if "gemma" in n:
                return "#f7e8ff"  # light purple
            return "#fff4e5"      # light orange
        bg = _brand_bg(disp)
        # Group header row with strong border and explicit black text color
        lines.append("    <tr>")
        lines.append(
            f"      <th colspan=\"{1+len(METRIC_ORDER)}\" style=\"padding:8px; text-align:left; border-top:2px solid #000; border-bottom:1px solid #000; background:{bg}; color:#000;\"><strong>{disp}</strong></th>")
        lines.append("    </tr>")

        # Rows per style
        model_styles = groups[m]
        # maintain preferred order but include any extra styles if present
        ordered_styles = [s for s in style_order if s in model_styles]
        ordered_styles += [s for s in model_styles.keys() if s not in ordered_styles]
        for s in ordered_styles:
            disp_style = "inseq" if (inseq_mode and s == "attention") else s
            g = model_styles.get(s, {})
            lines.append("    <tr>")
            lines.append(f"      <td style=\"border:1px solid #ddd; padding:4px;\">{disp_style}</td>")
            for key, _label in METRIC_ORDER:
                val = g.get(key)
                txt = f"{float(val):.3f}" if isinstance(val, (int, float)) else "&mdash;"
                lines.append(f"      <td style=\"border:1px solid #ddd; padding:4px; text-align:right;\">{txt}</td>")
            lines.append("    </tr>")
        # bottom separator between models
        lines.append("    <tr><td colspan=\"7\" style=\"border-bottom:2px solid #000; height:2px; padding:0;\"></td></tr>")

    lines.append("  </tbody>")
    lines.append("</table>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
