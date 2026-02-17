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


def load_metrics(path: Path):
    return json.loads(path.read_text(encoding="utf-8")).get("evaluated", [])


def build_wide_table(records, prefer_style_order=None):
    # Collect unique models and styles
    models = []
    styles = []
    for r in records:
        m = r.get("model")
        s = r.get("style")
        if m not in models:
            models.append(m)
        if s not in styles:
            styles.append(s)

    # Optional preferred style order
    if prefer_style_order:
        seen = set()
        ordered = []
        for s in prefer_style_order:
            if s in styles and s not in seen:
                ordered.append(s)
                seen.add(s)
        # append any remaining styles
        for s in styles:
            if s not in seen:
                ordered.append(s)
        styles = ordered
    else:
        styles.sort()

    # Build lookup: (model, style) -> global metric dict
    lut = {}
    for r in records:
        lut[(r.get("model"), r.get("style"))] = r.get("global", {})

    # Start HTML table (Markdown doesn't support colspan nicely)
    lines = []
    lines.append("<table>")
    # Header rows
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append('      <th rowspan="2">Style</th>')
    for m in models:
        disp = str(m).replace("__", "/")
        lines.append(f'      <th colspan="{len(METRIC_ORDER)}">{disp}</th>')
    lines.append("    </tr>")
    lines.append("    <tr>")
    for _ in models:
        for _, label in METRIC_ORDER:
            lines.append(f"      <th>{label}</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")

    # Body
    lines.append("  <tbody>")
    for s in styles:
        lines.append("    <tr>")
        lines.append(f"      <td>{s}</td>")
        for m in models:
            g = lut.get((m, s), {})
            for key, _ in METRIC_ORDER:
                val = g.get(key)
                cell = f"{float(val):.3f}" if isinstance(val, (int, float)) else "&mdash;"
                lines.append(f"      <td>{cell}</td>")
        lines.append("    </tr>")
    lines.append("  </tbody>")
    lines.append("</table>")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Create a wide HTML table (inside MD) for model×style global metrics")
    parser.add_argument("--input", default="results/eval/metrics_citebench.json")
    parser.add_argument("--output", default="results/eval/metrics_citebench_wide.md")
    args = parser.parse_args()

    recs = load_metrics(Path(args.input))
    md = ["# CITEBENCH Global Metrics (Wide)\n"]
    md.append(build_wide_table(recs, prefer_style_order=["zero", "zero-cot", "few", "few-cot"]))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

