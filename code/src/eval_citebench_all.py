import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_gt_maps(gt_items: List[Dict]) -> Tuple[Dict[Tuple[str, str], List[List[int]]], Dict[Tuple[str, str], List[List[int]]]]:
    by_qid: Dict[Tuple[str, str], List[List[int]]] = {}
    by_eid: Dict[Tuple[str, str], List[List[int]]] = {}
    for it in gt_items:
        dataset = str(it.get("dataset"))
        qid = str(it.get("qid")) if it.get("qid") is not None else None
        eid = str(it.get("example_id")) if it.get("example_id") is not None else None
        gt_cells = it.get("highlighted_cells") or []
        if qid is not None:
            by_qid[(dataset, qid)] = gt_cells
        if eid is not None:
            by_eid[(dataset, eid)] = gt_cells
    return by_qid, by_eid


def eval_pair(pred_cells: List[List[int]], gt_cells: List[List[int]]) -> Dict[str, int]:
    # Deduplicate
    pred_cells_set = set((int(r), int(c)) for r, c in pred_cells)
    gt_cells_set = set((int(r), int(c)) for r, c in gt_cells)

    pred_rows = {r for r, _ in pred_cells_set}
    pred_cols = {c for _, c in pred_cells_set}
    gt_rows = {r for r, _ in gt_cells_set}
    gt_cols = {c for _, c in gt_cells_set}

    row_tp = len(pred_rows & gt_rows)
    col_tp = len(pred_cols & gt_cols)
    cell_tp = len(pred_cells_set & gt_cells_set)

    return {
        "row_tp": row_tp,
        "row_pred_total": len(pred_rows),
        "row_gt_total": len(gt_rows),
        "col_tp": col_tp,
        "col_pred_total": len(pred_cols),
        "col_gt_total": len(gt_cols),
        "cell_tp": cell_tp,
        "cell_pred_total": len(pred_cells_set),
        "cell_gt_total": len(gt_cells_set),
    }


def add_counts(agg: Dict[str, int], inc: Dict[str, int]) -> None:
    for k, v in inc.items():
        agg[k] = agg.get(k, 0) + int(v)


def finalize_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    rp = counts.get("row_tp", 0)
    rpred = counts.get("row_pred_total", 0)
    rgt = counts.get("row_gt_total", 0)
    cp = counts.get("col_tp", 0)
    cpred = counts.get("col_pred_total", 0)
    cgt = counts.get("col_gt_total", 0)
    tp = counts.get("cell_tp", 0)
    tpred = counts.get("cell_pred_total", 0)
    tgt = counts.get("cell_gt_total", 0)
    return {
        "row_precision": (rp / rpred) if rpred else 0.0,
        "row_recall": (rp / rgt) if rgt else 0.0,
        "column_precision": (cp / cpred) if cpred else 0.0,
        "column_recall": (cp / cgt) if cgt else 0.0,
        "cell_precision": (tp / tpred) if tpred else 0.0,
        "cell_recall": (tp / tgt) if tgt else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all ICL results under a root against CITEBENCH.json")
    parser.add_argument("--results-root", default="results/ICL", help="Root folder containing model dirs")
    parser.add_argument("--gt", default="Datasets/CITEBENCH.json", help="Path to unified GT JSON")
    parser.add_argument("--output", default="results/eval/metrics_citebench.json", help="Output JSON path")
    parser.add_argument("--styles", nargs="*", default=["zero", "zero-cot", "few", "few-cot"], help="Which style basenames to include")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    gt_items = load_json(Path(args.gt))
    by_qid, by_eid = build_gt_maps(gt_items)

    # Prepare filesystem
    (Path(args.output).parent).mkdir(parents=True, exist_ok=True)

    summary: List[Dict] = []

    # Walk each model dir
    if not results_root.exists():
        print(f"Results root not found: {results_root}")
        return

    for model_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        model_name = model_dir.name

        for style in args.styles:
            # Prefer merged, then plain, then any style.*.json as fallback
            candidates = [
                model_dir / f"{style}.merged.json",
                model_dir / f"{style}.json",
            ]
            # If neither exists, look for any sharded json and merge in-memory
            if not any(p.exists() for p in candidates):
                shard_jsons = sorted(model_dir.glob(f"{style}.*.json"))
                if shard_jsons:
                    preds: List[Dict] = []
                    for fp in shard_jsons:
                        try:
                            preds.extend(load_json(fp))
                        except Exception:
                            continue
                else:
                    # Nothing found for this style
                    continue
            else:
                # Load from first existing candidate
                use_path = next(p for p in candidates if p.exists())
                preds = load_json(use_path)

            # Evaluate
            global_counts: Dict[str, int] = {}
            by_dataset_counts: Dict[str, Dict[str, int]] = {}

            missing = 0
            for rec in preds:
                dataset = str(rec.get("dataset"))
                qid = rec.get("qid")
                eid = rec.get("example_id")
                key_qid = (dataset, str(qid)) if qid is not None else None
                key_eid = (dataset, str(eid)) if eid is not None else None
                gt_cells = None
                # Prefer example_id if available: qid is not guaranteed unique (some datasets have duplicates).
                if key_eid and key_eid in by_eid:
                    gt_cells = by_eid[key_eid]
                elif key_qid and key_qid in by_qid:
                    gt_cells = by_qid[key_qid]
                if gt_cells is None:
                    missing += 1
                    continue

                pred_cells = rec.get("result") or []
                # No index offset; assume same convention as GT
                counts = eval_pair(pred_cells, gt_cells)
                add_counts(global_counts, counts)
                if dataset not in by_dataset_counts:
                    by_dataset_counts[dataset] = {}
                add_counts(by_dataset_counts[dataset], counts)

            record = {
                "model": model_name,
                "style": style,
                "missing": missing,
                "global": finalize_metrics(global_counts),
                "by_dataset": {ds: finalize_metrics(c) for ds, c in by_dataset_counts.items()},
                "raw_counts": {
                    "global": global_counts,
                    "by_dataset": by_dataset_counts,
                },
            }
            summary.append(record)

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"evaluated": summary}, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
