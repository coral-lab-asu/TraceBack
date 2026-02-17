import json


predicted_file_path = "totto_results/totto_answer_attribution_gpt-4o_traceback.json"


def evaluate_prediction(pred_result, gt_result):
    pred_result = [list(t) for t in set(tuple(lst) for lst in pred_result)]

    pred_rows = set(r for r, c in pred_result)
    pred_cols = set(c for r, c in pred_result)
    gt_rows = set(r for r, c in gt_result)
    gt_cols = set(c for r, c in gt_result)

    row_tp = len(pred_rows & gt_rows)
    col_tp = len(pred_cols & gt_cols)

    pred_cells = set((r, c) for r, c in pred_result)
    gt_cells = set((r, c) for r, c in gt_result)

    cell_tp = len(pred_cells & gt_cells)
    cell_pred_total = len(pred_cells)
    cell_gt_total = len(gt_cells)

    return {
        "row_tp": row_tp,
        "row_pred_total": len(pred_rows),
        "row_gt_total": len(gt_rows),
        "col_tp": col_tp,
        "col_pred_total": len(pred_cols),
        "col_gt_total": len(gt_cols),
        "cell_level_tp": cell_tp,
        "cell_pred_total": cell_pred_total,
        "cell_gt_total": cell_gt_total,
    }


def main():
    with open(predicted_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_row_tp = total_row_pred = total_row_gt = 0
    total_col_tp = total_col_pred = total_col_gt = 0
    total_cell_tp = total_cell_pred = total_cell_gt = 0

    total_examples = len(data)

    for item in data:
        pred_data = item.get("result") or []
        gt_data = item.get("highlighted_cell_ids")

        if not gt_data:
            total_examples -= 1
            continue

        # Shift predicted row indices to match 1-based GT convention
        for i in range(len(pred_data)):
            pred_data[i][0] += 1

        result = evaluate_prediction(pred_data, gt_data)

        total_row_tp += result["row_tp"]
        total_row_pred += result["row_pred_total"]
        total_row_gt += result["row_gt_total"]
        total_col_tp += result["col_tp"]
        total_col_pred += result["col_pred_total"]
        total_col_gt += result["col_gt_total"]
        total_cell_tp += result["cell_level_tp"]
        total_cell_pred += result["cell_pred_total"]
        total_cell_gt += result["cell_gt_total"]

    print(total_examples)
    metrics = {
        "row_precision": total_row_tp / total_row_pred if total_row_pred else 0.0,
        "row_recall": total_row_tp / total_row_gt if total_row_gt else 0.0,
        "column_precision": total_col_tp / total_col_pred if total_col_pred else 0.0,
        "column_recall": total_col_tp / total_col_gt if total_col_gt else 0.0,
        "cell_precision": total_cell_tp / total_cell_pred if total_cell_pred else 0.0,
        "cell_recall": total_cell_tp / total_cell_gt if total_cell_gt else 0.0,
    }

    print(metrics)


if __name__ == "__main__":
    main()

