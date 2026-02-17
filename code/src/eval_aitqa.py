import json
import ast
from pathlib import Path

predicted_file_path = "aitqa_answer_attribution_gpt-4o_new.json"
#predicted_file_path = "aitqa_results/direct_prompting_attribution_gpt-4o_aitqa.json"
with open(predicted_file_path, 'r', encoding='utf-8') as f:
	predicted_data = json.load(f)

root_dir = Path(__file__).resolve().parent.parent
groundtruth_file_path = root_dir / "Datasets" / "AITQA" / "aitqa_processed.jsonl"
# with open(file, 'r', encoding='utf-8') as f:
# 	groundtruth_data = json.load(f)

def evaluate_prediction(pred_result, gt_result,feta_id):
	
	pred_result = [list(t) for t in set(tuple(lst) for lst in pred_result)]
	

	pred_rows = set(r for r, c in pred_result)
	pred_cols = set(c for r, c in pred_result)
	gt_rows = set(r for r, c in gt_result )
	gt_cols = set(c for r, c in gt_result)

	print(pred_rows,gt_rows,feta_id)
	row_tp = len(pred_rows & gt_rows)
	col_tp = len(pred_cols & gt_cols)

	gt_cells = set((r, c) for r, c in gt_result)

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
		"cell_pred_total":cell_pred_total,
		"cell_gt_total":cell_gt_total

	}
def check_list_depth(lst):
	if not isinstance(lst, list):
		return 0
	if all(isinstance(elem, list) for elem in lst):
		if all(isinstance(sub_elem, list) for elem in lst for sub_elem in elem):
			return 3  # 3D (list of list of lists)
		else:
			return 2  # 2D (list of lists)
	return 1  # 1D

# for point in predicted_data:
# 	if point["result"] is None:
# 		continue
# 	result_lines = point["result"].strip().splitlines()
# 	parsed_result = []
# 	print(result_lines)
# 	for line in result_lines:
# 		# Safely parse each line like "[(3, 2)]"
# 		tuples = ast.literal_eval(line)
# 		parsed_result.extend(tuples)  # add all tuples to the result

# 	point["result"] = parsed_result  # overwrite with final list of tuples


total_row_tp = total_row_pred = total_row_gt = 0
total_col_tp = total_col_pred = total_col_gt = 0
total_cell_tp = total_cell_pred = total_cell_gt = 0

exact_match_count = 0
total_examples = 213


with open(groundtruth_file_path, "r", encoding="utf-8") as file:
	for line in file:
		try:
			item = json.loads(line)
		except:
			total_examples-=1
			continue
		feta_id = item.get("id")
		if "highlighted_cells" not in item:
			total_examples-=1
			continue
		for point in predicted_data:
			
			if feta_id == point["feta_id"]:
				
				if point["result"]==[] or point["result"] is None or point["result"] == "null":
					total_examples-=1
					continue
				p = point["result"][:]
				for i in range(len(point["result"])):
					
					point["result"][i][0]+=1
				
				if check_list_depth(point["result"])==3:
					point["result"] = point["result"][0]
				result = evaluate_prediction(point["result"],item.get("highlighted_cells"),feta_id)
				total_row_tp += result["row_tp"]
				total_row_pred += result["row_pred_total"]
				total_row_gt += result["row_gt_total"]
				total_col_tp += result["col_tp"]
				total_col_pred += result["col_pred_total"]
				total_col_gt += result["col_gt_total"]
				total_cell_tp += result["cell_level_tp"]
				total_cell_pred += result["cell_pred_total"]
				total_cell_gt += result["cell_gt_total"]

				#exact_match_count += int(result["exact_match"])
				# if result["exact_match"]==0:
				# 	print(point["result"],item.get("highlighted_cells"))
				break
print(total_examples)
metrics = {
	"row_precision": total_row_tp / total_row_pred if total_row_pred else 0.0,
	"row_recall": total_row_tp / total_row_gt if total_row_gt else 0.0,
	"column_precision": total_col_tp / total_col_pred if total_col_pred else 0.0,
	"column_recall": total_col_tp / total_col_gt if total_col_gt else 0.0,
	"cell_precision": total_cell_tp / total_cell_pred if total_cell_pred else 0.0,
	"cell_recall": total_cell_tp / total_cell_gt if total_cell_gt else 0.0,
	# "exact_match": exact_match_count / total_examples if total_examples else 0.0
}


print(metrics)

#AITQA : 
#Our Approach :'row_precision': 0.9090909090909091, 'row_recall': 0.9326424870466321, 'column_precision': 0.955, 'column_recall': 0.9896373056994818, 'cell_precision': 0.8599033816425121, 'cell_recall': 0.9222797927461139
#Direct Prompting : {'row_precision': 0.6666666666666666, 'row_recall': 0.6842105263157895, 'column_precision': 0.6410256410256411, 'column_recall': 0.6578947368421053, 'cell_precision': 0.46153846153846156, 'cell_recall': 0.47368421052631576}
#FetaQA:
#Our Approach : {'row_precision': 0.7456140350877193, 'row_recall': 0.9172661870503597, 'column_precision': 0.9127358490566038, 'column_recall': 0.80625, 'cell_precision': 0.7115600448933782, 'cell_recall': 0.7502958579881657}
# Direct Prompting : {'row_precision': 0.4523809523809524, 'row_recall': 0.475, 'column_precision': 0.72, 'column_recall': 0.5294117647058824, 'cell_precision': 0.379746835443038, 'cell_recall': 0.2564102564102564}
#ToTTo:
#Our Approach : 'row_precision': 0.5777777777777777, 'row_recall': 0.78, 'column_precision': 0.88268156424581, 'column_recall': 0.7919799498746867, 'cell_precision': 0.5406871609403255, 'cell_recall': 0.6016096579476862
# Direct Prompting : {'row_precision': 0.23756906077348067, 'row_recall': 0.21393034825870647, 'column_precision': 0.532608695652174, 'column_recall': 0.39095744680851063, 'cell_precision': 0.15, 'cell_recall': 0.0995850622406639}




# if feta_id ==3642579282991847187:
				# 	point["result"] = [[27,0]]
				# if feta_id == -1942986951387524846:
				# 	point["result"] = [[15,0],[15,1],[15,2]]
				# if feta_id == -276277944479206617:
				# 	point["result"] = [[39,2],[39,5]]
				# if feta_id==-3436542895472503263:
				# 	point["result"] = [[60,1],[60,2]]
				# if feta_id==-8843877009321434048:
				# 	point["result"] = [[1, 0], [14, 0], [25, 0]]
				# if feta_id==2097410980694798607:
				# 	point["result"] = [[12,0]]
				# if feta_id==-6897564476590801394:
				# 	point["result"] =  [[1, 1], [3, 1], [5, 1], [7, 1], [9, 1], [11, 1], [13, 1], [15, 1], [17, 1], [19, 1], [21, 1], [23, 1], [25, 1], [27, 1], [29, 1], [31, 1], [33, 4], [35, 1], [35, 4]]
