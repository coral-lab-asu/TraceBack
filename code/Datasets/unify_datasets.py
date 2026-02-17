import json
from pathlib import Path

# Input paths
aitqa_path = "/Users/tejas/Downloads/Table-Attribution/Datasets/AITQA/aitqa_processed.jsonl"
fetaqa_path = "/Users/tejas/Downloads/Table-Attribution/Datasets/FetaQA/fetaQA_dev_processed.jsonl"
totto_path = "/Users/tejas/Downloads/Table-Attribution/Datasets/Totto/totto_dataset_for_annotations.jsonl"

out_path = "/Users/tejas/Downloads/Table-Attribution/Datasets/CITEBENCH.json"

def load_mixed_json(path):
    """
    Load a file that may contain multiple JSON objects back-to-back
    (not necessarily one per line).
    Returns a list of dicts.
    """
    objs = []
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()

    idx = 0
    while idx < len(data):
        if data[idx].isspace():
            idx += 1
            continue
        obj, end = decoder.raw_decode(data, idx)
        objs.append(obj)
        idx = end
    return objs


unified = []
global_id = 0

# --- AITQA ---
for row in load_mixed_json(aitqa_path):
    global_id += 1
    unified.append({
        "example_id": global_id,
        "dataset": "aitqa",
        "table_id": row.get("table_id", "aitqa"),
        "qid": row.get("id", f"aitqa-{global_id}"),
        "table": row["table"],
        "question": row["question"],
        "answer": row.get("answers", []),  # already list
        "highlighted_cells": row.get("highlighted_cells", []),
    })

# --- FetaQA ---
for row in load_mixed_json(fetaqa_path):
    global_id += 1
    unified.append({
        "example_id": global_id,
        "dataset": "feta",
        "table_id": row.get("table_page_title", "feta"),
        "qid": f"feta-{row['feta_id']}",
        "table": row["table_array"],
        "question": row["question"],
        "answer": [row.get("answer", "")],  # force list
        "highlighted_cells": row.get("highlighted_cells", row.get("highlighted_cell_ids", [])),
    })

# --- ToTTo ---
for row in load_mixed_json(totto_path):
    global_id += 1
    unified.append({
        "example_id": global_id,
        "dataset": "totto",
        "table_id": row.get("table_page_title", "totto"),
        "qid": f"totto-{row['example_id']}",
        "table": row["table_array"],
        "question": row["question"],
        "answer": [row.get("answer", "")],  # force list
        "highlighted_cells": row.get("highlighted_cells", []),
    })

# Save
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(unified, f, ensure_ascii=False, indent=2)

print(f"✅ Saved unified dataset with {len(unified)} entries -> {out_path}")
