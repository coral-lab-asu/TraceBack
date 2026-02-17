import json
from pathlib import Path

def robust_load_json_or_jsonl(path):
    """Robustly load JSON or JSONL file."""
    text = open(path, "r", encoding="utf-8").read().strip()
    try:  # Try full JSON
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    # JSONL or concatenated
    decoder = json.JSONDecoder()
    idx, n = 0, len(text)
    items = []
    while idx < n:
        obj, end = decoder.raw_decode(text, idx)
        items.append(obj)
        idx = end
        while idx < n and text[idx].isspace():
            idx += 1
    return items


def normalize_item(raw):
    """Convert raw record to common schema."""
    if "table" in raw:  # AITQA
        return {
            "id": raw.get("id"),
            "table_id": raw.get("table_id", "aitqa"),
            "question": raw.get("question"),
            "answer": raw.get("answers", []),
            "table": raw["table"],
            "highlighted_cells": raw.get("highlighted_cells", []),  # fallback empty
        }
    elif "feta_id" in raw:  # FetaQA
        return {
            "id": f"feta-{raw['feta_id']}",
            "table_id": raw.get("table_page_title", "feta"),
            "question": raw["question"],
            "answer": raw.get("answer", ""),
            "table": raw["table_array"],
            "highlighted_cells": raw.get("highlighted_cells", raw.get("highlighted_cell_ids", [])),
        }
    elif "example_id" in raw:  # ToTTo
        return {
            "id": f"totto-{raw['example_id']}",
            "table_id": raw.get("table_page_title", "totto"),
            "question": raw["question"],
            "answer": raw.get("answer", ""),
            "table": raw["table_array"],
            "highlighted_cells": raw.get("highlighted_cells", []),
        }
    else:
        raise ValueError(f"Unknown format: {raw.keys()}")


def make_html(items, out_path="qa_table_viz.html"):
    html_template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QA Table Visualizer</title>
<style>
  :root {
    --border: #e0e0e0;
    --muted: #666;
    --highlight-bg: #fff3cd;
    --highlight-border: #f0ad4e;
  }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'; margin: 24px; }
  .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  select, input[type="checkbox"] { padding: 8px; font-size: 14px; }
  .card {
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    margin: 12px 0;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
  }
  .meta { font-size: 14px; color: var(--muted); margin-bottom: 6px; }
  table {
    border-collapse: separate;
    border-spacing: 0;
    margin-top: 8px;
    min-width: 600px;
  }
  th, td { padding: 8px 10px; border: 1px solid var(--border); }
  th { background: #f7f7f7; font-weight: 600; }
  .highlight { background: var(--highlight-bg) !important; border: 2px solid var(--highlight-border) !important; }
  .controls { margin-bottom: 8px; }
  .answer { margin-top: 6px; }
  .muted { color: var(--muted); }
</style>
</head>
<body>
  <h2>QA Table Visualizer</h2>

  <div class="controls row">
    <label>table_id:
      <select id="tableSelect"></select>
    </label>
    <label>question:
      <select id="qidSelect"></select>
    </label>
    <label><input type="checkbox" id="showAnswer" checked> Show answer</label>
  </div>

  <div id="card" class="card"></div>
  <div id="tableContainer"></div>

<script>
const DATA = __DATA__;

function uniqueTableIds(items) {
  return Array.from(new Set(items.map(d => d.table_id))).sort();
}

function itemsForTable(tid) {
  return DATA.filter(d => d.table_id === tid).sort((a,b) => a.id.localeCompare(b.id));
}

function findItem(tid, qid) {
  return itemsForTable(tid).find(d => d.id === qid) || null;
}

function renderOptions(el, values) {
  el.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    el.appendChild(opt);
  }
}

function renderCard(item, showAnswer) {
  const el = document.getElementById("card");
  if (!item) { el.innerHTML = "<b>No data selected.</b>"; return; }
  const ansHtml = showAnswer && item.answer && item.answer.length
    ? `<div class="answer"><b>Answer:</b> ${Array.isArray(item.answer) ? item.answer.join(", ") : item.answer}</div>`
    : "";
  el.innerHTML = `
    <div class="meta"><b>ID:</b> ${item.id} &nbsp; | &nbsp; <b>Table:</b> ${item.table_id}</div>
    <div style="font-size:16px;"><b>Question:</b> ${item.question}</div>
    ${ansHtml}
  `;
}

function renderTable(item) {
  const c = document.getElementById("tableContainer");
  c.innerHTML = "";
  if (!item || !item.table || !item.table.length) return;

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");

  // Header row
  const header = item.table[0];
  const trh = document.createElement("tr");
  for (const h of header) {
    const th = document.createElement("th");
    th.textContent = h;
    trh.appendChild(th);
  }
  thead.appendChild(trh);

  // Data rows
  for (let r = 1; r < item.table.length; r++) {
    const tr = document.createElement("tr");
    for (let cidx = 0; cidx < item.table[r].length; cidx++) {
      const td = document.createElement("td");
      td.textContent = item.table[r][cidx];
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  table.appendChild(thead);
  table.appendChild(tbody);
  c.appendChild(table);

  // Highlight cells
  if (Array.isArray(item.highlighted_cells)) {
    for (const pair of item.highlighted_cells) {
      const r = pair[0];
      const cidx = pair[1];
      const dataRowIndex = r - 1; // tbody starts after header
      if (dataRowIndex >= 0 &&
          dataRowIndex < tbody.rows.length &&
          cidx >= 0 &&
          cidx < tbody.rows[dataRowIndex].cells.length) {
        const cell = tbody.rows[dataRowIndex].cells[cidx];
        cell.classList.add("highlight");
      }
    }
  }
}

function init() {
  const tableSelect = document.getElementById("tableSelect");
  const qidSelect = document.getElementById("qidSelect");
  const showAnswer = document.getElementById("showAnswer");

  const tids = uniqueTableIds(DATA);
  renderOptions(tableSelect, tids);

  function refreshQids() {
    const tidsel = tableSelect.value;
    const items = itemsForTable(tidsel);
    const qids = items.map(d => d.id);
    renderOptions(qidSelect, qids);
  }

  function refreshAll() {
    const item = findItem(tableSelect.value, qidSelect.value);
    renderCard(item, showAnswer.checked);
    renderTable(item);
  }

  tableSelect.addEventListener("change", () => { refreshQids(); refreshAll(); });
  qidSelect.addEventListener("change", refreshAll);
  showAnswer.addEventListener("change", refreshAll);

  // Initial selections
  tableSelect.value = tids[0] || "";
  refreshQids();
  refreshAll();
}

init();
</script>
</body>
</html>
"""
    # inject data
    html = html_template.replace("__DATA__", json.dumps(items, ensure_ascii=False))
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"✅ Wrote {out_path}")



if __name__ == "__main__":
    path = "/Users/tejas/Downloads/Table-Attribution/Datasets/AITQA/aitqa_processed.jsonl"  # or FetaQA / ToTTo file
    raw_items = robust_load_json_or_jsonl(path)
    norm_items = [normalize_item(r) for r in raw_items]
    make_html(norm_items, out_path="./aqa_table_viz_AITQA.html")
