import os
import json
import pandas as pd
import time
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path
from LLM import Call_OpenAI, Call_Gemini, Call_DeepSeek, Call_HF, Call_Novita
from relevant_col_gen import get_relevant_cols
from query_attribution import attribution
from subqueries import get_subqueries
import logging


def _get_tqdm(disabled: bool):
	if disabled:
		return None
	try:
		from tqdm.auto import tqdm  # type: ignore
	except Exception:
		return None
	return tqdm


def _iter_jsonl(path: Path):
	with path.open("r", encoding="utf-8") as file:
		for line_no, line in enumerate(file, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				data = json.loads(line)
			except Exception:
				print(f"[Warn] Skipping malformed JSON line {line_no}")
				continue
			if isinstance(data, dict):
				yield line_no, data

def col_filter_table(cols, df):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Include only selected columns + row_index
    filtered_df = df[["row_id"] + cols]

    return filtered_df

def make_unique(headers):
	counts = {}
	unique = []
	for h in headers:
		counts[h] = counts.get(h, 0) + 1
		if counts[h] == 1:
			unique.append(h)
		else:
			unique.append(f"{h}_{counts[h]}")
	return unique

def convert_to_df(table_rows):
	max_cols = max(len(row) for row in table_rows)
	print('original:\n', table_rows)
	fill_value = ''
	# Step 2: Normalize each row to have max_cols entries
	padded_rows = [row + [fill_value] * (max_cols - len(row)) for row in table_rows]
	raw_header = padded_rows[0]

	header = make_unique(raw_header)
	padded_rows = [header] + padded_rows[1:]

	# Step 4: Add row ids
	row_numbers = list(range(len(padded_rows)))
	# Step 5: Build DataFrame with row number column
	df = pd.DataFrame(padded_rows, columns=header)
	df.insert(0, "row_id", row_numbers)
	# Try to coerce datatypes
	df = df.apply(pd.to_numeric, errors='ignore')
	return df

def map_to_original_cols(ans,column_headers,relevant_cols):
	col_dict = {}
	for column_index in range(len(column_headers)):
		col_dict[column_headers[column_index]] = column_index
	print("Column Headers : ",column_headers,"\nRelevant Columns : ",relevant_cols)
	mapped = []
	for cell in ans:
		if not cell:
			continue
		# Expect (row_idx, col_idx_within_relevant_cols)
		if not isinstance(cell, (list, tuple)) or len(cell) < 2:
			continue
		row_idx, col_idx = int(cell[0]), int(cell[1])
		# Guard against malformed column indices
		if col_idx < 0 or col_idx >= len(relevant_cols):
			continue
		header_name = relevant_cols[col_idx]
		if header_name not in col_dict:
			continue
		orig_col_idx = col_dict[header_name]
		mapped.append([row_idx, orig_col_idx])
	return mapped

def parse_attribution(text,column_headers,relevant_cols):
	pattern = re.compile(r'\[(.*?)\]$')

	final_answer = []

	# Split into lines and search for matches
	for line in text.splitlines():
		line = line.strip()
		print('line: ',line)
		'''
		match = pattern.search(line)

		if match:
			content = match.group(1).strip()  # The part inside [ ... ]
			content = '['+content+']'
			if content:
				try:
					lists = eval(content)
					print('lists', lists)

					# for l in lists:
					# 	if len(l) > 1:
					# 		final_answer.extend(l)
					# 	else:
					# 		final_answer.append(l)

				except:
					continue
				
			else:
				# If there's no content (i.e. an empty []), just store an empty tuple
				final_answer.append(())
		'''

		st = []
		for char in line:
			if char in ["(","[",","] or char.isnumeric():
				st.append(char)
			elif char in [")","]"]:
				res = ''
				while st:
					ch = st.pop()
					if ch in ["(","["]:
						break
					res+=ch
				try:
					tup = eval('('+res[::-1]+')')
					# Keep only (row, col) style 2-tuples; skip stray ints/singles
					if isinstance(tup, tuple) and len(tup) == 2:
						final_answer.append(tup)
				except Exception:
					continue


	final_ans = map_to_original_cols(final_answer,column_headers,relevant_cols)

	return final_ans


def sanitize_model_id(model_id: str) -> str:
	return model_id.replace("/", "__").replace(":", "_")


def build_llm(args):
	backend = str(args.backend).lower().strip()
	if backend == "openai":
		return Call_OpenAI(args.model)
	if backend == "gemini":
		return Call_Gemini(args.model)
	if backend == "deepseek":
		return Call_DeepSeek(args.model)
	if backend == "novita":
		return Call_Novita(
			args.model,
			max_tokens=int(args.max_new_tokens),
			temperature=float(args.temperature),
			top_p=float(args.top_p),
		)
	if backend == "hf":
		cache_dir = args.cache_dir.strip() or None
		return Call_HF(
			args.model,
			max_new_tokens=int(args.max_new_tokens),
			temperature=float(args.temperature),
			top_p=float(args.top_p),
			cache_dir=cache_dir,
		)
	raise ValueError(f"Unknown backend: {args.backend}")


def main():
	parser = argparse.ArgumentParser(description="Run TraceBack workflow on FetaQA.")
	parser.add_argument("--backend", default="openai", choices=["openai", "gemini", "deepseek", "novita", "hf"])
	parser.add_argument("--model", default="gpt-4o", help="Model name (OpenAI/Gemini/DeepSeek) or HF model id")
	parser.add_argument("--limit", type=int, default=0, help="Max examples to run (0 = all)")
	parser.add_argument("--resume", action="store_true", help="Resume from existing output JSON (skip processed ids)")
	parser.add_argument("--refresh-debug", action="store_true", help="With --resume: recompute entries missing traceback_debug")
	parser.add_argument("--output", default="", help="Output JSON path (default: derived from model)")
	parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output")
	parser.add_argument("--verbose", action="store_true", help="Print question/answer per example")
	parser.add_argument("--max-new-tokens", type=int, default=512, help="HF only: max new tokens per call")
	parser.add_argument("--temperature", type=float, default=0.0, help="HF only: sampling temperature")
	parser.add_argument("--top-p", type=float, default=1.0, help="HF only: nucleus sampling top_p")
	parser.add_argument("--cache-dir", default="", help="HF only: cache dir for model/tokenizer downloads")
	parser.add_argument("--no-row-filtering", action="store_true", help="Disable Step 2 (row filtering)")
	parser.add_argument("--no-nli-filtering", action="store_true", help="Disable NLI filtering for sub-queries")
	parser.add_argument("--nli-threshold", type=float, default=0.5, help="NLI entailment threshold for sub-query filtering (lower keeps more)")
	parser.add_argument("--no-mysql", action="store_true", help="Allow SQLite fallback instead of requiring MySQL")
	parser.add_argument("--mysql-table-name", default="", help="MySQL table name prefix (default: per-run unique)")
	args = parser.parse_args()

	all_results = []
	# Resolve dataset path relative to repo root (parent of this file's directory)
	root_dir = Path(__file__).resolve().parent.parent
	file_path = root_dir / 'Datasets' / 'FetaQA' / 'fetaQA_dev_processed.jsonl'
	model_slug = sanitize_model_id(args.model)
	output_file_path = Path(args.output) if args.output else (Path("fetaqa_results") / f'fetaqa_answer_attribution_{model_slug}_withrows_1.json')
	# Ensure repo root is importable so we can use the workflow runner.
	if str(root_dir) not in sys.path:
		sys.path.append(str(root_dir))
	from traceback_workflow import TraceBackWorkflowRunner

	model = build_llm(args)
	runner = TraceBackWorkflowRunner(
		call_llm=model.call,
		nli_threshold=float(args.nli_threshold),
		enable_row_filtering=not bool(args.no_row_filtering),
		enable_nli_filtering=not bool(args.no_nli_filtering),
		require_mysql=not bool(args.no_mysql),
		mysql_table_name=(args.mysql_table_name.strip() or f"traceback_tmp_feta_{os.getpid()}"),
	)
	processed = 0
	processed_ids = set()
	index_by_id = {}

	# Resume support
	if args.resume and output_file_path.exists():
		try:
			existing = json.loads(output_file_path.read_text(encoding="utf-8"))
			if isinstance(existing, list):
				all_results = existing
				for idx, rec in enumerate(all_results):
					fid = rec.get("feta_id")
					if fid is not None:
						index_by_id[fid] = idx
						if args.refresh_debug and not rec.get("traceback_debug"):
							continue
						processed_ids.add(fid)
				print(f"[Resume] Loaded {len(all_results)} existing records from {output_file_path}")
		except Exception as e:
			print(f"[Warn] Could not load existing output for resume: {e}")
	# Ensure logs directory exists next to this file
	log_dir = Path(__file__).resolve().parent / "logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	log_file = log_dir / f'fetaqa_{model_slug}_traceback.log'
	logging.basicConfig(filename=str(log_file), level=logging.INFO, format='%(asctime)s - %(message)s')

	# Pre-compute progress total (remaining examples)
	todo_total = 0
	for _line_no, data in _iter_jsonl(file_path):
		feta_id = data.get("feta_id")
		if processed_ids and feta_id in processed_ids:
			continue
		table_array = data.get("table_array")
		highlighted_cell_ids = data.get("highlighted_cells")
		if highlighted_cell_ids is None:
			continue
		if not table_array or not isinstance(table_array, list) or not table_array[0]:
			continue
		todo_total += 1

	effective_total = todo_total
	if args.limit and int(args.limit) > 0:
		effective_total = min(effective_total, int(args.limit))

	tqdm = _get_tqdm(bool(args.no_progress))
	pbar = tqdm(total=effective_total, desc=f"FetaQA | {model_slug}", unit="ex", dynamic_ncols=True) if (tqdm and effective_total) else None
			
	for line_no, data in _iter_jsonl(file_path):
		if args.limit and int(args.limit) > 0 and processed >= int(args.limit):
			break

		# Extract desired fields
		feta_id = data.get("feta_id")
		if processed_ids and feta_id in processed_ids:
			continue
		question = data.get("question")
		answer = data.get("answer")
		table_array = data.get("table_array")
		highlighted_cell_ids = data.get("highlighted_cells")
		if highlighted_cell_ids is None:
			continue
		if not table_array or not isinstance(table_array, list) or not table_array[0]:
			continue
		table_page_title = data.get("table_page_title")
		table_section_title = data.get("table_section_title")
		table_title = f"{table_page_title} - {table_section_title}"
		#table_title = f"AITQA-{feta_id}"
		if args.verbose:
			print("Question:",question)
			print("Answer:",answer)

		# Insert table into database
		# df = convert_to_df(table_array)
		# database = DataBase()
		# database.upload_table(table_title, df)

		logging.info(table_array)
		logging.info(question)

		logging.info('-'*50)
		logging.info(answer)
		logging.info(highlighted_cell_ids)
			
		try:
			out = runner.run_example(
				table=table_array,
				question=question,
				answer=answer,
				table_title=table_title,
			)
			corrected_attributed_queries = out.get("result_cells") or []
			traceback_debug = out.get("debug") or {}
			relevant_cols = (traceback_debug or {}).get("relevant_cols") or []
		except Exception as e:
			print(f"[Warn] TraceBack workflow failed for {feta_id}: {e}")
			corrected_attributed_queries = []
			relevant_cols = []
			traceback_debug = {"error": str(e)}

		logging.info('-'*50)
		logging.info(corrected_attributed_queries)

		logging.info('-'*50)

		logging.info('-'*50)
		logging.info('\n\n')
		# Final Answer attribution
		rec = {
			"feta_id": feta_id,
			"result": corrected_attributed_queries,
			"relevant_column_headers": relevant_cols,
			"highlighted_cell_ids": highlighted_cell_ids,
			"traceback_debug": traceback_debug,
		}
		if feta_id in index_by_id:
			all_results[index_by_id[feta_id]] = rec
		else:
			index_by_id[feta_id] = len(all_results)
			all_results.append(rec)
		# Ensure output directory exists
		output_file_path.parent.mkdir(parents=True, exist_ok=True)
		with output_file_path.open("w", encoding="utf-8") as f:
			json.dump(all_results, f, indent=4)
			

		processed += 1
		if pbar is not None:
			pbar.update(1)
		else:
			print(f"[{processed}/{effective_total or '?'}] Done id={feta_id} (line {line_no})")

	if pbar is not None:
		pbar.close()

if __name__ == "__main__":
	main()
