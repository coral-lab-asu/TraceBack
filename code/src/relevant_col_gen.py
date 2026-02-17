import os
import json
import pandas as pd
import time
import re
from datetime import datetime
from pathlib import Path
from LLM import Call_OpenAI, Call_Gemini, Call_DeepSeek

def load_prompt(txt_file_path):
	try:
		# Try reading with UTF-8 encoding
		with open(txt_file_path, 'r', encoding='utf-8') as file:
			return file.read().strip()
	except UnicodeDecodeError:
		print(f"UTF-8 decoding failed for {txt_file_path}. Trying ISO-8859-1...")
		try:
			# Fallback to ISO-8859-1 encoding
			with open(txt_file_path, 'r', encoding='ISO-8859-1') as file:
				return file.read().strip()
		except Exception as e:
			print(f"Error reading prompt file with fallback encoding: {str(e)}")
			return ""
	except FileNotFoundError:
		print(f"Prompt file not found at {txt_file_path}")
		return ""
	except Exception as e:
		print(f"Error reading prompt file: {str(e)}")
		return ""

def get_cols(cols_text):
	if not isinstance(cols_text, str):
		cols_text = str(cols_text)
	match = re.search(r"(?<=<Relevant Columns>: )(.*)", cols_text, re.IGNORECASE)
	if match:
		cols = match.group(1)
	else:
		print('No relevant cols found\n')
		cols = '[]'
	return cols

def run_inference( question, subqueries,answer,column_headers,table_title):
	# Resolve prompt relative to repo root (parent of Code/)
	root_dir = Path(__file__).resolve().parent.parent
	prompt_path = root_dir / "Prompts" / "relevant_cols.txt"
	prompt = load_prompt(prompt_path)
	
	if not prompt:
		print("Prompt not loaded. Exiting.")
		return []

	#message = f"Schema : {schema}\n{prompt}:\n Question: {data_point}"
	message = f"{prompt}\nInput : \nTable Title: {table_title}\n<Column Names>: {column_headers}\nSub-Queries : \n{subqueries} \nQuestion: {question}\nAnswer: {answer}\n\nOutput : \n"

	response = model_response(message)
	return response

def model_response(message):
	try:
		return model.call(message)
	except Exception as e:
		print(f"An unexpected error occurred in LLM: {e}")


model_name = 'gpt-4o'
model = Call_OpenAI(model_name)
def get_relevant_cols(question, subqueries,answer,column_headers,table_title):
	results = run_inference(question, subqueries,answer,column_headers,table_title)
	print(results)
	if not results:
		# Fallback: use all headers if model failed
		return list(column_headers)
	relevant_cols = get_cols(results)
	try:
		return eval(relevant_cols)
	except Exception:
		# Safe fallback
		return list(column_headers)
