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


def run_inference( question,answer,column_headers,table_title):
	# Resolve prompt relative to repo root (parent of Code/)
	root_dir = Path(__file__).resolve().parent.parent
	prompt_path = root_dir / "Prompts" / "subquery_prompt.txt"
	prompt = load_prompt(prompt_path)
	
	if not prompt:
		print("Prompt not loaded. Exiting.")
		return []

	#message = f"Schema : {schema}\n{prompt}:\n Question: {data_point}"
	message = f"{prompt}\nInput :\nTable-Schema :\n<Column Names>: {column_headers}\nTable Title: {table_title}\nQuestion: {question}\nAnswer: {answer}\n\nOutput : \n"

	response = model_response(message)
	return response

def model_response(message):
	try:
		return model.call(message)
	except Exception as e:
		print(f"An unexpected error occurred in LLM: {e}")


model_name = 'gpt-4o'
model = Call_OpenAI(model_name)

def get_subqueries(question,answer,column_headers,table_title):
	results = run_inference(question,answer,column_headers,table_title)

	return results
