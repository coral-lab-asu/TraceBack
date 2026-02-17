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

def load_relevant_cols(txt_file_path):
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


def run_inference(subqueries, table_title, table_array, column_headers,answer):
    # Resolve prompt relative to repo root (parent of Code_with_rows/)
    root_dir = Path(__file__).resolve().parent.parent
    prompt_path = root_dir / "Prompts" / "answer_attribution_aitqa.txt"
    prompt = load_prompt(prompt_path)

    if not prompt:
        print("Prompt not loaded. Exiting.")
        return []

    # message = f"Schema : {schema}\n{prompt}:\n Question: {data_point}"
    table_array = table_array[1:]
    message = f"{prompt}\nInput :\nTable-Schema :\n<Relevant-Columns>: {column_headers}\nTable Title: {table_title}\nTable Rows: {table_array.to_string()}\n<Questions>: {subqueries}\n<Original-Answer>:\n{answer}\n\nOutput : \n"

    response = model_response(message)
    return response


def model_response(message):
    try:
        return model.call(message)
    except Exception as e:
        print(f"An unexpected error occurred in LLM: {e}")


# Load the CSV file
all_results = []
file_path = '../Datasets/fetaQA/fetaQA-v1_dev.jsonl'

model_name = 'gpt-4o'
model = Call_OpenAI(model_name)
output_file_path = f'answer_attribution_{model_name}.json'

def attribution(subqueries, table_title, table_array, column_headers,answer):
    results = run_inference(subqueries, table_title, table_array, column_headers,answer)

    return results

