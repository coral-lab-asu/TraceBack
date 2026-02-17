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
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except UnicodeDecodeError:
        try:
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


def get_sql(sql):
    pattern = r"\b(?:CREATE TABLE|SELECT)\b.*?;"
    matches = re.findall(pattern, sql, re.DOTALL)
    for match in matches:
        return match
    return None


def run_inference(title, relevant_cols, table_array, question, answer, model):
    root_dir = Path(__file__).resolve().parent.parent
    prompt_path = root_dir / "Prompts" / "relevant_rows.txt"
    prompt = load_prompt(prompt_path)

    if not prompt:
        print("Prompt not loaded. Exiting.")
        return ""

    message = (
        f"{prompt}\nInput :\nTable-Schema :\n <Table Title>: {title}\n"
        f"<Column Names>: {relevant_cols}\nTable: {table_array.to_string()}\n"
        f"Question: {question}\nAnswer: {answer}\n\nOutput : \n"
    )

    try:
        return model.call(message)
    except Exception as e:
        print(f"An unexpected error occurred in LLM: {e}")
        return ""


def get_relevant_rows(database, title, relevant_cols, df, question, answer):
    """
    Row filtering using LLM-generated SQL (Evidence Span Extractor).
    This variant uses OpenAI GPT-4o via Call_OpenAI instead of Gemini.
    If the LLM call fails or produces no valid SQL, we fall back to
    returning the original df.
    """
    model = Call_OpenAI("gpt-4o")

    results = run_inference(title, relevant_cols, df, question, answer, model)
    sql = get_sql(results or "")
    if not sql:
        return df

    try:
        table = database.run_sql(sql)
    except Exception:
        return df

    if table is None or table.shape[0] == 0:
        return df
    return table
