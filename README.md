# TRACEBACK: Multi-Agent Decomposition for Fine-Grained Table Attribution

[![arXiv](https://img.shields.io/badge/arXiv-2602.13059-b31b1b.svg)](https://arxiv.org/abs/2602.13059)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://coral-lab-asu.github.io/TraceBack/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**TRACEBACK** is a multi-agent attribution framework that traces table-based answers back to their supporting cells, providing fine-grained attribution at the cell level rather than coarse row or column granularity.

---

## Overview

![TRACEBACK Overview](static/images/traceback_pipeline_clean.png)

Table QA systems can produce correct answers yet offer no way to verify **which cells** actually support them. Existing approaches either skip attribution entirely or operate at coarse row/column granularity, leaving fine-grained evidence trails unaddressed. TRACEBACK closes this gap with a modular, multi-agent pipeline for **cell-level attribution** in single-table QA.

**TRACEBACK** works in five key steps:

1. **Column Pruning** — Identify columns relevant to the question via an LLM agent, reducing noise from large tables.
2. **Row Filtering** — Generate and execute SQL to retain only the rows needed for answering (optional, via MySQL/SQLite).
3. **Sub-query Decomposition** — Break the question into atomic sub-queries, each targeting a single fact, with NLI-based filtering for faithfulness.
4. **Sub-query Attribution** — Align each sub-query to specific table cells, capturing both direct evidence and intermediate reasoning steps.
5. **Final Attribution** — Consolidate cell-level evidence across all sub-queries into a unified attribution map.

We also introduce **CITEBench**, a benchmark with phrase-to-cell annotations drawn from ToTTo, FetaQA, and AITQA, and **FairScore**, a reference-less metric that estimates attribution precision and recall without human cell labels.

This repository contains the code, prompts, datasets, and evaluation pipelines for reproducing and extending TRACEBACK, CITEBench, and FairScore.

---

## Repository Structure

```
code/
├── src/                        
│   ├── main_aitqa.py           # TraceBack runner for AITQA
│   ├── main_fetaqa.py          # TraceBack runner for FetaQA
│   ├── totto_traceback.py      # TraceBack runner for ToTTo
│   ├── LLM.py                  # Unified LLM backends (OpenAI, Gemini, DeepSeek, Novita, HF)
│   ├── database.py             # MySQL/SQLAlchemy interface for row filtering
│   ├── relevant_col_gen.py     # Step 1: Column relevance via LLM
│   ├── relevant_row_gen.py     # Step 2: Row filtering via SQL generation
│   ├── query_attribution.py    # Step 4: Subquery attribution
│   ├── subqueries.py           # Step 3: Subquery generation
│   ├── eval_aitqa.py           # AITQA precision/recall evaluation
│   ├── eval_fetaqa.py          # FetaQA precision/recall evaluation
│   ├── eval_totto.py           # ToTTo evaluation
│   ├── eval_traceback_md.py    # Combined markdown table (AITQA/FetaQA/ToTTo)
│   ├── eval_traceback_hf_all.py # HF model results aggregation
│   ├── traceback_citebench_full.py # TraceBack over CITEBENCH
│   ├── eval_citebench_all.py   # CITEBENCH evaluator
│   ├── eval_fairscore.py       # FAIRScore (reference-less) evaluation
│   ├── icl_runner.py           # ICL baseline for CITEBENCH
│   ├── metrics_to_md*.py       # CITEBENCH metrics → markdown converters
│   └── utils.py                # Shared utilities
├── traceback_workflow.py       # Core 5-step TraceBack workflow
├── Prompts/                    # TraceBack prompt templates
├── Prompts_2/                  # ICL baseline prompt templates
├── Datasets/                   # AITQA, FetaQA, ToTTo, CITEBENCH
└── README.md
```

## 1) Project Layout

Key scripts:
- `code/traceback_workflow.py`: shared 5-step TraceBack workflow used by dataset runners
- `code/src/main_aitqa.py`: TraceBack runner for AITQA
- `code/src/main_fetaqa.py`: TraceBack runner for FetaQA
- `code/src/totto_traceback.py`: TraceBack runner for ToTTo
- `code/src/eval_aitqa.py`, `code/src/eval_fetaqa.py`, `code/src/eval_totto.py`: per-dataset P/R evaluation
- `code/src/eval_traceback_md.py`: one-shot markdown table for AITQA/FetaQA/ToTTo
- `code/src/traceback_citebench_full.py`: TraceBack over unified CITEBENCH
- `code/src/eval_citebench_all.py`: CITEBENCH evaluator for style-based outputs
- `code/src/eval_fairscore.py`: FAIRScore evaluation
- `code/src/icl_runner.py`: CITEBENCH ICL baseline

Prompt files:
- TraceBack prompts: `code/Prompts/`
- ICL prompt templates: `code/Prompts_2/`

Datasets:
- `code/Datasets/AITQA/aitqa_processed.jsonl`
- `code/Datasets/FetaQA/fetaQA_dev_processed.jsonl`
- `code/Datasets/Totto/totto_processed.jsonl`
- `code/Datasets/CITEBENCH.json`

## 2) Environment

Required (core):
```bash
pip install pandas sqlalchemy pymysql tqdm openai
```

Optional by backend:
- Gemini backend: `pip install google-genai`
- Local HF backend: `pip install torch transformers accelerate`

Environment variables (as needed by backend):
- `OPENAI_API_KEY`
- `GEMINI_API_KEY` or `GEMINI_API_KEYS`
- `DEEPINFRA_API_KEY` or `DEEPSEEK_API_KEY`
- `NOVITA_API_KEY`
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (for gated HF models)

## 3) TraceBack on AITQA / FetaQA / ToTTo

### AITQA
Run:
```bash
python code/src/main_aitqa.py
```
Evaluate:
```bash
python code/src/eval_aitqa.py
```

### FetaQA
Run:
```bash
python code/src/main_fetaqa.py
```
Evaluate:
```bash
python code/src/eval_fetaqa.py
```

### ToTTo
Run:
```bash
python code/src/totto_traceback.py
```
Evaluate:
```bash
python code/src/eval_totto.py
```

### Local HF TraceBack runs
```bash
python code/src/main_aitqa.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --resume
python code/src/main_fetaqa.py --backend hf --model google/gemma-3-4b-it --resume
python code/src/totto_traceback.py --backend hf --model Qwen/Qwen2.5-3B-Instruct --resume
```

Notes:
- For paper-faithful Step 2 row filtering, use MySQL setup in `code/MYSQL_USAGE.txt`.
- For quick runs without MySQL, add `--no-mysql` or `--no-row-filtering`.
- Use `--output` to avoid overwriting existing prediction files.
- Use `--resume` to continue interrupted runs.

### Combined TraceBack markdown table (AITQA/FetaQA/ToTTo)
```bash
python code/src/eval_traceback_md.py --percent --output results/eval/metrics_traceback.md
```

## 4) CITEBENCH Experiments

### 4.1 TraceBack full pipeline on CITEBENCH
Run:
```bash
python code/src/traceback_citebench_full.py \
  --citebench code/Datasets/CITEBENCH.json \
  --outdir results/TraceBack_full \
  --model gpt-4o
```

Evaluate:
```bash
python code/src/eval_citebench_all.py \
  --results-root results/TraceBack_full \
  --gt code/Datasets/CITEBENCH.json \
  --styles traceback-full \
  --output results/eval/metrics_citebench_traceback_full.json
```

### 4.2 ICL baseline (CITEBENCH)
ICL prompt templates:
- `code/Prompts_2/answer_attr_zero.txt`
- `code/Prompts_2/answer_attr_zero_cot.txt`
- `code/Prompts_2/answer_attr_few.txt`
- `code/Prompts_2/answer_attr_few_cot.txt`

Run (OpenAI example):
```bash
python code/src/icl_runner.py \
  --backend openai \
  --models gpt-4o \
  --styles zero zero-cot few few-cot \
  --limit 0 \
  --resume
```

Evaluate:
```bash
python code/src/eval_citebench_all.py \
  --results-root results/ICL \
  --gt code/Datasets/CITEBENCH.json \
  --styles zero zero-cot few few-cot \
  --output results/eval/metrics_citebench_icl.json
```

## 5) FAIRScore (Reference-less)

Default run over TraceBack outputs:
```bash
python code/src/eval_fairscore.py --cells pred --backend openai --model gpt-4o
```

Parallel requests (be mindful of rate limits):
```bash
python code/src/eval_fairscore.py --cells pred --backend openai --model gpt-4o --workers 4 --max-inflight 4
```

Score both predicted and gold cells:
```bash
python code/src/eval_fairscore.py --cells both --backend openai --model gpt-4o
```

Outputs:
- Summary markdown: `results/eval/metrics_fairscore.md`
- Summary JSON: `results/eval/metrics_fairscore.json`
- Cache files: `results/fairscore/`

Use a unified CITEBENCH-style prediction file via `--preds`:
```bash
python code/src/eval_fairscore.py \
  --preds results/ICL/gpt-4o/few-cot.json \
  --pred-tag few-cot \
  --cells pred \
  --backend openai --model gpt-4o \
  --workers 4 --max-inflight 4 \
  --summary-md results/eval/metrics_fairscore_few-cot.md \
  --summary-json results/eval/metrics_fairscore_few-cot.json
```

## 6) MySQL (Optional, for strict TraceBack Step 2)

See `code/MYSQL_USAGE.txt` for full setup/start/stop instructions.

## Citation
If you use this repository in your research, please cite the accompanying paper (TRACEBACK). 
```bibtex
@misc{anvekar2026tracebackmultiagentdecompositionfinegrained,
      title={TraceBack: Multi-Agent Decomposition for Fine-Grained Table Attribution}, 
      author={Tejas Anvekar and Junha Park and Rajat Jha and Devanshu Gupta and Poojah Ganesan and Puneeth Mathur and Vivek Gupta},
      year={2026},
      eprint={2602.13059},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.13059}, 
}
```

## License
Please see the [LICENSE](LICENSE) file if provided. If absent, contact the authors for licensing information.

## Contributing
Contributions are welcome. Please open an issue or a pull request for fixes and improvements
