# AITR Lab 02, T21 Version A: verifying LLM citations

Vaibhav Sharma, MSE2026010. Course: AI Tools for Research.

**Question.** How well do three ways of checking LLM-generated citations agree with human judgement, and does a string-only detector trained on one model's citations still work on another model's citations?

- B0: rule baseline (red flags in the citation text)
- B1: string-only logistic regression detector (no database lookup), my own features (D11)
- B2: CrossRef + OpenAlex fuzzy matching, threshold fixed on pilot hand labels
- RQ2: B1 trained on Qwen2.5-7B-Instruct citations, tested on Mistral-7B-Instruct-v0.3 citations, and the reverse
- RQ3 (primary): precision, recall and F1 for "fabricated" of B0, B1, B2 against 200 hand labels

RQ1 (reproducing Naser 2026) was dropped because his data and code are not public (change notice D08).

## Layout

```
configs/protocol_v1.yaml   frozen settings (models, decoding, seeds, thresholds)
src/topics.py              topic list with OpenAlex prominence bands
src/generate.py            model answers (4-bit NF4 on a T4)
src/parse.py               answers -> citations
src/verify.py              B2 evidence and labels
src/b0_rules.py            B0
src/b1_detector.py         B1
src/analyse_a.py           RQ2, RQ3, cost, fabrication rate; tables and figures
scripts/env_test.py        Prompt 08 environment test
scripts/export_labels.py   blind labelling sheets
notebooks/                 Colab notebooks (Prompt 08, Prompt 09 pilot)
tests/                     unit tests (synthetic fixtures only)
results/                   env_report.json, results.csv, figures
paper/tables               LaTeX tables written by analyse_a.py
```

## Order of runs

```
python -m pytest -q
python -m src.topics
python -m src.generate --pilot
python -m src.parse --mode pilot
python -m src.verify --mode pilot
python scripts/export_labels.py --mode pilot     # then hand-label -> data/labels/gold_pilot.csv
python -m src.analyse_a --pilot                  # pick B2 threshold, write it in the config, freeze
python -m src.generate
python -m src.parse --mode main
python -m src.verify --mode main
python scripts/export_labels.py --mode main --n 200   # hand-label -> gold_main.csv
python -m src.analyse_a
```

## Secrets

`OPENALEX_API_KEY`, `CONTACT_EMAIL` (and optional `HF_TOKEN`) are read from environment variables, set from Colab Secrets. They are never written to any file in this repo.

## Environment

Tested on Colab, 17 Nov 2026: Python 3.13.16, torch 2.14.1+cu130, transformers 5.22.1, Tesla T4. Exact versions in `requirements.txt`.
