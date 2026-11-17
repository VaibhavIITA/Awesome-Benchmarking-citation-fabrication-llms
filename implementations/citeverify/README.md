# CiteVerify: rules, strings or databases for checking LLM citations

Implementation, data and paper for the AI Tools for Research final assignment, topic T21 (Version A).
Vaibhav Sharma, MSE2026010, Department of Software and Automation, Indian Institute of Information Technology Allahabad (IIIT-A).

Paper: [`paper/main.pdf`](paper/main.pdf) (source: `paper/main.tex`).

## Problem

Language models produce references that look real but often do not exist. Studies that measure this depend on a checker, and checkers are rarely compared on the same citations. This project compares three:

| Checker | Idea | Code |
|---|---|---|
| B0 rule baseline | Red flags in the citation string (no DOI, bad year, no authors, short title) | `src/b0_rules.py` |
| B1 string-only detector | Logistic regression on 16 string features, no database lookup | `src/b1_detector.py` |
| B2 database matching | CrossRef + OpenAlex title search, DOI check via CrossRef and DataCite | `src/verify.py` |

Research questions (numbering as in the paper):

- RQ1: How do B0, B1 and B2 compare in accuracy and cost against reference labels?
- RQ2: Does B1, trained on one model's citations, keep its performance on another model's citations?

## Main results (main run, 17 Nov 2026)

108 citations from Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 on 6 topics. Reference labels: 103 fabricated, 5 real works with errors, 0 fully correct.

| Checker | Recall (fabricated) | Specificity | AUC | Seconds per citation |
|---|---|---|---|---|
| B0 rules | 0.029 | 4 of 5 | 0.415 | under 0.001 |
| B1 string detector | 0.893 | 2 of 5 | 0.672 | under 0.001 |
| B2 database | 0.854 | 3 of 5 | 0.817 | 5.18 (3.69 API calls) |

All values: `results/results.csv`. Main caveat: the reference labels were made with AI assistance and partly from the same databases as B2 (see Limitations).

## Repository layout

```
configs/protocol_v1.yaml     frozen settings: models + pinned revisions, decoding, seeds, thresholds
src/                         topics, generate, parse, verify (B2), b0_rules, b1_detector, analyse_a
scripts/env_test.py          environment test (Prompt 08)
scripts/export_labels.py     blind labelling sheets
notebooks/                   Prompt08_env_test, Prompt09_pilot (Colab), Prompt13_fast_main_kaggle (Kaggle)
tests/                       36 unit tests (synthetic fixtures only)
data/topics_input.csv        90 candidate topics
data/topics_frozen.csv       66 topics with OpenAlex counts, bands, pilot flags
data/raw/                    model answers (pilot and main)
data/parsed/                 parsed citations (pilot and main)
data/labels/                 B2 evidence (auto_*.jsonl), blind sheets, key maps, reference labels (gold_*.csv)
results/                     results.csv, figures, threshold sweeps, B2 disagreements, environment reports
paper/                       main.tex, main.pdf, refs.bib, tables/, figures/
logs/                        console logs of every real run (p09_*, p10_*, p13_*)
research_decisions.md        every protocol decision (D01 to D31)
debugging_log.md             real bugs found and fixed
ai_assistance_log.md         what the AI assistant did and what I checked
labelling_rubric.md          rules for reference labels
```

## Setup

Python 3.13 was used. GPU only needed for generation.

```bash
git clone https://github.com/VaibhavIITA/Awesome-Benchmarking-citation-fabrication-llms.git
cd Awesome-Benchmarking-citation-fabrication-llms/implementations/citeverify
pip install -r requirements.in        # lower bounds
python -m pytest -q                   # expect: 36 passed
```

Exact environments: `requirements.txt` (Colab, pilot) and `results/requirements_kaggle.txt` (Kaggle, main run).

API access (only for `src.topics` and `src.verify`): set environment variables `OPENALEX_API_KEY` and `CONTACT_EMAIL` (in Colab or Kaggle Secrets). No key is stored in this repository; never commit one.

## Reproduce the tables and figures (no GPU, no API key)

This step only reads the saved outputs in `data/`.

```bash
python -m src.analyse_a
```

Writes `results/results.csv` (48 rows), `paper/tables/t1_transfer.tex`, `t2_verifiers.tex`, `t3_cost.tex`, `t4_fr.tex`, and the figures `f1_roc_transfer` and `f2_b2_sensitivity` (PNG in `results/figures`, PDF in `paper/figures`). Results are deterministic (fixed seeds).

Status: VERIFIED. Rerun by me on Google Colab (CPU) after cloning this repository on 17 Nov 2026; same tables as in the paper. Unit tests: 36 passed on Kaggle (logs/p13_pytest.log).

## Full pipeline (status of each command)

| Step | Command | Where it was run | Status |
|---|---|---|---|
| Environment test | `python scripts/env_test.py` | Colab T4 | VERIFIED (results/env_report.json) |
| Topic list | `python -m src.topics` | Colab | VERIFIED (logs/p09_topics.log) |
| Pilot generation | `python -m src.generate --pilot` | Colab T4, 26 min | VERIFIED (logs/p09_generate_pilot.log) |
| Pilot parse, B2 | `python -m src.parse --mode pilot`, `python -m src.verify --mode pilot` | Colab | VERIFIED |
| Pilot labelling sheet | `python scripts/export_labels.py --mode pilot` | Colab | VERIFIED |
| Threshold from pilot | `python -m src.analyse_a --pilot` | AI sandbox on my pilot files | VERIFIED (logs/p10_threshold.log) |
| Main generation, parse, B2, sheet | `notebooks/Prompt13_fast_main_kaggle.ipynb` (both models in parallel) | Kaggle 2 x T4, 20.4 min | VERIFIED (logs/p13_*.log) |
| Analysis | `python -m src.analyse_a` | AI sandbox, then Colab | VERIFIED |
| Main run on all 60 topics | set `topics.main_per_band: null` and `max_new_tokens: 2048`, rerun the Kaggle notebook | not run | [TEST REQUIRED] |
| Robustness seeds, quantization check | `python -m src.generate --robust` / `--quant` | not run | [TEST REQUIRED] |

Note: rerunning generation produces new model answers. Without the same GPU type, library versions and pinned revisions, the text can differ, so the numbers will not be identical. The saved answers in `data/raw/` are the ones the paper reports.

## Compute

| Step | Hardware | Time |
|---|---|---|
| Pilot generation (12 answers) | 1 x T4, Colab | 26 min (incl. model downloads) |
| Main generation (12 answers) | 2 x T4, Kaggle | 11 min |
| B2 checks | CPU + internet | about 5 s per citation (CrossRef limit 3 requests per second) |
| Analysis | CPU | under 1 min |

## Limitations

- Reference labels are AI-made (decisions D24, D27) and partly based on the same CrossRef/OpenAlex results as B2, so B2's scores are an upper bound.
- Small sample: 6 main topics, 108 citations, 5 non-fabricated works, one seed, one run.
- 7 of 12 main-run answers hit the 1,024-token limit.
- Two 7B models in 4-bit, one prompt; no commercial models or agentic search tools.
- Protocol changes before freezing are logged in `research_decisions.md` (for example D08: reproduction of Naser 2026 dropped because its data and code were not public; D25: reduced main run).

## Minimum reproduction checklist (for a second student)

1. Clone the repository and `cd implementations/citeverify`.
2. `pip install -r requirements.in`, then `python -m pytest -q`: expect 36 passed.
3. `python -m src.analyse_a`: expect "wrote 48 rows to results/results.csv".
4. Compare `paper/tables/t2_verifiers.tex` with Table 2 in `paper/main.pdf`: every number must match.
5. Open `results/b2_disagreements.csv`: expect 17 rows.
6. Check `configs/protocol_v1.yaml`: `frozen: true`, two pinned revisions, threshold 0.85, min_flags 1.
7. Optional, needs a GPU and API keys: rerun `notebooks/Prompt13_fast_main_kaggle.ipynb` on Kaggle (2 x T4) and compare counts (12 answers, about 108 citations); numbers may differ slightly ([TEST REQUIRED]).

## Credits

- Models: Qwen2.5-7B-Instruct (Qwen team) and Mistral-7B-Instruct-v0.3 (Mistral AI), both Apache-2.0, loaded from Hugging Face.
- Data services: CrossRef REST API, OpenAlex API, DataCite REST API.
- Libraries: PyTorch, Transformers, bitsandbytes, scikit-learn, pandas, rapidfuzz, matplotlib.
- Code written with an AI assistant (Claude, Anthropic) and tested and run by me; details in `ai_assistance_log.md`.

## License

[YOUR DECISION] Code: MIT. Data and paper: CC BY 4.0. Model answers in `data/raw/` are model outputs from Apache-2.0 models.

## Citation

```bibtex
@misc{sharma2026citeverify,
  author = {Sharma, Vaibhav},
  title  = {Rules, Strings or Databases? Checking Fabricated Citations from Two Open-Weight Language Models},
  year   = {2026},
  note   = {AI Tools for Research implementation paper, IIIT Allahabad},
  url    = {https://github.com/VaibhavIITA/Awesome-Benchmarking-citation-fabrication-llms/tree/main/implementations/citeverify}
}
```
