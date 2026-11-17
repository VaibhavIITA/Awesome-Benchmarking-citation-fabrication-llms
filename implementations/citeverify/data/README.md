# Data

All data in this folder was produced by this project; no restricted or third-party dataset is included.

| Path | What | Made by |
|---|---|---|
| topics_input.csv | 90 candidate computer-science topic names | written for this project |
| topics_frozen.csv | 66 topics with OpenAlex works count (fetched 17 Nov 2026), band, pilot and robustness flags | src/topics.py |
| raw/generations_pilot.jsonl | 12 pilot answers (Colab) | src/generate.py |
| raw/gen_main_qwen.jsonl, gen_main_mistral.jsonl, generations_main.jsonl | 12 main answers (Kaggle) | src/generate.py |
| parsed/citations_*.jsonl | one row per parsed citation | src/parse.py |
| labels/auto_*.jsonl | B2 evidence and labels per citation | src/verify.py |
| labels/auto_pilot_before_fix.jsonl | pilot B2 evidence before the accent and multi-copy fixes (kept for the debugging record) | src/verify.py |
| labels/to_label_*.csv, key_map_*.csv | blind labelling sheets and the key-to-citation maps | scripts/export_labels.py |
| labels/gold_pilot.csv, gold_main.csv | reference labels (AI-made, reviewed; decisions D24, D27), evidence per row in the note column | AI assistant, reviewed by the student |

Model answers come from Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3 (Apache-2.0). Citation metadata was looked up through the public CrossRef, OpenAlex and DataCite APIs; only matching scores and short titles are stored. The API cache (data/cache) is not committed.
