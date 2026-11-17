import json
from pathlib import Path

import yaml


def load_cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(x) for x in f if x.strip()]


def append_jsonl(p, row):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
