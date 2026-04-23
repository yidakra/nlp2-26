#!/bin/bash
#SBATCH --job-name=nlp2-xcomet
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Runs XCOMET-XL on all enriched files.
# Files must already exist (run midterm_fill_gaps.sh steps 1-3 first).
# Produces: outputs/metrics/xcomet_scores.json

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
set -o allexport; source .env; set +o allexport
source .venv/bin/activate

# Prefer pre-installed dependencies in the job environment.
# Only install transformers at runtime if it is missing or the version is out of range.
python3 - <<'PYEOF'
import sys
try:
    import transformers
    version = tuple(int(x) for x in transformers.__version__.split('.')[:2])
    if version < (4, 17):
        raise ImportError
except Exception:
    sys.exit(1)
PYEOF
if [ $? -ne 0 ]; then
    pip install "transformers>=4.17" -q
fi

python3 - <<'PYEOF'
import json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from scripts.eval import Evaluator, load_jsonl

import glob
PRIORITY_FILES = sorted(set(
    glob.glob("outputs/enriched/**/*.jsonl", recursive=True)
))

# Skip already-scored files
existing = {}
if os.path.exists("outputs/metrics/xcomet_scores.json"):
    with open("outputs/metrics/xcomet_scores.json") as f:
        existing = json.load(f)
PRIORITY_FILES = [f for f in PRIORITY_FILES if f not in existing]
print(f"Files to evaluate: {len(PRIORITY_FILES)} (skipping {len(existing)} already done)")

evaluator = Evaluator()
results = existing.copy()
os.makedirs("outputs/metrics", exist_ok=True)
for fpath in PRIORITY_FILES:
    if not os.path.exists(fpath):
        print(f"MISSING: {fpath}")
        continue
    rows = load_jsonl(fpath)
    eval_rows = [r for r in rows if r.get("src") and r.get("mt") and r.get("ref")]
    if not eval_rows:
        print(f"SKIP (no ref rows): {fpath}")
        continue
    print(f"Evaluating {fpath} ({len(eval_rows)} rows)...")
    summary = evaluator.evaluate(eval_rows, batch_size=4)
    results[fpath] = {"system": summary["system"], "n": len(eval_rows)}
    print(f"  XCOMET system score: {summary['system']:.4f}")
    # Save incrementally to preserve progress
    with open("outputs/metrics/xcomet_scores.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
print("\nSaved: outputs/metrics/xcomet_scores.json")
PYEOF
