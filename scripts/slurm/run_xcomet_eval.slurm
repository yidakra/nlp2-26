#!/bin/bash
#SBATCH --job-name=nlp2-xcomet
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Runs XCOMET-XL on the 9 priority-subset enriched files.
# Files must already exist (run midterm_fill_gaps.sh steps 1-3 first).
# Produces: outputs/metrics/xcomet_scores.json

set -euo pipefail
cd /home/scur0421/nlp2-26
set -o allexport; source .env; set +o allexport
source .venv/bin/activate

python3 - <<'PYEOF'
import json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
from scripts.eval import Evaluator, load_jsonl

PRIORITY_FILES = [
    "outputs/enriched/enes.noterm.baseline_fsk0.jsonl",
    "outputs/enriched/enes.noterm.strict_fsk0.jsonl",
    "outputs/enriched/enes.noterm.strict_fsk3.jsonl",
    "outputs/enriched/enes.proper.baseline_fsk0.jsonl",
    "outputs/enriched/enes.proper.strict_fsk0.jsonl",
    "outputs/enriched/enes.proper.strict_fsk3.jsonl",
    "outputs/enriched/enru.noterm.baseline_fsk0.jsonl",
    "outputs/enriched/enru.proper.baseline_fsk0.jsonl",
    "outputs/enriched/enru.proper.strict_fsk0.jsonl",
]

evaluator = Evaluator()
results = {}
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

out = "outputs/metrics/xcomet_scores.json"
os.makedirs("outputs/metrics", exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
PYEOF
