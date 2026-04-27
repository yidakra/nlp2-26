#!/usr/bin/env bash
set -euo pipefail

# Track 2 thinking-mode launcher: baseline inference with Qwen3 thinking enabled.
# Runs Qwen3.5-9B with enable_thinking=True on all 30 Track 2 test files.
# Uses max-new-tokens=8192 to accommodate reasoning chains.
#
# Usage (from repo root):
#   bash scripts/slurm/submit_track2_thinking.sh [model_id]

MODEL_ID="${1:-Qwen/Qwen3.5-9B}"
INPUT_ROOT="test-data/track2"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Missing ${INPUT_ROOT}. Run from repo root."
  exit 1
fi

shopt -s nullglob
for input_jsonl in "${INPUT_ROOT}"/*.jsonl; do
  fname=$(basename "${input_jsonl}" .jsonl)
  IFS='.' read -r -a parts <<< "${fname}"
  if [[ ${#parts[@]} -ne 3 ]]; then
    echo "ERROR: unexpected filename format: ${fname}" >&2
    continue
  fi
  # Expect input filenames like year.pair.mode.jsonl, where the basename is
  # split on '.' into [year, pair, mode]. For example: 2023.enzh.proper.jsonl.
  # This parsing assumes a single dot separator between those three parts and
  # strips the .jsonl extension via basename.
  year="${parts[0]}"
  pair="${parts[1]}"
  mode="${parts[2]}"

  run_group="track2-${pair}-${year}-${mode}-thinking-k0"

  jid=$(sbatch --parsable \
    --export=ALL,WANDB_PROJECT=nlp2-26,WANDB_RUN_GROUP="${run_group}",CODECARBON_PROJECT_NAME=nlp2-26,CODECARBON_COUNTRY_ISO_CODE=NLD,CODECARBON_OUTPUT_DIR=outputs/codecarbon \
    scripts/slurm/run_inference_eval.slurm \
    "${input_jsonl}" \
    "${pair}" \
    "${MODEL_ID}" \
    --prompt-strategy baseline \
    --rerank-strategy none \
    --num-candidates 1 \
    --few-shot-k 0 \
    --temperature 0.7 \
    --top-p 0.8 \
    --seed 42 \
    --max-new-tokens 8192 \
    --enable-thinking)

  echo "submitted year=${year} pair=${pair} mode=${mode} strategy=thinking job=${jid}"
done
