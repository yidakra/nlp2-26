#!/usr/bin/env bash
set -euo pipefail

# Track 2 RQ1 launcher: baseline + strict (k=0) inference on all Track 2 test files.
# Covers all 30 files: {2015..2023}.enzh and {2016..2024}.zhen × {noterm,proper,random}.
# No few-shot ablation — no Track 2 dev data exists.
#
# Usage (from repo root):
#   bash scripts/slurm/submit_track2.sh [model_id]

MODEL_ID="${1:-google/gemma-4-E2B-it}"
INPUT_ROOT="test-data/track2"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Missing ${INPUT_ROOT}. Run from repo root."
  exit 1
fi

# format: prompt_strategy:rerank_strategy:num_candidates:few_shot_k
STRATEGIES=(
  "baseline:none:1:0"
  "strict:term_coverage:3:0"
)

for input_jsonl in "${INPUT_ROOT}"/*.jsonl; do
  fname=$(basename "${input_jsonl}" .jsonl)   # e.g. 2023.enzh.noterm
  year="${fname%%.*}"                           # 2023
  rest="${fname#*.}"                            # enzh.noterm
  pair="${rest%%.*}"                            # enzh
  mode="${rest#*.}"                             # noterm

  if [[ ! "${year}" =~ ^[0-9]{4}$ || -z "${pair}" || -z "${mode}" ]]; then
    echo "ERROR: unexpected input filename format: ${fname}. Expected YEAR.PAIR.MODE" >&2
    continue
  fi

  for strategy_spec in "${STRATEGIES[@]}"; do
    IFS=':' read -r prompt_strategy rerank_strategy num_candidates few_shot_k <<<"${strategy_spec}"

    run_group="track2-${pair}-${year}-${mode}-${prompt_strategy}-k${few_shot_k}"

    jid=$(sbatch --parsable \
      --export=ALL,WANDB_PROJECT=nlp2-26,WANDB_RUN_GROUP="${run_group}",CODECARBON_PROJECT_NAME=nlp2-26,CODECARBON_COUNTRY_ISO_CODE=NLD,CODECARBON_OUTPUT_DIR=outputs/codecarbon \
      scripts/slurm/run_inference_eval.slurm \
      "${input_jsonl}" \
      "${pair}" \
      "${MODEL_ID}" \
      --prompt-strategy "${prompt_strategy}" \
      --rerank-strategy "${rerank_strategy}" \
      --num-candidates "${num_candidates}" \
      --few-shot-k "${few_shot_k}" \
      --temperature 0.7 \
      --top-p 0.8 \
      --seed 42 \
      --max-new-tokens 4096)

    echo "submitted year=${year} pair=${pair} mode=${mode} strategy=${prompt_strategy} job=${jid}"
  done
done
