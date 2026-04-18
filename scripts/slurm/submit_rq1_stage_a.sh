#!/usr/bin/env bash
set -euo pipefail

# Stage-A RQ1 launcher (test-data inference focus).
# Usage:
#   bash scripts/slurm/submit_rq1_stage_a.sh [pair] [model_id]
# Example:
#   bash scripts/slurm/submit_rq1_stage_a.sh ende google/gemma-4-E2B-it

PAIR="${1:-ende}"
MODEL_ID="${2:-google/gemma-4-E2B-it}"
INPUT_ROOT="test-data/track1"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Missing ${INPUT_ROOT}. Run from repo root."
  exit 1
fi

# format: prompt_strategy:rerank_strategy:num_candidates:few_shot_k
STRATEGIES=(
  "baseline:none:1:0"
  "strict:term_coverage:3:0"
)

# format: profile_name:temperature:top_p
DECODING_PROFILES=(
  "default:0.7:0.8"
  "lowtemp:0.3:0.9"
)

MODES=(noterm proper random)

for mode in "${MODES[@]}"; do
  input_jsonl="${INPUT_ROOT}/${PAIR}.${mode}.jsonl"
  if [[ ! -f "${input_jsonl}" ]]; then
    echo "Skipping missing input: ${input_jsonl}"
    continue
  fi

  for strategy_spec in "${STRATEGIES[@]}"; do
    IFS=':' read -r prompt_strategy rerank_strategy num_candidates few_shot_k <<<"${strategy_spec}"

    for decoding_spec in "${DECODING_PROFILES[@]}"; do
      IFS=':' read -r profile_name temperature top_p <<<"${decoding_spec}"

      run_group="snellius-rq1-${PAIR}-${mode}-${prompt_strategy}-${profile_name}"
      jid=$(sbatch --parsable \
        --export=ALL,WANDB_PROJECT=nlp2-26,WANDB_RUN_GROUP="${run_group}",CODECARBON_PROJECT_NAME=nlp2-26,CODECARBON_COUNTRY_ISO_CODE=NLD,CODECARBON_OUTPUT_DIR=outputs/codecarbon \
        scripts/slurm/run_inference_eval.slurm \
        "${input_jsonl}" \
        "${PAIR}" \
        "${MODEL_ID}" \
        --prompt-strategy "${prompt_strategy}" \
        --rerank-strategy "${rerank_strategy}" \
        --num-candidates "${num_candidates}" \
        --few-shot-k "${few_shot_k}" \
        --temperature "${temperature}" \
        --top-p "${top_p}" \
        --seed 42)

      echo "submitted mode=${mode} strategy=${prompt_strategy} rerank=${rerank_strategy} profile=${profile_name} job=${jid}"
    done
  done
done
