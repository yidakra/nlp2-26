#!/usr/bin/env bash
set -euo pipefail

# Submit Track 2 inference jobs using a fine-tuned LoRA adapter.
# Iterates over all 30 Track 2 test files and submits one SLURM job per file.
#
# Usage (from repo root):
#   bash scripts/slurm/submit_track2_adapter.sh <adapter_dir> [model_id]
#
# Examples:
#   bash scripts/slurm/submit_track2_adapter.sh /gpfs/home6/scur0421/outputs/cc_aligned_qwen
#   bash scripts/slurm/submit_track2_adapter.sh /gpfs/home6/scur0421/outputs/qwen_hk_legislation

ADAPTER_DIR="${1:?Usage: $0 <adapter_dir> [model_id]}"
MODEL_ID="${2:-Qwen/Qwen3.5-9B}"
INPUT_ROOT="test-data/track2"

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Missing ${INPUT_ROOT}. Run from repo root."
  exit 1
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Adapter directory not found: ${ADAPTER_DIR}"
  exit 1
fi

ADAPTER_TAG=$(basename "${ADAPTER_DIR}")

shopt -s nullglob
for input_jsonl in "${INPUT_ROOT}"/*.jsonl; do
  fname=$(basename "${input_jsonl}" .jsonl)
  IFS='.' read -r -a parts <<< "${fname}"
  if [[ ${#parts[@]} -ne 3 ]]; then
    echo "ERROR: unexpected filename format: ${fname}" >&2
    continue
  fi
  year="${parts[0]}"
  pair="${parts[1]}"
  mode="${parts[2]}"

  run_group="track2-${pair}-${year}-${mode}-adapter-${ADAPTER_TAG}-k0"

  jid=$(sbatch --parsable \
    --export=ALL,WANDB_PROJECT=nlp2-26,WANDB_RUN_GROUP="${run_group}",CODECARBON_PROJECT_NAME=nlp2-26,CODECARBON_COUNTRY_ISO_CODE=NLD,CODECARBON_OUTPUT_DIR=outputs/codecarbon,ADAPTER="${ADAPTER_DIR}" \
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
    --max-new-tokens 4096)

  echo "submitted year=${year} pair=${pair} mode=${mode} adapter=${ADAPTER_TAG} job=${jid}"
done
