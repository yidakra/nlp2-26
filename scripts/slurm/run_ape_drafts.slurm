#!/bin/bash
#SBATCH --job-name=nlp2-ape-draft
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Generate Qwen3.5-2B drafts on APE training-year inputs.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ape_drafts.slurm <input_jsonl> <pair>
#
# Example:
#   sbatch scripts/slurm/run_ape_drafts.slurm \
#     outputs/ape_train/inputs/2016.enzh.proper.jsonl enzh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -f .env ]]; then
  set -o allexport; source .env; set +o allexport
fi

INPUT_JSONL="${1:?Usage: $0 <input_jsonl> <pair>}"
PAIR="${2:?Usage: $0 <input_jsonl> <pair>}"
MODEL_ID="Qwen/Qwen3.5-2B"

source .venv/bin/activate

fname=$(basename "${INPUT_JSONL}" .jsonl)
OUT_DIR="outputs/ape_train/drafts"
mkdir -p "${OUT_DIR}"

export WANDB_PROJECT="${WANDB_PROJECT:-nlp2-26}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ape-draft}"
export CODECARBON_PROJECT_NAME="${CODECARBON_PROJECT_NAME:-nlp2-26}"
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-outputs/codecarbon}"
export CODECARBON_COUNTRY_ISO_CODE="${CODECARBON_COUNTRY_ISO_CODE:-NLD}"
mkdir -p "${CODECARBON_OUTPUT_DIR}"

if ! python -c "import wandb; wandb.login(relogin=False, anonymous='never')" >/dev/null 2>&1; then
  echo "W&B login not available."
  exit 1
fi

srun --ntasks=1 python scripts/inference.py \
  --model-id "${MODEL_ID}" \
  --src-tgt-pair "${PAIR}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "${OUT_DIR}/${fname}.Qwen3.5-2B.jsonl" \
  --batch-size 4 \
  --max-new-tokens 4096 \
  --prompt-strategy baseline \
  --temperature 0.7 \
  --top-p 0.8 \
  --seed 42
