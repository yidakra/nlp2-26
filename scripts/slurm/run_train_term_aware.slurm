#!/bin/bash
#SBATCH --job-name=nlp2-termtrain
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Example:
# sbatch scripts/slurm/run_train_term_aware.slurm \
#   /path/to/data_dir \
#   google/gemma-4-E2B-it \
#   outputs/term_aware_lora \
#   4 \
#   3 \
#   outputs/term_alignments

DATA_DIR="${1:-}"
MODEL_ID="${2:-}"
OUTPUT_DIR="${3:-outputs/term_aware_lora}"
BATCH_SIZE="${4:-4}"
EPOCHS="${5:-3}"
SAVE_TERMS_DIR="${6:-outputs/term_alignments}"

if [[ -z "${DATA_DIR}" || -z "${MODEL_ID}" ]]; then
  echo "Usage: sbatch $0 <data_dir> <model_id> <output_dir> [batch_size] [epochs] [save_terms_dir]"
  exit 1
fi

mkdir -p outputs
source .venv/bin/activate

export WANDB_PROJECT="${WANDB_PROJECT:-nlp2-26}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-term_train}"
export CODECARBON_PROJECT_NAME="${CODECARBON_PROJECT_NAME:-nlp2-26}"
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-outputs/codecarbon}"
export CODECARBON_COUNTRY_ISO_CODE="${CODECARBON_COUNTRY_ISO_CODE:-NLD}"
mkdir -p "${CODECARBON_OUTPUT_DIR}"

if ! python -c "import wandb; wandb.login(relogin=False, anonymous='never')" >/dev/null 2>&1; then
  echo "W&B login not available. Run 'wandb login' (or set WANDB_API_KEY) before submitting."
  exit 1
fi

python scripts/train_term_aware_nmt.py \
  --model "${MODEL_ID}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_terms_dir "${SAVE_TERMS_DIR}" \
  --wandb \
  --codecarbon \
  --wandb_group "${WANDB_RUN_GROUP}"
