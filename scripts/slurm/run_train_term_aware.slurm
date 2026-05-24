#!/bin/bash
#SBATCH --job-name=nlp2-termtrain
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=96G
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
MAX_LENGTH="${7:-4096}"
TARGET_CHUNK_TOKENS="${8:-2500}"
LORA_RANK="${9:-64}"

if [[ -z "${DATA_DIR}" || -z "${MODEL_ID}" ]]; then
  echo "Usage: sbatch $0 <data_dir> <model_id> <output_dir> [batch_size] [epochs] [save_terms_dir] [max_length] [target_chunk_tokens] [lora_rank]"
  exit 1
fi

mkdir -p outputs
source .venv/bin/activate

pip install pycccedict
pip install --upgrade torch
pip install --upgrade torchvision

export WANDB_PROJECT="${WANDB_PROJECT:-nlp2-26}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-term_train_hk_legislation}"
export CODECARBON_PROJECT_NAME="${CODECARBON_PROJECT_NAME:-nlp2-26}"
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-outputs/codecarbon}"
export CODECARBON_COUNTRY_ISO_CODE="${CODECARBON_COUNTRY_ISO_CODE:-NLD}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${CODECARBON_OUTPUT_DIR}"

if ! python -c "import wandb; wandb.login(relogin=False, anonymous='never')" >/dev/null 2>&1; then
  echo "W&B login not available. Run 'wandb login' (or set WANDB_API_KEY) before submitting."
  exit 1
fi

# Build the command
CMD=(
  python scripts/train_term_aware_nmt.py
  --model "${MODEL_ID}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lora_rank "${LORA_RANK}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --save_terms_dir "${SAVE_TERMS_DIR}"
  --max_length "${MAX_LENGTH}"
  --target_chunk_tokens "${TARGET_CHUNK_TOKENS}"
  --wandb
  --codecarbon
)

# Uncomment to resume from an adapter:
CMD+=(--adapter "./outputs/qwen_cc_redo")

echo "Running: ${CMD[*]}"
srun "${CMD[@]}"

