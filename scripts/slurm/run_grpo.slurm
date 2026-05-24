#!/bin/bash
#SBATCH --job-name=nlp2-grpo
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
# sbatch scripts/slurm/run_grpo_term_aware.slurm \
#   /path/to/data_dir \
#   google/gemma-4-E2B-it \
#   outputs/grpo_lora \
#   2 \
#   2 \
#   outputs/term_alignments \
#   outputs/term_aware_lora   # <-- SFT adapter to merge before GRPO

DATA_DIR="${1:-}"
MODEL_ID="${2:-}"
OUTPUT_DIR="${3:-outputs/grpo_lora}"
BATCH_SIZE="${4:-2}"
EPOCHS="${5:-2}"
SAVE_TERMS_DIR="${6:-outputs/term_alignments}"
ADAPTER="${7:-}"
MAX_LENGTH="${8:-4096}"
MAX_COMPLETION_LENGTH="${9:-512}"
TARGET_CHUNK_TOKENS="${10:-600}"
LORA_RANK="${11:-64}"
NUM_GENERATIONS="${12:-4}"
GENERATION_BATCH_SIZE="${13:-4}"
BETA="${14:-0.04}"
TEMPERATURE="${15:-0.7}"
MAX_TRAIN_SAMPLES="${16:-2000}"
MAX_EVAL_SAMPLES="${17:-100}"

if [[ -z "${DATA_DIR}" || -z "${MODEL_ID}" ]]; then
  echo "Usage: sbatch $0 <data_dir> <model_id> [output_dir] [batch_size] [epochs] [save_terms_dir] [adapter] [max_length] [max_completion_length] [target_chunk_tokens] [lora_rank] [num_generations] [beta] [temperature]"
  exit 1
fi

mkdir -p outputs
source .venv/bin/activate

pip install pycccedict
pip install torchvision
pip install --upgrade torch

export WANDB_PROJECT="${WANDB_PROJECT:-nlp2-26}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-grpo_term_hk_legislation}"
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
  python scripts/grpo_train.py
  --model "${MODEL_ID}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lora_rank "${LORA_RANK}"
  --data_dir "${DATA_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --save_terms_dir "${SAVE_TERMS_DIR}"
  --max_length "${MAX_LENGTH}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --target_chunk_tokens "${TARGET_CHUNK_TOKENS}"
  --num_generations "${NUM_GENERATIONS}"
  --generation_batch_size "${GENERATION_BATCH_SIZE}"
  --beta "${BETA}"
  --temperature "${TEMPERATURE}"
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --max_eval_samples "${MAX_EVAL_SAMPLES}"
  --wandb
  --codecarbon
)

# Attach SFT adapter if provided (strongly recommended)
if [[ -n "${ADAPTER}" ]]; then
  CMD+=(--adapter "${ADAPTER}")
else
  echo "WARNING: No --adapter provided. GRPO will train from the base model."
  echo "         This is usually suboptimal — consider passing your SFT checkpoint."
fi

echo "Running: ${CMD[*]}"
srun "${CMD[@]}"

