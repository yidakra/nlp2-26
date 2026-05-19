#!/bin/bash
#SBATCH --job-name=nlp2-ape-train
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Train the APE refiner LoRA on Qwen/Qwen3.5-2B.
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ape_train.slurm \
#     outputs/ape_train/ape_training.jsonl \
#     /gpfs/home6/scur0421/outputs/ape_refiner

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -f .env ]]; then
  set -o allexport; source .env; set +o allexport
fi

TRAINING_JSONL="${1:-outputs/ape_train/ape_training.jsonl}"
OUTPUT_DIR="${2:-/gpfs/home6/scur0421/outputs/ape_refiner}"

source .venv/bin/activate

export WANDB_PROJECT="${WANDB_PROJECT:-nlp2-26}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ape}"
export CODECARBON_PROJECT_NAME="${CODECARBON_PROJECT_NAME:-nlp2-26}"
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-outputs/codecarbon}"
export CODECARBON_COUNTRY_ISO_CODE="${CODECARBON_COUNTRY_ISO_CODE:-NLD}"
mkdir -p "${CODECARBON_OUTPUT_DIR}"

if ! python -c "import wandb; wandb.login(relogin=False, anonymous='never')" >/dev/null 2>&1; then
  echo "W&B login not available."
  exit 1
fi

python scripts/train_ape_refiner.py \
  --model "Qwen/Qwen3.5-2B" \
  --training-jsonl "${TRAINING_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --lora-rank 16 \
  --epochs 5 \
  --batch-size 2 \
  --max-length 4096 \
  --wandb \
  --codecarbon
