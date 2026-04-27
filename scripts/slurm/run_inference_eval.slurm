#!/bin/bash
#SBATCH --job-name=nlp2-infer-eval
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Example:
# sbatch scripts/slurm/run_inference_eval.slurm \
#   /path/to/input.jsonl \
#   enzh \
#   google/gemma-4-E2B-it

INPUT_JSONL="${1:-}"
PAIR="${2:-enzh}"
MODEL_ID="${3:-google/gemma-4-E2B-it}"

if [[ -z "${INPUT_JSONL}" ]]; then
  echo "Usage: sbatch $0 <input_jsonl> [pair] [model_id]"
  exit 1
fi

mkdir -p outputs
source .venv/bin/activate

export WANDB_PROJECT="${WANDB_PROJECT:-nlp2-26}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${PAIR}}"
export CODECARBON_PROJECT_NAME="${CODECARBON_PROJECT_NAME:-nlp2-26}"
export CODECARBON_OUTPUT_DIR="${CODECARBON_OUTPUT_DIR:-outputs/codecarbon}"
export CODECARBON_COUNTRY_ISO_CODE="${CODECARBON_COUNTRY_ISO_CODE:-NLD}"
mkdir -p "${CODECARBON_OUTPUT_DIR}"

if ! python -c "import wandb; wandb.login(relogin=False, anonymous='never')" >/dev/null 2>&1; then
  echo "W&B login not available. Run 'wandb login' (or set WANDB_API_KEY) before submitting."
  exit 1
fi

srun python scripts/inference.py \
  --model-id "${MODEL_ID}" \
  --adapter "./outputs/cc_aligned_qwen" \
  --src-tgt-pair "${PAIR}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "outputs/${PAIR}_$(date +%Y%m%d_%H%M%S).jsonl" \
  --batch-size 2 \
  --run-eval
