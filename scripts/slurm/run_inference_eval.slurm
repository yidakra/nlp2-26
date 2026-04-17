#!/bin/bash
#SBATCH --job-name=nlp2-infer-eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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

mkdir -p logs outputs
source .venv/bin/activate

export WANDB_PROJECT="${WANDB_PROJECT:-nlp2-26}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${PAIR}}"

if ! python -c "import wandb; wandb.login(relogin=False, anonymous='never')" >/dev/null 2>&1; then
  echo "W&B login not available. Run 'wandb login' (or set WANDB_API_KEY) before submitting."
  exit 1
fi

python scripts/inference.py \
  --model-id "${MODEL_ID}" \
  --src-tgt-pair "${PAIR}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "outputs/${PAIR}_$(date +%Y%m%d_%H%M%S).jsonl" \
  --batch-size 2 \
  --run-eval
