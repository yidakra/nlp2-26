#!/bin/bash
#SBATCH --job-name=nlp2-ape-infer
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# APE two-pass inference on a single Track 2 test file.
# Pass 1 draft must already exist (Qwen3.5-2B baseline output).
#
# Usage (from repo root):
#   sbatch scripts/slurm/run_ape_inference.slurm \
#     <input_jsonl> <pair> <draft_jsonl> <adapter_dir>
#
# Example:
#   sbatch scripts/slurm/run_ape_inference.slurm \
#     test-data/track2/2015.enzh.proper.jsonl \
#     enzh \
#     outputs/enriched/track2/2015.enzh.proper.baseline.Qwen3.5-2B.jsonl \
#     /gpfs/home6/scur0421/outputs/ape_refiner

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -f .env ]]; then
  set -o allexport; source .env; set +o allexport
fi

INPUT_JSONL="${1:?Usage: $0 <input_jsonl> <pair> <draft_jsonl> <adapter_dir>}"
PAIR="${2:?}"
DRAFT_JSONL="${3:?}"
ADAPTER_DIR="${4:?}"
MODEL_ID="Qwen/Qwen3.5-2B"

source .venv/bin/activate

mkdir -p outputs

export WANDB_PROJECT="${WANDB_PROJECT:-nlp2-26}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-ape-infer}"
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
  --adapter "${ADAPTER_DIR}" \
  --src-tgt-pair "${PAIR}" \
  --input-jsonl "${INPUT_JSONL}" \
  --draft-jsonl "${DRAFT_JSONL}" \
  --output-jsonl "outputs/${PAIR}_${SLURM_JOB_ID:-nojid}_$(date +%Y%m%d_%H%M%S_%N).jsonl" \
  --batch-size 2 \
  --max-new-tokens 4096 \
  --prompt-strategy ape \
  --temperature 0.7 \
  --top-p 0.8 \
  --seed 42
