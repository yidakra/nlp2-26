#!/bin/bash
#SBATCH --job-name=nlp2-infer-eval
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -f .env ]]; then
  set -o allexport; source .env; set +o allexport
else
  echo "Warning: .env not found in ${SLURM_SUBMIT_DIR}, skipping."
fi

# Example:
# (Gemma 4 requires transformers nightly; XCOMET jobs pin <5.0 separately)
# sbatch scripts/slurm/run_inference_eval.slurm \
#   /path/to/input.jsonl \
#   enzh \
#   google/gemma-4-E2B-it \
#   --prompt-strategy strict --num-candidates 4 --rerank-strategy term_coverage

INPUT_JSONL="${1:-}"
PAIR="${2:-enzh}"
MODEL_ID="${3:-google/gemma-4-E2B-it}"
EXTRA_ARGS=("${@:4}")

if [[ -z "${INPUT_JSONL}" ]]; then
  echo "Usage: sbatch $0 <input_jsonl> [pair] [model_id] [extra inference args ...]"
  exit 1
fi

mkdir -p outputs
if [[ ! -d .venv ]]; then
  echo "ERROR: .venv not found. Create the environment before submitting, e.g. via scripts/slurm/setup_env.sh." >&2
  exit 1
fi
source .venv/bin/activate
if ! python -c "import re; import transformers; v=tuple(int(re.sub(r'[^0-9]','',x) or 0) for x in transformers.__version__.split('.')[:2]); assert v>=(4,50)" >/dev/null 2>&1; then
  echo "ERROR: transformers>=4.50 is not installed in .venv. Prepare the environment with scripts/slurm/setup_env.sh and retry." >&2
  exit 1
fi

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

ADAPTER_ARGS=()
if [[ -n "${ADAPTER:-}" ]]; then
  ADAPTER_ARGS=(--adapter "${ADAPTER}")
fi

srun --ntasks=1 python scripts/inference.py \
  --model-id "${MODEL_ID}" \
  "${ADAPTER_ARGS[@]}" \
  --src-tgt-pair "${PAIR}" \
  --input-jsonl "${INPUT_JSONL}" \
  --output-jsonl "outputs/${PAIR}_${SLURM_JOB_ID:-nojid}_$(date +%Y%m%d_%H%M%S_%N).jsonl" \
  --batch-size 2 \
  --run-eval \
  "${EXTRA_ARGS[@]}"
