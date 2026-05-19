#!/usr/bin/env bash
# Submit the full APE (Automatic Post-Editing) pipeline for RQ4.
#
# Stages:
#   1. Create inference inputs for training years (runs locally)
#   2. Submit 10 Qwen3.5-2B draft jobs (training years)
#   3. Submit make_training job (depends on all draft jobs)
#   4. Submit APE refiner training job (depends on make_training)
#   5. Submit 30 APE test-inference jobs (depends on training)
#
# Usage (from repo root):
#   bash scripts/slurm/submit_ape_pipeline.sh [adapter_output_dir]
#
# Default adapter_output_dir: /gpfs/home6/scur0421/outputs/ape_refiner

set -euo pipefail

ADAPTER_DIR="${1:-/gpfs/home6/scur0421/outputs/ape_refiner}"
INPUTS_DIR="outputs/ape_train/inputs"
DRAFTS_DIR="outputs/ape_train/drafts"
TRAINING_JSONL="outputs/ape_train/ape_training.jsonl"
TEST_DATA_DIR="test-data/track2"
ENRICHED_DIR="outputs/enriched/track2"

# ── Stage 1: generate inference inputs ───────────────────────────────────────
echo "=== Stage 1: generating APE training inputs ==="
source .venv/bin/activate
python scripts/data/prepare_ape_data.py \
  --phase make_inputs \
  --inputs-dir "${INPUTS_DIR}"
deactivate 2>/dev/null || true

# ── Stage 2: submit 10 draft jobs ────────────────────────────────────────────
echo "=== Stage 2: submitting draft jobs ==="
DRAFT_JOB_IDS=()

declare -A PAIRS=(
  ["enzh"]="2016 2018 2020 2022 2024"
  ["zhen"]="2015 2017 2019 2021 2023"
)

for pair in enzh zhen; do
  for year in ${PAIRS[$pair]}; do
    inp="${INPUTS_DIR}/${year}.${pair}.proper.jsonl"
    jid=$(sbatch --parsable \
      --export=ALL,WANDB_PROJECT=nlp2-26,WANDB_RUN_GROUP=ape-draft,CODECARBON_PROJECT_NAME=nlp2-26,CODECARBON_COUNTRY_ISO_CODE=NLD,CODECARBON_OUTPUT_DIR=outputs/codecarbon \
      scripts/slurm/run_ape_drafts.slurm \
      "${inp}" "${pair}")
    DRAFT_JOB_IDS+=("${jid}")
    echo "  draft job: pair=${pair} year=${year} jid=${jid}"
  done
done

# Build --dependency string for all draft jobs
DRAFT_DEPS=$(printf ":%s" "${DRAFT_JOB_IDS[@]}")
DRAFT_DEPS="afterok${DRAFT_DEPS}"

# ── Stage 3: submit make_training job ────────────────────────────────────────
echo "=== Stage 3: submitting make_training job ==="
PREP_JID=$(sbatch --parsable \
  --dependency="${DRAFT_DEPS}" \
  --job-name=nlp2-ape-prep \
  --partition=rome \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=8G \
  --time=00:30:00 \
  --output=nlp2-ape-prep-%j.out \
  --error=nlp2-ape-prep-%j.err \
  --wrap="set -euo pipefail; cd \"$(pwd)\"; source .venv/bin/activate; \
python scripts/data/prepare_ape_data.py \
  --phase make_training \
  --inputs-dir \"${INPUTS_DIR}\" \
  --drafts-dir \"${DRAFTS_DIR}\" \
  --out-dir outputs/ape_train")
echo "  make_training job: jid=${PREP_JID}"

# ── Stage 4: submit APE training job ─────────────────────────────────────────
echo "=== Stage 4: submitting APE training job ==="
TRAIN_JID=$(sbatch --parsable \
  --dependency="afterok:${PREP_JID}" \
  --export=ALL,WANDB_PROJECT=nlp2-26,WANDB_RUN_GROUP=ape,CODECARBON_PROJECT_NAME=nlp2-26,CODECARBON_COUNTRY_ISO_CODE=NLD,CODECARBON_OUTPUT_DIR=outputs/codecarbon \
  scripts/slurm/run_ape_train.slurm \
  "${TRAINING_JSONL}" \
  "${ADAPTER_DIR}")
echo "  training job: jid=${TRAIN_JID}"

# ── Stage 5: submit 30 APE test-inference jobs ───────────────────────────────
echo "=== Stage 5: submitting APE test-inference jobs ==="
shopt -s nullglob
for input_jsonl in "${TEST_DATA_DIR}"/*.jsonl; do
  fname=$(basename "${input_jsonl}" .jsonl)
  IFS='.' read -r -a parts <<< "${fname}"
  if [[ ${#parts[@]} -ne 3 ]]; then
    echo "  SKIP unexpected filename: ${fname}" >&2
    continue
  fi
  year="${parts[0]}"
  pair="${parts[1]}"
  mode="${parts[2]}"

  draft_jsonl="${ENRICHED_DIR}/${year}.${pair}.${mode}.baseline.Qwen3.5-2B.jsonl"
  if [[ ! -f "${draft_jsonl}" ]]; then
    echo "  SKIP missing draft: ${draft_jsonl}" >&2
    continue
  fi

  jid=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JID}" \
    --export=ALL,WANDB_PROJECT=nlp2-26,WANDB_RUN_GROUP=ape-infer-${pair}-${year}-${mode},CODECARBON_PROJECT_NAME=nlp2-26,CODECARBON_COUNTRY_ISO_CODE=NLD,CODECARBON_OUTPUT_DIR=outputs/codecarbon \
    scripts/slurm/run_ape_inference.slurm \
    "${input_jsonl}" \
    "${pair}" \
    "${draft_jsonl}" \
    "${ADAPTER_DIR}")
  echo "  infer: year=${year} pair=${pair} mode=${mode} jid=${jid}"
done

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "After inference completes, run:"
echo "  source .venv/bin/activate && python scripts/data/compute_track2_metrics.py"
