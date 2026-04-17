#!/usr/bin/env bash
set -euo pipefail

# Minimal environment bootstrap for Snellius-like clusters.
# Usage:
#   bash scripts/slurm/setup_env.sh

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready. Activate with: source .venv/bin/activate"
