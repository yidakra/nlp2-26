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

# Install core requirements first. vLLM/flash-attn are optional for this pipeline and
# frequently require node-specific CUDA/PyTorch build constraints.
grep -Ev '^(vllm|flash-attn)$' requirements.txt > /tmp/nlp2_26_requirements_core.txt
cat > /tmp/nlp2_26_constraints.txt <<'EOF'
transformers<5
torch<2.6
PyYAML>=6
EOF
pip install -r /tmp/nlp2_26_requirements_core.txt -c /tmp/nlp2_26_constraints.txt

# Optional accelerator packages are disabled by default to keep bootstrap
# reproducible on login nodes and mixed cluster environments.
if [[ "${INSTALL_OPTIONAL_ACCEL:-0}" == "1" ]]; then
	if grep -q '^vllm$' requirements.txt; then
		if ! pip install vllm; then
			echo "Warning: vllm install failed. Continuing without it."
		fi
	fi

	if grep -q '^flash-attn$' requirements.txt; then
		if ! pip install flash-attn --no-build-isolation; then
			echo "Warning: flash-attn install failed. Continuing without it."
		fi
	fi
else
	echo "Skipping optional accelerators (vllm, flash-attn). Set INSTALL_OPTIONAL_ACCEL=1 to enable."
fi

echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Then log into W&B once: wandb login"
