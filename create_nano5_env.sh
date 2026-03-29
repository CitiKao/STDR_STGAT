#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${1:-sdtr-stgat-h100}"

# Avoid `ml purge` here. On Nano5 it can trigger `conda deactivate` before the
# shell hook is initialized, which leads to `Run 'conda init' before 'conda deactivate'`.
# Also disable `nounset` around `ml load`, because the site Lmod init script
# may reference SLURM_JOBID even on the login node.
set +u
ml load miniconda3/24.11.1
set -u
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

echo "[1/3] Installing MKL runtime dependencies..."
conda install -n "$ENV_NAME" -c defaults "intel-openmp<2024" "mkl<2024" -y

echo "[2/3] Installing PyTorch + CUDA 12.1..."
conda install -n "$ENV_NAME" -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y

echo "[3/3] Installing Python packages..."
conda run --no-capture-output -n "$ENV_NAME" python -m pip install numpy pandas pyarrow

echo "Environment '$ENV_NAME' is ready."
echo "You can now submit with:"
echo "  cd $SCRIPT_DIR"
echo "  sbatch submit_nano5_stgat.slurm"
