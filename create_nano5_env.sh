#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-sdtr-stgat-h100}"

# On Nano5, an already-loaded miniconda module may make `ml purge` call
# `conda deactivate`. Initialize the shell hook first when possible so the
# script does not require the user to run `conda init` manually.
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi

ml purge
ml load miniconda3/24.11.1
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda run -n "$ENV_NAME" conda install -c defaults "intel-openmp<2024" "mkl<2024" -y
conda run -n "$ENV_NAME" conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
conda run -n "$ENV_NAME" pip install numpy pandas pyarrow

echo "Environment '$ENV_NAME' is ready."
echo "You can now submit with:"
echo "  sbatch -A <PROJECT_ID> submit_nano5_stgat.slurm"
