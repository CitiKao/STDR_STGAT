#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-sdtr-stgat-h100}"

ml purge
ml load miniconda3/24.11.1

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
