#!/bin/bash
#SBATCH --job-name=sdtr_stgat_train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-sdtr-stgat-h100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-50}"
PRECISION="${PRECISION:-bf16}"
RUN_DIR="${SCRIPT_DIR}/runs/nano5_${SLURM_JOB_ID}"

mkdir -p "$RUN_DIR"

ml purge
ml load miniconda3/24.11.1

echo "job_id=${SLURM_JOB_ID}"
echo "job_node_list=${SLURM_JOB_NODELIST}"
echo "project_dir=${SCRIPT_DIR}"
echo "run_dir=${RUN_DIR}"
echo "env_name=${ENV_NAME}"
echo "batch_size=${BATCH_SIZE}"
echo "epochs=${EPOCHS}"
echo "precision=${PRECISION}"

conda run -n "$ENV_NAME" python "$SCRIPT_DIR/train_predictor.py" \
  --data-dir "$SCRIPT_DIR/data" \
  --log-dir "$RUN_DIR" \
  --device auto \
  --precision "$PRECISION" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS"

echo "Training finished. Outputs are in: $RUN_DIR"
