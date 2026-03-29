#!/bin/bash
#SBATCH --job-name=sdtr_stgat_train
#SBATCH --partition=normal
#SBATCH --account=ACD114200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gaodeefreetaipei@gmail.com

set -euo pipefail

# Under Slurm, the batch script is executed from a spool copy in /var/spool/slurmd.
# Use the original submission directory instead of BASH_SOURCE[0].
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ENV_NAME="${CONDA_ENV_NAME:-sdtr-stgat-h100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-50}"
PRECISION="${PRECISION:-bf16}"
RUN_DIR="${SCRIPT_DIR}/runs/nano5_${SLURM_JOB_ID}"

if [[ ! -f "$SCRIPT_DIR/train_predictor.py" ]]; then
  echo "Expected train_predictor.py under $SCRIPT_DIR" >&2
  echo "Submit this job from the project directory, for example:" >&2
  echo "  cd /work/<your_account>/SDTR_STGAT && sbatch submit_nano5_stgat.slurm" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

# Avoid `ml purge` here for the same reason as create_nano5_env.sh:
# some Nano5 shells call `conda deactivate` during purge before the conda hook
# is initialized.
# Disable `nounset` around `ml load` for compatibility with the site Lmod init
# script, which may reference SLURM_JOBID directly.
set +u
ml load miniconda3/24.11.1
set -u
eval "$(conda shell.bash hook)"

echo "job_id=${SLURM_JOB_ID}"
echo "job_node_list=${SLURM_JOB_NODELIST}"
echo "project_dir=${SCRIPT_DIR}"
echo "run_dir=${RUN_DIR}"
echo "env_name=${ENV_NAME}"
echo "batch_size=${BATCH_SIZE}"
echo "epochs=${EPOCHS}"
echo "precision=${PRECISION}"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' was not found." >&2
  echo "On the Nano5 login node, run:" >&2
  echo "  cd $SCRIPT_DIR && bash create_nano5_env.sh" >&2
  exit 1
fi

conda run -n "$ENV_NAME" python "$SCRIPT_DIR/train_predictor.py" \
  --data-dir "$SCRIPT_DIR/data" \
  --log-dir "$RUN_DIR" \
  --device auto \
  --precision "$PRECISION" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS"

echo "Training finished. Outputs are in: $RUN_DIR"
