#!/bin/bash
#SBATCH --job-name=fl_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

set -euo pipefail

module load Python/3.11.3-GCCcore-12.3.0    # check exact VSC module name
source "$VSC_DATA/venvs/fedids/bin/activate"

cd "$SLURM_SUBMIT_DIR"

python main.py train-federated \
    --config configs/experiment/fedavg_lstm_sgym.yaml \
    hydra.run.dir="$VSC_SCRATCH/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"