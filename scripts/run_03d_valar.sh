#!/bin/bash
#SBATCH --job-name=03d_lgbm
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --output=logs/03d_%j.out
#SBATCH --error=logs/03d_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

module load anaconda3/2025.06
export PATH="/home/ebasaran22/.conda/envs/engr422/bin:$PATH"
export CONDA_PREFIX="/home/ebasaran22/.conda/envs/engr422"

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

ls -lh data/X_train.csv data/X_test.csv data/y_train.csv data/y_test.csv \
       models/preprocessor_tree.pkl

jupyter nbconvert --to notebook --execute --inplace \
    notebooks/03d_lightgbm.ipynb \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=python3

ls -lh models/lightgbm.pkl models/lgbm_tuning_results.json
