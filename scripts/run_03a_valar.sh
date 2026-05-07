#!/bin/bash
#SBATCH --job-name=03a_linreg
#SBATCH --output=logs/03a_%j.out
#SBATCH --error=logs/03a_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Adjust the next two lines to match Valar's environment setup.
# module load python/3.11
# source ~/engr422-venv/bin/activate

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# data/ is gitignored. Make sure these CSVs are present before submitting.
ls -lh data/X_train.csv data/X_test.csv data/y_train.csv data/y_test.csv

jupyter nbconvert --to notebook --execute \
    notebooks/03a_baseline_linear_regression.ipynb \
    --output 03a_baseline_linear_regression.executed.ipynb \
    --ExecutePreprocessor.timeout=1800
