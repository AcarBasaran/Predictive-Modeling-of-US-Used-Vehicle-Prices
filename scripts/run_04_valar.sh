#!/bin/bash
#SBATCH --job-name=04_eval
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --output=logs/04_%j.out
#SBATCH --error=logs/04_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Dedicated project env (Python 3.11 + sklearn 1.8 / xgboost / seaborn / nbconvert).
# Prepending the env's bin to PATH directly is more robust than `source activate`,
# which silently fell back to system anaconda's sklearn 1.7 on at least one node
# and produced InconsistentVersionWarning on every unpickle.
module load anaconda3/2025.06
export PATH="/home/ebasaran22/.conda/envs/engr422/bin:$PATH"
export CONDA_PREFIX="/home/ebasaran22/.conda/envs/engr422"

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Required inputs (produced by 02 + 03a + 03b + 03c)
ls -lh data/X_test.csv data/y_test.csv \
       models/linear_regression.pkl \
       models/random_forest.pkl \
       models/xgboost.pkl
# xgboost_reduced.pkl (issue #18) is optional -- 04 handles its absence.

jupyter nbconvert --to notebook --execute --inplace \
    notebooks/04_evaluation.ipynb \
    --ExecutePreprocessor.timeout=3000 \
    --ExecutePreprocessor.kernel_name=python3
