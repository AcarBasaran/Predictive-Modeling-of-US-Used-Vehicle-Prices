#!/bin/bash
#SBATCH --job-name=02_preprocess
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --output=logs/02_preprocess_%j.out
#SBATCH --error=logs/02_preprocess_%j.err
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

# Required input
ls -lh data/vehicles.csv

# 02 writes these on success — listed for clarity, not required to exist yet:
#   data/X_train.csv  data/X_test.csv  data/y_train.csv  data/y_test.csv
#   models/preprocessor_linear.pkl  models/preprocessor_tree.pkl

jupyter nbconvert --to notebook --execute --inplace \
    notebooks/02_preprocessing.ipynb \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3

# Sanity check: confirm the artifacts 03a/b/c need actually got written
ls -lh data/X_train.csv data/X_test.csv data/y_train.csv data/y_test.csv
ls -lh models/preprocessor_linear.pkl models/preprocessor_tree.pkl
