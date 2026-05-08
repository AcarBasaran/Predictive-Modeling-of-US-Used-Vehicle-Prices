#!/bin/bash
#SBATCH --job-name=03b_rf
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --output=logs/03b_%j.out
#SBATCH --error=logs/03b_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

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

# Required inputs (produced by 02_preprocessing.ipynb)
ls -lh data/X_train.csv data/X_test.csv data/y_train.csv data/y_test.csv \
       models/preprocessor_tree.pkl

jupyter nbconvert --to notebook --execute --inplace \
    notebooks/03b_random_forest.ipynb \
    --ExecutePreprocessor.timeout=18000 \
    --ExecutePreprocessor.kernel_name=python3

# Sanity check: confirm the artifact 04_evaluation needs got written
ls -lh models/random_forest.pkl
