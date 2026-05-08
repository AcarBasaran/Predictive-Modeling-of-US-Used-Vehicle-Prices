#!/bin/bash
#SBATCH --job-name=01_eda
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --output=logs/01_eda_%j.out
#SBATCH --error=logs/01_eda_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

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

jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_eda.ipynb \
    --ExecutePreprocessor.timeout=1800 \
    --ExecutePreprocessor.kernel_name=python3
