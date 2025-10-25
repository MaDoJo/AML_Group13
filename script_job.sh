#!/bin/sh

#SBATCH --job-name=AML_no_DA
#SBATCH --output=logs/no_DA_output%j.out
#SBATCH --time=48:00:00
#SBATCH --partition=gpushort
#SBATCH --account=users
#SBATCH --gres=gpu:a100:1


export HOME=/scratch/s5112583

nvidia-smi

module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.4.0


# Uncomemment the commented lines below if you want to create a virtual environment and/or install dependencies


python -m venv projectEnvironment
source projectEnvironment/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install papermill ipykernel  # for running notebooks

# Register this virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=python3 --display-name "Python 3 (venv)"

# Run all cells in pipeline.ipynb and save executed version
papermill pipeline.ipynb pipeline_executed.ipynb