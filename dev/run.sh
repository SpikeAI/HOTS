#!/bin/bash
#SBATCH -t  3-10
#SBATCH --mail-user=antoine.grimaldi@univ-amu.fr
#SBATCH -J  logistic_regression
#SBATCH -A h146       
#SBATCH -p volta
#SBATCH --ntasks-per-node=32
#SBATCH --mem=150G

# Load modules :
module load userspace/all
module load python3/3.6.3
module load cuda/10.1

# Run your python file (optimization mode)
python -O  2021_03-09_meso_LR.py