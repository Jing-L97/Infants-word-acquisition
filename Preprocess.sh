#!/bin/bash
#SBATCH --job-name=Preprocess
#SBATCH --cpus-per-task=10    
#SBATCH --mem=200G
#SBATCH --time=30:00:00
#SBATCH --output=%x-%j.log
echo "Running job on $(hostname)"
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate temporal
python Preprocess.py
