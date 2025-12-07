#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node=64
#SBATCH --mail-user=pawojcik@andrew.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# load modules
module load anaconda3/2024.10-1

# activate python env
conda activate pyML

# nav to PDB folder
cd ${HOME}/lm_lab_proj/neuro-ml-capstone

python preprocess_image_stack.py