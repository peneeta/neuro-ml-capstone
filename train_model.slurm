#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gpus=v100-32:4
#SBATCH --mail-user=pawojcik@andrew.cmu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# load modules
module load anaconda3/2024.10-1
module load cuda

# activate python env
conda activate pyML

# nav
cd ${HOME}/lm_lab_proj/neuro-ml-capstone

srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train_model.py