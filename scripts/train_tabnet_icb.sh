#!/bin/bash

#SBATCH -J tabnet
#SBATCH --output=slurm_out/out.%j
#SBATCH --error=slurm_out/err.%j
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=6
#SBATCH --nice=1000


source $HOME/.profile
conda activate merlin
srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     python -u py_scripts/train_tabnet.py \
     --cluster="icb" \
     --version='version_xx'
