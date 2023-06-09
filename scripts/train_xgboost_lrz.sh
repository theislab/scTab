#!/bin/bash

#SBATCH -J xgboost
#SBATCH --output=slurm_out/xgboost_out.%j
#SBATCH --error=slurm_out/xgboost_err.%j
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --gres=gpu:1
#SBATCH --time 3-00:00:00
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=12


srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts='/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02,/dss/dssmcmlfs01/pn36po/pn36po-dss-0000/di93zer:/mnt/dssmcmlfs01' \
     --container-image='/dss/dsshome1/04/di93zer/merlin-2302.sqsh' \
     python -u /dss/dsshome1/04/di93zer/git/cellnet/scripts/py_scripts/train_xgboost.py \
     --cluster="lrz" \
     --version='version_1' --seed=1
