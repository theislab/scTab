#!/bin/bash

#SBATCH -J tabnet
#SBATCH --output=slurm_out/tabnet_out.%j
#SBATCH --error=slurm_out/tabnet_err.%j
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 3-00:00:00
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=6

DSSFS02_HOME="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer"
DSSMCML_HOME="/dss/dssmcmlfs01/pn36po/pn36po-dss-0000/di93zer"


SCRIPT="/dss/dsshome1/04/di93zer/git/cellnet/scripts/py_scripts/train_tabnet.py"
GIT_REPO="/dss/dsshome1/04/di93zer/git/cellnet"
ARGS="--version=version_1 --epochs=50"

srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts="/dss:/dss,${DSSFS02_HOME}:/mnt/dssfs02,${DSSMCML_HOME}:/mnt/dssmcmlfs01" \
     --container-image="/dss/dsshome1/04/di93zer/merlin-2302.sqsh" \
     --no-container-remap-root \
     bash -c "pip install -e ${GIT_REPO} --no-deps && python -u ${SCRIPT} --cluster=lrz ${ARGS}"
