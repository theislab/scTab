#!/bin/bash

#SBATCH -J scGPT
#SBATCH --output=slurm_out/scGPT_out.%j
#SBATCH --error=slurm_out/scGPT_err.%j
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 3-00:00:00
#SBATCH --mem=90GB
#SBATCH --cpus-per-task=6

DSSFS02_HOME="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer"
DSSMCML_HOME="/dss/dssmcmlfs01/pn36po/pn36po-dss-0000/di93zer"


srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts="/dss:/dss,${DSSFS02_HOME}:/mnt/dssfs02,${DSSMCML_HOME}:/mnt/dssmcmlfs01" \
     --container-image="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/scGPT-jupyter.sqsh" \
     --no-container-remap-root \
     bash -c "python -u /dss/dsshome1/04/di93zer/git/cellnet/scripts/py_scripts/scGPT-inference.py"
