#!/bin/bash

#SBATCH --account=hai_cellnet
#SBATCH --nodes=1
#SBATCH --output=slurm_out/out.%j
#SBATCH --error=slurm_out/err.%j
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

cd ~/"$USER" || exit
# load modules
ml purge
ml Stages/2023
ml GCC/11.3.0
ml OpenMPI/4.1.4
ml CUDA/11.7
ml cuDNN/8.6.0.163-CUDA-11.7
ml NCCL/default-CUDA-11.7
ml Python/3.10.4

source merlin-torch/bin/activate
cd git/cellnet/scripts || exit

srun -n 1 --cpus-per-task=12 --exclusive --output=slurm_out/out_gpu0.%j --error=slurm_out/err_gpu0.%j --cpu-bind=verbose,socket --gres=gpu:1 \
     python -u py_scripts/train_linear.py \
     --cluster="jsc" \
     --version='version_1' --epochs=35 --seed=1 &
srun -n 1 --cpus-per-task=12 --exclusive --output=slurm_out/out_gpu1.%j --error=slurm_out/err_gpu1.%j --cpu-bind=verbose,socket --gres=gpu:1 \
     python -u py_scripts/train_linear.py \
     --cluster="jsc" \
     --version='version_2' --epochs=35 --seed=2 &
srun -n 1 --cpus-per-task=12 --exclusive --output=slurm_out/out_gpu2.%j --error=slurm_out/err_gpu2.%j --cpu-bind=verbose,socket --gres=gpu:1 \
     python -u py_scripts/train_linear.py \
     --cluster="jsc" \
     --version='version_3' --epochs=35 --seed=3 &
srun -n 1 --cpus-per-task=12 --exclusive --output=slurm_out/out_gpu3.%j --error=slurm_out/err_gpu3.%j --cpu-bind=verbose,socket --gres=gpu:1 \
     python -u py_scripts/train_linear.py \
     --cluster="jsc" \
     --version='version_4' --epochs=35 --seed=4 &

wait
