#!/bin/bash

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

python -m venv --system-site-packages merlin
source merlin/bin/activate

python -m pip install cudf-cu11==23.02 rmm-cu11==23.02 dask-cudf-cu11==23.02 --extra-index-url https://pypi.nvidia.com/
python -m pip install torch torchvision torchaudio
python -m pip install merlin-dataloader["base"]
python -m pip install pytorch-lightning
python -m pip install torchmetrics
python -m pip install tensorboard
python -m pip install pytorch-tabnet
python -m pip install -e git/cellnet
