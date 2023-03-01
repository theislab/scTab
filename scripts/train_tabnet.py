import argparse
import os
from random import uniform
from time import sleep

import torch.cuda
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from lightning_fabric.utilities.seed import seed_everything

from cellnet.estimators import EstimatorCellTypeClassifier


def get_paths(cluster: str, model: str):
    if cluster == 'jsc':
        return (
            os.path.join('/p/scratch/hai_sfaira/tb_logs/', model),
            os.path.join('/p/scratch/hai_sfaira/tb_logs', model),
            '/p/scratch/ccstdl/theislab/merlin_cxg_simple_norm_parquet'
        )
    elif cluster == 'icb':
        return (
            os.path.join('/lustre/scratch/users/felix.fischer/tb_logs', model),
            os.path.join('/lustre/scratch/users/felix.fischer/tb_logs', model),
            '/lustre/scratch/users/felix.fischer/merlin_cxg_simple_norm_parquet'
        )
    else:
        raise ValueError(f'Only "jsc" or "icb" are supported as cluster. You supplied: {cluster}')


def get_model_checkpoint(checkpoint_path, checkpoint):
    if checkpoint is None:
        return None
    else:
        return os.path.join(checkpoint_path, 'default', checkpoint)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str)

    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--lambda_sparse', default=1e-6, type=float)
    parser.add_argument('--n_d', default=512, type=int)
    parser.add_argument('--n_a', default=128, type=int)
    parser.add_argument('--n_steps', default=5, type=int)
    parser.add_argument('--gamma', default=1.3, type=float)
    parser.add_argument('--n_independent', default=4, type=int)
    parser.add_argument('--n_shared', default=4, type=int)
    parser.add_argument('--virtual_batch_size', default=256, type=int)
    parser.add_argument('--mask_type', default='entmax', type=str)
    parser.add_argument('--lr_scheduler_step_size', default=2, type=int)
    parser.add_argument('--lr_scheduler_gamma', default=0.9, type=float)
    parser.add_argument('--version', default=None, type=str)

    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_interval', default=1, type=int)

    parser.add_argument('--seed', default=1, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    args = parse_args()
    print(args)

    # config parameters
    MODEL = 'cellnet_simple'
    CHECKPOINT_PATH, LOGS_PATH, DATA_PATH = get_paths(args.cluster, MODEL)

    sleep(uniform(0., 30.))  # add random sleep interval to avoid duplicated tensorboard log dirs
    estim = EstimatorCellTypeClassifier(DATA_PATH)
    seed_everything(args.seed)
    estim.init_datamodule(batch_size=args.batch_size)
    estim.init_trainer(
        trainer_kwargs={
            'max_epochs': 1000,
            'gradient_clip_val': 1.,
            'gradient_clip_algorithm': 'norm',
            'default_root_dir': CHECKPOINT_PATH,
            'resume_from_checkpoint': get_model_checkpoint(CHECKPOINT_PATH, args.resume_from_checkpoint),
            'accelerator': 'gpu',
            'devices': 1,
            'num_sanity_val_steps': 0,
            'check_val_every_n_epoch': 1,
            'logger': [TensorBoardLogger(LOGS_PATH, name='default', version=args.version)],
            'log_every_n_steps': 100,
            'detect_anomaly': False,
            'enable_progress_bar': True,
            'enable_model_summary': False,
            'enable_checkpointing': True,
            'callbacks': [
                TQDMProgressBar(refresh_rate=250),
                LearningRateMonitor(logging_interval='step'),
                ModelCheckpoint(filename='train_loss_{epoch}_{train_loss:.3f}', monitor='train_loss_epoch', mode='min',
                                every_n_epochs=args.checkpoint_interval, save_top_k=2),
                ModelCheckpoint(filename='val_f1_macro_{epoch}_{val_f1_macro:.3f}', monitor='val_f1_macro', mode='max',
                                every_n_epochs=args.checkpoint_interval, save_top_k=2),
                ModelCheckpoint(filename='val_loss_{epoch}_{val_loss:.3f}', monitor='val_loss', mode='min',
                                every_n_epochs=args.checkpoint_interval, save_top_k=2)
            ],
        }
    )
    estim.init_model(
        model_type='tabnet',
        model_kwargs={
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'lr_scheduler': torch.optim.lr_scheduler.StepLR,
            'lr_scheduler_kwargs': {
                'step_size': args.lr_scheduler_step_size,
                'gamma': args.lr_scheduler_gamma,
                'verbose': True
            },
            'optimizer': torch.optim.AdamW,
            'lambda_sparse': args.lambda_sparse,
            'n_d': args.n_d,
            'n_a': args.n_a,
            'n_steps': args.n_steps,
            'gamma': args.gamma,
            'n_independent': args.n_independent,
            'n_shared': args.n_shared,
            'virtual_batch_size': args.virtual_batch_size,
            'mask_type': args.mask_type,
        },
    )
    print(ModelSummary(estim.model))
    estim.train()
