import argparse
from random import uniform
from time import sleep

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch import seed_everything

from cellnet.estimators import EstimatorCellTypeClassifier
from utils import get_paths, get_model_checkpoint


torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=str)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--logging_dir', type=str, default='cxg_2023_05_15_mlp')

    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--sub_sample_frac', default=1., type=float)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--n_hidden', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--augment_training_data', default=True, type=lambda x: x.lower() in ['true', '1', '1.'])
    parser.add_argument('--lr_scheduler_step_size', default=1, type=int)
    parser.add_argument('--lr_scheduler_gamma', default=0.9, type=float)
    parser.add_argument('--version', default=None, type=str)

    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--checkpoint_interval', default=1, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)

    parser.add_argument('--seed', default=1, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # config parameters
    MODEL = args.logging_dir
    CHECKPOINT_PATH, LOGS_PATH, DATA_PATH = get_paths(args.cluster, MODEL)
    if args.data_path is not None:
        DATA_PATH = args.data_path

    sleep(uniform(0., 30.))  # add random sleep interval to avoid duplicated tensorboard log dirs
    estim = EstimatorCellTypeClassifier(DATA_PATH)
    seed_everything(args.seed)
    estim.init_datamodule(batch_size=args.batch_size, sub_sample_frac=args.sub_sample_frac)
    estim.init_trainer(
        trainer_kwargs={
            'max_epochs': args.epochs,
            'gradient_clip_val': 1.,
            'gradient_clip_algorithm': 'norm',
            'accelerator': 'gpu',
            'devices': 1,
            'num_sanity_val_steps': 0,
            'check_val_every_n_epoch': args.check_val_every_n_epoch,
            'logger': [TensorBoardLogger(LOGS_PATH, name='default', version=args.version)],
            'log_every_n_steps': 200,
            'detect_anomaly': False,
            'enable_progress_bar': True,
            'enable_model_summary': False,
            'enable_checkpointing': True,
            'default_root_dir': CHECKPOINT_PATH,
            'callbacks': [
                TQDMProgressBar(refresh_rate=250),
                LearningRateMonitor(logging_interval='step'),
                ModelCheckpoint(filename='last_{epoch}', every_n_epochs=args.checkpoint_interval),
                ModelCheckpoint(filename='val_f1_macro_{epoch}_{val_f1_macro:.3f}', monitor='val_f1_macro', mode='max',
                                every_n_epochs=args.checkpoint_interval, save_top_k=2),
                ModelCheckpoint(filename='val_loss_{epoch}_{val_loss:.3f}', monitor='val_loss', mode='min',
                                every_n_epochs=args.checkpoint_interval, save_top_k=2)
            ],
        }
    )
    estim.init_model(
        model_type='mlp',
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
            'hidden_size': args.hidden_size,
            'n_hidden': args.n_hidden,
            'augment_training_data': args.augment_training_data
        },
    )
    print(ModelSummary(estim.model))
    estim.train(ckpt_path=get_model_checkpoint(CHECKPOINT_PATH, args.resume_from_checkpoint))
