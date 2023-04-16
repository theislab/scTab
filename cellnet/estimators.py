from os.path import join
from typing import Dict, List

import dask.dataframe as dd
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.tuner.tuning import Tuner

from cellnet.datamodules import MerlinDataModule
from cellnet.models import TabnetClassifier, LinearClassifier


class EstimatorCellTypeClassifier:

    datamodule: MerlinDataModule
    model: pl.LightningModule
    trainer: pl.Trainer

    def __init__(self, data_path: str):
        self.data_path = data_path

    def init_datamodule(
            self,
            batch_size: int = 4096,
            dataloader_kwargs_train: Dict = None,
            dataloader_kwargs_inference: Dict = None,
            merlin_dataset_kwargs_train: Dict = None,
            merlin_dataset_kwargs_inference: Dict = None
    ):
        self.datamodule = MerlinDataModule(
            self.data_path,
            columns=['idx', 'cell_type'],
            batch_size=batch_size,
            dataloader_kwargs_train=dataloader_kwargs_train,
            dataloader_kwargs_inference=dataloader_kwargs_inference,
            dataset_kwargs_train=merlin_dataset_kwargs_train,
            dataset_kwargs_inference=merlin_dataset_kwargs_inference
        )

    def init_model(self, model_type: str, model_kwargs):
        if model_type == 'tabnet':
            self.model = TabnetClassifier(**{**self.get_fixed_model_params(), **model_kwargs})
        elif model_type == 'linear':
            self.model = LinearClassifier(**{**self.get_fixed_model_params(), **model_kwargs})
        else:
            raise ValueError(f'model_type has to be in ["linear", "tabnet"]. You supplied: {model_type}')

    def init_trainer(self, trainer_kwargs):
        self.trainer = pl.Trainer(**trainer_kwargs)

    def _check_is_initialized(self):
        if not self.model:
            raise RuntimeError('You need to call self.init_model before calling self.train')
        if not self.datamodule:
            raise RuntimeError('You need to call self.init_datamodule before calling self.train')
        if not self.trainer:
            raise RuntimeError('You need to call self.init_trainer before calling self.train')

    def get_fixed_model_params(self):
        return {
            'gene_dim': len(pd.read_parquet(join(self.data_path, 'var.parquet'))),
            'type_dim': len(pd.read_parquet(join(self.data_path, 'categorical_lookup/cell_type.parquet'))),
            'feature_means': np.load(join(self.data_path, 'norm/zero_centering/means.npy')),
            'class_weights': np.load(join(self.data_path, 'class_weights.npy')),
            'child_matrix': np.load(join(self.data_path, 'cell_type_hierarchy/child_matrix.npy')),
            'sample_labels': dd.read_parquet(join(self.data_path, 'train'), columns='cell_type').compute().to_numpy(),
            'train_set_size': sum(self.datamodule.train_dataset.partition_lens),
            'val_set_size': sum(self.datamodule.val_dataset.partition_lens),
            'batch_size': self.datamodule.batch_size,
        }

    def find_lr(self, lr_find_kwargs, plot_results: bool = False):
        self._check_is_initialized()
        tuner = Tuner(self.trainer)
        lr_finder = tuner.lr_find(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
            **lr_find_kwargs
        )
        if plot_results:
            lr_finder.plot(suggest=True)

        return lr_finder.suggestion(), lr_finder.results

    def train(self, ckpt_path: str = None):
        self._check_is_initialized()
        self.trainer.fit(
            self.model,
            train_dataloaders=self.datamodule.train_dataloader(),
            val_dataloaders=self.datamodule.val_dataloader(),
            ckpt_path=ckpt_path
        )

    def validate(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer.validate(self.model, dataloaders=self.datamodule.val_dataloader(), ckpt_path=ckpt_path)

    def test(self, ckpt_path: str = None):
        self._check_is_initialized()
        return self.trainer.test(self.model, dataloaders=self.datamodule.test_dataloader(), ckpt_path=ckpt_path)

    def predict(self, ckpt_path: str = None) -> np.ndarray:
        self._check_is_initialized()
        predictions_batched: List[torch.Tensor] = self.trainer.predict(
            self.model,
            dataloaders=self.datamodule.predict_dataloader(),
            ckpt_path=ckpt_path
        )
        return torch.vstack(predictions_batched).numpy()
