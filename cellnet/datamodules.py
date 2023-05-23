import os
from math import ceil
from os.path import join
from typing import Dict, List

import lightning.pytorch as pl
import merlin.io
from merlin.dataloader.torch import Loader
from merlin.dtypes import boolean
from merlin.dtypes import float32, int64
from merlin.schema import ColumnSchema, Schema


PARQUET_SCHEMA = {
    'X': float32,
    'soma_joinid': int64,
    'is_primary_data': boolean,
    'dataset_id': int64,
    'donor_id': int64,
    'assay': int64,
    'cell_type': int64,
    'development_stage': int64,
    'disease': int64,
    'tissue': int64,
    'tissue_general': int64,
    'tech_sample': int64,
    'idx': int64,
}


def merlin_dataset_factory(path: str, columns: List[str], dataset_kwargs: Dict[str, any]):
    return merlin.io.Dataset(
        path,
        engine='parquet',
        schema=Schema(
            [
                ColumnSchema(
                    'X', dtype=PARQUET_SCHEMA['X'],
                    is_list=True, is_ragged=False,
                    properties={'value_count': {'max': 19331}}
                )
            ] +
            [ColumnSchema(col, dtype=PARQUET_SCHEMA[col]) for col in columns]
        ),
        **dataset_kwargs
    )


def set_default_kwargs_dataloader(kwargs: Dict[str, any] = None, training: bool = True):
    assert isinstance(training, bool)
    if kwargs is None:
        kwargs = {}
    if 'parts_per_chunk' not in kwargs:
        kwargs['parts_per_chunk'] = 8 if training else 1
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = training
    if'shuffle' not in kwargs:
        kwargs['shuffle'] = training

    return kwargs


def set_default_kwargs_dataset(kwargs: Dict[str, any] = None, training: bool = True):
    if kwargs is None:
        kwargs = {}
    if all(['part_size' not in kwargs, 'part_mem_fraction' not in kwargs]):
        kwargs['part_size'] = '100MB' if training else '325MB'

    return kwargs


def _get_data_files(base_path: str, split: str, sub_sample_frac: float):
    if sub_sample_frac == 1.:
        # if no subsampling -> just return base path and merlin takes care of the rest
        return join(base_path, split)
    else:
        files = [file for file in os.listdir(join(base_path, split)) if file.endswith('.parquet')]
        files = [join(base_path, split, file) for file in sorted(files, key=lambda x: int(x.split('.')[1]))]
        return files[:ceil(sub_sample_frac * len(files))]


class MerlinDataModule(pl.LightningDataModule):

    def __init__(
            self,
            path: str,
            columns: List[str],
            batch_size: int,
            sub_sample_frac: float = 1.,
            dataloader_kwargs_train: Dict[str, any] = None,
            dataloader_kwargs_inference: Dict[str, any] = None,
            dataset_kwargs_train: Dict[str, any] = None,
            dataset_kwargs_inference: Dict[str, any] = None
    ):
        super(MerlinDataModule).__init__()
        for col in columns:
            assert col in PARQUET_SCHEMA

        self.dataloader_kwargs_train = set_default_kwargs_dataloader(dataloader_kwargs_train, training=True)
        self.dataloader_kwargs_inference = set_default_kwargs_dataloader(dataloader_kwargs_inference, training=False)

        self.train_dataset = merlin_dataset_factory(
            _get_data_files(path, 'train', sub_sample_frac),
            columns,
            set_default_kwargs_dataset(dataset_kwargs_train, training=True)
        )
        self.val_dataset = merlin_dataset_factory(
            _get_data_files(path, 'val', sub_sample_frac),
            columns,
            set_default_kwargs_dataset(dataset_kwargs_inference, training=False)
        )
        self.test_dataset = merlin_dataset_factory(
            join(path, 'test'), columns, set_default_kwargs_dataset(dataset_kwargs_inference, training=False))

        self.batch_size = batch_size

    def train_dataloader(self):
        return Loader(self.train_dataset, batch_size=self.batch_size, **self.dataloader_kwargs_train)

    def val_dataloader(self):
        return Loader(self.val_dataset, batch_size=self.batch_size, **self.dataloader_kwargs_inference)

    def test_dataloader(self):
        return Loader(self.test_dataset, batch_size=self.batch_size, **self.dataloader_kwargs_inference)

    def predict_dataloader(self):
        return Loader(self.test_dataset, batch_size=self.batch_size, **self.dataloader_kwargs_inference)
