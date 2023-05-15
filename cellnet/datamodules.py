import os
from math import ceil
from os.path import join
from typing import Dict, List

import lightning.pytorch as pl
import merlin.io
import pyarrow as pa
from merlin.dataloader.torch import Loader
from merlin.dtypes import float32, int64
from merlin.schema import ColumnSchema, Schema


PARQUET_SCHEMA = pa.schema([
    ('X', pa.list_(pa.float32())),
    ('id', pa.int64()),
    ('assay_sc', pa.int64()),
    ('tech_sample', pa.int64()),
    ('cell_type', pa.int64()),
    ('cell_type_ontology_term_id', pa.int64()),
    ('disease', pa.int64()),
    ('development_stage', pa.int64()),
    ('organ', pa.int64()),
    ('idx', pa.int64()),
])


def _merlin_dataset_factory(path: str, columns: List[str], dataset_kwargs: Dict):
    return merlin.io.Dataset(
            path,
            engine='parquet',
            schema=Schema(
                [
                    ColumnSchema(
                        'X', dtype=float32,
                        is_list=True, is_ragged=False,
                        properties={'value_count': {'max': 19357}}
                    )
                ] +
                [ColumnSchema(col, dtype=int64) for col in columns]
            ),
            **dataset_kwargs
        )


def _set_default_kwargs_dataloader(kwargs: Dict[str, any], train: bool = True):
    if kwargs is None:
        kwargs = {}

    parts_per_chunk = 8 if train else 1
    drop_last = True if train else False
    shuffle = True if train else False

    if 'parts_per_chunk' not in kwargs:
        kwargs['parts_per_chunk'] = parts_per_chunk
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = drop_last
    if'shuffle' not in kwargs:
        kwargs['shuffle'] = shuffle

    return kwargs


def _set_default_kwargs_dataset(kwargs: Dict[str, any], train: bool = True):
    if kwargs is None:
        kwargs = {}

    part_size = '100MB' if train else '325MB'

    if all(['part_size' not in kwargs, 'part_mem_fraction' not in kwargs]):
        kwargs['part_size'] = part_size

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
            dataloader_kwargs_train: Dict = None,
            dataloader_kwargs_inference: Dict = None,
            dataset_kwargs_train: Dict = None,
            dataset_kwargs_inference: Dict = None
    ):
        super(MerlinDataModule).__init__()

        for col in columns:
            assert col in PARQUET_SCHEMA.names

        self.dataloader_kwargs_train = _set_default_kwargs_dataloader(dataloader_kwargs_train, train=True)
        self.dataloader_kwargs_inference = _set_default_kwargs_dataloader(dataloader_kwargs_inference, train=False)

        self.train_dataset = _merlin_dataset_factory(
            _get_data_files(path, 'train', sub_sample_frac),
            columns,
            _set_default_kwargs_dataset(dataset_kwargs_train, train=True)
        )
        self.val_dataset = _merlin_dataset_factory(
            _get_data_files(path, 'val', sub_sample_frac),
            columns,
            _set_default_kwargs_dataset(dataset_kwargs_inference, train=False)
        )
        self.test_dataset = _merlin_dataset_factory(
            join(path, 'test'), columns, _set_default_kwargs_dataset(dataset_kwargs_inference, train=False))

        self.batch_size = batch_size

    def train_dataloader(self):
        return Loader(self.train_dataset, batch_size=self.batch_size, **self.dataloader_kwargs_train)

    def val_dataloader(self):
        return Loader(self.val_dataset, batch_size=self.batch_size, **self.dataloader_kwargs_inference)

    def test_dataloader(self):
        return Loader(self.test_dataset, batch_size=self.batch_size, **self.dataloader_kwargs_inference)

    def predict_dataloader(self):
        return Loader(self.test_dataset, batch_size=self.batch_size, **self.dataloader_kwargs_inference)
