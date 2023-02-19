from os.path import join
from typing import Dict, List

import merlin.io
import pyarrow as pa
import pytorch_lightning as pl
from merlin.dataloader.torch import Loader
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
])


def _merlin_dataset_factory(path: str, columns: List[str], dataset_kwargs: Dict):
    return merlin.io.Dataset(
            path,
            engine='parquet',
            schema=Schema(
                [
                    ColumnSchema(
                        'X', dtype='float32',
                        is_list=True, is_ragged=False,
                        properties={'value_count': {'max': 19357}}
                    )
                ] +
                [ColumnSchema(col, dtype='int64') for col in columns]
            ),
            **dataset_kwargs
        )


class MerlinDataModule(pl.LightningDataModule):

    def __init__(
            self,
            path: str,
            columns: List[str],
            batch_size: int,
            drop_last: bool = True,
            merlin_dataset_kwargs_train: Dict = None,
            merlin_dataset_kwargs_inference: Dict = None
    ):
        super(MerlinDataModule).__init__()
        if merlin_dataset_kwargs_train is None:
            merlin_dataset_kwargs_train = {}
        if merlin_dataset_kwargs_inference is None:
            merlin_dataset_kwargs_inference = {}

        for col in columns:
            assert col in PARQUET_SCHEMA.names

        # set default partition size if not supplied
        default_part_size = '650MB'
        if all([
            'part_size' not in merlin_dataset_kwargs_train, 'part_mem_fraction' not in merlin_dataset_kwargs_train
        ]):
            merlin_dataset_kwargs_train['part_size'] = default_part_size
        if all([
            'part_size' not in merlin_dataset_kwargs_inference,
            'part_mem_fraction' not in merlin_dataset_kwargs_inference
        ]):
            merlin_dataset_kwargs_inference['part_size'] = default_part_size

        self.train_dataset = _merlin_dataset_factory(join(path, 'train'), columns, merlin_dataset_kwargs_train)
        self.val_dataset = _merlin_dataset_factory(join(path, 'val'), columns, merlin_dataset_kwargs_inference)
        self.test_dataset = _merlin_dataset_factory(join(path, 'test'), columns, merlin_dataset_kwargs_inference)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def train_dataloader(self):
        return Loader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last
        )

    def val_dataloader(self):
        return Loader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

    def test_dataloader(self):
        return Loader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

    def predict_dataloader(self):
        return Loader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
