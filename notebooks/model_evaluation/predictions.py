import gc
from os.path import join

import anndata
import dask.dataframe as dd
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from cellnet.estimators import EstimatorCellTypeClassifier
from cellnet.models import TabnetClassifier, LinearClassifier, MLPClassifier


def get_count_matrix(ddf):
    x = (
        ddf['X']
        .map_partitions(
            lambda xx: pd.DataFrame(np.vstack(xx.tolist())),
            meta={col: 'f4' for col in range(19331)}
        )
        .to_dask_array(lengths=[1024] * ddf.npartitions)
    )

    return x


def eval_tabnet(ckpts, data_path):
    estim = EstimatorCellTypeClassifier(data_path)
    estim.init_datamodule(batch_size=2048)
    estim.trainer = pl.Trainer(logger=[], accelerator='gpu', devices=1)

    preds = []
    for ckpt in ckpts:
        estim.model = TabnetClassifier.load_from_checkpoint(ckpt, **estim.get_fixed_model_params('tabnet'))
        probas = estim.predict(estim.datamodule.test_dataloader())
        preds.append(np.argmax(probas, axis=1))
        gc.collect()

    return preds


def eval_linear(ckpts, data_path):
    estim = EstimatorCellTypeClassifier(data_path)
    estim.init_datamodule(batch_size=2048)
    estim.trainer = pl.Trainer(logger=[], accelerator='gpu', devices=1)

    preds = []
    for ckpt in ckpts:
        estim.model = LinearClassifier.load_from_checkpoint(ckpt, **estim.get_fixed_model_params('linear'))
        probas = estim.predict(estim.datamodule.test_dataloader())
        preds.append(np.argmax(probas, axis=1))
        gc.collect()

    return preds


def eval_xgboost(ckpts, data_path):
    x_test = np.load(join(data_path, 'pca/x_pca_training_test_split_256.npy'))

    preds = []
    for ckpt in ckpts:
        clf = xgb.XGBClassifier()
        clf.load_model(ckpt)
        clf.set_params(predictor='gpu_predictor')
        preds.append(clf.predict(x_test))

    return preds


def eval_mlp(ckpts, data_path):
    estim = EstimatorCellTypeClassifier(data_path)
    estim.init_datamodule(batch_size=2048)
    estim.trainer = pl.Trainer(logger=[], accelerator='gpu', devices=1)

    preds = []
    for ckpt in ckpts:
        estim.model = MLPClassifier.load_from_checkpoint(ckpt, **estim.get_fixed_model_params('mlp'))
        probas = estim.predict(estim.datamodule.test_dataloader())
        preds.append(np.argmax(probas, axis=1))
        gc.collect()

    return preds


def eval_celltypist(ckpts, data_path):
    import celltypist

    ddf = dd.read_parquet(join(data_path, 'test'), split_row_groups=True)
    x = get_count_matrix(ddf)
    var = pd.read_parquet(join(data_path, 'var.parquet'))

    preds = []
    for ckpt in ckpts:
        preds_ckpt = []
        # run this in batches to keep the memory footprint in check
        for i, idxs in enumerate(np.array_split(np.arange(x.shape[0]), 20)):
            # data is already normalized
            adata_test = anndata.AnnData(
                X=x[idxs, :].map_blocks(csr_matrix).compute(),
                var=var.set_index('feature_name')
            )
            preds_ckpt.append(celltypist.annotate(adata_test, model=ckpt))

        preds.append(
            np.concatenate([batch.predicted_labels.to_numpy().flatten() for batch in preds_ckpt])
        )

    return preds
