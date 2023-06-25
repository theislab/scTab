import argparse
import json
from os.path import join

import dask.dataframe as dd
import numpy as np
import xgboost as xgb

from utils import get_paths


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cluster', type=str)
    parser.add_argument('--version', type=str)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--n_estimators', type=int, default=800)
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--subsample', type=float, default=0.75)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--early_stopping_rounds', type=int, default=10)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # config parameters
    MODEL = 'cxg_2023_05_15_xgboost'
    CHECKPOINT_PATH, LOGS_PATH, DATA_PATH = get_paths(args.cluster, MODEL)
    # save hparams to json
    with open(join(CHECKPOINT_PATH, f'{args.version}_hparams.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # load training + val data
    x_train = np.load(join(DATA_PATH, 'pca/x_pca_training_train_split_256.npy'))
    y_train = dd.read_parquet(join(DATA_PATH, 'train'), columns='cell_type').compute().to_numpy()
    x_val = np.load(join(DATA_PATH, 'pca/x_pca_training_val_split_256.npy'))
    y_val = dd.read_parquet(join(DATA_PATH, 'val'), columns='cell_type').compute().to_numpy()
    class_weights = {i: weight for i, weight in enumerate(np.load(join(DATA_PATH, 'class_weights.npy')))}
    weights = np.array([class_weights[label] for label in y_train])

    clf = xgb.XGBClassifier(
        tree_method='gpu_hist',
        n_estimators=args.n_estimators,
        eta=args.eta,
        subsample=args.subsample,
        max_depth=args.max_depth,
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed
    )
    clf = clf.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)])
    clf.save_model(join(CHECKPOINT_PATH, f'{args.version}.json'))
