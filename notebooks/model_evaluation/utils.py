import os
import re
from os.path import join
from typing import Dict, List

import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import classification_report


@njit
def correct_labels(y_true: np.ndarray, y_pred: np.ndarray, child_matrix: np.ndarray):
    """
    Update predictions.
    If prediction is actually a child node of the true label -> update prediction to true value.

    E.g: Label='T cell' and prediction='CD8 positive T cell' -> update prediction to 'T cell'
    """
    updated_predictions = y_pred.copy()
    # precalculate child nodes
    child_nodes = {i: np.where(child_matrix[i, :])[0] for i in range(child_matrix.shape[0])}

    for i, (pred, true_label) in enumerate(zip(y_pred, y_true)):
        if pred in child_nodes[true_label]:
            updated_predictions[i] = true_label
        else:
            updated_predictions[i] = pred

    return updated_predictions


def get_best_ckpts(logs_path, versions):
    best_ckpts = []

    for version in versions:
        # sort first -> in case both f1-score are the same -> take the one which was trained for fewer epochs
        files = sorted([file for file in os.listdir(join(logs_path, version, 'checkpoints')) if 'val_f1_macro' in file])
        f1_scores = [float(re.search('val_f1_macro=(.*?).ckpt', file).group(1)) for file in files]
        best_ckpt = files[np.argmax(f1_scores)]
        best_ckpts.append(join(logs_path, version, 'checkpoints', best_ckpt))

    return best_ckpts


def macro_f1_per_group(y_true, y_pred, group_variable, grouping: Dict[str, List[str]]):
    assert len(y_true) == len(y_pred) == len(group_variable)
    groups = []
    f1_macro = []

    for group, group_assignments in grouping.items():
        y_pred_group = y_pred[np.isin(group_variable, group_assignments).squeeze()]
        y_true_group = y_true[np.isin(group_variable, group_assignments).squeeze()]
        clf_report = pd.DataFrame(classification_report(
            y_true=y_true_group,
            y_pred=y_pred_group,
            labels=np.unique(y_true_group),
            output_dict=True,
            zero_division=0
        )).T
        groups.append(group)
        f1_macro.append(clf_report.loc['macro avg', 'f1-score'])

    return pd.DataFrame({'group': groups, 'f1_score': f1_macro})


"""
Constants
"""

BIONETWORK_GROUPING = {
    'adipose': ['adipose tissue'],
    'breast': ['breast', 'exocrine gland'],
    'eye': ['eye'],
    'gut': [
        'digestive system', 'small intestine', 'colon', 'intestine', 'stomach', 'esophagus', 'large intestine',
        'omentum', 'spleen', 'peritoneum', 'mucosa', 'abdomen', 'exocrine gland', 'endocrine gland'
    ],
    'heart': ['heart', 'vasculature'],
    'blood_and_immune': ['immune system', 'lymph node', 'blood', 'bone marrow', 'spleen'],
    'kidney': ['kidney'],
    'liver': ['liver'],
    'lung': ['lung', 'respiratory system', 'pleural fluid'],
    'musculoskeletal': ['musculature', 'bone marrow', 'vasculature'],
    'nervous_system': ['brain', 'endocrine gland'],
    'oral_and_craniofacial': ['tongue', 'nose', 'mucosa', 'exocrine gland', 'saliva', 'endocrine gland'],
    'pancreas': ['pancreas', 'exocrine gland', 'endocrine gland'],
    'reproduction': [
        'reproductive system', 'uterus', 'fallopian tube', 'ovary', 'prostate gland', 'endocrine gland',
        'ascitic fluid', 'urinary bladder', 'peritoneum', 'bladder organ', 'placenta'
    ],
    'skin': ['skin of body']
}
