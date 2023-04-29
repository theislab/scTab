import numpy as np

from numba import njit


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
