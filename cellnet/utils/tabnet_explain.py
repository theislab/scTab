import numba
import numpy as np
from scipy.sparse import csc_matrix
from tqdm import tqdm

from cellnet.tabnet.tab_network import TabNet


@numba.njit
def _get_nnz_idxs_per_row(x: np.ndarray):
    nnz_idxs = []
    for ix in range(x.shape[0]):
        nnz_idxs.append(np.where(x[ix, :] > 0.)[0])

    return nnz_idxs


def create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
    """
    This is a computational trick.
    In order to rapidly sum importances from same embeddings
    to the initial index.

    Parameters
    ----------
    input_dim : int
        Initial input dim
    cat_emb_dim : int or list of int
        if int : size of embedding for all categorical feature
        if list of int : size of embedding for each categorical feature
    cat_idxs : list of int
        Initial position of categorical features
    post_embed_dim : int
        Post embedding inputs dimension

    Returns
    -------
    reducing_matrix : np.ndarray
        Matrix of dim (post_embed_dim, input_dim) to perform reduce
    """

    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim - 1] * len(cat_idxs)
    else:
        all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]

    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i + acc_emb])
        else:
            indices_trick.append(
                range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1)
            )
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1

    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    return csc_matrix(reducing_matrix)


def explain(
    model: TabNet,
    dataloader,
    only_return_nnz_idxs: bool = True,
    normalize: bool = False,
    device: str = 'cuda'
):
    reducing_matrix = create_explain_matrix(
        model.classifier.input_dim,1,[], model.classifier.input_dim)
    model.to(device)
    model.eval()

    res_explain = []
    for batch_idx, data in tqdm(enumerate(dataloader)):
        data = data[0]['X'].to(device)
        M_explain, _ = model.classifier.forward_masks(data)
        original_feat_explain = csc_matrix.dot(M_explain.cpu().detach().numpy(), reducing_matrix)
        if not only_return_nnz_idxs:
            res_explain.append(original_feat_explain)
        else:
            res_explain += _get_nnz_idxs_per_row(original_feat_explain)

    if not only_return_nnz_idxs:
        res_explain = np.vstack(res_explain)
        if normalize:
            res_explain /= np.sum(res_explain, axis=1)[:, None]

    return res_explain


def get_feature_masks(
    model: TabNet,
    dataloader,
    device: str = 'cuda'
):
    model.to(device)
    model.eval()

    res_explain = {i: [] for i in range(model.n_steps)}
    for batch_idx, data in tqdm(enumerate(dataloader)):
        data = data[0]['X'].to(device)
        _, masks = model.forward_masks(data)
        for i in range(model.n_steps):
            res_explain[i].append(masks[i].cpu().detach().numpy())

    return {i: np.vstack(res_explain[i]) for i in range(model.n_steps)}
