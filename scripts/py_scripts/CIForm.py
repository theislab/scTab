import warnings
from os.path import join

import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
import anndata
import math
import scanpy as sc
import dask.dataframe as dd
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import (DataLoader,Dataset)
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import random
from sklearn import preprocessing


##Set random seeds
def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(2021)


"""
Code taken and slightly adapted from: https://github.com/zhanglab-wbgcas/CIForm/blob/main/Tutorial/Tutorial_Inter.ipynb
"""


##Gene embedding,
# Function
# The pre-processed scRNA-seq data is converted into a form acceptable to the Transformer encoder
# Parameters
# gap: The length of a sub-vector
# adata: pre-processed scRNA-seq data. The rows represent the cells and the columns represent the genes
# Traindata_paths: the paths of the cell type labels file(.csv) corresponding to the training data
def getXY(gap, adata):
    # Converting the gene expression matrix into sub-vectors
    # (n_cells,n_genes) -> (n_cells,gap_num,gap)  gap_num = int(gene_num / gap) + 1
    X = adata.X  # getting the gene expression matrix
    single_cell_list = []
    for single_cell in X:
        feature = []
        length = len(single_cell)
        # spliting the gene expression vector into some sub-vectors whose length is gap
        for k in range(0, length, gap):
            if (k + gap <= length):
                a = single_cell[k:k + gap]
            else:
                a = single_cell[length - gap:length]
            # scaling each sub-vectors
            a = preprocessing.scale(a)
            feature.append(a)
        feature = np.asarray(feature)
        single_cell_list.append(feature)

    single_cell_list = np.asarray(single_cell_list)  # (n_cells,gap_num,gap)

    return single_cell_list, adata.obs.cell_type.to_numpy()


##Function
# Converting label annotation to numeric form
##Parameters
# cells:  all cell type labels
# cell_types: all cell types of Training datasets
def getNewData(cells, cell_types):
    labels = []
    for i in range(len(cells)):
        cell = cells[i]
        cell = str(cell).upper()

        if (cell_types.__contains__(cell)):
            indexs = cell_types.index(cell)
            labels.append(indexs + 1)
        else:
            labels.append(0)  # 0 denotes the unknowns cell types

    return np.asarray(labels)


class TrainDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = torch.from_numpy(self.data)
        label = torch.from_numpy(self.label)

        return data[index], label[index]


class TestDataSet(Dataset):
    def __init__(self, data):
        self.data = data

        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = torch.from_numpy(self.data)
        return data[index]


##Positional Encoder Layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        ##the sine function is used to represent the odd-numbered sub-vectors
        pe[:, 0::2] = torch.sin(position * div_term)
        ##the cosine function is used to represent the even-numbered sub-vectors
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


##CIForm
##function
# annotating cell type identification of scRNA-seq data
##parameters
# input_dim  :Default is equal to gap
# nhead      :Number of heads in the attention mechanism
# d_model    :Default is equal to gap
# num_classes:Number of cell types
# dropout    :dropout rate which is used to prevent model overfitting
class CIForm(nn.Module):
    def __init__(self, input_dim, nhead=2, d_model=80, num_classes=2, dropout=0.1):
        super().__init__()
        # TransformerEncoderLayer with self-attention
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=1024, nhead=nhead, dropout=dropout
        )

        # Positional Encoding with self-attention
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Classification layer
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, mels):
        out = mels.permute(1, 0, 2)
        # Positional Encoding layer
        out = self.positionalEncoding(out)
        # Transformer Encoder layer layer
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        # Pooling layer
        out = out.mean(dim=1)
        # Classification layer
        out = self.pred_layer(out)
        return out


##main
##parameters
# s                  :the length of a sub-vector
# referece_datapath  :the paths of referece datasets
# Train_names        :the names of referece datasets
# Testdata_path      :the path pf test dataset
# Testdata_name      :the name of test dataset
def main(s, adata_train, adata_test):
    gap = s  # the length of a sub-vector
    d_models = s
    heads = 64  # the number of heads in self-attention mechanism

    lr = 0.0001  # learning rate
    dp = 0.1  # dropout rate
    batch_sizes = 256  # the size of batch
    n_epochs = 20  # the number of epoch

    # Getting the data which input into the CIForm
    train_data, labels = getXY(gap, adata_train)
    test_data, _ = getXY(gap, adata_test)
    num_classes = 164

    # Constructing the CIForm model
    model = CIForm(input_dim=d_models, nhead=heads, d_model=d_models, num_classes=num_classes, dropout=dp)
    # Setting loss function
    criterion = nn.CrossEntropyLoss()
    # Setting optimization function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Setting the training dataset
    train_dataset = TrainDataSet(data=train_data, label=labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, pin_memory=True)
    # Setting the test dataset
    test_dataset = TestDataSet(data=test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False, pin_memory=True)

    # starting training CIForm.Using training data to train CIForm
    # n_epochs: the times of Training
    model.train()
    for epoch in range(n_epochs):
        for batch in tqdm(train_loader):
            # A batch consists of scRNA-seq data and corresponding cell type annotation.
            data, labels = batch
            logits = model(data)
            labels = torch.tensor(labels, dtype=torch.long)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), '/mnt/dssfs02/tb_logs/CIForm/CIForm.tar')
    ##Starting the validation model, which predicts the cell types in the test dataset
    model.eval()
    y_predict = []
    for batch in tqdm(test_loader):
        data = batch
        with torch.no_grad():
            logits = model(data)
        # Getting the predicted cell type
        preds = logits.argmax(1)
        preds = preds.cpu().numpy().tolist()
        y_predict.extend(preds)

    with open('/mnt/dssfs02/tb_logs/CIForm/CIForm_preds.npy', 'wb') as f:
        np.save(f, np.array(y_predict))


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


def get_adata(split, hvg_mask=None, max_cells: int = None):
    data_path = '/mnt/dssmcmlfs01/merlin_cxg_2023_05_15_sf-log1p'

    ddf = dd.read_parquet(join(data_path, split), split_row_groups=True)
    if hvg_mask is None:
        x = get_count_matrix(ddf)[:max_cells, :].compute()
        var = pd.read_parquet(join(data_path, 'var.parquet'))
    else:
        x = get_count_matrix(ddf)[:max_cells, hvg_mask].compute()
        var = pd.read_parquet(join(data_path, 'var.parquet')).iloc[hvg_mask]
    obs = dd.read_parquet(join(data_path, split), columns=['cell_type']).compute().iloc[:max_cells]

    return anndata.AnnData(X=x, obs=obs, var=var)


if __name__ == '__main__':
    adata_train = get_adata('train', max_cells=750_000)
    sc.pp.highly_variable_genes(adata_train, n_top_genes=2000)
    hvgs = adata_train.var.highly_variable.to_numpy()
    adata_train = adata_train[:, hvgs].copy()
    adata_test = get_adata('test', hvg_mask=hvgs, max_cells=None)

    s = 1024  # the length of a sub-vector
    main(s, adata_train, adata_test)
