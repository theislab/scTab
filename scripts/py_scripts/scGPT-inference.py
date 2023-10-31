from os.path import join
from pathlib import Path

import scanpy as sc
import scgpt as scg


SAVE_PATH = '/mnt/dssfs02/scGPT-splits'
model_dir = Path("/mnt/dssfs02/scTab-checkpoints/scGPT")
cell_type_key = "cell_type"
gene_col = "index"


for i in range(10):
    adata = sc.read_h5ad(join(SAVE_PATH, 'train', f'{i}.h5ad'))
    adata = scg.tasks.embed_data(
        adata,
        model_dir,
        cell_type_key=cell_type_key,
        gene_col=gene_col,
        batch_size=64,
        return_new_adata=True,
    ).write_h5ad(join(SAVE_PATH, 'train', f'{i}_embed.h5ad'))


for i in range(30):
    adata = sc.read_h5ad(join(SAVE_PATH, 'test', f'{i}.h5ad'))
    adata = scg.tasks.embed_data(
        adata,
        model_dir,
        cell_type_key=cell_type_key,
        gene_col=gene_col,
        batch_size=64,
        return_new_adata=True,
    ).write_h5ad(join(SAVE_PATH, 'test', f'{i}_embed.h5ad'))
