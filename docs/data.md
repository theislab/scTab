# Data set creation
**Author:** Felix Fischer (GitHub @felix0097) \
**Date:** 12.06.2023

## Data set curation
* Based on Cell-by-Gene (CxG) census version `2023-05-15` 
  * CxG census: https://chanzuckerberg.github.io/cellxgene-census/index.html)
    ```python
    import cellxgene_census

    census = cellxgene_census.open_soma(census_version='2023-05-15')
    ```
  * The CxG census can be used to query CxG data with a standardised API: 
    https://chanzuckerberg.github.io/cellxgene-census/python-api.html
  * The census version `2023-05-15` is a long-term supported (LTS) release and will be hosted by cellxgene for at least 
    5 years
* Gene space subset to `19331` protein coding genes (see `notebooks/store_creation/features.parquet` for full list)
* Cells from the census are subset with following criterions:
  1. Data has to be primary data to prevent label leakage between the train and test set: `is_primary_data == True`
  2. Sequencing protocol has to be in `Protocols` (`assay in PROTOCOLS`) with
       ```pytyhon
        PROTOCOLS = [
           "10x 5' v2", 
           "10x 3' v3", 
           "10x 3' v2", 
           "10x 5' v1", 
           "10x 3' v1", 
           "10x 3' transcription profiling", 
           "10x 5' transcription profiling"
       ] 
       ```
  3. Annotated cell type has to be a subtype of `native cell`
  4. There have to be at least `5000` cells for a specific cell type
  5. Each cell type has to be observed in at least `30` donors to reliably quantify whether the classifier can 
  generalize to new donors.
  6. Each cell type needs to have at least `7` parent nodes according to the cell type ontology. This criterion is used 
     as heuristic filter out to filter out to granular cell type labels
  granular / general cell type labels
* Split data into train, val and test set
  * Splits are based on donors:  
    * E.g. each donor is either in the train, val or test set (Unlike for random subsampling)
    * A donor based split better represents how the classifier generalises to unseen donors / data sets
    * A donor based split roughly resembles a random split when looking at the overall proportion of cells in the 
    train, val and test set.cIf the number of donors is large enough this hold wells. On average 68% of the samples per 
    cell type are in the train set in the current setting (ideal would be 70% with a 70% - 15% - 15% split). The worst 
    outlier cell type is that only 37% of the cells are in the training set. 
  * Split fraction: train=0.7, val=0.15, test=0.15
* The code to reproduce the data set creation can be found under 
  `notebooks/store_creation/01_create_train_val_test_splits.ipynb`


## Data preprocessing
Preprocessing includes the following steps:
1. Normalize each cell to have `10000` counts (size factor normalization)
   ```python
   import numpy as np
      
      
   def sf_normalize(X):
       X = X.copy()
       counts = np.array(X.sum(axis=1))
       # avoid zero division error
       counts += counts == 0.
       # normalize to 10000. counts
       scaling_factor = 10000. / counts
       np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)
        
       return X
   ```
2. After size factor normalization the data is Log1p transformed
   ```python
    import numpy as np
        
        
    def log1p_norm(x):
      return np.log1p(x)
   ```


## Data statistics

* `164` cell types
* `5052` unique donors
* Data set size: `22.189.056` cells
  * train: `15.240.192` cells
  * val: `3.500.,032` cells
  * test: `3.448.832` cells
* `19331` genes (protein coding genes)
* `56` different tissues (`197` with more fine-grained tissue annotation)


## Data preparation pipeline

* The data preparation pipeline can be found under `notebooks/store_creation`:
  1. `01_create_train_val_test_splits.ipynb`: Subset and download data from CxG census. And split downloaded data into 
  train, val and test set
  2. optional `02_fit_quantile_norm.ipynb`: Fit quantile normalization model (only necessary for quantile normalized 
     data - can be skipped with default size factor + log1p normalization)
  3. `03_write_store_merlin.ipynb`: Save data into on-disk format that can be used by Nvidia Merlin dataloader 
  (https://github.com/NVIDIA-Merlin/dataloader)
  4. `04_create_hierarchy_matrices.ipynb`: Create child node lookup matrix to find subtypes based on cell type 
  ontology
  5. `05_compute_pca.ipynb`: Compute PCA embeddings for visualization (50 components) and model training 
  (256 components)
  6. `06_check_written_store.ipynb`: Sanity check written data
