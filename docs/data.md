# Data set curation
* Based on Cell-by-Gene (CxG) census version `2023-05-08`
* Gene space subset to `19331` protein coding genes (see `notebooks/store_creation/features.parquet` for full list)
* Cells are subset with following criterion
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
  5. Each cell type has to be observed in at least `30` donors to reliably quantify whether the classifier can generalize to new donors.
  6. Each cell type needs to have at least `7` parent nodes according to the cell type ontology to filter out too granular / general cell type labels
* Split into train, val and test set
  * Split is based on donor:  
    * E.g. each donor is either in the train, val or test set (Unlike for random subsampling)
    * A donor based split better represents how the classifier generalises to unseen donors / data sets
  * Split fraction: train=0.7, val=0.15, test=0.15


# Data preprocessing
* Preprocessing includes the following steps:
  1. Normalize each cell to have `10000` counts
  2. Quantile transformation
     * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
     * Fitted on training data subsetted to `1.000.000` cells (weighted sampling based on donor_id)
  3. Zero-centering: Normalize each column to have zero mean


# Data statistics
* `157` cell types
* `4771` unique donors
* Data set sizes
  * train: `14.564.352` cells
  * val: `3.402.752` cells
  * test: `3.211.264` cells
* `19331` genes (protein coding genes)
* `54` different tissues (`190` with more fine-grained annotation)
