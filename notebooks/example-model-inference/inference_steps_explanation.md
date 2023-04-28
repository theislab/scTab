# Inference Steps

All data can be found on `bayes` under `/home/felixfischer/model_inference`

## Phase 1: Data Preprocessing
1. Collect Data
    * Raw count data is required (no normalization)
    * Data should be supplied in cellxgene `.h5ad` format (raw counts under `.raw.X`)
2. Align gene feature space
    * Order of columns in count matrix is given in `var.parquet` file. Reorder genes accordingly. Genes have to be in exactly same order!
    * Zero fill genes if they're missing in the supplied data
    * For the code in this notebook to work correctly, both data sets need to have the same ensembl release (release 104 in this case)
3. Normalize data
   * Normalization includes the following steps
     1. Normalize each cell to have 10000 counts. This is already included in the preprocessing pipeline from step b. So, no need to run this separately. 
     2. Quantile transformation 
         * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
         * Model is already fitted during training (`preproc_pipeline.pickle` file). Only inference is required.
     3. Zero Centering
        * Feature means are saved in `means.npy` file
        * Zero centering: `x = x - feature_means`
        * This is done in Phase 3 to keep memory footprint low as zero centering breaks the sparsity structure of the count matrix
4. Wrap count data array into PyTorch data loader

## Phase 2: Load Trained Model
1. Load model checkpoint via `torch.load()`
2. Initialize new model
   1. Load model architecture / hyper-parameters from `hparams.yaml` file
   2. Initialize model according to model architecture from step i.
3. Load weights + set model to inference model
   1. Load pretrained weights via `model.load_state_dict(weights)`
   2. Set model to eval mode `model.eval()`

## Phase 3: Run Model Inference

1. Run model inference
2. Map integer predictions to string labels via `cell_type.parquet` file
