# Inference Steps

## Phase 1: Data Preprocessing
1. Collect Data
    * Raw count data is required (no normalization)
    * Data should be supplied in cellxgene `.h5ad` format (raw counts under `.raw.X`)
2. Align gene feature space
    * Order of columns in count matrix is given in `var.parquet` file. Reorder genes accordingly. Genes have to be in 
      exactly same order!
    * Zero fill genes if they're missing in the supplied data
    * For the code in this notebook to work correctly, both data sets need to have the same ensembl release 
      (release 104 in this case)
3. Wrap count data array into PyTorch data loader

## Phase 2: Load Trained Model
1. Load model checkpoint via `torch.load()`
2. Initialize new model
   1. Load model architecture / hyperparameters from `hparams.yaml` file
   2. Initialize model according to model architecture from step i.
3. Load weights + set model to inference model
   1. Load pretrained weights via `model.load_state_dict(weights)`
   2. Set model to eval mode `model.eval()`

## Phase 3: Run Model Inference

1. Run model inference
2. Map integer predictions to string labels via `cell_type.parquet` file
