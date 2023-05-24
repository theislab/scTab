# Overview models

## TabNet model
* Based on https://arxiv.org/abs/1908.07442
* Trained with cross-entropy loss
* Input features: 19331 protein coding genes
* Output: class probabilities


## Linear model
* Linear model (single fully connected layer) trained with cross-entropy loss
* Input features: 19331 protein coding genes
* Output: class probabilities


## XGBoost model
* Based on official XGBoost model: https://xgboost.readthedocs.io/en/stable/
* Trained with `multi:softprob` objective
* Input features:
  * 256 PCA components (calculated based on all protein coding genes / 19331 genes)
  * PCA is computed based on training data
* Output: class probabilities
