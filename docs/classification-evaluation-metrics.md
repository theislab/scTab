# Evaluation metrics

* F1-score: https://en.wikipedia.org/wiki/F-score
* macro F1-score
  * Macro average == calculate F1-score per cell type and take the average of the per cell type F1-scores
  * Macro average better reflects performance across all cell types as there is a strong class imbalance in the data set
  * macro F1-score is often taken as the default metric to evaluate cell type classification performance
* weighted F1-score
  * Weighted average == calculate F1-score per cell type and take the weighted average of the per cell type F1-scores
  (classes with more samples get a higher weight)
  * Weighted average better reflects how many of the cells are classified correctly


# Evaluation data

* Evaluation covers the same cell types as seen during model training
* Evaluation data consists of donors the model has not seen during training 
  * Random donor holdout and not random subsampling!
  * This better represents how well the classifier generalises to unseen donors than just random subsampling of cells


# Dealing with different cell type annotation granularity

* Different data sets are often annotated with vastly different granularity e.g. `T cell` vs 
`CD4-positive, alpha-beta T cell`
* To account for this, the following rule applies when evaluating whether a prediction is right or wrong:
  * Prediction is `right` or `true` if:
    * Classifier predicts the same label as the author
    * Classifier predicts a subtype of the true label
      * This is considered as a right prediction as the prediction agrees with the true label up to the annotation 
      granularity the author provided
      * e.g. classifier predicts `CD4-positive, alpha-beta T cell` when the author annotated the cell with `T cell`
  * Prediction is `wrong` or `false` if:
    * Classifier predicts a parent cell type of the true label
      * This is considered a wrong prediction as the author supplied a more fine-grained label
      * e.g. classifier predicts `T cell` instead of `CD4-positive, alpha-beta T cell`
    * Anything else
* The code to find the child nodes based on the cell ontology (https://www.ebi.ac.uk/ols/ontologies/cl) can be 
found under `cellnet/utils/cell_ontology.py`


# Evaluation results

* The cell type classification evaluation notebooks for TabNet + reference models can be found here:
  * TabNet: `notebooks/model_evaluation/classification-tabnet.ipynb`
  * Linear reference model: `notebooks/model_evaluation/classification-linear.ipynb`
  * XGBoost reference model: `notebooks/model_evalutation/classification-xgboost.ipynb`

* The evaluation notebooks contain the following metrics:
  * Overall classification performance measured by macro F1-score (shows overall performance of the classifier)
  * Plot of per cell type F1-score (can be used to spot cell types where the model currently struggles with)
  * TSNE visualization of predicted and true labels
    * TSNE visualization is calculated based on the first 50 PCA components of the test set
    * Additionally, binary indicator whether a prediction is `right` or `wrong` is overlaid on the TSNE plots

* Evaluation notebooks for TabNet model:
  * `notebooks/model_evaluation/model-scaling-tabnet.ipynb`: Classification performance vs training data size for TabNet
  model
  * `notebooks/model_evaluation/classificationf-tabnet-ensembl.ipynb`: Deep ensemble of TabNet models + evaluation of 
  uncertainty quantification of predictions
