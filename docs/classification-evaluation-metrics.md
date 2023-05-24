# Evaluation metrics

* Classification is evaluated based on macro F1-score
  * Macro average == calculate F1-score per cell type and take the average of the per cell type F1-scores
  * Macro average better reflects performance across all cell types as there is a strong class imbalance in the data set
  * F1-score: https://en.wikipedia.org/wiki/F-score
  * macro F1-score is often taken as the default metric to evaluate cell type classification performance


# Evaluation data

* Evaluation covers the same cell types as seen during model training
* Evaluation data consists of donors the model has not seen during training 
  * Random donor holdout and not random subsampling!
  * This better represents how well the classifier generalises to unseen donors than just random subsampling of cells


# Dealing with different cell type annotation granularity

* Different data sets are often annotated with vastly different granularity e.g. `T cell` vs `CD4-positive, alpha-beta T cell`
* To account for this, the following rule applies when evaluating whether a prediction is right or wrong:
  * Prediction is `right` or `true` if:
    * Classifier predicts the same label as the author
    * Classifier predicts a subtype of the true label
      * This is considered as a right prediction as the prediction agrees with the true label up to the annotation granularity the author provided
      * e.g. classifier predicts `CD4-positive, alpha-beta T cell` when the author annotated the cell with `T cell`
  * Prediction is `wrong` or `false` if:
    * Classifier predicts a parent cell type of the true label
      * This is considered a wrong prediction as the author supplied a more fine-grained label
      * e.g. classifier predicts `T cell` instead of `CD4-positive, alpha-beta T cell`
    * Anything else
* The code to find the child nodes based on the cell ontology (https://www.ebi.ac.uk/ols/ontologies/cl) can be found under `cellnet/utils/cell_ontology.py`
