cellnet
=======
De novo cell type prediction model for single-cell RNA-seq data that can be trained across a large-scale collection of 
curated datasets.

Project structure
-----
* ``cellnet``: code for models + data loading infrastructure
* ``docs``: 
  * ``data.md``: Details about data preparation
  * ``models.md``: Details about used models
  * ``classification-evaluation-metrics.md``: Details about used evaluation metrics
* ``notebooks``:
  * ``data_augmentation``: Notebooks related to data augmentation &rarr; calculation of augmentation vectors +
  evaluation 
  * ``model_evaluation``: Notebooks containing all evaluation code from this paper
  * ``loss_curve_plotting``: Notebooks to plot and compare loss curves
  * ``store_creation``: Notebooks used to create and reproduce the datasets used in this paper
  * ``training``: Notebooks to train models
* ``notebooks-tutorials``: 
  * ``data_loading.ipynb``: Example notebook about how to use data loading
  * ``model_inference.ipynb``: Example notebook how to use trained models for inference
* ``scripts``: Scripts used to train models

Installation
------------
Run the following command the project folder to install the ``cellnet`` package:
``pip install -e .``

To install GPU dependencies install the dependencies from the ``requirements-gpu.txt`` file first. 
To do so, use ``--extra-index-url https://pypi.nvidia.com/`` argument when installing packages via pip.

Licence
-------
MIT license

Authors
-------
`cellnet` was written by `Felix Fischer <felix.fischer@helmholtz-muenchen.de>`_.
