import abc
import gc
from typing import Callable, Dict

import functorch
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNet
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class BaseClassifier(pl.LightningModule, abc.ABC):

    classifier: nn.Module  # classifier mapping von gene_dim to type_dim - outputs logits

    def __init__(
        self,
        # fixed params
        gene_dim: int,
        type_dim: int,
        feature_means: np.ndarray,
        class_weights: np.ndarray,
        parent_matrix: np.ndarray,
        child_matrix: np.ndarray,
        # params from datamodule
        train_set_size: int,
        val_set_size: int,
        batch_size: int,
        # model specific params
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        gc_frequency: int = 5
    ):
        super(BaseClassifier, self).__init__()

        self.gene_dim = gene_dim
        self.type_dim = type_dim
        self.train_set_size = train_set_size
        self.val_set_size = val_set_size
        self.batch_size = batch_size
        self.gc_freq = gc_frequency

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.register_buffer('feature_means', torch.tensor(feature_means))

        metrics = MetricCollection({
            'f1_micro': MulticlassF1Score(num_classes=type_dim, average='micro'),
            'f1_macro': MulticlassF1Score(num_classes=type_dim, average='macro'),
            'acc_micro': MulticlassAccuracy(num_classes=type_dim, average='micro'),
            'acc_macro': MulticlassAccuracy(num_classes=type_dim, average='macro')
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights.astype('f4')))

        self.parent_lookup = nn.Embedding.from_pretrained(torch.tensor(parent_matrix.astype('f8')), freeze=True)
        self.get_hierarchy_corrected_preds = functorch.vmap(self._get_hierarchy_corrected_pred)
        self.child_lookup = nn.Embedding.from_pretrained(torch.tensor(child_matrix.astype('f8')), freeze=True)
        self.get_hierarchy_corrected_targets = functorch.vmap(self._get_hierarchy_corrected_target)

    @abc.abstractmethod
    def _step(self, batch) -> (torch.Tensor, torch.Tensor):
        """Calculate predictions (int64 tensor) and loss"""
        pass

    def _get_hierarchy_corrected_pred(self, logit, label):
        pred = torch.argmax(logit)
        label_is_parent_node_or_node = self.parent_lookup(pred) * F.one_hot(label, self.type_dim)

        return torch.where(torch.gt(torch.sum(label_is_parent_node_or_node), 0.), label, pred)

    def _get_hierarchy_corrected_target(self, logit, target):
        pred_is_child_node_or_node = self.child_lookup(target) * F.one_hot(torch.argmax(logit), self.type_dim)

        return torch.where(
            torch.gt(torch.sum(pred_is_child_node_or_node), 0.), torch.argmax(pred_is_child_node_or_node), target
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        with torch.no_grad():
            batch = batch[0]
            # zero center data
            batch['X'] = batch['X'] - self.feature_means

        return batch

    def forward(self, batch):
        return self.classifier(batch['X'])

    def training_step(self, batch, batch_idx):
        preds, loss = self._step(batch)
        self.log('train_loss', loss, on_epoch=True)
        self.log_dict(self.train_metrics(preds, batch['cell_type']), on_epoch=True)

        if batch_idx % self.gc_freq == 0:
            gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss = self._step(batch)
        self.log('val_loss', loss)
        metrics = self.val_metrics(preds, batch['cell_type'])
        self.log_dict(metrics)
        self.log('hp_metric', metrics['val_f1_macro'])

        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def test_step(self, batch, batch_idx):
        preds, loss = self._step(batch)
        self.log('test_loss', loss)
        self.log_dict(self.test_metrics(preds, batch['cell_type']))

        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def on_train_epoch_end(self) -> None:
        gc.collect()

    def on_validation_epoch_end(self) -> None:
        gc.collect()

    def configure_optimizers(self):
        optimizer_config = {'optimizer': self.optim(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)}
        if self.lr_scheduler is not None:
            lr_scheduler_kwargs = {} if self.lr_scheduler_kwargs is None else self.lr_scheduler_kwargs
            interval = lr_scheduler_kwargs.pop('interval', 'epoch')
            monitor = lr_scheduler_kwargs.pop('monitor', 'val_loss_epoch')
            frequency = lr_scheduler_kwargs.pop('frequency', 1)
            scheduler = self.lr_scheduler(optimizer_config['optimizer'], **lr_scheduler_kwargs)
            optimizer_config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': interval,
                'monitor': monitor,
                'frequency': frequency
            }

        return optimizer_config


class TabnetClassifier(BaseClassifier):

    def __init__(
        self,
        # fixed params
        gene_dim: int,
        type_dim: int,
        feature_means: np.ndarray,
        class_weights: np.ndarray,
        parent_matrix: np.ndarray,
        child_matrix: np.ndarray,
        # params from datamodule
        train_set_size: int,
        val_set_size: int,
        batch_size: int,
        # model specific params
        learning_rate: float = 0.005,
        weight_decay: float = 0.1,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Callable = None,
        lr_scheduler_kwargs: Dict = None,
        # tabnet params
        lambda_sparse: float = 1e-6,
        n_d: int = 256,
        n_a: int = 128,
        n_steps: int = 5,
        gamma: float = 1.3,
        n_independent: int = 4,
        n_shared: int = 4,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 128,
        momentum: float = 0.02,
        mask_type: str = 'entmax'
    ):
        super(TabnetClassifier, self).__init__(
            gene_dim=gene_dim,
            type_dim=type_dim,
            feature_means=feature_means,
            class_weights=class_weights,
            parent_matrix=parent_matrix,
            child_matrix=child_matrix,
            train_set_size=train_set_size,
            val_set_size=val_set_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )
        self.save_hyperparameters(ignore=['feature_means', 'class_weights', 'parent_matrix'])

        self.lambda_sparse = lambda_sparse
        self.classifier = TabNet(
            input_dim=gene_dim,
            output_dim=type_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=epsilon,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            mask_type=mask_type,
        )
        self.predict_bottleneck = False

    def _step(self, batch, training=True):
        logits, m_loss = self(batch)

        with torch.no_grad():
            preds = self.get_hierarchy_corrected_preds(logits, batch['cell_type'])

        if training:
            loss = self.ce_loss(logits, batch['cell_type']) - self.lambda_sparse * m_loss
        else:
            with torch.no_grad():
                # correct targets during evaluation
                targets = self.get_hierarchy_corrected_targets(logits, batch['cell_type'])
            loss = self.ce_loss(logits, targets) - self.lambda_sparse * m_loss

        return preds, loss

    def predict_embedding(self, batch):
        embedder = self.classifier.embedder
        encoder = self.classifier.tabnet.encoder
        steps_output, _ = encoder(embedder(batch['X']))
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        return res

    def predict_cell_types(self, batch):
        return F.softmax(self(batch)[0], dim=1)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.predict_bottleneck:
            return self.predict_embedding(batch)
        else:
            return self.predict_cell_types(batch)
