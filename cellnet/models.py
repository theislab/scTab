import abc
import gc
from typing import Callable, Dict, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score

from cellnet.tabnet.tab_network import TabNet


def _sf_log1p_norm(x):
    # norm to 10000 counts per cell and log1p transform
    counts = torch.sum(x, dim=1, keepdim=True)
    # avoid zero devision error
    counts += counts == 0.
    scaling_factor = 10000. / counts
    return torch.log1p(scaling_factor * x)


class BaseClassifier(pl.LightningModule, abc.ABC):

    classifier: nn.Module  # classifier mapping von gene_dim to type_dim - outputs logits

    def __init__(
        self,
        # fixed params
        gene_dim: int,
        type_dim: int,
        feature_means: np.ndarray,
        class_weights: np.ndarray,
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
        gc_frequency: int = 5,
        normalize_inputs=False
    ):
        assert isinstance(normalize_inputs, bool)
        super(BaseClassifier, self).__init__()

        self.gene_dim = gene_dim
        self.type_dim = type_dim
        self.train_set_size = train_set_size
        self.val_set_size = val_set_size
        self.batch_size = batch_size
        self.gc_freq = gc_frequency
        self.normalize_inputs = normalize_inputs

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.optim = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.register_buffer('feature_means', torch.tensor(feature_means.astype('f4')))

        metrics = MetricCollection({
            'f1_micro': MulticlassF1Score(num_classes=type_dim, average='micro'),
            'f1_macro': MulticlassF1Score(num_classes=type_dim, average='macro'),
        })
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

        self.register_buffer('class_weights', torch.tensor(class_weights.astype('f4')))
        self.register_buffer('child_lookup', torch.tensor(child_matrix.astype('i8')))

    @abc.abstractmethod
    def _step(self, batch, training=True) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def hierarchy_correct(self, preds, targets) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_is_child_node_or_node = torch.sum(
            self.child_lookup[targets, :] * F.one_hot(preds, self.type_dim), dim=1
        ) > 0

        return (
            torch.where(pred_is_child_node_or_node, targets, preds),  # corrected preds
            torch.where(pred_is_child_node_or_node, preds, targets)  # corrected targets
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        with torch.no_grad():
            batch = batch[0]
            batch['cell_type'] = torch.squeeze(batch['cell_type'])
            batch['idx'] = torch.squeeze(batch['idx'])
            if self.normalize_inputs:
                batch['X'] = _sf_log1p_norm(batch['X'])

        return batch

    def forward(self, x: torch.Tensor):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        preds, loss = self._step(batch, training=True)
        self.log('train_loss', loss)
        f1_macro = self.train_metrics['f1_macro'](preds, batch['cell_type'])
        f1_micro = self.train_metrics['f1_micro'](preds, batch['cell_type'])
        self.log('train_f1_macro_step', f1_macro)
        self.log('train_f1_micro_step', f1_micro)

        if batch_idx % self.gc_freq == 0:
            gc.collect()

        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss = self._step(batch, training=False)
        self.log('val_loss', loss)
        self.val_metrics['f1_macro'].update(preds, batch['cell_type'])
        self.val_metrics['f1_micro'].update(preds, batch['cell_type'])
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def test_step(self, batch, batch_idx):
        preds, loss = self._step(batch, training=False)
        self.log('test_loss', loss)
        self.test_metrics['f1_macro'].update(preds, batch['cell_type'])
        self.test_metrics['f1_micro'].update(preds, batch['cell_type'])
        if batch_idx % self.gc_freq == 0:
            gc.collect()

    def on_train_epoch_end(self) -> None:
        self.log('train_f1_macro_epoch', self.train_metrics['f1_macro'].compute())
        self.train_metrics['f1_macro'].reset()
        self.log('train_f1_micro_epoch', self.train_metrics['f1_micro'].compute())
        self.train_metrics['f1_micro'].reset()
        gc.collect()

    def on_validation_epoch_end(self) -> None:
        f1_macro = self.val_metrics['f1_macro'].compute()
        self.log('val_f1_macro', f1_macro)
        self.log('hp_metric', f1_macro)
        self.val_metrics['f1_macro'].reset()
        self.log('val_f1_micro', self.val_metrics['f1_micro'].compute())
        self.val_metrics['f1_micro'].reset()
        gc.collect()

    def on_test_epoch_end(self) -> None:
        self.log('test_f1_macro', self.test_metrics['f1_macro'].compute())
        self.test_metrics['f1_macro'].reset()
        self.log('test_f1_micro', self.test_metrics['f1_micro'].compute())
        self.test_metrics['f1_micro'].reset()
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


class LinearClassifier(BaseClassifier):

    def __init__(
        self,
        # fixed params
        gene_dim: int,
        type_dim: int,
        feature_means: np.ndarray,
        class_weights: np.ndarray,
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
        normalize_inputs=False
    ):
        super(LinearClassifier, self).__init__(
            gene_dim=gene_dim,
            type_dim=type_dim,
            feature_means=feature_means,
            class_weights=class_weights,
            child_matrix=child_matrix,
            train_set_size=train_set_size,
            val_set_size=val_set_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            normalize_inputs=normalize_inputs
        )
        self.save_hyperparameters(ignore=['feature_means', 'class_weights', 'parent_matrix', 'child_matrix'])

        self.classifier = nn.Linear(gene_dim, type_dim)

    def _step(self, batch, training=True):
        x = batch['X'] - self.feature_means
        logits = self(x)
        targets = batch['cell_type']
        preds = torch.argmax(logits, dim=1)
        if training:
            loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, targets)

        return preds, loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch['X'] - self.feature_means
        return F.softmax(self(x), dim=1)


class TabnetClassifier(BaseClassifier):

    def __init__(
        self,
        # fixed params
        gene_dim: int,
        type_dim: int,
        feature_means: np.ndarray,
        class_weights: np.ndarray,
        child_matrix: np.ndarray,
        sample_labels: np.ndarray,
        augmentations: np.ndarray,
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
        normalize_inputs: bool = False,
        # tabnet params
        lambda_sparse: float = 1e-5,
        n_d: int = 256,
        n_a: int = 128,
        n_steps: int = 3,
        gamma: float = 1.3,
        n_independent: int = 3,
        n_shared: int = 3,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 256,
        momentum: float = 0.02,
        mask_type: str = 'entmax',
        augment_training_data: bool = True,
        correct_targets: bool = False,
        min_class_freq: int = 100
    ):
        if augmentations is None and augment_training_data:
            raise ValueError('No augmentations provided and augment_training_data set to "True"')

        super(TabnetClassifier, self).__init__(
            gene_dim=gene_dim,
            type_dim=type_dim,
            feature_means=feature_means,
            class_weights=class_weights,
            child_matrix=child_matrix,
            train_set_size=train_set_size,
            val_set_size=val_set_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            normalize_inputs=normalize_inputs
        )
        self.save_hyperparameters(
            ignore=['feature_means', 'class_weights', 'child_matrix', 'sample_labels', 'augmentations'])

        self.lambda_sparse = lambda_sparse
        classifier = TabNet(
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
        self.classifier = classifier

        self.augment_training_data = augment_training_data
        if self.augment_training_data:
            self.register_buffer('augmentations', torch.tensor(augmentations.astype('f4')))

        self.correct_targets = correct_targets
        if self.correct_targets:
            self.register_buffer('sample_labels', torch.tensor(sample_labels))
            self.min_class_freq = int(min_class_freq)

        self.predict_bottleneck = False

    def _step(self, batch, training=True):
        if self.augment_training_data & training:
            x = self._augment_data(batch['X']) - self.feature_means
        else:
            x = batch['X'] - self.feature_means
        logits, m_loss = self(x)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            preds_corrected, targets_corrected = self.hierarchy_correct(preds, batch['cell_type'])
            if self.correct_targets:
                # recalculate class weights with updated labels
                with torch.no_grad():
                    self.sample_labels[batch['idx']] = targets_corrected
                    labels, counts = torch.unique(self.sample_labels, return_counts=True)
                    # use minimum count of self.min_class_freq
                    label_counts = self.min_class_freq * torch.ones(
                        self.type_dim, dtype=torch.int64, device=labels.device)
                    label_counts[labels] = counts
                    weight = label_counts.sum() / (self.type_dim * label_counts)
            else:
                weight = self.class_weights

        if training:
            if self.correct_targets:
                loss = F.cross_entropy(logits, targets_corrected, weight=weight) - self.lambda_sparse * m_loss
            else:
                loss = F.cross_entropy(logits, batch['cell_type'], weight=weight) - self.lambda_sparse * m_loss
        else:
            loss = F.cross_entropy(logits, targets_corrected)

        return preds_corrected, loss

    def _augment_data(self, x: torch.Tensor):
        augmentations = self.augmentations[
            torch.randint(0, self.augmentations.shape[0], (x.shape[0], ), device=x.device), :
        ]
        sign = 2. * (torch.bernoulli(.5 * torch.ones(x.shape[0], 1, device=x.device)) - .5)

        return torch.clamp(x + (sign * augmentations), min=0., max=1.)

    def predict_embedding(self, x: torch.Tensor):
        steps_output, _ = self.classifier.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)

        return res

    def predict_cell_types(self, x: torch.Tensor):
        return F.softmax(self(x)[0], dim=1)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch['X'] - self.feature_means
        if self.predict_bottleneck:
            return self.predict_embedding(x)
        else:
            return self.predict_cell_types(x)
