import torch
import torchmetrics

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
import wandb


def compute_accuracy(model: torch.nn.Module, dataloader: DataLoader) -> float:
    """Computes the accuracy of a model on a dataset."""

    # Set the model to evaluation mode.
    model = model.eval()

    correct = 0
    total_examples = 0

    # Loop over the data, compare predictions with labels.
    for features, labels in dataloader:
        # Make sure no gradients are computed.
        with torch.no_grad():
            logits = model(features)

        # Predictions are the argmax over the last dimension, since our logits
        # are of shape (batch_size, num_classes).
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


class LightningModel(LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate: float, num_classes: int = None):
        """Initializes the LightningModel.

        Args:
            model: A PyTorch model.
            learning_rate: The learning rate to use for training.
            num_classes: Number of classes for classification. If None, will be inferred.
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.model = model
        self.num_classes = num_classes

        # Initialize metrics - will be updated in setup if num_classes is None
        if num_classes is not None:
            self._init_metrics(num_classes)
        else:
            self.train_acc = None
            self.val_acc = None
            self.test_acc = None

    def _init_metrics(self, num_classes):
        """Initialize metrics with the correct number of classes."""
        task = "binary" if num_classes == 2 else "multiclass"
        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes)

        # Per-class metrics
        self.val_precision = torchmetrics.Precision(task=task, num_classes=num_classes, average="none")
        self.val_recall = torchmetrics.Recall(task=task, num_classes=num_classes, average="none")
        self.val_f1 = torchmetrics.F1Score(task=task, num_classes=num_classes, average="none")

    def setup(self, stage=None):
        """Setup method called before training starts."""
        if self.num_classes is None:
            # Try to infer num_classes from model output
            if hasattr(self.model, 'num_labels'):
                self.num_classes = self.model.num_labels
            elif hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'out_features'):
                self.num_classes = self.model.classifier.out_features
            elif hasattr(self.model, 'fc') and hasattr(self.model.fc, 'out_features'):
                self.num_classes = self.model.fc.out_features
            else:
                # Default to binary classification
                self.num_classes = 2
                print(f"Warning: Could not infer num_classes, defaulting to {self.num_classes}")
            
            self._init_metrics(self.num_classes)
            print(f"Initialized metrics with {self.num_classes} classes")

    def forward(self, x):
        """Forward pass of the model."""
        return self.model(x)

    def _shared_step(self, batch):
        """Shared step for training, validation, and test steps."""
        features, true_labels = batch
        logits = self(features)

        # Validate logits dimensions
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"Model output dimension ({logits.shape[1]}) doesn't match num_classes ({self.num_classes})")
        
        # Validate label range
        if true_labels.max() >= self.num_classes or true_labels.min() < 0:
            raise ValueError(f"Labels contain values outside valid range [0, {self.num_classes-1}]. "
                           f"Found min: {true_labels.min()}, max: {true_labels.max()}")

        loss = F.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        if self.train_acc is not None:
            self.train_acc(predicted_labels, true_labels)
            self.log(
                "train_acc",
                self.train_acc,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)

        if self.val_acc is not None:
            self.val_acc(predicted_labels, true_labels)
            self.log("val_acc", self.val_acc, prog_bar=True)

        # Log per-class metrics
        precision = self.val_precision(predicted_labels, true_labels)
        recall = self.val_recall(predicted_labels, true_labels)
        f1 = self.val_f1(predicted_labels, true_labels)

        for class_idx in range(self.num_classes):
            self.log(f"val_precision_class_{class_idx}", precision[class_idx])
            self.log(f"val_recall_class_{class_idx}", recall[class_idx])
            self.log(f"val_f1_class_{class_idx}", f1[class_idx])

    def test_step(self, batch, batch_idx):
        """Test step for the model."""
        loss, true_labels, predicted_labels = self._shared_step(batch)
        if self.test_acc is not None:
            self.test_acc(predicted_labels, true_labels)
            self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        return optimizer

class HFLightningModel(LightningModel):
    def __init__(
        self,
        model: torch.nn.Module,
        label_name: str = "labels",
        learning_rate: float = 5e-5,
        num_classes: int = None,
        task_type: str = "sequence_classification"  # "sequence_classification" or "token_classification"
    ):
        """
        Flexible Lightning module for sequence or token classification.

        Args:
            model: Pretrained HuggingFace model.
            label_name: Name of the label key in the batch.
            learning_rate: Learning rate for optimizer.
            num_classes: Number of classes.
            task_type: "sequence_classification" or "token_classification".
        """
        super().__init__(model, learning_rate, num_classes)
        self.model = model
        self.label_name = label_name
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.task_type = task_type

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    def _shared_step(self, batch):
        labels = batch[self.label_name]
        outputs = self(batch)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        if self.task_type == "sequence_classification":
            # Sequence classification: [batch_size, num_classes]
            if logits.shape[1] != self.num_classes:
                raise ValueError(
                    f"Model output dimension ({logits.shape[1]}) doesn't match num_classes ({self.num_classes})"
                )
            loss = F.cross_entropy(logits, labels)
            predicted_labels = torch.argmax(logits, dim=1)

        elif self.task_type == "token_classification":
            # Token classification: [batch_size, seq_len, num_classes]
            if logits.shape[2] != self.num_classes:
                raise ValueError(
                    f"Model output dimension ({logits.shape[2]}) doesn't match num_classes ({self.num_classes})"
                )
            # Flatten batch and seq_len for loss
            loss = F.cross_entropy(
                logits.view(-1, self.num_classes),
                labels.view(-1),
                ignore_index=-100  # ignore special tokens
            )
            predicted_labels = torch.argmax(logits, dim=-1)

        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, preds = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, preds = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class FlexibleLightningModel(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float,
        num_classes: int = None,
        label_name: str = "labels",
        task_type: str = "sequence_classification",  # or "token_classification"
    ):
        """
        Flexible Lightning model for sequence or token classification.

        Args:
            model: PyTorch or HuggingFace model.
            learning_rate: Learning rate for optimizer.
            num_classes: Number of classes (required for metrics).
            label_name: Key name for labels in batch.
            task_type: "sequence_classification" or "token_classification".
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.label_name = label_name
        self.task_type = task_type

        if num_classes is not None:
            self._init_metrics(num_classes)
        else:
            self.train_acc = None
            self.val_acc = None
            self.test_acc = None

    def _init_metrics(self, num_classes):
        """Initialize metrics for both task types."""
        task = "binary" if num_classes == 2 else "multiclass"
        ignore_index = -100 if self.task_type == "token_classification" else None

        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, ignore_index=ignore_index)
        self.val_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, ignore_index=ignore_index)
        self.test_acc = torchmetrics.Accuracy(task=task, num_classes=num_classes, ignore_index=ignore_index)

        # Per-class metrics
        self.val_precision = torchmetrics.Precision(task=task, num_classes=num_classes, average="none", ignore_index=ignore_index)
        self.val_recall = torchmetrics.Recall(task=task, num_classes=num_classes, average="none", ignore_index=ignore_index)
        self.val_f1 = torchmetrics.F1Score(task=task, num_classes=num_classes, average="none", ignore_index=ignore_index)

    def forward(self, batch):
        """Forward pass."""
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    def _shared_step(self, batch):
        """Shared logic for training, validation, and testing."""
        labels = batch[self.label_name]
        outputs = self(batch)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        if self.task_type == "sequence_classification":
            if logits.shape[1] != self.num_classes:
                raise ValueError(f"Model output dimension ({logits.shape[1]}) doesn't match num_classes ({self.num_classes})")
            loss = F.cross_entropy(logits, labels)
            predicted_labels = torch.argmax(logits, dim=1)

        elif self.task_type == "token_classification":
            if logits.shape[2] != self.num_classes:
                raise ValueError(f"Model output dimension ({logits.shape[2]}) doesn't match num_classes ({self.num_classes})")
            loss = F.cross_entropy(logits.view(-1, self.num_classes), labels.view(-1), ignore_index=-100)
            predicted_labels = torch.argmax(logits, dim=-1)

        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

        return loss, labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, labels, preds = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        if self.train_acc is not None:
            self.train_acc(preds, labels)
            self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, labels, preds = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)

        if self.val_acc is not None:
            self.val_acc(preds, labels)
            self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)

        # Log per-class metrics
        f1 = self.val_f1(preds, labels)
        precision = self.val_precision(preds, labels)
        recall = self.val_recall(preds, labels)

        for class_idx in range(self.num_classes):
            if class_idx == 1:
                # log once with prog_bar=True, tracked by both progress bar + wandb
                self.log(f"val_f1_class_{class_idx}", f1[class_idx], on_epoch=True, prog_bar=True)
            else:
                self.log(f"val_f1_class_{class_idx}", f1[class_idx], on_epoch=True)

            # precision & recall for all classes (including 1)
            self.log(f"val_precision_class_{class_idx}", precision[class_idx], on_epoch=True)
            self.log(f"val_recall_class_{class_idx}", recall[class_idx], on_epoch=True)

        return {"loss": loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx):
        loss, labels, preds = self._shared_step(batch)
        if self.test_acc is not None:
            self.test_acc(preds, labels)
            self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def train_model_lightning(
    group: str,
    project_name: str,
    metric_to_monitor: str = "val_acc",
    mode: str = "max",
    max_epochs: int = 30,
) -> Trainer:
    """Trains a PyTorch model using PyTorch Lightning.

    The model is trained using the provided train and validation Data
    Loaders. The training process is logged using the provided logger.
    The model is checkpointed based on the validation accuracy.

    Args:
        lightning_model: The LightningModule to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        logger: The logger to use for logging.
        max_epochs: The maximum number of epochs to train for.

    Returns:
        The trained Trainer object.
    """

    wandb_logger = WandbLogger(
        project=project_name,
        log_model="all",
        group=group
    )

    callbacks = [
        ModelCheckpoint(
            save_top_k=1,
            mode=mode,
            monitor=metric_to_monitor,
            save_last=True,
        )
    ]

    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        accelerator="auto",
        logger=wandb_logger,
        deterministic=True,
        gradient_clip_val=1.0  # Enable gradient clippin
    )

    return trainer
