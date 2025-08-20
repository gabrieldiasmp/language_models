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


def train_model_vanilla(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
) -> tuple[list[float], list[float], list[float]]:
    """Trains a PyTorch model on a dataset, vanilla style."""

    # Define some parameters. These could be parameters to the function.
    LEARNING_RATE = 0.05
    LOGGING_BATCHES_FREQUENCY = 250

    # Get an optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Define lists to store loss and accuracies.
    loss_list, train_acc_list, val_acc_list = [], [], []
    for epoch in range(num_epochs):
        model = model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_idx % LOGGING_BATCHES_FREQUENCY:
                print(
                    f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                    f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                    f" | Train Loss: {loss:.2f}"
                )
            loss_list.append(loss.item())

        train_acc = compute_accuracy(model, train_loader)
        val_acc = compute_accuracy(model, val_loader)
        print(f"Train Acc.: {train_acc*100:.2f}% | Val. Acc.: {val_acc*100:.2f}%")
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

    return loss_list, train_acc_list, val_acc_list


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

    def __init__(self, model: torch.nn.Module, label_name: str, learning_rate: float, num_classes: int = None):
        super().__init__(model, learning_rate, num_classes)
        self.label_name = label_name

    def forward(self, batch):
        return self.model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    
    def _shared_step(self, batch):
        labels = batch[self.label_name]

        outputs = self(batch)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

        # Validate logits dimensions
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"Model output dimension ({logits.shape[1]}) doesn't match num_classes ({self.num_classes})")
        
        # Validate label range
        if labels.max() >= self.num_classes or labels.min() < 0:
            raise ValueError(f"Labels contain values outside valid range [0, {self.num_classes-1}]. "
                           f"Found min: {labels.min()}, max: {labels.max()}")

        loss = F.cross_entropy(logits, labels)
        predicted_labels = torch.argmax(logits, dim=1)
        return loss, labels, predicted_labels

def train_model_lightning(
    lightning_model: LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    group: str,
    project_name: str,
    logger: Logger | None = None,
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
            mode="max",
            monitor="val_acc",
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

    # trainer.fit(
    #     model=lightning_model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=val_loader,
    # )

    return trainer
