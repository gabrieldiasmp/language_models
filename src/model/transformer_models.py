import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForSequenceClassification


def get_huggingface_model(
    model_str: str, num_classes: int, train_last_layers_only: bool = False
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_str, num_labels=num_classes
    )

    if train_last_layers_only:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True

    return model
