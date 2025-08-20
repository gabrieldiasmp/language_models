import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def tokenize_hugging_face(dataset, text_column_name: str = "sentence", model_str: str = "distilbert-base-uncased"):

    tokenizer = AutoTokenizer.from_pretrained(
        model_str, model_max_length=512
    )

    dataset_tokenized = dataset.map(
        lambda batch: tokenizer(batch[text_column_name], truncation=True, padding=True),
        batched=True,
        batch_size=None,
    )

    return dataset_tokenized

    
class HFTextDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.dataset.num_rows