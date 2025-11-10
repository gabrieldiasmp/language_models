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

def tokenize_ner_models(examples, model_id: str = "distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

class HFTextDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.dataset.num_rows