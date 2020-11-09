import os
import csv
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import pickle
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn import metrics
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
from tqdm import tqdm
from train_24_hour_helper import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', never_split=['[unused0]'])
tokenizer.add_tokens('[unused0]')

print("Parsing CSV file")
labeled_examples = parse_labeled_examples()

print("Parsing imputed CSV file")
labeled_examples.extend(parse_imputed_examples())

full_time_set = create_full_time_set(labeled_examples)
train_time_set, val_time_set, test_time_set = split_full_time_set(
    full_time_set)

# Function to calculate the accuracy of our predictions vs labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# returns input_ids, attention_masks, and labels


def tokenize_all_examples(time_set):
    sentences, labels = zip(*time_set)

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,                      # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=384,           # Pad & truncate all sentences.
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,   # Construct attn. masks.
            return_tensors='pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    return input_ids, attention_masks, labels


def test_24_hour_model(all_examples):
    input_ids, attention_masks, labels = tokenize_all_examples(all_examples)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 32

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        dataset,  # The validation samples.
        sampler=SequentialSampler(dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        # Use the 12-layer BERT model, with an uncased vocab.
        "./bert_models/full_24_hour_model/",
        num_labels=24,  # 24 class model for us
        # Whether the model returns attentions weights.
        output_attentions=False,
        # Whether the model returns all hidden-states.
        output_hidden_states=False,
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    return predictions, true_labels


pred, true = test_24_hour_model(test_time_set)
pred = np.concatenate(pred)

pred_labels = np.argmax(pred, axis=1)
true_labels = np.concatenate(true)
c = evaluate_error(true_labels, pred_labels)
for hour in range(24):
    print("{} {:.2f}".format(hour, c[hour]))
print(np.mean(list(c.values())))
