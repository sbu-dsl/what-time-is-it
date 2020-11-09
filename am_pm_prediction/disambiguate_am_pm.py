import os
import csv
import configparser
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
from train_am_pm_helper import format_time


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', never_split=['[unused0]'])
tokenizer.add_tokens('[unused0]')

config = configparser.ConfigParser()
config.read("../config.ini")
time_csv_path = config["Paths"]["time_csv"]
time_imputed_csv_path = config["Paths"]["time_imputed_csv"]

print("Parsing CSV file")
unlabeled_examples = []
with open(time_csv_path, "r") as f:
    reader = csv.DictReader(f)
    for num, row in enumerate(reader):
        hour = int(row['hour_reference'])
        phrase = row['time_phrase']
        window = row['tok_context'].split()
        is_unlabeled = row['is_ambiguous']
        time_pos_start = int(row['time_pos_start'])
        time_pos_end = int(row['time_pos_end'])
        new_window = window[:time_pos_start] + \
            ['[unused0]'] + window[time_pos_end:]
        window = ' '.join(new_window)
        if is_unlabeled == "True":
            unlabeled_examples.append((window, hour, row))


hourly_set = defaultdict(list)
hourly_row_info = defaultdict(list)
for sent, time_int, full_info in unlabeled_examples:
    if time_int == 0:
        time_int = 1
    if time_int == 12:
        time_int = 0
    hourly_set[time_int].append(sent)
    hourly_row_info[time_int].append(full_info)

am_pm_set = defaultdict(list)
for time_int in hourly_set:
    for sent in hourly_set[time_int]:
        if time_int < 12:
            am_pm_set[time_int].append((sent, 0))
        else:
            am_pm_set[time_int-12].append((sent, 1))

print("AM vs PM size")
total_size = 0
for hour in range(12):
    total_size += len(am_pm_set[hour])
    print(hour, len(am_pm_set[hour]))
print(total_size)


def unlabeled_to_labeled_am_pm_set(unlabeled_am_pm_set):
    labeled_set = []
    for hour in range(12):
        print(hour, " start")
        sentences, labels = zip(*unlabeled_am_pm_set[hour])
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
            # Pull out batches sequentially.
            sampler=SequentialSampler(dataset),
            batch_size=batch_size  # Evaluate with this batch size.
        )

        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        model = BertForSequenceClassification.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            './bert_models/am_pm_{}_model_save/'.format(hour),
            num_labels=2,  # 2 class model for us
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
        pred_labels = np.argmax(np.concatenate(predictions), axis=1)
        true_labels = np.concatenate(true_labels)
        labeled_set.append((sentences, pred_labels, true_labels, predictions))
    return labeled_set


result = unlabeled_to_labeled_am_pm_set(am_pm_set)
predY = []
for hour in range(12):
    _, pred, _, _ = result[hour]
    predY.append(pred)

with open(time_imputed_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['guten_id',
                  'hathi_id',
                  'hour_reference',
                  'time_phrase',
                  'time_pos_start',
                  'time_pos_end',
                  'tok_context']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for hour in range(12):
        meta_infos = hourly_row_info[hour]
        pred_labs = predY[hour]
        if len(meta_infos) != len(pred_labs):
            print("Error: mismatch in prediction length, hour", hour)
            break
        for i in range(len(meta_infos)):
            row_data = meta_infos[i]
            del row_data['is_ambiguous']
            am_pm = pred_labs[i]
            curr_hour = int(row_data['hour_reference'])
            if am_pm == 0:
                if curr_hour == 0:
                    row_data['hour_reference'] = 1
                elif curr_hour == 12:
                    row_data['hour_reference'] = 0
            elif am_pm == 1:
                if curr_hour == 0:
                    curr_hour = 1
                elif curr_hour == 12:
                    curr_hour = 0
                row_data['hour_reference'] = curr_hour + 12
            writer.writerow(row_data)
