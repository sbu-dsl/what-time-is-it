import csv
import random
import time
import datetime
import re
import pickle
import configparser
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)


def parse_labeled_examples():
    config = configparser.ConfigParser()
    config.read("../config.ini")
    time_csv_path = config["Paths"]["time_csv"]
    labeled_examples = []
    with open(time_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            guten_id = row['guten_id']
            if guten_id:
                continue
            hour = int(row['hour_reference'])
            phrase = row['time_phrase']
            window = row['tok_context'].split()
            is_unlabeled = row['is_ambiguous']
            time_pos_start = int(row['time_pos_start'])
            time_pos_end = int(row['time_pos_end'])
            new_window = window[:time_pos_start] + \
                ['[unused0]'] + window[time_pos_end:]
            window = ' '.join(new_window)
            if is_unlabeled == "False":
                labeled_examples.append((window, phrase, hour))
    return labeled_examples


def parse_imputed_examples():
    config = configparser.ConfigParser()
    config.read("../config.ini")
    time_csv_path = config["Paths"]["time_imputed_csv"]
    labeled_examples = []
    with open(time_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            guten_id = row['guten_id']
            if guten_id:
                continue
            hour = int(row['hour_reference'])
            phrase = row['time_phrase']
            window = row['tok_context'].split()
            time_pos_start = int(row['time_pos_start'])
            time_pos_end = int(row['time_pos_end'])
            new_window = window[:time_pos_start] + \
                ['[unused0]'] + window[time_pos_end:]
            window = ' '.join(new_window)
            labeled_examples.append((window, phrase, hour))
    return labeled_examples


def create_full_time_set(labeled_examples, up_bound=20000):
    full_time_set = defaultdict(list)
    for sent, phrase, time_int in labeled_examples:
        full_time_set[time_int].append(sent)

    for hour in range(24):
        np.random.shuffle(full_time_set[hour])
        full_time_set[hour] = full_time_set[hour][:up_bound]

    return full_time_set


def split_full_time_set(full_time_set, test_size=0.3, val_size=0.1):
    time_set = []
    for hour in range(24):
        for sent in full_time_set[hour]:
            time_set.append((sent, hour))
    full_train_time_set, test_time_set = train_test_split(
        time_set, test_size=test_size, random_state=seed_val)
    train_time_set, val_time_set = train_test_split(
        full_train_time_set, test_size=val_size, random_state=seed_val)
    return train_time_set, val_time_set, test_time_set


def evaluate_error(Y, Ypred):
    hour_error = defaultdict(int)
    hour_count = defaultdict(int)
    for i in range(len(Y)):
        minv = min(Y[i], Ypred[i])
        maxv = max(Y[i], Ypred[i])
        dist = min(maxv-minv, minv+24-maxv)
        hour_error[Y[i]] += dist
        hour_count[Y[i]] += 1
    for i in range(24):
        hour_error[i] /= hour_count[i]
    return hour_error


replace_puncts = {'`': "'", '′': "'", '“': '"', '”': '"', '‘': "'"}

strip_chars = [',', '.', '"', ':', ')', '(', '-', '|', ';', "'", '[', ']', '>', '=', '+', '\\', '•',  '~', '@',
               '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
               '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
               '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
               '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

puncts = ['!', '?', '$', '&', '/', '%', '#', '*', '£']


def clean_str(x):
    x = str(x)

    x = x.lower()

    x = re.sub(
        r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "url", x)

    for k, v in replace_puncts.items():
        x = x.replace(k, f' {v} ')

    for punct in strip_chars:
        x = x.replace(punct, ' ')

    for punct in puncts:
        x = x.replace(punct, f' {punct} ')

    x = x.replace(" '", " ")
    x = x.replace("' ", " ")

    return x


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# takes in two lists of lists


def evaluate_model_statistics(Y_actual, Y_predict):
    zero = 0
    one = 1
    accuracies = []
    am_f1_scores = []
    pm_f1_scores = []
    total_correct = 0
    # treat AM as target class
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0
    total = 0
    for i in range(12):
        accuracies.append(np.mean(Y_predict[i] == Y_actual[i]))
        am_f1, pm_f1 = metrics.f1_score(
            Y_actual[i], Y_predict[i], average=None)
        am_f1_scores.append(am_f1)
        pm_f1_scores.append(pm_f1)

        total_TP += np.sum([x == y for x,
                            y in zip(Y_predict[i], Y_actual[i]) if x == zero])
        total_TN += np.sum([x == y for x,
                            y in zip(Y_predict[i], Y_actual[i]) if x == one])
        total_FP += np.sum([x != y for x,
                            y in zip(Y_predict[i], Y_actual[i]) if x == zero])
        total_FN += np.sum([x != y for x,
                            y in zip(Y_predict[i], Y_actual[i]) if x == one])

        total_correct += np.sum(Y_predict[i] == Y_actual[i])
        total += len(Y_predict[i])
    print(total_TP, total_TN, total_FP, total_FN)
    am_precision = total_TP / (total_TP + total_FP)
    am_recall = total_TP / (total_TP + total_FN)
    pm_precision = total_TN / (total_TN + total_FN)
    pm_recall = total_TN / (total_TN + total_FP)
    micro_accuracy = total_correct / total
    macro_accuracy = np.mean(accuracies)
    micro_am_f1_score = 2*(am_precision * am_recall) / \
        (am_precision + am_recall)
    macro_am_f1_score = np.mean(am_f1_scores)
    micro_pm_f1_score = 2*(pm_precision * pm_recall) / \
        (pm_precision + pm_recall)
    macro_pm_f1_score = np.mean(pm_f1_scores)
    print("Macro Accuracy:", np.mean(accuracies))
    print("Micro Accuracy:", total_correct / total)
    print()
    print("Macro AM F1:", np.mean(am_f1_scores))
    print("Micro AM F1:", micro_am_f1_score)
    print()
    print("Macro PM F1:", np.mean(pm_f1_scores))
    print("Micro PM F1:", micro_pm_f1_score)
    return (micro_accuracy,
            macro_accuracy,
            micro_am_f1_score,
            macro_am_f1_score,
            micro_pm_f1_score,
            macro_pm_f1_score)
