import csv
import re
import random
import datetime
import configparser
from collections import defaultdict
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)


def parse_labeled_unlabeled_examples():
    config = configparser.ConfigParser()
    config.read("../config.ini")
    time_csv_path = config["Paths"]["time_csv"]
    labeled_examples = []
    unlabeled_examples = []
    with open(time_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
                unlabeled_examples.append((window, phrase, hour))
            else:
                labeled_examples.append((window, phrase, hour))
    return labeled_examples, unlabeled_examples


def parse_am_pm_set(labeled_examples, up_bound=30000):
    # am_pm_set is a dictionary of lists
    # key: int between 0 to 11 representing hours
    # lists of tuples (sent, 0/1) depending on AM (0) or PM (1)
    am_pm_set = defaultdict(list)
    for example in labeled_examples:
        sent, phrase, time_int = example
        if time_int < 12:
            am_pm_set[time_int].append((sent, 0))
        else:
            am_pm_set[time_int - 12].append((sent, 1))

    # limit number of midnight/noon examples
    np.random.shuffle(am_pm_set[0])
    am_pm_set[0] = am_pm_set[0][:up_bound]

    return am_pm_set


def construct_merged_am_pm_set(am_pm_set, time_window):
    merged_am_pm_set = defaultdict(list)
    for i in range(12):
        for j in range(-time_window, time_window + 1):
            curr = (i + j) % 12
            if i + j < 0 or i + j >= 12:
                rev_set = [(x, 0) if y == 1 else (x, 1)
                           for x, y in am_pm_set[curr]]
                merged_am_pm_set[i].extend(rev_set)
            else:
                merged_am_pm_set[i].extend(am_pm_set[curr])
    return merged_am_pm_set


def train_test_split_am_pm_set(am_pm_set, test_size=0.3, random_state=1792):
    train_am_pm_set = {}
    test_am_pm_set = {}
    for hour in am_pm_set:
        hour_set = am_pm_set[hour]
        train_set, test_set = train_test_split(
            hour_set, test_size=test_size, random_state=random_state)
        train_am_pm_set[hour] = train_set
        test_am_pm_set[hour] = test_set

    return train_am_pm_set, test_am_pm_set


replace_puncts = {'`': "'", '′': "'", '“': '"', '”': '"', '‘': "'"}

strip_chars = [',', '.', '"', ':', ')', '(', '-', '|', ';', "'", '[', ']', '>',
               '=', '+', '\\', '•', '~', '@', '·', '_', '{', '}', '©', '^', '®',
               '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′',
               'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−',
               '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
               '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■',
               '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸',
               '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
               '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤',
               'ï', 'Ø', '¹', '≤', '‡', '√', ]

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


def split_train_and_val(am_pm_set, ratio=0.1):
    train_am_pm_set = {}
    val_am_pm_set = {}
    for hour in range(12):
        x_train, x_val = train_test_split(
            am_pm_set[hour], test_size=ratio, random_state=42)
        x_train = np.array(x_train)
        x_val = np.array(x_val)
        train_am_pm_set[hour] = x_train
        val_am_pm_set[hour] = x_val
    return train_am_pm_set, val_am_pm_set


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
    micro_am_f1_score = 2 * (am_precision * am_recall) / \
        (am_precision + am_recall)
    macro_am_f1_score = np.mean(am_f1_scores)
    micro_pm_f1_score = 2 * (pm_precision * pm_recall) / \
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
