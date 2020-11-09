import csv
import random
import pickle
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn import metrics
from sklearn.model_selection import train_test_split
from train_am_pm_helper import *


def train_baseline(train_am_pm_set):
    models = []
    for hour in range(12):
        num_zero = 0
        num_one = 0
        for sent, lab in train_am_pm_set[hour]:
            if lab == 0:
                num_zero += 1
            else:
                num_one += 1
        if num_zero >= num_one:
            models.append(0)
        else:
            models.append(1)
    return models


def test_baseline(models, test_am_pm_set):
    yactual = defaultdict(list)
    ypred = defaultdict(list)
    for hour in range(12):
        for sent, lab in test_am_pm_set[hour]:
            yactual[hour].append(lab)
            ypred[hour].append(models[hour])
        yactual[hour] = np.array(yactual[hour])
        ypred[hour] = np.array(ypred[hour])
    return yactual, ypred


labeled_examples, unlabeled_examples = parse_labeled_unlabeled_examples()
am_pm_set = parse_am_pm_set(labeled_examples)
train_am_pm_set, test_am_pm_set = train_test_split_am_pm_set(am_pm_set)
models = train_baseline(train_am_pm_set)
actual, pred = test_baseline(models, test_am_pm_set)
evaluate_model_statistics(actual, pred)
