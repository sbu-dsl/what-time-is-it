import os
import csv
import time
import datetime
import random
import pickle
import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from train_24_hour_helper import *

print("Parsing CSV file")
labeled_examples = parse_labeled_examples()

print("Parsing imputed CSV file")
labeled_examples.extend(parse_imputed_examples())

full_time_set = create_full_time_set(labeled_examples)
train_time_set, val_time_set, test_time_set = split_full_time_set(
    full_time_set)


def convert_time_set_to_nb(time_set):
    return zip(*time_set)


train_x, train_y = convert_time_set_to_nb(train_time_set)
test_x, test_y = convert_time_set_to_nb(test_time_set)
text_nb_clf = Pipeline([
    ('vect', CountVectorizer(binary=True)),
    ('clf', MultinomialNB(alpha=1))
])
text_nb_clf.fit(train_x, train_y)
pred_y = text_nb_clf.predict(test_x)
errors = evaluate_error(test_y, pred_y)
for hour in range(24):
    print("{} {:.2f}".format(hour, errors[hour]))
print(np.mean(list(errors.values())))
