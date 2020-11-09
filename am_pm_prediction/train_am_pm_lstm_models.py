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
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.initializers import Constant
from train_am_pm_helper import *

labeled_examples, unlabeled_examples = parse_labeled_unlabeled_examples()
am_pm_set = parse_am_pm_set(labeled_examples)
train_am_pm_set, test_am_pm_set = train_test_split_am_pm_set(am_pm_set)
merged_am_pm_set = construct_merged_am_pm_set(train_am_pm_set, 1)

config = configparser.ConfigParser()
config.read("../config.ini")
glove_input_file = config["Paths"]["glove_input_file"]
word2vec_output_file = glove_input_file + '.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

glove_model = KeyedVectors.load_word2vec_format(
    word2vec_output_file, binary=False)

max_features = 20000
sequence_length = 300
tokenizer = Tokenizer(num_words=max_features, split=' ',
                      oov_token='<unw>', filters=' ')
texts = []
for hour in range(12):
    for sent, lab in am_pm_set[hour]:
        texts.append(clean_str(sent))
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index)) + 1
embedding_dim = 100

embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i > max_features:
        continue
    if word in glove_model.vocab:
        embedding_vector = glove_model[word]
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.randn(embedding_dim)


def convert_am_pm_set_to_lstm(am_pm_set):
    x = [[] for _ in range(12)]
    y = [[] for _ in range(12)]
    X = []
    Y = []
    for hour in range(12):
        for sent, lab in am_pm_set[hour]:
            x[hour].append(clean_str(sent))
            y[hour].append(int(lab))
        X.append(pad_sequences(
            tokenizer.texts_to_sequences(x[hour]), sequence_length))
        Y.append(y[hour])
    return np.array(X), np.array(Y)


train_am_pm_set, val_am_pm_set = split_train_and_val(
    train_am_pm_set, ratio=0.1)
merged_train_am_pm_set = construct_merged_am_pm_set(train_am_pm_set, 1)
x_trains, y_trains = convert_am_pm_set_to_lstm(merged_train_am_pm_set)
x_vals, y_vals = convert_am_pm_set_to_lstm(val_am_pm_set)
x_tests, y_tests = convert_am_pm_set_to_lstm(test_am_pm_set)

models = []
for hour in range(12):
    print("Started", hour)
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=sequence_length,
                        trainable=True
                        ))
    model.add(LSTM(300, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['acc'])
    x_train = np.array(x_trains[hour])
    y_train = np.array(y_trains[hour])
    x_val = np.array(x_vals[hour])
    y_val = np.array(y_vals[hour])
    x_test = np.array(x_tests[hour])
    y_test = np.array(y_tests[hour])
    model.fit(x_train, y_train, epochs=5, verbose=1,
              validation_data=(x_val, y_val))
    model.save("lstm_models/{}_am_pm_lstm_model.hd5".format(hour))
    models.append(model)

y_preds = []
for hour in range(12):
    y_pred = models[hour].predict_classes(np.array(x_tests[hour]))
    y_preds.append(y_pred.flatten())

evaluate_model_statistics(y_tests, y_preds)
