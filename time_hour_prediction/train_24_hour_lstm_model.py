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
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.initializers import Constant
from tqdm import tqdm
from train_24_hour_helper import *

print("Parsing CSV file")
labeled_examples = parse_labeled_examples()

print("Parsing imputed CSV file")
labeled_examples.extend(parse_imputed_examples())

full_time_set = create_full_time_set(labeled_examples)
train_time_set, val_time_set, test_time_set = split_full_time_set(
    full_time_set)


def convert_time_set_to_lstm(time_set):
    x = []
    y = []
    for sent, lab in time_set:
        x.append(clean_str(sent))
        lab_hot = [0] * 24
        lab_hot[lab] = 1
        y.append(lab_hot)
    X = pad_sequences(tokenizer.texts_to_sequences(x), sequence_length)
    return X, np.array(y)


max_features = 20000
sequence_length = 300
tokenizer = Tokenizer(num_words=max_features, split=' ',
                      oov_token='<unw>', filters=' ')
texts = []
for sent, lab in train_time_set:
    texts.append(clean_str(sent))
tokenizer.fit_on_texts(texts)

config = configparser.ConfigParser()
config.read("../config.ini")
glove_input_file = config["Paths"]["glove_input_file"]
word2vec_output_file = glove_input_file + '.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)
glove_model = KeyedVectors.load_word2vec_format(
    word2vec_output_file, binary=False)

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index)) + 1
print(num_words)
embedding_dim = 300

embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i > max_features:
        continue
    if word in glove_model.vocab:
        embedding_vector = glove_model[word]
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.randn(embedding_dim)

model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=sequence_length,
                    trainable=True
                    ))
model.add(LSTM(300, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(24, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['acc'])
x_train, y_train = convert_time_set_to_lstm(train_time_set)
x_val, y_val = convert_time_set_to_lstm(val_time_set)
x_test, y_test = convert_time_set_to_lstm(test_time_set)
model.fit(x_train, y_train, epochs=5, verbose=1,
          validation_data=(x_val, y_val))
model.save("lstm_models/full_24_hour_model.hd5")

y_pred = model.predict_classes(np.array(x_test)).flatten()
y_testt = [np.argmax(x) for x in y_test]
errors = evaluate_error(y_testt, y_pred)
for hour in range(24):
    print("{} {:.2f}".format(hour, errors[hour]))
print(np.mean(list(errors.values())))
