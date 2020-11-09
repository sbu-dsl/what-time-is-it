from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from train_am_pm_helper import *


def convert_am_pm_set_to_nb(am_pm_set):
    x = [[] for _ in range(12)]
    y = [[] for _ in range(12)]
    for hour in am_pm_set:
        for sent, lab in am_pm_set[hour]:
            x[hour].append(sent)
            y[hour].append(lab)
    return x, y


def train_nb(train_am_pm_set):
    nb_xtrain, nb_ytrain = convert_am_pm_set_to_nb(train_am_pm_set)
    nb_models = []
    for i in range(12):
        text_nb_clf = Pipeline([
            ('vect', CountVectorizer(binary=True)),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        nb_models.append(text_nb_clf.fit(nb_xtrain[i], nb_ytrain[i]))
    return nb_models


def test_nb(nb_models, test_am_pm_set):
    nb_xtest, nb_ytest = convert_am_pm_set_to_nb(test_am_pm_set)
    nby_pred = []
    for hour in range(12):
        nby_pred.append(nb_models[hour].predict(nb_xtest[hour]))
    return nb_ytest, nby_pred


labeled_examples, unlabeled_examples = parse_labeled_unlabeled_examples()
am_pm_set = parse_am_pm_set(labeled_examples)
train_am_pm_set, test_am_pm_set = train_test_split_am_pm_set(am_pm_set)

merged_am_pm_set = construct_merged_am_pm_set(train_am_pm_set, 1)
for i in range(12):
    print(i, len(train_am_pm_set[i]), len(merged_am_pm_set[i]))

models = train_nb(merged_am_pm_set)
actual, pred = test_nb(models, test_am_pm_set)
evaluate_model_statistics(actual, pred)
