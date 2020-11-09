import os
import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from lxml import etree
from multiprocessing import Pool
from itertools import product


def normalize(x):
    return x / np.sum(x)


def minmax_norm(x):
    pos_x = x - np.min(x)
    return pos_x / np.sum(pos_x)


def tok_to_prob(tok):
    standard_ratio = 1
    prob = np.zeros(24)
    if tok.isnumeric():
        time = int(tok)
        while time >= 24:
            time -= 24
        prob[time] = standard_ratio
        if time < 12:
            prob[time+12] = standard_ratio  # could be pm
    elif tok == "MO":
        for i in range(7, 12):
            prob[i] = standard_ratio
    elif tok == "AF":
        for i in range(12, 17):
            prob[i] = standard_ratio
    elif tok == "EV":
        for i in range(17, 21):
            prob[i] = standard_ratio
    elif tok == "NI":
        for i in range(0, 3):
            prob[i] = standard_ratio
        for i in range(3, 7):
            prob[i] = standard_ratio
        for i in range(21, 24):
            prob[i] = standard_ratio
    else:
        raise ValueError("token {} not recognized".format(tok))
    return normalize(prob)


def gen_model_probs_from_raw(paras, model_scores, window=1, threshold=-1):
    model_probs = []
    for j, scores in enumerate(model_scores):
        norm_probs = minmax_norm(np.array(scores))
        if threshold >= 0:
            if len(paras[j].split()) < threshold:
                norm_probs = normalize(np.ones(24))
        smoothed_probs = np.zeros(24)
        for i in range(24):
            total_prob = norm_probs[i]
            for j in range(1, window+1):
                lidx = (i-j) % 24
                ridx = (i+j) % 24
                total_prob += norm_probs[lidx] + norm_probs[ridx]
            smoothed_probs[i] = total_prob / (2*window + 1)
        model_probs.append(smoothed_probs)
    return model_probs


def get_raw_model_scores(csv_file):
    paras = []
    raw_model_scores = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            paras.append(row['para'])
            raw_model_scores.append([float(x) for x in row['scores'].split()])
    return raw_model_scores, paras


def gen_model_probs(csv_file, threshold=-1):
    raw_scores, paras = get_raw_model_scores(csv_file)
    return gen_model_probs_from_raw(paras, raw_scores, threshold)


def is_valid_key(key):
    return key.isnumeric() or key in ["MO", "AF", "EV", "NI"]


def gen_local_probs(annot_xml_path, num_paras):
    with open(annot_xml_path, "rb") as f:
        root = etree.fromstring(f.read())

    para_to_time = {}
    for entity in root.iter('entity'):
        if entity.get('ner') == "TIME":
            timex_val = entity.get('timex_value')
            if timex_val:
                t_idx = timex_val.find('T')
                if t_idx != -1:
                    key = timex_val[t_idx+1:t_idx+3]
                    if is_valid_key(key):
                        para_num = int(
                            entity.getparent().getparent().get('num'))
                        para_to_time[para_num] = key

    local_probs = np.array([np.ones(24) / 24 for _ in range(num_paras)])
    known_loc = []
    for num in para_to_time:
        local_probs[num] = tok_to_prob(para_to_time[num])
        known_loc.append(num)

    idx = 0
    known_idx = 0
    while known_idx < len(known_loc):
        lidx = idx
        ridx = known_loc[known_idx]
        if ridx != lidx:
            diffv = (local_probs[ridx] - local_probs[lidx]) / (ridx-lidx)
            for j in range(lidx+1, ridx):
                local_probs[j] = local_probs[j-1] + diffv
        idx = known_loc[known_idx]
        known_idx += 1
    return local_probs


def gen_prob_arr(model_probs, local_probs, w1, w2):
    prob_arr = []
    for i in range(len(model_probs)):
        prob_arr.append((w1*model_probs[i] + w2*local_probs[i]))
    return prob_arr


def gen_model_times(annot_xml_path, time_bert_pred_csv_path, break_size=40, window_size=250, mix=0.5):
    model_probs = gen_model_probs(time_bert_pred_csv_path)
    local_probs = gen_local_probs(annot_xml_path, len(model_probs))
    prob_arr = gen_prob_arr(local_probs, model_probs, mix, 1-mix)
    dpx, dpy = gen_optimized_times(prob_arr)
    return dpy


def annotate_time(annot_xml_path, time_csv_path, time_annot_output_path):
    para_to_time = gen_model_times(annot_xml_path, time_csv_path)
    parser = etree.XMLParser(huge_tree=True)
    tree = etree.parse(str(annot_xml_path), parser=parser)
    book = tree.getroot()
    for para in book.iter('p'):
        para_num = int(para.get('num'))
        para.set("hour", str(para_to_time[para_num]))
    with open(time_annot_output_path, "wb") as f:
        f.write(etree.tostring(book, pretty_print=True))
    print(time_annot_output_path)
