import os
import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from lxml import etree
from multiprocessing import Pool
from itertools import product
from predict_book_len_times_helper import *


def gen_optimized_times(prob_arr, break_size=40):
    num_paras = len(prob_arr)
    naive_sizes = []
    naive_times = []
    i = 0
    while i < num_paras:
        probs = prob_arr[i:i+break_size]
        hour = np.argmax(np.sum(probs, axis=0))
        naive_sizes.append(len(probs))
        naive_times.append(hour)
        i += break_size
    dpx = list(range(num_paras))
    dpy = []
    for i in range(len(naive_times)):
        dpy.extend([naive_times[i]] * naive_sizes[i])
    return dpx, dpy
