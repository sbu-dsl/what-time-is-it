import os
import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from lxml import etree
from multiprocessing import Pool
from itertools import product


def gen_optimized_dp_times(prob_arr, num_breaks):
    num_paras = len(prob_arr)
    prefix_sum = np.cumsum(prob_arr, axis=0)
    dp = [[(-1, -1, -1) for _ in range(num_breaks+1)]
          for _ in range(num_paras)]
    for i in range(num_paras):
        top_time = np.argmax(prefix_sum[i])
        top_score = prefix_sum[i][top_time]
        dp[i][0] = (top_score, top_time, -1)
    for k in range(1, num_breaks+1):
        for i in range(k, num_paras):
            overall_top_score = -1
            overall_top_time = -1
            overall_top_prev = -1
            for j in range(k, i+1):
                curr_scores = prefix_sum[i] - prefix_sum[j-1]
                top_time = np.argmax(curr_scores)
                top_score = curr_scores[top_time]
                full_score = dp[j-1][k-1][0] + top_score
                if full_score > overall_top_score:
                    overall_top_score = full_score
                    overall_top_time = top_time
                    overall_top_prev = j-1
            dp[i][k] = (overall_top_score, overall_top_time, overall_top_prev)
    break_locs = []
    para_num = num_paras - 1
    for k in range(num_breaks, -1, -1):
        score, time, prev = dp[para_num][k]
        break_locs.append((para_num, time))
        para_num = prev
    break_locs.append((-1, -1))
    all_times = []
    curr_para_num, curr_time = break_locs[0]
    for para_num, time in break_locs[1:]:
        all_times.extend((curr_para_num - para_num) * [curr_time])
        curr_para_num = para_num
        curr_time = time
    all_times.reverse()
    return list(range(num_paras)), all_times
