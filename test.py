#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import numpy as np
import data_utils as dus
from elm import GenELMClassifier
from random_layer import MLPRandomLayer

ps_thresh = 0.01  # pearson rate & spearman rate threshold
ol_percent = 0.1  # outlier percentage
act_func = 'sigmoid'

data = np.loadtxt('data/dataTrain_test.csv',dtype=np.float64,delimiter=',',unpack=False)

X = data[:, 1:-1]
y = data[:, -1]

# normalization
X = dus.normalize(X)

data_group = {'original': (X, y)}

# clip
idx_clip = dus.clip_list(X, ps_thresh)
X_clip = np.delete(X, idx_clip, 1)
data_group['clip'] = (X_clip, y)

# todo AUC filter: filter_1
data_group['filter'] = (X[:50000, :], y[:50000])
data_group['clip&filter'] = (X_clip[:50000, :], y[:50000])

# outlier filter: filter_2
idx_outliers = dus.idx_outlier(X_clip[:50000, :], ol_percent)
X_dense = np.delete(X_clip[:50000, :], idx_outliers, 0)
y_dense = np.delete(y[:50000], idx_outliers, 0)
data_group['dense'] = (X_dense, y_dense)

result = [['func', 'h-num', 'data', 'score', 'time']]

hn_list = [1000, 2000, 4000, 8000, 10000]

for hn in hn_list:
    sig_rl = MLPRandomLayer(n_hidden=hn, activation_func=act_func)
    for dk in data_group:
        clf = GenELMClassifier(hidden_layer=sig_rl)
        cur_X, cur_y = data_group[dk]
        s_time = time.time()
        clf.fit(cur_X, cur_y)
        score = np.round(clf.score(cur_X, cur_y)*100, 2)
        c_time = np.round(time.time() - s_time, 2)
        result.append(['sigmoid', hn, dk, score, c_time])
        print("func: {}, hn: {}, data: {}, score: {}, time: {}".format('sigmoid', hn, dk, score, c_time))

with open('data/result.csv', 'w') as fo:
    for r in result:
        fo.write("{},{},{},{},{}\n".format(r[0], r[1], r[2], str(r[3])+'%', r[4]))

    fo.close()
