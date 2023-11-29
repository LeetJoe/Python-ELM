#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import numpy as np
from elm import GenELMClassifier
from random_layer import MLPRandomLayer


data = np.loadtxt('data/dataTrain_test.csv',dtype=np.float64,delimiter=',',unpack=False)

X = data[:, 1:-1]
y = data[:, -1]
nrow, ncol = X.shape

# standardization
for i in range(0, ncol):
    cmin = np.min(X[:, i])
    cmax = np.max(X[:, i])
    X[:, i] = (X[:, i] - cmin)/(cmax - cmin)

''' no prompt
X_mean = np.mean(X, axis=0)
X_ori = X - X_mean
X_ori_norm = np.linalg.norm(X_ori, axis=1)
X = np.delete(X, np.argsort(X_ori_norm)[-10000:], 0)
y = np.delete(y, np.argsort(X_ori_norm)[-10000:])
'''

data_group = {'original': (X, y)}

idx_clip = [10, 11, 12, 15, 16, 19, 21, 27, 38, 40, 41]  # 0.5
# idx_clip = [8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 24, 26, 27, 28, 30, 31, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45]  # 5
X_clip = np.delete(X, idx_clip, 1)
data_group['clip'] = (X_clip, y)

data_group['filter'] = (X[:50000, :], y[:50000])
data_group['clip&filter'] = (X_clip[:50000, :], y[:50000])

result = [['func', 'h-num', 'data', 'score', 'time']]

for hn in [500, 1000, 2000, 4000, 8000]:
    sig_rl = MLPRandomLayer(n_hidden=hn, activation_func='sigmoid')
    clf = GenELMClassifier(hidden_layer=sig_rl)
    for dk in data_group:
        cur_X, cur_y = data_group[dk]
        s_time = time.time()
        clf.fit(cur_X, cur_y)
        score = np.round(clf.score(cur_X, cur_y)*100, 2)
        c_time = np.round(time.time() - s_time, 2)
        result.append(['sigmoid', hn, dk, score, c_time])
        print("func: {}, hn: {}, data: {}, score: {}, time: {}".format('sigmoid', hn, dk, score, c_time))

with open('result.csv', 'w') as fo:
    for r in result:
        fo.write("{},{},{},{},{}\n".format(r[0], r[1], r[2], str(r[3])+'%', r[4]))

    fo.close()
