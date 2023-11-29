#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import os
import pickle
import numpy as np
import data_utils as dus
from elm import GenELMClassifier
from random_layer import MLPRandomLayer


def pred_save(model, col_clip, in_file, out_file):
    data = np.loadtxt(in_file, dtype=np.float64, delimiter=',', unpack=False)
    X = data[:, 1:]
    X = np.delete(X, col_clip, 1)

    # normalization
    X = dus.normalize(X)
    y = model.predict(X)

    with open(out_file, 'w') as fo:
        fo.write("id,label\n")
        for r in range(len(y)):
            fo.write("{},{}\n".format(r + 1, np.round(y[r], 2)))

        fo.close()


ps_thresh = 0.01  # pearson rate & spearman rate threshold
ol_percent = 0.1  # outlier percentage
act_func = 'sigmoid'
hn = 8000
save_model = False
load_model = False

data = np.loadtxt('data/dataTrain_test.csv', dtype=np.float64, delimiter=',', unpack=False)

X = data[:, 1:-1]
y = data[:, -1]

# normalization
X = dus.normalize(X)

idx_clip = dus.clip_list(X, ps_thresh)
X_clip = np.delete(X, idx_clip, 1)
idx_outliers = dus.idx_outlier(X_clip[:50000, :], ol_percent)
X_dense = np.delete(X_clip[:50000, :], idx_outliers, 0)
y_dense = np.delete(y[:50000], idx_outliers, 0)

model_file = 'data/model_{}.sav'.format(hn)

s_time = time.time()
if load_model:
    if not os.path.exists(model_file):
        print('Error! Model file {} not found.'.format(model_file))
        exit(-1)
    with open(model_file, 'rb') as fi:
        clf = pickle.load(fi)
        print('Model loaded from {}.'.format(model_file))
else:
    sig_rl = MLPRandomLayer(n_hidden=hn, activation_func=act_func)
    clf = GenELMClassifier(hidden_layer=sig_rl)
    clf.fit(X_dense, y_dense)

score = np.round(clf.score(X_dense, y_dense)*100, 2)
c_time = np.round(time.time() - s_time, 2)
print("func: {}, hn: {}, score: {}, time: {}".format('sigmoid', hn, score, c_time))

if (not load_model) and save_model:
    with open(model_file, 'wb') as fo:
        pickle.dump(clf, fo)
    print('Saved model into {}...'.format(model_file))

pred_save(clf, idx_clip, 'data/dataA_test.csv', 'data/predictA_{}.csv'.format(hn))
pred_save(clf, idx_clip, 'data/dataB_test.csv', 'data/predictB_{}.csv'.format(hn))
