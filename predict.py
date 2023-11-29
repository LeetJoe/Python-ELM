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
ol_percent = 0.1  # outlier percentage: we predict there were 10% noise data randomly mixed in the train data
act_func = 'sigmoid'
auc_train = 50000
hn = 8000  # 100000 will be better
save_model = False
load_model = False

data = np.loadtxt('data/dataTrain_test.csv', dtype=np.float64, delimiter=',', unpack=False)

X = data[:, 1:-1]
y = data[:, -1]

# normalization
X = dus.normalize(X)

idx_clip = dus.clip_list(X, y, ps_thresh)
X_clip = np.delete(X, idx_clip, 1)
idx_outliers = dus.idx_outlier(X_clip[:auc_train, :], idx_clip, ol_percent)
X_dense = np.delete(X_clip[:auc_train, :], idx_outliers, 0)
y_dense = np.delete(y[:auc_train], idx_outliers, 0)

params = {
    'ps': ps_thresh,
    'ol': ol_percent,
    'f': act_func,
    'hn': hn,
    'clip': idx_clip,
    'idol': idx_outliers,
    'auct': auc_train
}
model_file = 'data/model_{}.sav'.format(hn)

s_time = time.time()
if load_model:
    if not os.path.exists(model_file):
        print('Error! Model file {} not found.'.format(model_file))
        exit(-1)
    with open(model_file, 'rb') as fi:
        model_loaded = pickle.load(fi)
        clf = model_loaded['model']
        print('Model loaded from {}.'.format(model_file))
else:
    sig_rl = MLPRandomLayer(n_hidden=hn, activation_func=act_func)
    clf = GenELMClassifier(hidden_layer=sig_rl)
    print('Start training with {} hidden nodes...'.format(hn))
    clf.fit(X_dense, y_dense)

score = np.round(clf.score(X_dense, y_dense)*100, 2)
c_time = np.round(time.time() - s_time, 2)
print("func: {}, hn: {}, score: {}, time: {}".format('sigmoid', hn, score, c_time))

if (not load_model) and save_model:
    with open(model_file, 'wb') as fo:
        pickle.dump({'model': clf, 'params': params}, fo)
    print('Saved model into {}...'.format(model_file))

print('Predicting data A...')
result_file_A = 'data/predictA_{}.csv'.format(hn)
pred_save(clf, idx_clip, 'data/dataA_test.csv', result_file_A)
print('Predict result of data A saved in {}.'.format(result_file_A))

print('Predicting data B...')
result_file_B = 'data/predictB_{}.csv'.format(hn)
pred_save(clf, idx_clip, 'data/dataB_test.csv', 'data/predictB_{}.csv'.format(hn))
print('Predict result of data B saved in {}.'.format(result_file_A))
