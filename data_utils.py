#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IPython.core.pylabtools import figsize


def show_hist(data):
    figsize(100, 300)
    num_feature = len(data[0, :])
    for i in range(num_feature):
        plt.subplot(int(np.ceil(num_feature/4)), 4, i + 1)
        plt.hist(data[:, i], bins=len(np.unique(data[:, i])), log=True, label=str(i))
        plt.legend()

    plt.show()


def pearsonr(x, y):
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    r = np.cov(x, y, ddof=1)[0, 1] / (sx * sy)
    return r


def clip_list(x, y, thresh=0.05):
    score_ps = np.array([pearsonr(x[:, i], y) for i in range(len(x[0]))])

    idx_no_ps = np.argwhere(np.abs(score_ps) < thresh)
    idx_no_ps = idx_no_ps.reshape(len(idx_no_ps))

    # spearman
    score_spr = np.array([stats.spearmanr(x[:, i], y)[0] for i in range(len(x[0]))])

    idx_no_spr = np.argwhere(np.abs(score_spr) < thresh)
    idx_no_spr = idx_no_spr.reshape(len(idx_no_spr))

    return np.intersect1d(idx_no_ps, idx_no_spr)


# normalization
def normalize(x):
    for i in range(len(x[0])):
        cmin = np.min(x[:, i])
        cmax = np.max(x[:, i])
        x[:, i] = (x[:, i] - cmin)/(cmax - cmin)
    return x


def idx_outlier(x, idc, percent=0.1, auc_idx=34000):
    nl_data = np.loadtxt('data/dataNoLabel_test.csv', dtype=np.float64, delimiter=',', unpack=False)
    nl_data = np.delete(nl_data, idc, 1)
    x_mean = np.mean(normalize(nl_data[:auc_idx, 1:]), axis=0)
    lx = x - x_mean
    x_norm = np.linalg.norm(lx, axis=1)

    num = int(np.floor(len(lx)*percent))

    return np.argsort(x_norm)[-num:]


def auc_test(fmodel, data_file, test_step=1000, no_label=False):
    """
    auc of no label data changed after index 34000 from ~0.19 to ~0.25, so we choose 0~33999 rows of no label data;
    auc of train data changed after index 50000 from ~0.2 to ~0.47, so we drop data out of 50000 rows.
    """
    with open(fmodel, 'rb') as fi:
        model_loaded = pickle.load(fi)
        clf = model_loaded['model']
        params = model_loaded['params']
        print('Model loaded from {}.'.format(fmodel))

    data = np.loadtxt(data_file, dtype=np.float64, delimiter=',', unpack=False)

    # X = data[:, 1:-1]
    # y = data[:, -1]
    if no_label:
        X = data[:, 1:]
        y = data[:, 1]
    else:
        X = data[:, 1:-1]
        y = data[:, -1]

    X = normalize(X)
    X = np.delete(X, params['clip'], 1)

    nrows, ncols = X.shape

    i = 0
    while i < nrows:
        e = min(i + test_step, nrows)
        y_pred = clf.predict(X[i:e, :])
        auc = np.round(np.sum(y_pred) / min(test_step, e - i), 2)
        if no_label:
            print('{}~{}, auc: {}'.format(i, e, auc))
        else:
            y_scr = clf.score(X[i:e, :], y[i:e])
            print('{}~{}, score: {}, auc: {}'.format(i, e, np.round(y_scr, 4), auc))

        i = e


# auc_test('data/model_8000.sav', 'data/dataNoLabel_test.csv', no_label=True)
# auc_test('data/model_8000.sav', 'data/dataTrain_test.csv')
