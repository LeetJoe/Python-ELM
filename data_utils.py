#!/usr/bin/python
# -*- coding: utf-8 -*-

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


def clip_list(data, thresh=0.05):
    X = data[:, 1:-1]
    y = data[:, -1]

    nrow, ncol = X.shape

    score_ps = np.array([pearsonr(X[:, i], y) for i in range(ncol)])

    idx_no_ps = np.argwhere(np.abs(score_ps) < thresh)
    idx_no_ps = idx_no_ps.reshape(len(idx_no_ps))

    # spearman
    score_spr = np.array([stats.spearmanr(X[:, i], y)[0] for i in range(ncol)])

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


def idx_outlier(x, percent=0.1):
    x_mean = np.mean(x, axis=0)
    x -= x_mean
    x_norm = np.linalg.norm(x, axis=1)

    num = int(np.floor(len(x)*percent))

    return np.argsort(x_norm)[-num:]


