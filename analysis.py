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
        # plt.subplot(50, np.ceil(i/10)+1, i%10+1)
        ax = plt.subplot(int(np.ceil(num_feature/4)), 4, i + 1)
        plt.hist(data[:, i], bins=len(np.unique(data[:, i])), log=True, label=str(i))
        plt.legend()

    plt.show()


def pearsonr(x, y):
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    r = np.cov(x, y, ddof=1)[0, 1] / (sx * sy)
    return r


data = np.loadtxt('data/dataTrain_test.csv',dtype=np.float64,delimiter=',',unpack=False)

X = data[:, 1:-1]
y = data[:, -1]

nrow, ncol = X.shape

# show_hist(X)

score_ps = np.array([np.round(pearsonr(X[:, i], y)*100, 2) for i in range(ncol)])

idx_no_ps = np.argwhere(np.abs(score_ps) < 5)
idx_no_ps = idx_no_ps.reshape(len(idx_no_ps))


# spearman
score_spr = np.array([np.round(stats.spearmanr(X[:, i], y)[0]*100, 2) for i in range(ncol)])

idx_no_spr = np.argwhere(np.abs(score_spr) < 5)
idx_no_spr = idx_no_spr.reshape(len(idx_no_spr))

print(np.intersect1d(idx_no_ps, idx_no_spr))

# x = np.linspace(np.min(data) - 1, np.max(data) + 1, nrow)
# plt.plot(x, np.sort(data), lw=1)

exit(-1)



# standardization
for i in range(0, ncol):
    X[:, i] /= np.max(X[:, i])
    # X[:, i] = X[:, i]*2 - 1

''' no prompt
X_mean = np.mean(X, axis=0)
X_ori = X - X_mean
X_ori_norm = np.linalg.norm(X_ori, axis=1)
X = np.delete(X, np.argsort(X_ori_norm)[-10000:], 0)
y = np.delete(y, np.argsort(X_ori_norm)[-10000:])
'''
# print(X.shape)


