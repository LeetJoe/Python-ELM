#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
import csv

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from elm import GenELMClassifier
from random_layer import RBFRandomLayer, MLPRandomLayer


def make_classifiers(nh=10):
    names = [
        # 'ELM({},tanh)'.format(nh),
        # "ELM({},tanh,LR)".format(nh),
        # "ELM({},sinsq)".format(nh),
        # "ELM({},tribas)".format(nh),
        "ELM(hardlim)",
        # "ELM({},rbf(0.1))".format(nh*2),
        'sigmoid',
        # 'log_reg'
    ]

    # pass user defined transfer func
    sinsq = (lambda x: np.power(np.sin(x), 2.0))
    srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)

    # use internal transfer funcs
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')

    srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')

    srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    sigmoid = MLPRandomLayer(n_hidden=nh, activation_func='sigmoid')

    # use gaussian RBF
    srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)

    log_reg = LogisticRegression()

    classifiers = [
        # GenELMClassifier(hidden_layer=srhl_tanh),
        # GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
        # GenELMClassifier(hidden_layer=srhl_sinsq),
        # GenELMClassifier(hidden_layer=srhl_tribas),
        GenELMClassifier(hidden_layer=srhl_hardlim),
        # GenELMClassifier(hidden_layer=srhl_rbf),
        GenELMClassifier(hidden_layer=sigmoid), # 0.84
        # GenELMClassifier(hidden_layer=log_reg),
    ]

    return names, classifiers


data = np.loadtxt('data/dataTrain_test.csv',dtype=np.float64,delimiter=',',unpack=False)

X = data[:, 1:-1]
y = data[:, -1]

nrow, ncol = X.shape

for i in range(0, ncol):
    a = np.max(X[:, i])
    if a > 1:
        X[:, i] /= a
    # X[:, i] = X[:, i]*2 - 1

names, classifiers = make_classifiers(4000)

# nh = 2000 , best is sigmoid with 84%, nh = 4000, best is sigmoid with 85.2%

for name, clf in zip(names, classifiers):
    clf.fit(X, y)
    score = clf.score(X, y)

    print(name + ", {}".format(score))




















exit(-1)

def get_data_bounds(X):
    h = .02  # step size in the mesh

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return (x_min, x_max, y_min, y_max, xx, yy)


def plot_data(ax, X_train, y_train, X_test, y_test, xx, yy):
    cm = ListedColormap(['#FF0000', '#0000FF'])
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


def plot_contour(ax, X_train, y_train, X_test, y_test, xx, yy, Z):
    cm = pl.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    ax.set_title(name)
    ax.text(xx.max() - 0.3, yy.min() + 0.3, ('%.2f' % score).lstrip('0'),
            size=13, horizontalalignment='right')


def make_datasets():
    return [make_moons(n_samples=200, noise=0.3, random_state=0),
            make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1),
            make_linearly_separable()]


def make_classifiers():

    names = ["ELM(10,tanh)", "ELM(10,tanh,LR)", "ELM(10,sinsq)",
             "ELM(10,tribas)", "ELM(hardlim)", "ELM(20,rbf(0.1))"]

    nh = 10

    # pass user defined transfer func
    sinsq = (lambda x: np.power(np.sin(x), 2.0))
    srhl_sinsq = MLPRandomLayer(n_hidden=nh, activation_func=sinsq)

    # use internal transfer funcs
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')

    srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')

    srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    # use gaussian RBF
    srhl_rbf = RBFRandomLayer(n_hidden=nh*2, rbf_width=0.1, random_state=0)

    log_reg = LogisticRegression()

    classifiers = [GenELMClassifier(hidden_layer=srhl_tanh),
                   GenELMClassifier(hidden_layer=srhl_tanh, regressor=log_reg),
                   GenELMClassifier(hidden_layer=srhl_sinsq),
                   GenELMClassifier(hidden_layer=srhl_tribas),
                   GenELMClassifier(hidden_layer=srhl_hardlim),
                   GenELMClassifier(hidden_layer=srhl_rbf)]

    return names, classifiers


def make_linearly_separable():
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=1,
                               n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)

###############################################################################

datasets = make_datasets()
names, classifiers = make_classifiers()

i = 1
figure = pl.figure(figsize=(18, 9))

# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4,
                                                        random_state=0)

    x_min, x_max, y_min, y_max, xx, yy = get_data_bounds(X)

    # plot dataset first
    ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
    plot_data(ax, X_train, y_train, X_test, y_test, xx, yy)

    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will asign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        plot_contour(ax, X_train, y_train, X_test, y_test, xx, yy, Z)

        i += 1

figure.subplots_adjust(left=.02, right=.98)
pl.show()
