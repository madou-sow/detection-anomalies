#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Implemented by SOW
#  Modification iforestSow.py  donne IsolationTreeFromiforestSow.py

import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import sys
import time

# np.random.seed(21)
# random.seed(21)
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
#

 
class IsolationTree:
    def __init__(self, height_limit, current_height):

        self.height_limit = height_limit
        self.current_height = current_height
        self.split_by = None
        self.split_value = None
        self.right = None
        self.left = None
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1

    def fit_improved(self, X: np.ndarray, improved=False):
        """
        Add Extra while loop
        """

        if len(X) <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            self.size = len(X)

            return self

        split_by = random.choice(np.arange(X.shape[1]))
        min_x = X[:, split_by].min()
        max_x = X[:, split_by].max()

        if min_x == max_x:
            self.exnodes = 1
            self.size = len(X)

            return self
        condition = True

        while condition:

            split_value = min_x + random.betavariate(0.5,0.5)*(max_x-min_x)

            a = X[X[:, split_by] < split_value]
            b = X[X[:, split_by] >= split_value]
            if len(X) < 10 or a.shape[0] < 0.25 * b.shape[0] or b.shape[0] < 0.25 * a.shape[0] or (
                    a.shape[0] > 0 and b.shape[0] > 0):
                condition = False

            self.size = len(X)
            self.split_by = split_by
            self.split_value = split_value

            self.left = IsolationTree(self.height_limit, self.current_height + 1).fit_improved(a, improved=False)
            self.right = IsolationTree(self.height_limit, self.current_height + 1).fit_improved(b, improved=False)
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """

        if len(X) <= 1 or self.current_height >= self.height_limit:
            self.exnodes = 1
            self.size = X.shape[0]

            return self

        split_by = random.choice(np.arange(X.shape[1]))
        X_col = X[:, split_by]
        min_x = X_col.min()
        max_x = X_col.max()

        if min_x == max_x:
            self.exnodes = 1
            self.size = len(X)

            return self

        else:

            split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)

            w = np.where(X_col < split_value, True, False)
            del X_col

            self.size = X.shape[0]
            self.split_by = split_by
            self.split_value = split_value

            self.left = IsolationTree(self.height_limit, self.current_height + 1).fit(X[w], improved=True)
            self.right = IsolationTree(self.height_limit, self.current_height + 1).fit(X[~w], improved=True)
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self



def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1

    while threshold > 0:
        y_pred = [1 if p[0] >= threshold else 0 for p in scores]
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        TPR = tp / (tp + fn)
        FPR = fp / (fp + tn)
        if TPR >= desired_TPR:
            return threshold, FPR

        threshold = threshold - 0.001

    return threshold, FPR


def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0

def path_length_tree(x, t,e):
    e = e
    if t.exnodes == 1:
        e = e+ c(t.size)
        return e
    else:
        a = t.split_by
        if x[a] < t.split_value :
            return path_length_tree(x, t.left, e+1)

        if x[a] >= t.split_value :
            return path_length_tree(x, t.right, e+1)
