#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Implemented by SOW
#  Modification iforestSow.py  donne IsolationTreeEnsembleFromiforestSow.py


import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import sys
import time

# np.random.seed(21)
# random.seed(21)
# Follows algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):

        self.sample_size = sample_size
        self.n_trees = n_trees
        self.height_limit = np.log2(sample_size)
        self.trees = []

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            len_x = len(X)
            col_x = X.shape[1]
            self.trees = []

        if improved:
            for i in range(self.n_trees):
                sample_idx = random.sample(list(range(len_x)), self.sample_size)
                temp_tree = IsolationTree(self.height_limit, 0).fit_improved(X[sample_idx, :], improved=True)
                self.trees.append(temp_tree)
        else:
            for i in range(self.n_trees):
                sample_idx = random.sample(list(range(len_x)), self.sample_size)
                temp_tree = IsolationTree(self.height_limit, 0).fit(X[sample_idx, :], improved=False)
                self.trees.append(temp_tree)

        return self

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        pl_vector = []
        if isinstance(X, pd.DataFrame):
            X = X.values

        for x in (X):
            pl = np.array([path_length_tree(x, t, 0) for t in self.trees])
            pl = pl.mean()

            pl_vector.append(pl)

        pl_vector = np.array(pl_vector).reshape(-1, 1)

        return pl_vector

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """

        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."

        scores = 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))
        predictions = [1 if p[0] >= threshold else 0 for p in scores]

        return predictions
