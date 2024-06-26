#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### Implemented by SOW

from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.utils import check_random_state
from skmultiflow.utils import get_dimensions

import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
import sys
import time

# IsolationTree : 

class IsolationTree:
    def __init__(self, height_limit, current_height):

        self.depth = height_limit
        self.current_height = current_height
        self.split_by = None
        self.split_value = None
        self.right = None
        self.left = None
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1

    def fit(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """

        if len(X) <= 1 or self.current_height >= self.depth:
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

            self.left = IsolationTree(self.depth, self.current_height + 1).fit(X[w])
            self.right = IsolationTree(self.depth, self.current_height + 1).fit(X[~w])
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self


          
# IsolationTreeEnsemble : Contain an Isolation Tree Ensemble object, current model for IsolationForestStream
 

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees, random_state):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.depth = np.log2(sample_size)
        self.trees = []
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)
        self.is_learning_phase_on = True 

    def fit(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        len_x = len(X)

        for i in range(self.n_trees):
            sample_idx = random.sample(list(range(len_x)), self.sample_size )
            temp_tree = IsolationTree(self.depth, 0).fit(X[sample_idx])
            self.trees.append(temp_tree)

        return self
   
    def path_length(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        pl_vector = []

        for x in (X):
            pl = np.array([path_length_tree(x, t, 0) for t in self.trees])
            pl = pl.mean()

            pl_vector.append(pl)

        pl_vector = np.array(pl_vector).reshape(-1, 1)

        return pl_vector

    def anomaly_score(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores:int, threshold:float):
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        predictions = 1 if scores >= threshold else 0

        return predictions

## To implement this class, we took inspiration from Scikit-MultiFLow HSTrees implementation to follow its requirements.

class IsolationForestStream(BaseSKMObject, ClassifierMixin):

  def __init__(self, window_size=100, n_estimators=25, anomaly_threshold=0.5, drift_threshold=0.5, random_state=None):

        super().__init__()
        self.n_estimators = n_estimators
        self.ensemble = None
        self.random_state = random_state
        self.window_size = window_size
        self.samples_seen = 0
        self.anomaly_rate = 0.20
        self.anomaly_threshold = anomaly_threshold
        self.drift_threshold = drift_threshold
        self.window = None
        self.prec_window = None
        self.cpt = 0

  def partial_fit(self, X, y, classes=None, sample_weight=None):

          """ Partially (incrementally) fit the model.
          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features)
              The features to train the model.
          y: numpy.ndarray of shape (n_samples)
              An array-like with the class labels of all samples in X.
          classes: None
              Not used by this method.
          sample_weight: None
              Not used by this method.
          Returns
          -------
          self
          """

          ## get the number of observations
          number_instances, _ = X.shape

          if(self.samples_seen==0):
            ## ToDo ? Give a sample of self.window_size in attribute of iForest
            iforest = IsolationTreeEnsemble(self.window_size,self.n_estimators,self.random_state)
            self.ensemble = iforest


          for i in range(number_instances):
              self._partial_fit(X[i], y[i])

          return self


  def _partial_fit(self, X, y):

          """ Trains the model on samples X and corresponding targets y.
          Private function where actual training is carried on.
          Parameters
          ----------
          X: numpy.ndarray of shape (1, n_features)
              Instance attributes.
          y: int
              Class label for sample X. Not used in this implementaion which is Unsupervised
          """

          """
          Reshape X and add it to our window if it isn't full.
          If it's full, give window to our precedent_window.
          If we are at the end our window, fit if we're learning
          Check the anomaly score of our window
          Update if self.anomaly_rate > self.drift_threshold

          """
          X = np.reshape(X,(1,len(X)))

          if self.samples_seen % self.window_size == 0:
            ## Update the two windows (precedent one and current windows)
            self.prec_window = self.window
            self.window = X
          else:
            self.window = np.concatenate((self.window,X))


          if self.samples_seen % self.window_size == 0 and self.samples_seen !=0:
              #Fit the ensemble if it's not empty
              if(self.cpt<self.n_estimators):
                self.ensemble.fit(self.prec_window)
                self.cpt += 1
                  ## Update the current anomaly score
              self.anomaly_rate = self.anomaly_scores_rate(self.prec_window) ## Anomaly rate
              #print(self.anomaly_rate) ##

                  ## Update the model if the anomaly rate is greater than the threshold (u in the original paper [3])
              if self.anomaly_rate > self.drift_threshold: ## Use Anomaly RATE ?
                self.update_model(self.prec_window) # This function will discard completly the old model and create a new one

          self.samples_seen += 1

  def update_model(self,window):
    """ Update the model (fit a new isolation forst) if outhe current anomaly rate (in the previous sliding window)
     is higher than self.drift_threshold
        Parameters:
          window: numpy.ndarray of shape (self.window_size, n_features)
        Re-Initialize our attributes and our ensemble, fit with the current window

    """

    ## ToDo ? Give a sample of self.window_size in attribute of iForest
    self.is_learning_phase_on = True
    iforest = IsolationTreeEnsemble(self.window_size,self.n_estimators,self.random_state)
    self.ensemble = iforest
    self.ensemble.fit(window)
    print("Update")


  def anomaly_scores_rate(self, window):
    """
    Given a 2D matrix of observations, compute the anomaly rate
    for all instances in the window and return an anomaly rate of the given window.

    Parameters :
    window: numpy.ndarray of shape (self.window_size, n_features)
    """

    score_tab = 2.0 ** (-1.0 * self.ensemble.path_length(window) / c(len(window)))
    score = 0
    for x in score_tab:
      if x > self.anomaly_threshold:
        score += 1
    return score / len(score_tab)

  def predict(self, X):
    """
    Given an instance, Predict the anomaly (1 or 0) based on the last sample of the window by using predict_proba if our model have fit,
    else return None

    """
    if(self.samples_seen <= self.window_size):

      return [-1] ## Return the last element

    X = np.reshape(X,(1,len(X[0])))
    self.prec_window = np.concatenate((self.prec_window ,X)) ## Append the instances in the sliding window

    prediction =  self.ensemble.predict_from_anomaly_scores(self.predict_proba(self.prec_window),self.anomaly_threshold) ## return 0 or 1

    return [prediction]

  def predict_proba(self, X):
    """
    Calculate the anomaly score of the window if our model have fit, else return None
    Parameters :
    X: numpy.ndarray of shape (self.window_size, n_features)

    """
    if(self.samples_seen <= self.window_size):
        return [-1]
    return self.ensemble.anomaly_score(self.prec_window)[-1] # Anomaly return an array with all scores of each data, taking -1 return the last instance (X) anomaly score






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
