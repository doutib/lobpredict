
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from multiprocessing import Pool, TimeoutError
from multiprocessing import cpu_count
from datetime import timedelta

from  sklearn.ensemble import RandomForestClassifier

import sys
import csv
import itertools
import time


# In[13]:

def rf(X_train_cols,
       X_train,
       Y_train,
       X_test,
       Y_test,
       n_estimators=10,
       criterion="gini",
       max_features="auto",
       max_depth=-1,
       n_jobs=1):
    """
    Parameters
    ----------
    X_train_cols  : list of feature column names
        from the training set
    X_train       : pandas data frame
        data frame of features for the training set
    Y_train       : pandas data frame
        data frame of labels for the training set
    X_test        : pandas data frame
        data frame of features for the test set
    Y_test        : pandas data frame
        data frame of labels for the test set
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.
    criterion : string, optional (default=”gini”)
        The function to measure the quality of a split.
        Supported criteria are “gini” for the Gini impurity and “entropy”
        for the information gain.
    max_features : int, float, string or None, optional (default=”auto”)
        The number of features to consider when looking for the best split:
        If int, then consider max_features features at each split.
        If float, then max_features is a percentage and int(max_features * n_features)
        features are considered at each split.
        If “auto”, then max_features=sqrt(n_features).
        If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
        If “log2”, then max_features=log2(n_features).
        If None, then max_features=n_features.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree.
        If None, then nodes are expanded until all leaves are pure or
        until all leaves contain less than min_samples_split samples.
        Ignored if max_leaf_nodes is not None.
    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both fit and predict.
        If -1, then the number of jobs is set to the number of cores.

    Result:
    -------
    numpy array
        logloss    : averaged logarithmic loss
        miss_err   : missclassification error rate
        prec       : precision
        recall     : recall
        f1         : f1 score
        parameters : previous parameters in the order previously specified
    """
    if max_depth==-1:
        max_depth = None

    labels = np.unique(Y_train)

    ## # Run rf
    # Define classifier
    rf = RandomForestClassifier(n_estimators = n_estimators,
                                criterion    = criterion,
                                max_features = max_features,
                                max_depth    = max_depth,
                                n_jobs       = n_jobs)
    # Fit
    rf.fit(X_train, Y_train)

    # Predict
    Y_hat   = rf.predict(X_test)
    Y_probs = rf.predict_proba(X_test)

    ## # Misclassification error rate
    miss_err = 1-accuracy_score(Y_test, Y_hat)
    ## # Log Loss
    eps = 10^(-15)
    logloss = log_loss(Y_test, Y_probs, eps = eps)

    ##confusion_matrix
    confusion_matrix1 = confusion_matrix(y_true=Y_test, y_pred=Y_hat
                                         , labels=labels)

    # classification_report
    classification_report1 = classification_report(y_true=Y_test, y_pred=Y_hat)

    # Variable importance
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Return tuple of (rank, feature name, variable importance)
    var_importance = [(f+1, X_train_cols[f], importances[indices[f]]) for f in range(X_train.shape[1])]

    # Output results in a list format
    result = []
    result.append("confusion_matrix")
    result.append(confusion_matrix1)
    result.append("classification_report")
    result.append(classification_report1)
    result.append("number of trees")
    result.append(n_estimators)
    result.append("max depth")
    result.append(max_depth)
    result.append("logloss")
    result.append(logloss)
    result.append("miss_err")
    result.append(miss_err)
    result.append("var_importance")
    result.append(var_importance)
    return result
