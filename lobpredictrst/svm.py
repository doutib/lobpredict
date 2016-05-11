
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, TimeoutError
from multiprocessing import cpu_count
from datetime import timedelta

from sklearn.svm import SVC

import sys
import csv
import itertools
import time


# In[3]:

def svm(X_train,
        Y_train,
        X_test,
        Y_test,
        C=1.0,
        kernel='rbf',
        degree=3,
        gamma='auto',
        tol=0.001,
        decision_function_shape='ovr'):

    """
    Parameters
    ----------
    X_train       : pandas data frame
        data frame of features for the training set
    Y_train       : pandas data frame
        data frame of labels for the training set
    X_test        : pandas data frame
        data frame of features for the test set
    Y_test        : pandas data frame
        data frame of labels for the test set
        proportion of the traning set
    kernel                  : str
        Specifies the kernel type to be used in the algorithm.
        It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’.
    degree                  : int
        Degree of the polynomial kernel function (‘poly’).
        Ignored by all other kernels.
    gamma                   : int
        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        If gamma is ‘auto’ then 1/n_features will be used instead.
    tol                     : float
        Tolerance for stopping criterion.
    decision_function_shape : 'ovo’, ‘ovr’
        Whether to return a one-vs-rest (‘ovr’) decision function: shape=(n_samples, n_classes)
        or the original one-vs-one (‘ovo’) decision function: shape=(n_samples, n_classes * (n_classes - 1) / 2)

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

    labels = np.unique(Y_train)

    ## # Scale Data
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)
    X_train = scaler.fit_transform(X_train)

    ## # Run svm
    # Define classifier
    clf = SVC(C=C,
              kernel=kernel,
              degree=degree,
              gamma=gamma,
              probability=True,
              tol=tol,
              decision_function_shape=decision_function_shape)
    # Fit
    clf.fit(X_train, Y_train)
    # Predict
    Y_hat = clf.predict(X_test)
    Y_probs = clf.predict_proba(X_test)

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

    # Output results in a list format
    result = []
    result.append("confusion_matrix")
    result.append(confusion_matrix1)
    result.append("classification_report")
    result.append(classification_report1)
    result.append("kernel")
    result.append(kernel)
    result.append("C")
    result.append(C)
    result.append("logloss")
    result.append(logloss)
    result.append("miss_err")
    result.append(miss_err)

    return result
