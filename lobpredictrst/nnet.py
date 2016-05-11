
# coding: utf-8

# In[4]:

import numpy as np
import pandas as pd

from sknn.mlp import Classifier, Layer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, TimeoutError
from multiprocessing import cpu_count
from datetime import timedelta

import sys
import csv
import itertools
import time


# In[2]:

def two_layers_nnet(X_train,
                    Y_train,
                    X_test,
                    Y_test,
                    method1="Tanh",
                    neurons1=5,
                    method2="",
                    neurons2=0,
                    decay=0.0001,
                    learning_rate=0.001,
                    n_iter=25,
                    random_state=1):
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
    method1       : str
        method used for the first layer
    neurons1      : int
        number of neurons of the first layer
    method2       : None
        method used for the first layer
    neurons2      : int
        number of neurons of the first layer
    decay         : float
        weight decay
    learning_rate : float
        learning rate
    n_iter        : int
        number of iterations
    random_state  : int
        seed for weight initialization
        
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
    
    # Layers
    if neurons2 == 0 :
        layers=[Layer(method1, weight_decay = decay, units = neurons1),
                Layer("Softmax")]
    else:
        layers=[Layer(method1, weight_decay = decay, units = neurons1),
                Layer(method2, weight_decay = decay, units = neurons2),
                Layer("Softmax")]
        
    ## # Run nnet
    # Define classifier
    nn = Classifier(layers,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    n_iter=n_iter)
    # Fit
    nn.fit(X_train, Y_train)
    # Predict
    Y_hat = nn.predict(X_test)
    Y_probs = nn.predict_proba(X_test)
    
    ## # Misclassification error rate
    miss_err = 1-accuracy_score(Y_test, Y_hat)
    ## # Log Loss
    eps = 10^(-15)
    logloss = log_loss(Y_test, Y_probs, eps = eps)
    
    ## # Precision
    prec = precision_score(y_true=Y_test, y_pred=Y_hat, labels=labels, average='micro')
    ## # Recal
    recall = recall_score(y_true=Y_test, y_pred=Y_hat, labels=labels, average='micro') 
    ## # F1
    f1 = f1_score(y_true=Y_test, y_pred=Y_hat, labels=labels, average='micro')
    
    # Summarized results
    result = np.array([logloss,
                       miss_err,
                       prec,
                       recall,
                       f1,
                       method1,
                       neurons1,
                       method2,
                       neurons2,
                       decay,
                       learning_rate,
                       n_iter,
                       random_state])
    return result


# In[3]:



def processInput(xxx_todo_changeme): 
    # Define parameters names
    (X_train,Y_train,X_test,Y_test,parameters,index) = xxx_todo_changeme
    method1,neurons1,method2,neurons2,decay,learning_rate,n_iter,random_state=parameters[index]
    
    # Run nnet
    result = two_layers_nnet(X_train,
                             Y_train,
                             X_test,
                             Y_test,
                             method1,
                             neurons1,
                             method2,
                             neurons2,
                             decay,
                             learning_rate,
                             n_iter,
                             random_state)
    return result


def two_layers_nnet_simulation(X_train,
                               Y_train,
                               X_test,
                               Y_test,
                               method1,
                               neurons1,
                               method2,
                               neurons2,
                               decay,
                               learning_rate,
                               n_iter,
                               random_state):
    """
    Parameters:
    -----------
    Same parameters as two_layers_nnet, in a list format.
    
    Result:
    ------
    List of Lists of results from two_layers_nnet.
        One list corresponds to one set of parameters
    """
    
    print('Lauching Simulation...')
    start = time.time()
    
    # Combinations
    param = np.array([method1,
                      neurons1,
                      method2,
                      neurons2,
                      decay,
                      learning_rate,
                      n_iter,
                      random_state])
    
    parameters = list(itertools.product(*param))
    
    indexes = list(range(len(parameters)))
    print("Number of sets of parameters: %s.\n" %len(parameters))
    
    print('Parameters:\n-----------')
    print(np.array(parameters))
    

    # Number of clusters
    num_cpu = cpu_count()          
    print("\nNumber of identified CPUs: %s.\n" %num_cpu)
    num_clusters = min(num_cpu,len(parameters))
    
    ## # Parallelization
    tuples_indexes = tuple([(X_train,Y_train,X_test,Y_test,parameters,index) for index in indexes])

    # Start clusters
    print('Start %s clusters.\n' % num_clusters)
    print('Running...')
    pool = Pool(processes=num_clusters)
    results = pool.map(processInput, tuples_indexes) 
    pool.terminate()
    
    # Results
    print('Results:\n--------')
    print(results)
    end = time.time()
    elapsed = end - start
    print('End of Simulation.\nElapsed time: %s' %str(timedelta(seconds=elapsed)))
    print('Write into csv...')
    
    return results


# In[4]:

def two_layers_nnet_predict(X_train,
                            Y_train,
                            X_test,
                            method1="Tanh",
                            neurons1=5,
                            method2="",
                            neurons2=0,
                            decay=0.0001,
                            learning_rate=0.001,
                            n_iter=25,
                            random_state=1):
    """
    Parameters
    ----------
    X_train       : pandas data frame
        data frame of features for the training set
    Y_train       : pandas data frame
        data frame of labels for the training set
    X_test        : pandas data frame
        data frame of features for the test set
    method1       : str
        method used for the first layer
    neurons1      : int
        number of neurons of the first layer
    method2       : None
        method used for the first layer
    neurons2      : int
        number of neurons of the first layer
    decay         : float
        weight decay
    learning_rate : float
        learning rate
    n_iter        : int
        number of iterations
    random_state  : int
        seed for weight initialization
        
    Result:
    -------
    tuple of numpy arrays
        (predicted classes, predicted probabilities)
    """

    labels = np.unique(Y_train)
    
    ## # Scale Data
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)
    X_train = scaler.fit_transform(X_train)

    ## # Split data set into train/test
    
    # Layers
    if neurons2 == 0 :
        layers=[Layer(method1, weight_decay = decay, units = neurons1),
                Layer("Softmax")]
    else:
        layers=[Layer(method1, weight_decay = decay, units = neurons1),
                Layer(method2, weight_decay = decay, units = neurons2),
                Layer("Softmax")]
        
    ## # Run nnet
    # Define classifier
    nn = Classifier(layers,
                    learning_rate=learning_rate,
                    random_state=random_state,
                    n_iter=n_iter)
    # Fit
    nn.fit(X_train, Y_train)
    # Predict
    Y_hat = nn.predict(X_test)
    Y_probs = nn.predict_proba(X_test)
    
    # Summarized results
    result = (Y_hat,Y_probs)
    return result

