
# coding: utf-8

# # The best model parameters are given by
# ```
# author : SHAMINDRA
# data_source_dir : SC_shuffle
# test_type : validation
# model_type : RF
# RF:
#     n_estimators : 100
#     criterion : 'gini'
#     max_features : 'auto'
#     max_depth : 20
#     n_jobs : 1
# SVM:
#     kernel : 'rbf'
#     degree : 3
#     gamma : 'auto'
#     tol : 0.001
# NNET:
#     method1 : 'Tanh'
#     neurons1 : 24
#     method2 : 'Tanh'
#     neurons2 : 39
#     decay : 0.0001
#     learning_rate : 0.001
#     n_iter : 25
#     random_state : 1
# ```

# In[1]:

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import imp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

from  sklearn.ensemble import RandomForestClassifier

import pandas as pd
import yaml
import re


# We looked at the top features from the best performing random forest. They are as below:

# In[2]:

# The top variables are:
var_importance = [(1, 'P_1_bid', 0.020001165389254737)
                  , (2, 'V_1_bid', 0.018358575666246449)
                  , (3, 'P_1_ask', 0.017058479215839299)
                  , (4, 'V_1_ask', 0.016953559068869958)
                  , (5, 'P_2_bid', 0.016908649059514971)
                  , (6, 'V_2_bid', 0.016219220215427665)
                  , (7, 'P_2_ask', 0.015039647893425838)
                  , (8, 'V_2_ask', 0.014497773408233052)
                  , (9, 'P_3_bid', 0.014321084019596746)
                  , (10, 'V_3_bid', 0.014158850118003859)
                  , (11, 'P_3_ask', 0.014101386932514923)
                  , (12, 'V_3_ask', 0.013911823640617986)
                  , (13, 'P_4_bid', 0.013838322603744435)
                  , (14, 'V_4_bid', 0.013668619218980316)
                  , (15, 'P_4_ask', 0.013413471959983998)]


# In[3]:

# Train test datasets
train_ds_ref      = "data/output/model_clean_data/SC_shuffle/train_test_validation.tar.gz"
test_ds_ref       = "data/output/model_clean_data/SC_shuffle/strategy_validation.tar.gz"

# Open test and train sets
df_train = pd.read_csv(train_ds_ref
                       , compression='gzip', index_col = None)
df_test  = pd.read_csv(test_ds_ref
                       , compression='gzip', index_col = None)

# Drop the first columns - they are not useful
df_train_clean = df_train.iloc[:,1:]
df_test_clean  = df_test.iloc[:,1:]


# In[4]:

def predict_simple_linear(df_train_clean, df_test_clean):
    X_train_cols  =  list(df_train_clean[['P_1_bid', 'V_1_bid', 'P_1_ask', 'V_1_ask', 'P_2_bid', 'V_2_bid', 'P_2_ask'
                          , 'V_2_ask']].columns.values)

    X_train  =  np.array(df_train_clean[['P_1_bid', 'V_1_bid', 'P_1_ask', 'V_1_ask', 'P_2_bid', 'V_2_bid', 'P_2_ask'
                              , 'V_2_ask']])
    Y_train  =  np.array(df_train_clean[['labels']])[:,0]

    X_test  =  np.array(df_test_clean[['P_1_bid', 'V_1_bid', 'P_1_ask', 'V_1_ask', 'P_2_bid', 'V_2_bid', 'P_2_ask'
                                       , 'V_2_ask']])
    Y_test  =  np.array(df_test_clean[['labels']])[:,0]
    
    # Define the labels
    labels = np.unique(Y_train)

    ## # Scale Data
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)
    X_train = scaler.fit_transform(X_train)

    # Set up the data
    logreg = linear_model.LogisticRegression(C=1e5)

    # Fit
    logreg.fit(X_train, Y_train)

    # Predict
    Y_hat   = logreg.predict(X_test)
    Y_probs = logreg.predict_proba(X_test)

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
    result.append("logloss")
    result.append(logloss)
    result.append("miss_err")
    result.append(miss_err)
    result.append("Y_hat")
    result.append(Y_hat)
    
    return result


# In[5]:

linear_simple_predict = predict_simple_linear(df_train_clean = df_train_clean
                                              , df_test_clean = df_test_clean)


# In[6]:

print(linear_simple_predict[3])


# In[6]:

# Get the predicted outcomes
linear_simple_predict_vals = linear_simple_predict[len(linear_simple_predict) -1]
#len(list(linear_simple_predict_vals))
len(list(linear_simple_predict_vals))-len(df_test_clean)


# In[7]:

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
    result.append("Y_hat")
    result.append(Y_hat)    
    return result


# In[8]:

def execute_model(inp_yaml):
    """Apply trees in the forest to X, return leaf indices.
        Parameters
        ----------
        inp_yaml : A yaml file with model specifications

        Returns
        -------
        parameters_dict : A python dictionary with the model specifications
                          to be used to encode metadata for the model
                          and pass into specific model functions e.g. random
                          forest
        """

    # Read in and parse all parameters from the YAML file
    yaml_params = read_model_yaml(inp_yaml)

    # Define output file name based on input
    folder_name = re.split("/", inp_yaml)[2]
    file_name   = re.split("/", inp_yaml)[3][:-5]
    output_txt_file = 'data/output/' + folder_name + '/' + file_name + '.txt'

    #-------------------------------------------------
    # Create Train and Test Datasets
    #-------------------------------------------------

    data_source_dir = yaml_params["data_source_dir"]
    test_type       = yaml_params["test_type"]

    print('data source dir is: %s' % (data_source_dir))
    print('test type is: %s' % (test_type))

    if test_type == "test":
        train_ds_name = "train.tar.gz"
        test_ds_name  = "test.tar.gz"
    elif test_type == "validation":
        train_ds_name = "train_test.tar.gz"
        test_ds_name  = "validation.tar.gz"
    else:
        #train_ds_name = "train_test_validation.tar.gz"
        #TODO - delete
        train_ds_name = "train_test.tar.gz"
        test_ds_name  = "strategy_validation.tar.gz"

    train_ds_ref      = "data/output/model_clean_data/" + data_source_dir + "/" + train_ds_name
    test_ds_ref       = "data/output/model_clean_data/" + data_source_dir + "/" + test_ds_name

    print('training dataset is: %s' % (train_ds_ref))
    print('test dataset is: %s' % (test_ds_ref))

    # Open test and train sets
    df_train = pd.read_csv(train_ds_ref
                           , compression='gzip', index_col = None)
    df_test  = pd.read_csv(test_ds_ref
                           , compression='gzip', index_col = None)

    # Drop the first columns - they are not useful
    df_train_clean = df_train.iloc[:,1:]
    df_test_clean  = df_test.iloc[:,1:]

    # Traning data column names - used for variale importance
    X_train_cols  =  list(df_train_clean.drop(['labels', 'index', 'Time'], axis=1).columns.values)

    # Define test/training set
    X_train  =  np.array(df_train_clean.drop(['labels', 'index', 'Time'], axis = 1))
    Y_train  =  np.array(df_train_clean[['labels']])[:,0]
    X_test   =  np.array(df_test_clean.drop(['labels', 'index', 'Time'], axis = 1))
    Y_test   =  np.array(df_test_clean[['labels']])[:,0]


    #-------------------------------------------------
    # Run RF (RANDOM FOREST)
    #-------------------------------------------------

    if yaml_params["model_type"] == "RF":

        # Extract the RF model variables from the YAML file
        n_estimators  = yaml_params["parameters"]["n_estimators"]
        criterion     = yaml_params["parameters"]["criterion"]
        max_features  = yaml_params["parameters"]["max_features"]
        max_depth     = yaml_params["parameters"]["max_depth"]
        n_jobs        = yaml_params["parameters"]["n_jobs"]

        print('number of trees is: %d' % (n_estimators))
        print('max depth is: %d' % (max_depth))

        print("running RF WITHOUT simulation...")

        # Run simulation
        result = rf(X_train_cols   = X_train_cols
                    , X_train      = X_train
                    , Y_train      = Y_train
                    , X_test       = X_test
                    , Y_test       = Y_test
                    , n_estimators = n_estimators
                    , criterion    = criterion
                    , max_features = max_features
                    , max_depth    = max_depth)

        print("finished - rf without simulation")

        return result
        # Write into text file
        #write_results_txt(output_txt_file, result)

    #-------------------------------------------------
    # Run SVM (SUPPORT VECTOR MACHINE)
    #-------------------------------------------------

    # Extract the SVM model variables from the YAML file
    if yaml_params["model_type"] == "SVM":
        kernel  = yaml_params["parameters"]["kernel"]
        degree  = yaml_params["parameters"]["degree"]
        gamma   = yaml_params["parameters"]["gamma"]
        tol     = yaml_params["parameters"]["tol"]
        C       = yaml_params["parameters"]["C"]

        print('The value of C is: %.2f' % (C))

        print("running SVM WITHOUT simulation...")

        # Run a single simulation
        result = svm(X_train        = X_train
                     , Y_train      = Y_train
                     , X_test       = X_test
                     , Y_test       = Y_test
                     , kernel       = kernel
                     , C            = C
                     , degree       = degree
                     , gamma        = gamma
                     , tol          = tol
                     , decision_function_shape='ovr')

        # Write into text file
        #write_results_txt(output_txt_file, result)
        return result
        print("finished - SVM without simulation")


# In[9]:

def read_model_yaml(inp_yaml):
    """Apply trees in the forest to X, return leaf indices.
        Parameters
        ----------
        inp_yaml : A yaml file with model specifications

        Returns
        -------
        parameters_dict : A python dictionary with the model specifications
                          to be used to encode metadata for the model
                          and pass into specific model functions e.g. random
                          forest
        """
    # with open("../model/spec/SS/SS_RF_1.yaml") as stream:
    with open(inp_yaml) as stream:
        data = yaml.load(stream)
        parameters_dict = {
        "author"                   : data["author"]
        , "data_source_dir"        : data["data_source_dir"]
        , "model_type"             : data["model_type"]
        , "test_type"              : data["test_type"]        
        , "parameters"             : data[data["model_type"]]
        }
    return parameters_dict


# In[10]:

result_best_RF = execute_model(inp_yaml = "./model/spec/TB/TB_RF_SC_shuffle_strategy_4.yaml")


# In[11]:

print(result_best_RF[3])


# In[12]:

test_ds_ref       = "data/output/model_clean_data/SC_shuffle/strategy_validation.tar.gz"
df_test           = pd.read_csv(test_ds_ref, compression='gzip', index_col = None)
df_test2          = df_test
df_test2["predicted"] = list(result_best_RF[len(result_best_RF) - 1])


# In[13]:

# Direct Python to plot all figures inline (i.e., not in a separate window)
get_ipython().magic('matplotlib inline')

# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Division of two integers in Python 2.7 does not return a floating point result. The default is to round down 
# to the nearest integer. The following piece of code changes the default.
from __future__ import division

def profit_calculator(data, delta_t = 30, simple = False):
    """Calculate the profit of trading strategy based on precisely the prediction of the model
        Parameters
        ----------
        data    : a data frame with "predicted" "P_1_bid" "P_1_ask"
        delta_t : time gap between 
        simple  : a dummy, True, means we make transection decisions only every delta_t period. False, means we track the current 
                  hand every period, only if we don't have anything at hand, we make new transactions

        Returns
        -------
        profit        : a numeric, the net profit at the end
        
        """    
    if simple == True:
        data_effective = data.loc[np.arange(len(data)) % delta_t == 0]
        bid            = data_effective['P_1_bid']
        ask            = data_effective['P_1_ask']
        trade_decision = data_effective['predicted'][:-1]
        buy_profit     = np.array(bid[1:]) - np.array(ask[:-1])
        sell_profit    = np.array(bid[:-1]) - np.array(ask[1:])
        profit         = sum((np.array(trade_decision) > 0) * buy_profit + (np.array(trade_decision) < 0) * sell_profit)
        return profit
    else:
        buy_profit           = np.array(data['P_1_bid'][delta_t:]) - np.array(data['P_1_ask'][:(-1 * delta_t)])
        sell_profit           = np.array(data['P_1_bid'][:(-1 * delta_t)]) - np.array(data['P_1_ask'][delta_t:])
        trade_decision_draft = data['predicted'][:(-1 * delta_t)]
        T                    = len(buy_profit)
        current_state        = [0] * T
        trade_decision       = [0] * T
        profit               = 0
        for i in range(T):
            if current_state[i] == 1:
                trade_decision[i] = 0
            else:
                trade_decision[i] = trade_decision_draft[i]
                    
            if i < T-1:
                current_state[i+1] = int(sum(abs(np.array(trade_decision[max(0, i + 1 - delta_t):i + 1]))) != 0)
        profit = sum((np.array(trade_decision) > 0) * buy_profit + (np.array(trade_decision) < 0) * sell_profit)
        return profit         


# In[14]:

profit_calculator(data = df_test2, delta_t = 1000, simple = False)


# In[15]:

profit_calculator(data = df_test2, delta_t = 1000, simple = True)

