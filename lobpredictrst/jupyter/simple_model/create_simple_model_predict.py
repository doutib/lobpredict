
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

# In[66]:

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
import pandas as pd


# We looked at the top features from the best performing random forest. They are as below:

# In[48]:

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


# In[33]:

# Open test and train sets
df_train = pd.read_csv(train_ds_ref
                       , compression='gzip', index_col = None)
df_test  = pd.read_csv(test_ds_ref
                       , compression='gzip', index_col = None)

# Drop the first columns - they are not useful
df_train_clean = df_train.iloc[:,1:]
df_test_clean  = df_test.iloc[:,1:]


# In[34]:

X_train_cols  =  list(df_train_clean[['P_1_bid', 'V_1_bid', 'P_1_ask', 'V_1_ask', 'P_2_bid', 'V_2_bid', 'P_2_ask'
                          , 'V_2_ask']].columns.values)

X_train  =  np.array(df_train_clean[['P_1_bid', 'V_1_bid', 'P_1_ask', 'V_1_ask', 'P_2_bid', 'V_2_bid', 'P_2_ask'
                          , 'V_2_ask']])
Y_train  =  np.array(df_train_clean[['labels']])[:,0]

X_test  =  np.array(df_test_clean[['P_1_bid', 'V_1_bid', 'P_1_ask', 'V_1_ask', 'P_2_bid', 'V_2_bid', 'P_2_ask'
                          , 'V_2_ask']])
Y_test  =  np.array(df_test_clean[['labels']])[:,0]


# In[38]:

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


# In[46]:

print(result[3])
print(Y_hat)
print(Y_probs)


# #### The predicted output for our most successful RF model is as follows

# ```
# classification_report
# 
#              precision    recall  f1-score   support
# 
#          -1       0.99      0.98      0.98     18373
#           0       0.97      0.98      0.97     16950
#           1       0.99      0.98      0.98     15265
# 
# avg / total       0.98      0.98      0.98     50588
# ```

# In[49]:

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


# In[62]:

linear_simple_predict = predict_simple_linear(df_train_clean = df_train_clean
                                              , df_test_clean = df_train_clean)


# In[64]:

# Get the predicted outcomes
linear_simple_predict_vals = linear_simple_predict[len(linear_simple_predict) -1]
len(list(linear_simple_predict_vals))


# In[67]:

modl = imp.load_source('execute_model', '../../execute_model.py')


# In[ ]:



