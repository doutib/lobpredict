
# coding: utf-8

# In[1]:

import sys
import imp
import yaml
import csv
import pandas as pd
import re
import numpy as np
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[2]:

def label_percent(v):
    l=len(v)
    pos=sum(x > 0 for x in v)/l
    neg=sum(x < 0 for x in v)/l
    zero=sum(x == 0 for x in v)/l
    print('Pos:'+str("{0:.2f}".format(pos))+"; Neg:"+str("{0:.2f}".format(neg))+'; Zero:'+str("{0:.2f}".format(zero))+';')


# In[3]:

# Open test and train sets
df_train = pd.read_csv("data/output/model_clean_data/train_2.tar.gz", compression='gzip', index_col = None)
df_test  = pd.read_csv("data/output/model_clean_data/test_2.tar.gz" , compression='gzip', index_col = None)


# In[4]:

# df_train.head()


# In[5]:

# df_test.columns


# In[6]:

#X_test_new = df_test.drop(df_test.columns[['labels', 'train.csv', 'index']], axis = 1)
x = df_test.drop(['labels', 'test.csv', 'index'], axis=1)


# In[7]:

x.head()


# In[8]:

# Define test/training set
X_test   =  np.array(df_test.drop(['labels', 'test.csv', 'index', 'Time'], axis = 1))
Y_test   =  np.array(df_test[['labels']])[:,0]
X_train  =  np.array(df_train.drop(['labels', 'train.csv', 'index', 'Time'], axis = 1))
Y_train  =  np.array(df_train[['labels']])[:,0]


# In[9]:

X_train_cols  =  list(df_train.drop(['labels', 'train.csv', 'index', 'Time'], axis=1).columns.values)
X_train_cols


# In[10]:

label_percent(Y_train)
Y_train.size


# In[11]:

label_percent(Y_test)
Y_test.size


# In[12]:

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100, max_depth=10)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(X_train, Y_train)

# Take the same decision trees and run it on the test data
output = forest.predict(X_test)

classification_report1 = classification_report(y_true=Y_test, y_pred=output)
print(classification_report1)


# In[13]:

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# In[16]:

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print("%d. feature %s (%f)" % (f + 1, X_train_cols[f], importances[indices[f]]))


# In[18]:

var_importance = [(f+1, X_train_cols[f], importances[indices[f]]) for f in range(X_train.shape[1])]


# In[20]:

print("\n".join(var_importance))


# In[ ]:

print(output)


# In[ ]:

label_percent(output)


# In[ ]:

from sklearn.cross_validation import StratifiedShuffleSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
sss = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)
len(sss)


# In[ ]:

sss


# In[ ]:

print(sss)


# In[ ]:



