
# coding: utf-8

# In[1]:

import sys
import imp
import yaml
import csv
import pandas as pd
import re
from rf import *
from svm import *
#from nnet import *
modl = imp.load_source('read_model_yaml', 'read_model_yaml.py')


# In[2]:

# Import argument
#inp_yaml = "model/spec/SS/SS_RF_1.yaml"
inp_yaml = sys.argv[1]


# In[3]:

# Open test and train sets
df_test  = pd.read_csv("data/output/model_clean_data/test.tar.gz",compression='gzip', index_col = None)
df_train = pd.read_csv("data/output/model_clean_data/train.tar.gz",compression='gzip', index_col = None)

# Define test/training set
X_test   =  np.array(df_test.drop(['labels'], axis = 1))
Y_test   =  np.array(df_test[['labels']])[:,0]
X_train  =  np.array(df_train.drop(['labels'], axis = 1))
Y_train  =  np.array(df_train[['labels']])[:,0]


# In[4]:

import inspect
print(inspect.getsource(rf))


# In[5]:

def write_results_txt(filename, result):
    """
    Write results into csv file.
    
    Parameters
    ----------
    filename : string
        filename to output the result
    results : list or numpy array
        results of some simulation
    labels : list
        labels for the results, i.e. names of parameters and metrics
    """
    with open(filename, "w") as fp:
        for item in result:
            fp.write("%s\n\n" % item)


def run_model(inp_yaml,X_train,Y_train,X_test,Y_test):
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
    
    # Define output file name based on input
    folder_name = re.split("/","model/spec/SS/SS_RF_1.yaml")[2]
    file_name   = re.split("/","model/spec/SS/SS_RF_1.yaml")[3][:-5]
    output      = 'data/output/'+folder_name+'/'+file_name+'.txt'
    
    # Read in and parse all parameters from the YAML file
    yaml_params = modl.read_model_yaml(inp_yaml)
    
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
        
       
        # Run many simulations in parallel using as many cores as necessary
        if yaml_params["simulations"]:
            print("running RF WITH simulation...")
            # Run simulation
            result = rf_simulation(X_train        = X_train
                                   , Y_train      = Y_train
                                   , X_test       = X_test
                                   , Y_test       = Y_test
                                   , n_estimators = n_estimators
                                   , criterion    = criterion
                                   , max_features = max_features
                                   , max_depth    = max_depth)

            print("finished - RF WITH simulation")
            # Write into csv
            write_results_txt(output, result)
            
        # Run a single simulation
        else:
            print("running RF WITHOUT simulation...")
            # Run simulation
            result = rf(X_train        = X_train
                        , Y_train      = Y_train
                        , X_test       = X_test
                        , Y_test       = Y_test
                        , n_estimators = n_estimators
                        , criterion    = criterion
                        , max_features = max_features
                        , max_depth    = max_depth)
            
            print("finished - rf without simulation")
            # Write into csv
            write_results_txt(output, result)
            
    #-------------------------------------------------
    # Run SVM (SUPPORT VECTOR MACHINE)
    #-------------------------------------------------
    
    # Extract the SVM model variables from the YAML file        
    if yaml_params["model_type"] == "SVM":        
        kernel  = yaml_params["parameters"]["kernel"] 
        degree  = yaml_params["parameters"]["degree"]  
        gamma   = yaml_params["parameters"]["gamma"] 
        tol     = yaml_params["parameters"]["tol"]
        
        # Define labels of output
        labels = ["logloss"
                  , "miss_err"
                  , "prec"
                  , "recall"
                  , "f1"
                  , "C"
                  , "kernel"
                  , "degree" 
                  , "gamma"
                  , "tol"
                  , "decision_function_shape"]
        
        # Run many simulations in parallel using as many cores as necessary
        if yaml_params["simulations"]:
            # Run simulation
            result = svm_simulation(X_train                   = X_train
                                    , Y_train                 = Y_train
                                    , X_test                  = X_test
                                    , Y_test                  = Y_test
                                    , kernel                  = kernel
                                    , C                       = 1.0
                                    , degree                  = degree 
                                    , gamma                   = gamma
                                    , tol                     = tol
                                    , decision_function_shape ='ovr')

            # Write into csv
            write_results_csv(output, result, labels)
        
        # Run a single simulation
        else:
            # Run simulation
            result = svm(X_train        = X_train
                         , Y_train      = Y_train
                         , X_test       = X_test
                         , Y_test       = Y_test
                         , kernel       = kernel
                         , C            = 1.0
                         , degree       = degree
                         , gamma        = gamma
                         , tol          = tol
                         , decision_function_shape='ovr')
            
            # Write into csv
            write_results_csv(output, result, labels)
        


# In[6]:

# Run the model
run_model(inp_yaml,X_train,Y_train,X_test,Y_test)


# In[ ]:



