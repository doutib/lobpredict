
# coding: utf-8

# In[8]:

import imp
import yaml
import csv
#modl = imp.load_source('read_model_yaml', '../read_model_yaml.py')
modl = imp.load_source('read_model_yaml', '../../read_model_yaml.py')


# In[10]:

a = modl.read_model_yaml("../../model/spec/SS/SS_RF_1.yaml")


# In[11]:

a.keys()


# In[12]:

a.values()


# In[13]:

# Run model
import imp
import yaml
import sys
modl = imp.load_source('read_model_yaml', '../read_model_yaml.py')

inp_yaml = sys.argv[1]

def write(filename,results,labels,level=1):
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
    level : int, either 1 or 2
        first dimension of the 'results' array
        
    """
    ## # Write into csv file
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(labels)
        if level == 2:
            writer.writerows(results)
        else:
            writer.writerow(results)

    

def run_model(inp_yaml):
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
    
    
    # TODO: Fix this
    X_train = 
    Y_train = 
    X_test  = 
    Y_test  = 
    
    yaml_params = modl.read_model_yaml(inp_yaml)
    
    # TODO: define output file name
    filename = 
    
    if yaml_params["model_type"] = "RF":
        n_estimators  = yaml_params["parameters"]["n_estimators"]
        criterion     = yaml_params["parameters"]["criterion"]   
        max_features  = yaml_params["parameters"]["max_features"]          
        max_depth     = yaml_params["parameters"]["max_depth"]      
        n_jobs        = yaml_params["parameters"]["n_jobs"]
        
        # Define labels of output
        labels = ["logloss",
                  "miss_err",
                  "prec",
                  "recall",
                  "f1",
                  "C",
                  "n_estimators",
                  "criterion",   
                  "max_features",
                  "max_depth"]
        
        # Run many simulations in parallel using as many cores as necessary
        if yaml_params["simulations"]:
            # Run simulation
            result = rf_simulation(X_train      = X_train,
                                  Y_train      = Y_train,
                                  X_test       = X_test,
                                  Y_test       = Y_test,
                                  n_estimators = n_estimators,
                                  criterion    = criterion,
                                  max_features = max_features,
                                  max_depth    = max_depth)
            # Write into csv
            write(filename,result,labels,level=1)
        
        # Run a single simulation
        else:
            # Run simulation
            result = rf(X_train      = X_train,
                       Y_train      = Y_train,
                       X_test       = X_test,
                       Y_test       = Y_test,
                       n_estimators = n_estimators,
                       criterion    = criterion,
                       max_features = max_features,
                       max_depth    = max_depth)
            # Write into csv
            write(filename,result,labels,level=2)
        

# Run the model!    
run_model(inp_yaml = inp_yaml)


# In[ ]:



