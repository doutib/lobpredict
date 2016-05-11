import sys
import imp
import yaml
import csv
import pandas as pd
import re
from rf import *
from svm import *
modl = imp.load_source('read_model_yaml', 'read_model_yaml.py')

# Parse the YAML file location as the first parameter
inp_yaml = sys.argv[1]

def write_results_txt(filename, result):
    """
    Write results into csv file.

    Parameters
    ----------
    filename : string
        filename to output the result
    labels : list
        labels for the results, i.e. names of parameters and metrics
    """
    with open(filename, "w") as fp:
        for item in result:
            fp.write("%s\n\n" % item)


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
    yaml_params = modl.read_model_yaml(inp_yaml)

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
        train_ds_name = "train_test_validation.tar.gz"
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

        # Write into text file
        write_results_txt(output_txt_file, result)

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
        write_results_txt(output_txt_file, result)

        print("finished - SVM without simulation")

# Run the execute model code
execute_model(inp_yaml)
