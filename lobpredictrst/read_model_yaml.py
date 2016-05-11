import yaml
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
