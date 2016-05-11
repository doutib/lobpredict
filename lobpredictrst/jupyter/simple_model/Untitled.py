
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

# In[3]:

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

var_importance


# In[ ]:



