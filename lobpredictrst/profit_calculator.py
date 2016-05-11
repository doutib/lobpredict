# Direct Python to plot all figures inline (i.e., not in a separate window)
%matplotlib inline

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