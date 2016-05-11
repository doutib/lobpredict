
# coding: utf-8

# # Report

# ## Data Cleaning
# 
# The data we used in this project comes from the limit order book (LOB) and the message book of Apple stock. In LOB, there are 10 levels of ask/bid prices and volumes. The data is quite clean since there are no missing values nor outliers. We used 9 features as Kercheval and Zhang in their study. The first 40 columns are the original ask/bid prices and volumns after renaming. Then the next four features are in the time insensitive set. It contains bid-ask spreads and mid-prices, price differences, mean prices and volumes accumulated differences. The last four are time-sensitive features including price and volume derivatives, average intensity of each type, relative intensity indicators, accelerations(market/limit). 
# 
# In time-sensitive features, the biggest problem we encountered is the choice of $\Delta t$. Also, the choice of $\Delta t$ is correlated with labels. Mainly we would like to predict stock prices by mid-price movement or price spread crossing. Price spread crossing is defined as following. (1) An upward price spread crossing appears when the best bid price at $t+\Delta t$ is greater than the best ask price at time $t$, which is $P_{t+\Delta t}^{Bid}>P_{t}^{Ask}$. (2) A downward price spread crossing appears when the best ask price at $t+\Delta t$ is smaller than the best bid price at time $t$, which is $P_{t+\Delta t}^{Ask}>P_{t}^{bid}$. (3) If the spreads of best ask price and best bid price are still crossing each other, than we consider it is no price spread crossing, which is stable status. In this case, compared to mid-price movements, price spread crossing is less possible to have upward or downward movements, particularly in high frequency trading since big $\Delta t$ might be useless. According to our test, even we use 1000 rows as $\Delta t$, we still get $92%$ stables.
# 
# 
# 

# ## Data preprocessing
# 
# Data 

# 

# In[ ]:



