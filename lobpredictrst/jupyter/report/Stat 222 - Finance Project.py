
# coding: utf-8

# # Stat 222: Finance Project (Spring 2016)

# ##### Team: Fengshi Niu, Shamindra Shrotriya, Yueqi (Richie) Feng, Thibault Doutre

# ## Abstract

# We created an open source Python Package 'lobpredictrst' to predict mid price movements for the AAPL LOB stock
# - In data preprocessing part, we follow closely to the  Kercheval and Zhang (2015). We create categorical (up, stationary, down) price movement labels using midprice change and bid-ask price spread crossing between delta t, which we pick, respectively and create features according to the paper's feature 1-6.
# - We use SVM and random forest with the standard cross-validation procedure to get the prediction models and create a straightforward trading strategy accordingly
# - We use several measures to evaluate the precision of our different prediction models and the profit generated from the trading strategies. Our best trading strategy is selected according to the net profit.

# ## Introduction and Background

# Describe what the project is about and roughly our approach in each of the following 5 sections
# 
# - In this project, we analyze the limit order book (LOB) of AAPL, fit a predictive model of price movements under 30 time-stamped scale using random forest and SVM respectively, 30? millsecond according to data from 9:30 to 11:00, and create a high frequency trading strategy based on it. In the end, we run the strategy on data from 11:00 to 12:28 and evaluate the net profit
# - The following sections are data preprocessing, model fitting, model assessment, and trading algorithm implementation

# ## Data Preprocessing

# ### Data Description

# The data we used in this project comes from the limit order book (LOB) and the message book of Apple stock. In LOB, there are 10 levels of ask/bid prices and volumes. The data is quite clean since there are no missing values nor outliers.
# 
# - Add the time and limitations (only one morning, split chronologically, does not allow for seasonality effects)
# 
# - Mention that we experimented with the MOB data but chose not to add the features in the end. It was sparser than originally thought

# ### Transformation Process
# 
# We used 6 features as Kercheval and Zhang in their study. The first 40 columns are the original ask/bid prices and volumns after renaming. Then the next four features are in the time insensitive set. It contains bid-ask spreads and mid-prices, price differences, mean prices and volumes accumulated differences. The last four are time-sensitive features including price and volume derivatives, average intensity of each type, relative intensity indicators, accelerations(market/limit). 
# 
# - Insert the Kercheval and Zhang table (screenshot) here. With a caption acknowleding the source paper.
# 
# - Insert summary statistics for key columns in the training set to give a exploratory feel of the underlying raw features.
# 
# In time-sensitive features, the biggest problem we encountered is the choice of $\Delta t$. Also, the choice of $\Delta t$ is correlated with labels. Mainly we would like to predict stock prices by mid-price movement or price spread crossing. Price spread crossing is defined as following. (1) An upward price spread crossing appears when the best bid price at $t+\Delta t$ is greater than the best ask price at time $t$, which is $P_{t+\Delta t}^{Bid}>P_{t}^{Ask}$. (2) A downward price spread crossing appears when the best ask price at $t+\Delta t$ is smaller than the best bid price at time $t$, which is $P_{t+\Delta t}^{Ask}>P_{t}^{bid}$. (3) If the spreads of best ask price and best bid price are still crossing each other, than we consider it is no price spread crossing, which is stable status. In this case, compared to mid-price movements, price spread crossing is less possible to have upward or downward movements, particularly in high frequency trading since big $\Delta t$ might be useless. According to our test, even we use 1000 rows as $\Delta t$, we still get $92\%$ stationary labels.

# ### Deciding dt - Methodology and Reasoning

# In the previous section, we explain the importance of picking good $\Delta t$. In this section, we explain how we pick it in detail. 
# 
# $\Delta t$ affect our model and strategy in at least two ways:
# - In our prediction model, we use price at $t+\Delta t$ and $\Delta t$ to create our label
# - In our trading strategy, we long/short one share at some point and clear it exactly $\Delta t$ later
# 
# The tradeoff we are facing can be understanded this way:
# 
# In high frequency trade, any current information for profit opportunity is only valueable within an extremely short period of time. And any profit opportunity is completely exploited within a few millsecond. This implies:
# - Cost for using large $\Delta t$
#     - prediction models based on ancient information will hardly work, because we essentially using no information
#     - the trading strategy with large $\Delta t$ won't generate profit. Even the prediction model in some sense find a profitable opportunity, when we excute our transaction, the opportunity has already be taken by other people
# 
# There is an important benefit of large $\Delta t$. Very small $\Delta t$ results in extremely high proportion of 'stationary' label, meaning that the price measure doesn't change. Highly imbalanced label makes machine learning algorithm too easy and make the information less efficiently used. It actually induces the machine learning algorithm to cheat by ignoring the features and only predicting too much 'stationary'.
# - Benefit for using large $\Delta t$
#     - solve the label imbalance problem and help machine learning algorithms to learn the data more efficiently
# 
# In practice, we look at the proportion of each category of labels 'up', 'stationary', 'down' for different $\Delta t$. The plot is shown below. Looking at the graph, we see that the proportion of 'stationary' falls down quickly for Midprice lables but very slowly for bid-ask spread crossing. In the end, we pick $\Delta t_{MD} = 30$, because the proportion of 'stationary' falls down quickly before 30 and and slowly after 30. 30 is too large and gives us litter enough 'stationary'. We pick $\Delta t_{SC} = 1000$, because the proportion at 1000 are about 0.33, 0.33, 0.33. We really like this balance property. However, we acknowlege that it is probably too large. However, as a machine learning excercise, we decide to care more about whether the algorithm is going to work better or not and sacrifices some really essential practical issues.
# 
# For future extensions of this work, we can consider picking $\Delta t_{SC} = 30$ (say) and oversampling up/ down movements to get a better data for modelling purposes. This would be another way to help mitigate the risk of the class imbalance problem inherent in the SC approach.
# 
# 
# 
# 
# Delete the following old skeleton version
# '''
# - Describe (using graphs and tables) the process used to determine the final dt column
# - Acknowledge that Johnny allowed us to use rows instead of time to determine dt
# - The is all of the exploratory analysis performed by FN to determine the 'optimal' dt
# - Explain that we settled on dt = 30 rows i.e. from the graph it is clear that the **optimal dt and why**
# - Explain that we looked at spread crossing but did not model on it becuase we could not get a good spread of the labels - pretty much all of them were stationary
# - We used dt = 1000 for SC because it gave same degree of label stability (in distribution) as MP crossing
# - We understand that this may be too high for actual HFT purposes but for future extensions of this work, we can consider picking dt = 30 (say) and oversampling up/ down movements to get a better data for modelling purposes. This would be another way to help mitigate the risk of the class imbalance problem inherent in the SC approach.
# '''
# 
# <img src="delta_t_MP.png">
# <img src="delta_t_SC.png">

# ### Preparation of transformed data for model fitting

# - Explain the split of the data - ensure that the proportions are preserved
#     - **train**
#     - **test**
#     - **validation**
#     - **train + test**
#     - **train + test + validation**
#     - **strategy**    
# - We tried it on Chronological, but then decided to use a simple random shuffling approach
# - There is a notable limitation of this - namely that there are very highly correlated features occuring in chronolological chunks. By randomly shiffling, we not only stabilise the distribution of labels (good) but also the **emperical distribution CHECK THIS!!!** of features (not ideal). For future extensions of this work try to shuffling by these chronologically consecutive chunks across the training and test/ validation sets.
# - Explain that the 12:00-12:30PM data is discarded per Johnny's suggestion

# ## Model Fitting

# ### Background of SVM - Theoretical Background

# - From ISLR include a description of SVM
# - Explain its core strengths and weaknesses

# ### Background of Random Forests - Theoretical Background

# - From ISLR include a description of RF
# - Explain its core strengths and weaknesses and why we use it i.e. adapts well for numerical features that are not necessarily linearly separable

# ### Approach for Testing Models

# - Explain the key parameter changes for SVM and RF
# - TD: Go through YAML files and produce table summarising key parameter changes

# ## Model Assessment

# ### SVM Output

# - Insert summary tables here i.e. F1, precision and recall
# - Explain how the output improves as we change our summary metrics

# ### RF Output

# - Insert summary tables here i.e. F1, precision and recall
# - Explain how the output improves as we change our summary metrics

# ### GBM Output (If we have the time)

# - Insert summary tables here i.e. F1, precision and recall
# - Explain how the output improves as we change our summary metrics

# ## Interpretations

# - Understand the key features driving the model selection i.e. look at variable importance
# - Take the top few features and fit a simple logistic regression for up-stationary-down movements

# ## Trading Strategies Implementation
# 
# According to the requirement of the project:" 1. You can place only market orders, and assume that they can be executed immediately at the best bid/ask. 2. Your position can only be long/short at most one share." These requirement means 
# - We can only buy at best ask price and sell at best bid price in the future
# - Whenever we our current position is not 0, we cannot long/short a new share
# 
# We construct two trading strategies based on predictions of our best random forest models with bid-ask spread crossing labels. The two strategies are called simple strategy and finer strategy. 
# - Whenever we make a trading decision on a new share, we long a new share if the model prediction is up, short a new share if the prediction is down, don't do anything for any new share if the prediction is stationary. 
# - We clear any old share $\Delta t$ timestamps after we long/short it originally.
# - In the simpler strategy, we only take a trading decision on a new share every other $\Delta t$ timestamp, i.e. we consider whether to do anything at $t_0, t_{\Delta t}, \ldots, t_{n\Delta t}, \ldots$
# - In the finer strategy, we take a trading decision on a new share whenever our position is 0.

# ## Discussion

# ## Conclusion

# - From a machine learning perspective, the black box approaches included SVM and RF. We also tried GBM
# - The ensemble methods of RF and GBM gave the best prediction as quantified by the F1, recall and precision scores
# - We designed a simple trading strategy and tested it on the data from 11:00-12:00 and noted that this transparent approach yielded an accuracy of 70% compared to the more black box approaches which gave us 85% accuracy
# - We note that the models should be built over a longer period of time and also randomly split by time, to ensure that we adapt to seasonal trends in a consistent manner. Perhaps longitudinal classification methods could be applied here to capture the temporal component in LOB price movements

# ## Acknowledgments

# - We would like to thank Jarrod Millman and Matthew Brett for helping us set up the Python package `lobpredictrst`
# - We would like to thank Johnny for explaining the theoretical underpinnings of RF and SVM
