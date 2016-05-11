
# coding: utf-8

# In[1]:

#first commit -Richie
import pandas as pd
import numpy as np


# In[2]:

data_message = pd.read_csv('../../data/raw_data/AAPL_05222012_0930_1300_message.tar.gz',compression='gzip')
data_lob = pd.read_csv('../../data/raw_data/AAPL_05222012_0930_1300_LOB_2.tar.gz',compression='gzip')


# In[3]:

#drop redundant time
col_names=data_lob.columns
delete_list=[i for i in col_names if 'UPDATE_TIME' in i]
for i in delete_list:
    data_lob=data_lob.drop(i,1)


# In[4]:

#functions for renaming
def rename(txt):
    txt=txt[16:].split('..')[0]
    index=0
    ask_bid=''
    p_v=''
    if txt[-2].isdigit():
        index=txt[-2:]
    else:
        index=txt[-1]
    if txt[:3]=="BID":
        ask_bid='bid'
    else:
        ask_bid='ask'
    if txt[4:9]=="PRICE":
        p_v='P'
    else:
        p_v='V'
    return('_'.join([p_v,index,ask_bid]))


# In[5]:

#rename columns
col_names=data_lob.columns
new_col_names=[]
new_col_names.append('index')
new_col_names.append('Time')
for i in col_names[2:]:
    new_col_names.append(rename(i))
len(new_col_names)
data_lob.columns=new_col_names


# In[6]:

#feature: bid-ask spreads and mid price
for i in list(range(1, 11)):
    bid_ask_col_name='_'.join(['spreads',str(i)])
    p_i_ask='_'.join(['P',str(i),'ask'])
    p_i_bid='_'.join(['P',str(i),'bid'])
    data_lob[bid_ask_col_name]=data_lob[p_i_ask]-data_lob[p_i_bid]
    
    mid_price_col_name = '_'.join(['mid_price',str(i)])
    data_lob[mid_price_col_name]=(data_lob[p_i_ask]+data_lob[p_i_bid])/2
    


# In[7]:

#convert time
def timetransform(r):
    # transform the time to millisecond, starting from 0
    timestr = r
    return (int(timestr[11:13]) - 9) * 60**2 + (int(timestr[14:16]) - 30) * 60 + float(timestr[17:])
time = list(data_lob['Time']) 
time_new = [timetransform(i) for i in time] 
data_lob["Time"] = time_new


# In[8]:

time = list(data_message['Time']) 
time_new = [timetransform(i) for i in time] 
data_message["Time"] = time_new


# In[9]:

#price difference 
data_lob['P_diff_ask_10_1']=data_lob['P_10_ask']-data_lob['P_1_ask']
data_lob['P_diff_bid_1_10']=data_lob['P_1_bid']-data_lob['P_1_bid']

for i in list(range(1, 10)):
    P_diff_ask_i_name='_'.join(['P','diff','ask',str(i),str(i+1)])
    P_diff_bid_i_name='_'.join(['P','diff','bid',str(i),str(i+1)])
    P_i_ask='_'.join(['P',str(i),'ask'])
    P_i1_ask='_'.join(['P',str(i+1),'ask'])
    P_i_bid='_'.join(['P',str(i),'bid'])
    P_i1_bid='_'.join(['P',str(i+1),'bid'])
    data_lob[P_diff_ask_i_name]=abs(data_lob[P_i1_ask]-data_lob[P_i_ask])
    data_lob[P_diff_bid_i_name]=abs(data_lob[P_i1_bid]-data_lob[P_i_bid])


# In[10]:

#mean price and volumns
p_ask_list=['_'.join(['P',str(i),'ask']) for i in list(range(1, 11))]
p_bid_list=['_'.join(['P',str(i),'bid']) for i in list(range(1, 11))]
v_ask_list=['_'.join(['V',str(i),'ask']) for i in list(range(1, 11))]
v_bid_list=['_'.join(['V',str(i),'bid']) for i in list(range(1, 11))]


data_lob['Mean_ask_price']=0.0
data_lob['Mean_bid_price']=0.0
data_lob['Mean_ask_volumn']=0.0
data_lob['Mean_bid_volumn']=0.0
for i in list(range(0, 10)):
    data_lob['Mean_ask_price']+=data_lob[p_ask_list[i]]
    data_lob['Mean_bid_price']+=data_lob[p_bid_list[i]]
    data_lob['Mean_ask_volumn']+=data_lob[v_ask_list[i]]
    data_lob['Mean_bid_volumn']+=data_lob[v_bid_list[i]]

data_lob['Mean_ask_price']/=10
data_lob['Mean_bid_price']/=10
data_lob['Mean_ask_volumn']/=10
data_lob['Mean_bid_volumn']/=10


# In[11]:

#accumulated difference
p_ask_list=['_'.join(['P',str(i),'ask']) for i in list(range(1, 11))]
p_bid_list=['_'.join(['P',str(i),'bid']) for i in list(range(1, 11))]
v_ask_list=['_'.join(['V',str(i),'ask']) for i in list(range(1, 11))]
v_bid_list=['_'.join(['V',str(i),'bid']) for i in list(range(1, 11))]

data_lob['Accum_diff_price']=0.0
data_lob['Accum_diff_volumn']=0.0
for i in list(range(0, 10)):
    data_lob['Accum_diff_price']+=data_lob[p_ask_list[i]]-data_lob[p_bid_list[i]]
    data_lob['Accum_diff_volumn']+=data_lob[v_ask_list[i]]-data_lob[v_bid_list[i]]

data_lob['Accum_diff_price']/=10
data_lob['Accum_diff_volumn']/=10


# In[12]:

# #price and volumn derivatives

# p_ask_list=['_'.join(['P',str(i),'ask']) for i in list(range(1, 11))]
# p_bid_list=['_'.join(['P',str(i),'bid']) for i in list(range(1, 11))]
# v_ask_list=['_'.join(['V',str(i),'ask']) for i in list(range(1, 11))]
# v_bid_list=['_'.join(['V',str(i),'bid']) for i in list(range(1, 11))]

# #data_lob['Time_diff']=list(np.zeros(30)+1)+list(np.array(data_lob['Time'][30:])-np.array(data_lob['Time'][:-30]))

# for i in list(range(0, 10)):
#     P_ask_i_deriv='_'.join(['P','ask',str(i+1),'deriv'])
#     P_bid_i_deriv='_'.join(['P','bid',str(i+1),'deriv'])
#     V_ask_i_deriv='_'.join(['V','ask',str(i+1),'deriv'])
#     V_bid_i_deriv='_'.join(['V','bid',str(i+1),'deriv'])
    
#     data_lob[P_ask_i_deriv]=list(np.zeros(30))+list(np.array(data_lob[p_ask_list[i]][30:])-np.array(data_lob[p_ask_list[i]][:-30]))
#     data_lob[P_bid_i_deriv]=list(np.zeros(30))+list(np.array(data_lob[p_bid_list[i]][30:])-np.array(data_lob[p_bid_list[i]][:-30]))
#     data_lob[V_ask_i_deriv]=list(np.zeros(30))+list(np.array(data_lob[v_ask_list[i]][30:])-np.array(data_lob[v_ask_list[i]][:-30]))
#     data_lob[V_bid_i_deriv]=list(np.zeros(30))+list(np.array(data_lob[v_bid_list[i]][30:])-np.array(data_lob[v_bid_list[i]][:-30]))  
    
#     data_lob[P_ask_i_deriv]/=30
#     data_lob[P_bid_i_deriv]/=30
#     data_lob[V_ask_i_deriv]/=30
#     data_lob[V_bid_i_deriv]/=30


# #price and volumn derivatives

p_ask_list=['_'.join(['P',str(i),'ask']) for i in list(range(1, 11))]
p_bid_list=['_'.join(['P',str(i),'bid']) for i in list(range(1, 11))]
v_ask_list=['_'.join(['V',str(i),'ask']) for i in list(range(1, 11))]
v_bid_list=['_'.join(['V',str(i),'bid']) for i in list(range(1, 11))]

#data_lob['Time_diff']=list(np.zeros(30)+1)+list(np.array(data_lob['Time'][30:])-np.array(data_lob['Time'][:-30]))

for i in list(range(0, 10)):
    P_ask_i_deriv='_'.join(['P','ask',str(i+1),'deriv'])
    P_bid_i_deriv='_'.join(['P','bid',str(i+1),'deriv'])
    V_ask_i_deriv='_'.join(['V','ask',str(i+1),'deriv'])
    V_bid_i_deriv='_'.join(['V','bid',str(i+1),'deriv'])
    
    data_lob[P_ask_i_deriv]=list(np.zeros(1000))+list(np.array(data_lob[p_ask_list[i]][1000:])-np.array(data_lob[p_ask_list[i]][:-1000]))
    data_lob[P_bid_i_deriv]=list(np.zeros(1000))+list(np.array(data_lob[p_bid_list[i]][1000:])-np.array(data_lob[p_bid_list[i]][:-1000]))
    data_lob[V_ask_i_deriv]=list(np.zeros(1000))+list(np.array(data_lob[v_ask_list[i]][1000:])-np.array(data_lob[v_ask_list[i]][:-1000]))
    data_lob[V_bid_i_deriv]=list(np.zeros(1000))+list(np.array(data_lob[v_bid_list[i]][1000:])-np.array(data_lob[v_bid_list[i]][:-1000]))  
    
    data_lob[P_ask_i_deriv]/=1000
    data_lob[P_bid_i_deriv]/=1000
    data_lob[V_ask_i_deriv]/=1000
    data_lob[V_bid_i_deriv]/=1000


# In[ ]:

#set labels
diff=data_lob['mid_price_1']
diff_30=np.array(diff[30:])-np.array(diff[:-30])
label=[]
for i in diff_30:
    if i>0.01:
        label.append('1')
    elif i<(-0.01):
        label.append('-1')
    else:
        label.append('0')


data_lob['labels']=label+list(np.zeros(30))


# In[13]:

#set spread crossing labels
p_now_bid = list(data_lob['P_1_bid'][:-1000])
p_now_ask = list(data_lob['P_1_ask'][:-1000])

p_next_bid=list(data_lob['P_1_bid'][1000:])
p_next_ask=list(data_lob['P_1_ask'][1000:])

label_SC=[]
for i in range(len(p_now_bid)):
    if p_next_bid[i]>=p_now_ask[i]:
        label_SC.append('+1')
    elif p_next_ask[i]<=p_now_bid[i]:
        label_SC.append('-1')
    else:
        label_SC.append('0')


data_lob['labels']=label_SC+list(np.zeros(1000))


# In[14]:

#drop first and last 30 rows
data_out=data_lob[:-1000]
data_out=data_out[1000:]

# data_out=data_lob[:-30]
# data_out=data_out[30:]



# In[15]:

# split train and test
# split2=203350-30
# split1=int(split2*0.5)
# split_test=split1+int(split2*0.25)
# split3=279656-30


# shuttle=data_out[:split2].reindex(np.random.permutation(data_out[:split2].index))

# train=shuttle[:split1]
# test=shuttle[split1:split_test]
# train_test=shuttle[:split_test]
# validation=shuttle[split_test:]
# strategy_validation=data_out[split2:split3]

# split2=203350-30
# split1=int(split2*0.5)
# split_test=split1+int(split2*0.25)
# split3=279656-30

# train=data_out[:split1]
# test=data_out[split1:split_test]
# train_test=data_out[:split_test]
# validation=data_out[split_test:split2]
# strategy_validation=data_out[split2:split3]


#split for SC
# split2=203349-1000
# split1=int(split2*0.5)
# split_test=split1+int(split2*0.25)
# split3=279656-1000

# train=data_out[:split1]
# test=data_out[split1:split_test]
# train_test=data_out[:split_test]
# validation=data_out[split_test:split2]
# strategy_validation=data_out[split2:split3]

#split shuttle for SC

split2=203349-1000
split1=int(split2*0.5)
split_test=split1+int(split2*0.25)
split3=279656-1000


shuttle=data_out[:split2].reindex(np.random.permutation(data_out[:split2].index))

train=shuttle[:split1]
test=shuttle[split1:split_test]
train_test=shuttle[:split_test]
validation=shuttle[split_test:]
strategy_validation=data_out[split2:split3]



# In[17]:


train_sample=train.sample(int(0.1*len(train)))
test_sample=test.sample(int(0.1*len(test)))
train_test_sample=train_test.sample(int(0.1*len(train_test)))
validation_sample=validation.sample(int(0.1*len(validation)))
strategy_validation_sample=strategy_validation.sample(int(0.1*len(strategy_validation)))



# In[22]:

label_percent(np.array(train_test_sample['labels']).astype('int'))


# In[ ]:

data_out[data_out['Time']>(5400+3600)]


# In[ ]:

diff=data_lob['mid_price_1']
diff_10=np.array(diff[10:])-np.array(diff[:-10])
diff_20=np.array(diff[20:])-np.array(diff[:-20])
diff_30=np.array(diff[30:])-np.array(diff[:-30])
diff_40=np.array(diff[40:])-np.array(diff[:-40])
print('Delta=10:')
label_percent(diff_10)
print('Delta=20:')
label_percent(diff_20)
print('Delta=30:')
label_percent(diff_30)
print('Delta=40:')
label_percent(diff_40)


# Delta=10:
# Pos:0.31; Neg:0.33; Zero:0.36; 
# Delta=20:
# Pos:0.39; Neg:0.42; Zero:0.19;
# Delta=30:
# Pos:0.42; Neg:0.46; Zero:0.11;
# Delta=40:
# Pos:0.44; Neg:0.48; Zero:0.07;


# In[19]:

def label_percent(v):
    l=len(v)
    pos=sum(x > 0 for x in v)/l
    neg=sum(x < 0 for x in v)/l
    zero=sum(x == 0 for x in v)/l
    print('Pos:'+str("{0:.2f}".format(pos))+"; Neg:"+str("{0:.2f}".format(neg))+'; Zero:'+str("{0:.2f}".format(zero))+';')


# In[24]:

# train.to_csv('../../data/output/model_clean_data/train.csv')
# test.to_csv('../../data/output/model_clean_data/test.csv')
# validation.to_csv('../../data/output/model_clean_data/validation.csv')
# train_test.to_csv('../../data/output/model_clean_data/train_test.csv')
# strategy_validation.to_csv('../../data/output/model_clean_data/strategy_validation.csv')

train_sample.to_csv('../../data/output/model_clean_data/train.csv')
test_sample.to_csv('../../data/output/model_clean_data/test.csv')
validation_sample.to_csv('../../data/output/model_clean_data/validation.csv')
train_test_sample.to_csv('../../data/output/model_clean_data/train_test.csv')
strategy_validation_sample.to_csv('../../data/output/model_clean_data/strategy_validation.csv')


# In[ ]:

data_out[:1000].to_csv('/Users/Richie/Desktop/sample.csv')


# In[64]:

#combine train test validation
data_train_test1 = pd.read_csv('../../data/output/model_clean_data/SC_shuffle/train_test.tar.gz',compression='gzip')
data_validation1 = pd.read_csv('../../data/output/model_clean_data/SC_shuffle/validation.tar.gz',compression='gzip')

combine1= pd.concat([data_train_test1,data_validation1],join='inner')
combine1.to_csv('../../data/output/model_clean_data/train_test_validation_1.csv')

data_train_test2 = pd.read_csv('../../data/output/model_clean_data/SC_chrono/train_test.tar.gz',compression='gzip')
data_validation2 = pd.read_csv('../../data/output/model_clean_data/SC_chrono/validation.tar.gz',compression='gzip')

combine2= pd.concat([data_train_test2,data_validation2],join='inner')
combine2.to_csv('../../data/output/model_clean_data/train_test_validation_2.csv')

data_train_test3 = pd.read_csv('../../data/output/model_clean_data/SC_sample/train_test.tar.gz',compression='gzip')
data_validation3 = pd.read_csv('../../data/output/model_clean_data/SC_sample/validation.tar.gz',compression='gzip')

combine3= pd.concat([data_train_test3,data_validation3],join='inner')
combine3.to_csv('../../data/output/model_clean_data/train_test_validation_3.csv')

data_train_test4 = pd.read_csv('../../data/output/model_clean_data/MP_shuffle/train_test.tar.gz',compression='gzip')
data_validation4 = pd.read_csv('../../data/output/model_clean_data/MP_shuffle/validation.tar.gz',compression='gzip')

combine4= pd.concat([data_train_test4,data_validation4],join='inner')
combine4.to_csv('../../data/output/model_clean_data/train_test_validation_4.csv')

data_train_test5 = pd.read_csv('../../data/output/model_clean_data/MP_chrono/train_test.tar.gz',compression='gzip')
data_validation5 = pd.read_csv('../../data/output/model_clean_data/MP_chrono/validation.tar.gz',compression='gzip')

combine5= pd.concat([data_train_test5,data_validation5],join='inner')
combine5.to_csv('../../data/output/model_clean_data/train_test_validation_5.csv')

data_train_test6 = pd.read_csv('../../data/output/model_clean_data/MP_sample/train_test.tar.gz',compression='gzip')
data_validation6 = pd.read_csv('../../data/output/model_clean_data/MP_sample/validation.tar.gz',compression='gzip')

combine6= pd.concat([data_train_test6,data_validation6],join='inner')
combine6.to_csv('../../data/output/model_clean_data/train_test_validation_6.csv')


# In[ ]:



