#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
from pylab import rcParams
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import plot_model
import math
from datetime import date


# In[2]:


response = pd.read_html("https://in.finance.yahoo.com/quote/GOOG/history?period1=1539109800&period2=1570645800&interval=1d&filter=history&frequency=1d")


# In[3]:


response


# In[4]:


df = pd.DataFrame()


# In[5]:


for data in response:
    df = data


# In[6]:


df


# In[7]:


df = df[:-1]


# In[8]:


df


# In[9]:


df = df.reindex(index=df.index[::-1])


# In[10]:


df


# In[11]:


test_size = 0.2
N = 9
lstm_units = 128
dropout_prob = 1
optimizer = 'nadam'
epochs = 50
batch_size = 8


# In[12]:


df.head()


# In[13]:


datetime.datetime.strptime(df['Date'][0],'%d-%b-%Y')


# In[14]:


df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%d-%b-%Y')


# In[15]:


df.head()


# In[16]:


df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]


# In[17]:


df.head()


# In[18]:


df.sort_values(by='date', inplace=True, ascending=True)


# In[19]:


df.head()


# In[20]:


rcParams['figure.figsize'] = 10, 8 # width 10, height 8
ax = df.plot(x='date', y='low', style='b-', grid=True)
ax = df.plot(x='date', y='open', style='r-', grid=True, ax=ax)
ax.set_xlabel('date')
ax.set_ylabel("USD")


# In[21]:


num_test = int(len(df) * test_size)
num_train = len(df) - num_test


# In[22]:


num_test, num_train


# In[23]:


train = df[:num_train][['date','open','high','low','adj._close**']]


# In[24]:


test = df[num_train:][['date','open','high','low','adj._close**']]


# In[25]:


train.shape


# In[26]:


test.shape


# In[27]:


train.head()


# In[28]:


test.head()


# In[29]:


scaler = StandardScaler()
train_scaled = scaler.fit_transform(np.array(train[['open','high','low','adj._close**']]))


# In[30]:


train_scaled


# In[31]:


train_scaled.shape


# In[32]:


def get_x_y(data, N, offset):
    # Split data into x (features) and y (target)
    
    x, y = [], []
    for i in range(offset, len(data)):
        x.append(data[i-N:i])
        y.append(data[i,1])
    
    x = np.array(x)
    y = np.array(y)
    
    return x, y


# In[33]:


x_train_scaled, y_train_scaled = get_x_y(train_scaled, N, N)


# In[34]:


x_train_scaled.shape


# In[35]:


y_train_scaled.shape


# In[36]:


y_train_scaled


# In[37]:


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_scaled.shape[1],4)))
model.add(Dropout(0.5)) # Add dropout with a probability of 0.5
model.add(LSTM(units=50))
model.add(Dropout(1)) # Add dropout with a probability of 0.5
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train_scaled, y_train_scaled, epochs=1, batch_size=1, verbose=2)


# In[38]:


model.summary()


# In[39]:


def get_x_scaled_y(data, N, offset):
    """
    Split data into x (features) and y (target)
    We scale x to have mean 0 and std dev 1, and return this.
    We do not scale y here.
    Inputs
        data     : pandas series to extract x and y
        N
        offset
    Outputs
        x_scaled : features used to predict y. Scaled such that each element has mean 0 and std dev 1
        y        : target values. Not scaled
        mu_list  : list of the means. Same length as x_scaled and y
        std_list : list of the std devs. Same length as x_scaled and y
    """
    
    x_scaled, y, mu_list, std_list = [], [], [], []
    for i in range(offset, len(data)):
        mu_list.append(np.mean(data[i-N:i]))
        std_list.append(np.std(data[i-N:i]))
        x_scaled.append((data[i-N:i] - mu_list[i-offset]) / std_list[i-offset])
        y.append(data[i,1])
    
    x_scaled = np.array(x_scaled)
    y = np.array(y)
    
    return x_scaled, y, mu_list, std_list


# In[40]:


x_test_scaled, y_test, mu_test_list, std_test_list = get_x_scaled_y(np.array(df[['open','high','low','adj._close**']]), N, num_train)


# In[41]:


x_test_scaled.shape


# In[42]:


y_test.shape


# In[43]:


y_test


# In[44]:


def train_pred_eval_model(x_train_scaled,                           y_train_scaled,                           x_cv_scaled,                           y_cv,                           mu_cv_list,                           std_cv_list,                           lstm_units=50,                           dropout_prob=0.5,                           optimizer='adam',                           epochs=1,                           batch_size=1):
    '''
    Train model, do prediction, scale back to original range and do evaluation
    Use LSTM here.
    Returns rmse, mape and predicted values
    Inputs
        x_train_scaled  : e.g. x_train_scaled.shape=(451, 9, 1). Here we are using the past 9 values to predict the next value
        y_train_scaled  : e.g. y_train_scaled.shape=(451, 1)
        x_cv_scaled     : use this to do predictions 
        y_cv            : actual value of the predictions
        mu_cv_list      : list of the means. Same length as x_scaled and y
        std_cv_list     : list of the std devs. Same length as x_scaled and y 
        lstm_units      : lstm param
        dropout_prob    : lstm param
        optimizer       : lstm param
        epochs          : lstm param
        batch_size      : lstm param
    Outputs
        rmse            : root mean square error
        mape            : mean absolute percentage error
        est             : predictions
    '''
    
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(x_train_scaled.shape[1], 4)))
    model.add(Dropout(dropout_prob))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(dropout_prob))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit(x_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
    
    #Do Prediction
    est_scaled = model.predict(x_cv_scaled)
    est = (est_scaled * np.array(std_cv_list).reshape(-1, 1)) + np.array(mu_cv_list).reshape(-1, 1)
    
    rmse = math.sqrt(mean_squared_error(y_cv, est))
    mape = get_mape(y_cv, est)
    
    return rmse, mape, est


# In[45]:


def get_mape(y_true, y_pred):
    # Mean Absolute Percentage Error
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[46]:


rmse, mape, est = train_pred_eval_model(x_train_scaled,                                         y_train_scaled,                                         x_test_scaled,                                         y_test,                                         mu_test_list,                                         std_test_list,                                         lstm_units=128,                                         dropout_prob=1,                                         optimizer='nadam',                                         epochs=50,                                         batch_size=8)


# In[47]:


rmse


# In[48]:


mape


# In[49]:


est


# In[50]:


y_test


# In[51]:


# Plot adjusted close over time
rcParams['figure.figsize'] = 10, 8 # width 10, height 8

est_df = pd.DataFrame({'est': est.reshape(-1), 
                       'date': df[num_train:]['date']})

ax = train.plot(x='date', y='high', style='b-', grid=True)
ax = test.plot(x='date', y='high', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")


# In[52]:


# Plot adjusted close over time, for test set only
rcParams['figure.figsize'] = 10, 8 # width 10, height 8
ax = train.plot(x='date', y='high', style='b-', grid=True)
ax = test.plot(x='date', y='high', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")
ax.set_xlim([date(2019, 9, 12), date(2019, 10, 9)])
ax.set_ylim([1150, 1250])
ax.set_title("Zoom in to test set")


# In[106]:


test


# In[ ]:




