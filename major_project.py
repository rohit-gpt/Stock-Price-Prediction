#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 19:58:26 2019

@author: rohitgupta
"""

import numpy as np
import pandas as pd
import datetime
from pylab import rcParams
from sklearn.preprocessing import StandardScaler
import time

from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import plot_model
import math
from datetime import date

df = pd.read_csv("df_RNN.csv", index_col=[0])

scaler = StandardScaler()

df1 = df.loc[:, ['open','high','low','adj_close**','sentiments']]
df1 = scaler.fit_transform(df1)

df1 = np.array(df1)

df.loc[:, 'date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')

def get_x_y(data, N, offset):
    X, y = [], []
    for i in range(offset, len(data)):
        X.append(data[i-N:i])
        y.append(data[i,1])
    X = np.array(X)
    y = np.array(y)
    
    X_train, y_train = [], []
    
    for i in range(70):
        X_train.append(X[i])
    
    y_train = y[:70]
    
    return X_train, y_train

X, y = get_x_y(df1, 7, 7)

X_train = []
for i in range(70):
    X_train.append(X[i])

y_train = y[:70]

X_test = []
for i in range(70, len(X)):
    X_test.append(X[i])
    
y_test = y[70:]

X_train = np.array(X_train)
X_test = np.array(X_test)




########### TUNING N ####################

param_label = 'N'
param_list = range(3, 60)

error_rate = {param_label: [], 'rmse': [], 'mape_pct': []}
tic = time.time()
for param in tqdm_notebook(param_list):
    
    # Split train into x and y
    x_train, y_train = get_x_y(train_scaled, param, param)

    # Split cv into x and y
    x_cv_scaled, y_cv, mu_cv_list, std_cv_list = get_x_scaled_y(np.array(train_cv['adj_close']).reshape(-1,1), param, num_train)
    
    # Train, predict and eval model
    rmse, mape, _ = train_pred_eval_model(x_train_scaled, \
                                          y_train_scaled, \
                                          x_cv_scaled, \
                                          y_cv, \
                                          mu_cv_list, \
                                          std_cv_list, \
                                          lstm_units=lstm_units, \
                                          dropout_prob=dropout_prob, \
                                          optimizer=optimizer, \
                                          epochs=epochs, \
                                          batch_size=batch_size)
    
    # Collect results
    error_rate[param_label].append(param)
    error_rate['rmse'].append(rmse)
    error_rate['mape_pct'].append(mape)
    
error_rate = pd.DataFrame(error_rate)
toc = time.time()
print("Minutes taken = " + str((toc-tic)/60.0))
error_rate



model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_scaled.shape[1],5)))
model.add(Dropout(0.5)) # Add dropout with a probability of 0.5
model.add(LSTM(units=50))
model.add(Dropout(0.5)) # Add dropout with a probability of 0.5
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)


# Plot adjusted close over time
est_df = pd.DataFrame({'est': est.reshape(-1), 
                       'date': df[-23:]['date']})

test = pd.DataFrame(columns=['date','high'])

test['date'] = df['date'][-23:]
test['high'] = y_test


ax = train.plot(x='date', y='high', style='b-', grid=True)
ax = test.plot(x='date', y='high', style='g-', grid=True)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['test', 'predictions'])
ax.set_xlabel("date")
ax.set_ylabel("USD")





est = model.predict(X_test)
