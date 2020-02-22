#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:46:28 2019

@author: rohitgupta
"""

import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
import bs4

from fastnumbers import isfloat
from fastnumbers import fast_float

from multiprocessing.dummy import Pool as ThreadPool

import matplotlib.pyplot as plt
import seaborn as sns
import json
from tidylib import tidy_document

response = requests.get("https://finance.yahoo.com/quote/GOOG/history?period1=1459449000&period2=1569954600&interval=1d&filter=history&frequency=1d", timeout=240, headers=h)

h={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.89 Safari/537.36'}

#res=requests.get(url,headers=h)

response = pd.read_html("https://finance.yahoo.com/quote/GOOG/history?period1=1459449000&period2=1569954600&interval=1d&filter=history&frequency=1d")

dataset = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])


dataset = pd.DataFrame()


for data in response:
    dataset = data

dataset = dataset[:-1]
#reverse the dataset
#dataset = dataset.reindex(index=dataset.index[::-1])
#dataset.reset_index(drop = True,inplace = True)
#dataset = dataset[dataset.Open != "80 Dividend"]


soup = BeautifulSoup(response.content, "html.parser")

arr = []
i = 0
for row in soup.find_all('tr'):
    if len(row.contents) == 7:
        for x in row.contents:
            arr.append(x.contents[0].contents[0])
        dataset.loc[i] = arr
        arr = []
        i += 1


dataset.drop(dataset.index[0], inplace=True)

#dataset['Open'] = dataset['Open'].astype(float)

dataset.isna().sum()

dataset = dataset.apply(pd.to_numeric, errors="ignore")

dataset.dtypes

dataset = dataset.reindex(index=dataset.index[::-1])

dataset.to_csv("google.csv")

dataset = dataset.loc[:,train_cols]

dataset = sc.fit_transform(dataset)

from sklearn.model_selection import train_test_split

training, testing = train_test_split(dataset, test_size=0.1, shuffle=False)

train_cols = ["Open","High","Low","Close*"]

training_set = training.iloc[:,:].values

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range= (0,1))

training_set_scaled = sc.fit_transform(training_set)




X_train = []
y_train = []

for i in range(60, len(training_set)):
    X_train.append(training_set[i-60:i,:])
    y_train.append(training_set[i, 3])
    
X_train, y_train = np.array(X_train), np.array(y_train)

#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))






from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences=True, input_shape=(X_train.shape[1], 4)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer='adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 150, batch_size = 32)








real_stock_price = testing.iloc[:, 3:4].values #Adj Close

dataset_total = dataset[:,:]

inputs = dataset[len(dataset) - len(testing) - 60:]

inputs = sc.transform(inputs)

#inputs = inputs.reshape(-1, 1)

X_test = []
for i in range(60, 70):
    X_test.append(inputs.iloc[i-60:i, :].values)
    
X_test = np.array(X_test)

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

real_stock_price = sc.fit_transform(real_stock_price)

plt.plot(real_stock_price, color='red', label="Real Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Stock Price")
plt.show()


###############################################################################
#DATA PREPROCESSING

df_train, df_test = train_test_split(dataset, train_size=0.9, test_size=0.1, shuffle=False)

train = dataset.iloc[:89,:]

X_train = train.loc[:,train_cols].values

# scale the feature MinMax, build array

#x = df_train.loc[:,train_cols].values

min_max_scaler = MinMaxScaler()

X_train = min_max_scaler.fit_transform(X_train)

x_train = min_max_scaler.fit_transform(x)
x_test = min_max_scaler.transform(df_test.loc[:,train_cols])



###############################################################################
def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))
    
    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS+i]
        y[i] = mat[TIME_STEPS+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

