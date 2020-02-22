#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 22:28:38 2019

@author: rohitgupta
"""

import requests
import bs4
import urllib
import webbrowser
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
from matplotlib import pyplot as plt
import datetime
from datetime import date
import time
import seaborn as sns
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

nltk.download('vader_lexicon')
sia=SIA()
#===============================================================================
#dataset
#===============================================================================
response = pd.read_html("https://finance.yahoo.com/quote/GOOG/history?period1=1555525800&period2=1571337000&interval=1d&filter=history&frequency=1d")
df = pd.DataFrame()
for data in response:
    df = data
df = df[:-1]
df = df.reindex(index=df.index[::-1])
datetime.datetime.strptime(df['Date'][0],'%b %d, %Y')
df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%b %d, %Y')
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
df.sort_values(by='date', inplace=True, ascending=True)
df['low'] = df['low'].astype(float)
df['open'] = df['open'].astype(float)
df['high'] = df['high'].astype(float)
df['adj_close**'] = df['adj_close**'].astype(float)
df.drop(['close*'],axis = 1,inplace = True)
df.drop(['volume'],axis = 1, inplace = True)
df_linear = df
df_csv = df
#===============================================================================
#google news scrapping and sentimental analysis
#================================================================================
final_scores = []
for i in range(56,100):
    curr_date = df['date'][100-i-1]
    print(type(curr_date))
    date_time = curr_date.strftime("%m-%d-%Y")
    print(date_time)
    url1='https://www.google.com/search?q='
    company_name='google+stock'
    url2='&rlz=1C1CHZL_enIN820IN820&biw=1366&bih=625&sxsrf=ACYBGNR_Yffd_Zho481hN4gWm15kmPcsFA%3A1570634911397&source=lnt&tbs=cdr%3A1%2Ccd_min%3A'
    min_date=date_time.split('-')		# in m(m)-d(d)-yyyy
    url3=min_date[0]+'%2F'+min_date[1]+'%2F'+min_date[2]
    max_date=date_time.split('-')		# in m(m)-d(d)-yyyy
    url5=max_date[0]+'%2F'+max_date[1]+'%2F'+max_date[2]
    print(url3)
    url4='%2Ccd_max%3A'
    url6='&tbm=nws'
    url=url1+company_name+url2+url3+url4+url5+url6
    print(url)
    h={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.89 Safari/537.36'}
    res=requests.get(url,headers=h)
    soup=bs4.BeautifulSoup(res.text,'lxml')
    print(url5)
    links=soup.select('.r a')
    print(links)
    x=int(input('Enter the number of search results to be opened: '))
    tabs=min(x,len(links))
    score=0
    scores=[]
    tabs = 8
    tabs = min(tabs,len(links))
    pattern = "https://economictimes.indiatimes.com"
    for i in range(tabs):
    	webbrowser.open(links[i].get('href'))
    	print(links[i].get('href'))
        
        r=requests.get(links[i].get('href'),headers=h)
        s=bs4.BeautifulSoup(r.text,'lxml')
        
        z = re.match(pattern,links[i].get('href'))
        if(z):
            mydivs = s.findAll("div", {"class": "section1"})
            for div in mydivs:
                for i in div.find_all("div", {"class": "Normal"}):
                    par = i.getText()
                    par = par[:1500]
            mydivs = s('div',limit = 15)
        else:
            par=(s('p',limit=7))
    
            y=[re.sub(r'<.+?>',r'',str(a)) for a in par]
            par=''
            for i in y:
                par=par+i
        print(par)
        print("===============================================================")
        if(len(par)>200):
            scores.append(sia.polarity_scores(par)['compound'])
            
            score+=sia.polarity_scores(par)['compound']   
        else:
            scores.append(0)
            score+=0
            
    score/=len(scores)
    final_scores.append(score)
print(score)
final_scores.insert(66,final_scores[65])
ax=plt.axes()
ax.plot(range(100),final_scores)
plt.show()

df_final_score = pd.DataFrame(final_scores)
df_final_score.to_csv('FinalScores.csv')