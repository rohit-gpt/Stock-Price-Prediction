#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:07:10 2019

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

sns.set_style('whitegrid')

def ffloat(string):
    if string is None:
        return np.nan
    if type(string)==float or type(string)==np.float64:
        return string
    if type(string)==int or type(string)==np.int64:
        return string
    return fast_float(string.split(" ")[0].replace(',','').replace('%',''), default=np.nan)

def ffloat_list(string_list):
    return list(map(ffloat,string_list))

def remove_multiple_spaces(string):
    if type(string)==str:
        return ' '.join(string.split())
    return string

from IPython.core.display import HTML
HTML("<b>Rendered HTML</b>")

response = requests.get("https://www.moneycontrol.com/india/stockpricequote/auto-2-3-wheelers/heromotocorp/HHM", timeout=240)
page_content = BeautifulSoup(response.content, "html.parser")
HTML(str(page_content.find("h1")))

price_div = page_content.find("ul",attrs={"class":'clearfix op_list'})
HTML(str(price_div))