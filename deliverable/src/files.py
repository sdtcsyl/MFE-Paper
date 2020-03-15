# -*- coding: utf-8 -*-
"""
@author: Yulu Su
"""

import os 
import pandas as pd
import numpy as np
from dateutil.parser import parse

# src folder
src_path = os.path.dirname(os.path.realpath('__file__'))

dir_path = os.path.dirname(src_path) # parent folder
data_path = dir_path +'\data' #data folder
image_path = dir_path + '\image' #image folder path

def read_data(file_name):
    data = pd.read_excel(data_path + '\\' + file_name)
    data = data.set_index('Date')
    data = data.iloc[::-1] # reverse the data according to the index date
    lit = ['Open', 'High', 'Close', 'Low']  #columns name
    data = data[lit]
 
    d_one = data.index      #turn index to datetime type
    d_two = []
    d_three = []
    for i in d_one:
        d_two.append(i.strftime("%Y-%m-%d"))
    for i in range(len(d_two)):
        d_three.append(parse(d_two[i]))
    data2 = pd.DataFrame(data,index=d_three,dtype=np.float64)   
    #build a new DataFrame whose index is yyyy-mm-dd
    return data2