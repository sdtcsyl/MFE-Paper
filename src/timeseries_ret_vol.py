# -*- coding: utf-8 -*-
"""
@author: Yulu Su
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import files


#read CME lean hogs futures' data from excel
lh = pd.read_excel(files.data_path + r'\LeanHogsFutures.xlsx')
lh = lh.set_index('Date')
lh = lh.iloc[::-1] # reverse the data according to the index date

df = pd.DataFrame(lh['Close'])
df = df.dropna()
df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
#series.hist()
#series.plot.hist(bins=200)

# 42, 252 days moving average mean
df['42d'] = df['Close'].rolling(42).mean()
df['252d'] = df['Close'].rolling(252).mean()
df[['Close', '42d', '252d']].plot(figsize=(8, 5))
#plt.savefig(files.image_path + '\LH_close_42dRet_252dRet.png')

# annualized volatility
df['Mov_Vol'] = df['Return'].rolling(252).std()* math.sqrt(252)

# plot in dataframe 
df[['Close', 'Mov_Vol', 'Return']].plot(subplots=True, style='b', figsize=(8, 7), 
  title=['CME Lean Hogs Future Close Price','CME Lean Hogs Future Annualized Volatility',
         'CME Lean Hogs Future Return'])
#plt.savefig(files.image_path + '\LH_close_vol_return.png')

