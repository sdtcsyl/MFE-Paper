# -*- coding: utf-8 -*-
"""
@author: Yulu Su
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.unitroot import ADF, KPSS, DFGLS, PhillipsPerron, ZivotAndrews, VarianceRatio

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



data = 100*df['Return'].resample('M').mean() # resample daily data to monthly data
adf = ADF(data)
print(adf.summary().as_text())
kpss = KPSS(data)
print(kpss.summary().as_text())
dfgls = DFGLS(data)
print(dfgls.summary().as_text())
pp = PhillipsPerron(data)
print(pp.summary().as_text())
za = ZivotAndrews(data) 
print(za.summary().as_text())
vr = VarianceRatio(data, 12)
print(vr.summary().as_text())

print(data.describe())
print('Lean Hogs Future skewness is {}'.format(data.skew(axis=0))) 
print('Lean Hogs Future kurtosis is {}'.format(data.kurtosis(axis=0)))


import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import scipy.stats as stats
import seaborn as sns

# Plot figure with subplots of different sizes
fig = plt.figure(1)
# set up subplot grid
gridspec.GridSpec(2,2)
# large subplot
plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=2)
plt.title('LogReturn of Lean Hogs Futures Distribution')
sns.distplot(data, bins=20,kde=True)
# small subplot 3
ax0 = plt.subplot2grid((2,2), (0,1))
ax1 = plt.subplot2grid((2,2), (1,1))
probplot = sm.ProbPlot(data, dist=stats.t, fit=True)
probplot.ppplot(line='45', ax=ax0)
probplot.qqplot(line='45', ax=ax1)
ax0.set_title('P-P Plot')
ax1.set_title('Q-Q Plot')
plt.show()
