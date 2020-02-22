# -*- coding: utf-8 -*-
"""
@author: Yulu Su
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import statsmodels.formula.api as sm

import files

#read CME lean hogs futures' data from excel
lh = files.read_data(r'LeanHogsFutures.xlsx')

#read CME corn futures' data from excel
cn = files.read_data(r'CornFutures.xlsx')

#read CME live cattle futures' data from excel
lc = files.read_data(r'LivecCattleFutures.xlsx')

#read S&P 500 data from excel
sp = files.read_data(r'S&P500.xlsx')

#read Gold Commodities data from excel
gc = files.read_data(r'GoldCommodity.xlsx')

#read Cotton Futures data from excel
ct = files.read_data(r'CottonFutures.xlsx')

#combine Close price of lean hogs futures and corn future to a dataframe
start_time = dt.datetime(2000, 1, 1)
data = pd.DataFrame({'LeanHogs':lh['Close'][lh.index > start_time]})
data = data.join(pd.DataFrame({'Corn' : cn['Close'][cn.index > start_time]}))
data = data.join(pd.DataFrame({'LiveCattle' : lc['Close'][lc.index > start_time]}))
data = data.join(pd.DataFrame({'SP500' : sp['Close'][sp.index > start_time]}))
data = data.join(pd.DataFrame({'Gold' :gc['Close'][gc.index > start_time]}))
data = data.join(pd.DataFrame({'Cotton' : ct['Close'][ct.index > start_time]}))

#forward fill the NA
data = data.fillna(method='ffill')

#plot the Close price
data.plot(subplots=True, grid=True, style='b', figsize=(8, 6))

#get returns
rets = np.log(data / data.shift(1))

#plot the returns
rets.plot(subplots=True, grid=True, style='b', figsize=(8, 6))

xdat = rets['LeanHogs']
y1dat = rets['Corn']
y2dat = rets['LiveCattle']
y3dat = rets['SP500']
y4dat = rets['Gold']
y5dat = rets['Cotton']

result = sm.ols(formula="LeanHogs ~ Corn + LiveCattle + SP500 + Gold + Cotton", data=rets).fit()
print(result.params)
print(result.summary())


plt.plot(xdat, y1dat, 'r.')
ax = plt.axis() # grab axis values
x = np.linspace(ax[0], ax[1] + 0.01)
plt.plot(x, result.params[0] + result.params[1] * x, 'b', lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel('Corn Future Price')
plt.ylabel('Lean Hogs Price')


rets.corr()


#rooling correlation plot
fig, axes = plt.subplots(len(rets.columns), len(rets.columns), figsize=(10,2.5), 
                         dpi=100, sharex=True, sharey=True)
colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue']

for i in range(0,len(rets.columns),1):
    for j in range(0,len(rets.columns),1):
        col = rets.columns[j]
        
        if i > j:  
            plt.subplot(len(rets.columns), len(rets.columns), i*6 + j + 1 )
            plt.plot(rets[rets.columns[i]].rolling(252).corr(rets[rets.columns[j]]))
        
        if i == j:             
            x = rets[col]
            axes[i,j].hist(x, bins=70, label=str(col), color=colors[i])
         
        if i < j:  
            formula_str = rets.columns[i] + ' ~ ' + rets.columns[j]
            result = sm.ols(formula=formula_str, data=rets).fit()
            print(result.params)
            print(result.summary())
            plt.subplot(len(rets.columns), len(rets.columns), i*6 + j + 1 )
            plt.plot(rets[rets.columns[j]], rets[rets.columns[i]], 'r.')
            ax = plt.axis() # grab axis values
            x = np.linspace(ax[0], ax[1] + 0.01)
            plt.plot(x, result.params[0] + result.params[1] * x, 'b', lw=2)
            plt.grid(True)
            plt.axis('tight')
            print(formula_str)
            
        if i == 0:
            axes[i,j].set_title(rets.columns[j])
            plt.title(rets.columns[j])
            
        if j == 0:
            axes[i,j].set_ylabel(rets.columns[i])
            plt.ylabel(rets.columns[i])     

plt.suptitle('Scatterplot matrix')


##Scatterplot Matrix
#import seaborn as sns
#sns.set(style="ticks")
#
#df = rets
#sns.pairplot(df, kind="reg", diag_kind="kde",  markers="+",
#                  diag_kws=dict(shade=True))