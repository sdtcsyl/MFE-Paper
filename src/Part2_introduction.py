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
lh = files.read_data(r'LeanHogsFutures.xlsx')
lh = pd.DataFrame(lh['Close'].dropna())  # reverse the data according to the index date
lh['Return'] = np.log(lh['Close'] / lh['Close'].shift(1))* 100
df = lh['1992':'2004']

df = df.dropna()


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
df[['Close', 'Mov_Vol', 'Return']].plot(subplots=True,  figsize=(8, 7), 
  title=['CME Lean Hogs Future Close Price','CME Lean Hogs Future Annualized Volatility',
         'CME Lean Hogs Future Return'])
#plt.savefig(files.image_path + '\LH_close_vol_return.png')



data = lh['Close'].resample('M').mean() # resample daily data to monthly data
data = data['1992':'2004']
data =  np.log(data / data.shift(1)).dropna()*100 # d 1
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

import matplotlib.pyplot as plt
import numpy as np

rooms=['Stage 1','Stage 2', 'Stage 3']
colors=['lightblue', 'lightblue', 'lightblue']

input_files=['hoglifecycle.txt']
day_labels=['Life Cycle of Hogs']


for input_file, day_label in zip(input_files, day_labels):
    fig=plt.figure(figsize=(10,5.89))
    for line in open(input_file, 'r'):
        data=line.split()
        print(data)
        event=data[-1]
        data=list(map(float, data[:-1]))
        room=data[0]-0.48
        start=data[1]+data[2]
        end=start+data[3]
        # plot event
        plt.fill_between([room, room+0.96], [start, start], [end,end], color=colors[int(data[0]-1)], edgecolor='k', linewidth=0.5)
        # plot beginning time
        plt.text(room+0.02, start ,'{0}'.format(int(start)), va='top', fontsize=12)
        plt.text(room+0.02, end ,'{0}'.format(int(end)), va='top', fontsize=12)
        # plot event name
        plt.text(room+0.48, (start+end)*0.5, event, ha='center', va='center', fontsize=14, fontweight='bold')

    # Set Axis
    ax=fig.add_subplot(111)
    #ax.yaxis.grid()
    ax.axvline(x=1.5, ymin=0, ymax=300,color='grey')
    ax.axvline(x=2.5, ymin=0, ymax=300,color='grey')
    #ax.xaxis.grid()
    ax.set_xlim(0.5,len(rooms)+0.5)
    ax.set_ylim(0, 290,10)
    ax.set_xticks(range(1,len(rooms)+1))
    ax.set_xticklabels(rooms)
    ax.set_ylabel('Hogs Weight (Pounds)')

    # Set Second Axis
    ax2=ax.twiny().twinx()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(rooms)
    ax2.set_ylabel('Hogs Weight (Pounds)')

    plt.title(day_label,y=1.07)




