# -*- coding: utf-8 -*-
"""
@author: Yulu Su
"""

#https://towardsdatascience.com/garch-processes-monte-carlo-simulations-for-analytical-forecast-27edf77b2787
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from statsmodels.tsa.stattools import  acf, pacf
from arch.unitroot import ADF, KPSS, DFGLS, PhillipsPerron, ZivotAndrews, VarianceRatio

import files

data = files.read_data(r'LeanHogsFutures.xlsx')
data.describe()

data = pd.DataFrame(data['Close'].dropna()) #[data.index > dt.datetime(2010, 1, 1)].dropna())
data = data.resample('M').mean() # resample daily data to monthly data
data = data['1992':'2004']

def garch_plot1(lh):
    # Plot figure with subplots of different sizes
    fig = plt.figure(1)
    # set up subplot grid
    gridspec.GridSpec(3,2)
    # large subplot
    plt.subplot2grid((3,2), (0,0), colspan=2, rowspan=1)
    plt.title('Lean Hogs Time Series Analysis Plots')
    plt.plot(lh)
    # small subplot 1
    plt.subplot2grid((3,2), (1,0))
    lag_acf = acf(lh, nlags=40)
    plt.stem(lag_acf)
    plt.axhline(y=0, linestyle='-',color='black')
    plt.axhline(y=-1.96/np.sqrt(len(lh)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(lh)),linestyle='--',color='gray')
    plt.ylabel('ACF')
    # small subplot 2
    plt.subplot2grid((3,2), (1,1))
    lag_pacf = pacf(lh, nlags=40, method='ols')
    plt.stem(lag_pacf)
    plt.axhline(y=0, linestyle='-',color='black')
    plt.axhline(y=-1.96/np.sqrt(len(lh)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(lh)),linestyle='--',color='gray')
    plt.ylabel('PACF')
    # small subplot 3
    ax0 = plt.subplot2grid((3,2), (2,0))
    ax1 = plt.subplot2grid((3,2), (2,1))
    probplot = sm.ProbPlot(lh, dist='lognorm', fit=True)
    probplot.ppplot(line='45', ax=ax0)
    probplot.qqplot(line='45', ax=ax1)
    ax0.set_title('P-P Plot')
    ax1.set_title('Q-Q Plot')
    plt.show()

garch_plot1(data['Close'])

lh =  np.log(data / data.shift(1)).dropna() # d 1

garch_plot1(lh['Close'])
print('Lean Hogs Future skewness is {}'.format(lh.skew(axis=0)[0])) 
print('Lean Hogs Future kurtosis is {}'.format(lh.kurtosis(axis=0)[0]))


sns.distplot(lh['Close'], color='blue') #density plot
plt.title('1986–2018 Lean Hogs Future return frequency')
plt.xlabel('Possible range of data values')
# Pull up summary statistics
print(lh.describe())

adf = ADF(lh['Close'])
print(adf.summary().as_text())
kpss = KPSS(lh['Close'])
print(kpss.summary().as_text())
dfgls = DFGLS(lh['Close'])
print(dfgls.summary().as_text())
pp = PhillipsPerron(lh['Close'])
print(pp.summary().as_text())
za = ZivotAndrews(lh['Close'])
print(za.summary().as_text())
vr = VarianceRatio(lh['Close'], 12)
print(vr.summary().as_text())

from arch import arch_model

X = 100* lh

import datetime as dt
am = arch_model(X, p=4, o=0, q=0, vol='Garch', dist='StudentsT')
res = am.fit(last_obs=dt.datetime(2003,12,31))
forecasts = res.forecast(horizon=1,   start='2004-1-1')
cond_mean = forecasts.mean['2004':]
cond_var = forecasts.variance['2004':]/31
print(res.summary())

# acf and pacf test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# acf test
std_resid = res.resid[:-13] / res.conditional_volatility[:-13]
plot_acf(std_resid,lags=30, ax=plt.subplot(1, 2, 1) )
plt.title("ACF plot of Lean Hogs Future Standardized residuals")
# pacf test
plot_pacf(std_resid,lags=30, ax=plt.subplot(1, 2, 2))
plt.title("PACF of Lean Hogs Future Standardized residuals")

res2 = std_resid**2
# acf test
plot_acf(res2,lags=30, ax=plt.subplot(1, 2, 1) )
plt.title("ACF plot of Lean Hogs Future Squared residuals")
# pacf test
plot_pacf(res2,lags=30, ax=plt.subplot(1, 2, 2))
plt.title("PACF of Lean Hogs Future Squared residuals")


adf = ADF(res.resid[:-13]**2)
print(adf.summary().as_text()) #p = 0.001, residue it is stationary

plt.plot(X['2000':])
plt.plot(cond_var)
plt.title('GARCH model 12 months predications')


from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(X[-12:], cond_var)))