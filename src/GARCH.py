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
#data = data['2010':'2018']

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

#VR tests was performed to test whether the return series is a pure random walk versus
# having some predictability. I have compared here 1-month to 12-months returns, and 
# the null that the series is a pure random walk is rejected. Rejection of the null 
# with a negative test statistic VA(-11.07) indicates the presence of serial correlation 
# in the time series. The tests of unit roots and stationarity with ADF, KPSS, DFGLS, PP and 
# ZA statistics are all shown to be significant, indicating the application of the GARCH-type 
# model to fit the return series is appropriate.


#Non-linear dynamics
closes_recent = lh[-2000:]
plt.plot(closes_recent)
# calculate Hurst of recent prices
lag1 = 2
lags = range(lag1, 20)
tau = [np.sqrt(np.std(np.subtract(closes_recent[lag:], closes_recent[:-lag]))) for lag in lags]
plt.plot(np.log(lags), np.log(tau))
m = np.polyfit(np.log(lags), np.log(tau), 1)
hurst = m[0]*2
print ('hurst = ',hurst)
#
#from pyrqa.time_series import TimeSeries
#from pyrqa.settings import Settings
#from pyrqa.computing_type import ComputingType
#from pyrqa.neighbourhood import FixedRadius
#from pyrqa.metric import EuclideanMetric
#from pyrqa.computation import RQAComputation
#time_series = TimeSeries(data['Close'],
#embedding_dimension=2,
#time_delay=2)
#settings = Settings(time_series,computing_type=ComputingType.Classic,neighbourhood=FixedRadius(0.65),
#similarity_measure=EuclideanMetric,
#theiler_corrector=1)
#computation = RQAComputation.create(settings,
#verbose=True)
#result = computation.run()
#result.min_diagonal_line_length = 2
#result.min_vertical_line_length = 2
#result.min_white_vertical_line_lelngth = 2
#print(result)
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model

def _get_best_model(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None
    pq_rng = range(5) # [0,1,2,3,4,5]
    d_rng = range(5) # [0,1]
    for i in pq_rng:
        for d in d_rng:
            for j in pq_rng:
                try:
                    tmp_mdl = ARIMA(TS, order=(i,d,j)).fit(
                    method='mle', trend='nc'
                    )
                    tmp_aic = tmp_mdl.aic
                    if tmp_aic < best_aic:
                        best_aic = tmp_aic
                        best_order = (i, d, j)
                        best_mdl = tmp_mdl
                except: continue
    print('aic: {:6.2f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl

X = 100* lh
TS = X
res_tup = _get_best_model(TS) #aic: 2788.90 | order: (3, 0, 1)

gam = arch_model(res_tup[2].resid, p=3, o=0, q=1, vol='Garch', dist='StudentsT')
gres = gam.fit(update_freq=5, disp='off')
print(gres.summary())
adf = ADF(gres.resid**2)
print(adf.summary().as_text()) #p = 0.00, residue it is stationary


am = arch_model(X, p=3, q=1, o=0,power=2.0, vol='Garch', dist='StudentsT')
res = am.fit(update_freq=5)
print(res.summary())
eam = arch_model(X, p=3,q=1, o=0, power=2.0, vol='EGARCH', dist='StudentsT')
eres = eam.fit(update_freq=5)
print(res.summary())
gjam = arch_model(X, p=3, o=1, q=0, power=2.0,  dist='StudentsT')
gjres = gjam.fit(update_freq=5, disp='off')
print(gjres.summary())

std_resid = res.resid / res.conditional_volatility
unit_var_resid = res.resid / res.resid.std()
plt.subplots(2,1)
plt.subplot(2, 1, 1)
plt.plot(std_resid)
plt.subplot(2, 1, 2)
plt.plot(unit_var_resid)

plt.xlim(-2, 2)
sns.kdeplot(std_resid**2, shade=True)
sns.kdeplot(std_resid, shade=True)
sns.kdeplot(unit_var_resid, shade=True)
plt.legend(['Squared Residual', 'Unit variance residual', 'Std Residual'], loc='best')
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(std_resid)
plt.title('Standardized residuals')
plot_acf(std_resid**2)
plt.title('Squared residuals')


am = arch_model(X,mean='HAR',lags=[1,5,22],vol='Constant')
sim_data = am.simulate([0.1,0.4,0.3,0.2,1.0], 250)
am = arch_model(sim_data['data'],mean='HAR',lags=[1,5,22], vol='Constant')
res = am.fit()
fig = res.hedgehog_plot(type='mean')
