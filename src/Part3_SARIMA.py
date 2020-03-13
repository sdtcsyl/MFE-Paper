# -*- coding: utf-8 -*-
"""
@author: Yulu Su
"""
#https://towardsdatascience.com/how-to-forecast-sales-with-python-using-sarima-model-ba600992fa7d

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import pandas as pd
import statsmodels.api as sm

import files

data = files.read_data(r'LeanHogsFutures.xlsx')
data = pd.DataFrame(data['Close'].dropna()) #[data.index > dt.datetime(2010, 1, 1)].dropna())
data = data.resample('M').mean() # resample daily data to monthly data
data = data['1992':'2004']
data.to_csv(r'LeanHogsFu.csv')
lh =  np.log(data / data.shift(1)).dropna()*100 # d 1
lh.to_csv(r'leanhogsfudif.csv')
lh.plot(figsize=(19, 4))
plt.show()

#test stationarity
from statsmodels.tsa.stattools import adfuller
#Perform Dickey-Fuller test:
print('Results of Dickey-Fuller Test:')
dftest = adfuller(pd.Series(lh['Close']), autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

#Results of Dickey-Fuller Test:
#Test Statistic                -8.388264e+00
#p-value                        2.405295e-13 p_value is extremely small and we reject the h0 that the timeseries is unstationary
##Lags Used                     1.000000e+00
#Number of Observations Used    1.050000e+02
#Critical Value (1%)           -3.494220e+00
#Critical Value (5%)           -2.889485e+00
#Critical Value (10%)          -2.581676e+00

# acf and pacf test
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(lh, nlags=40)
lag_pacf = pacf(lh, nlags=40, method='ols')

plt.figure(figsize=(15,5))
plt.subplot(121)
plt.stem(lag_acf)
plt.axhline(y=0, linestyle='-',color='black')
plt.axhline(y=-1.96/np.sqrt(len(lh)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(lh)),linestyle='--',color='gray')
plt.xlabel('Lag')
plt.ylabel('ACF')

plt.subplot(122)
plt.stem(lag_pacf)
plt.axhline(y=0, linestyle='-',color='black')
plt.axhline(y=-1.96/np.sqrt(len(lh)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(lh)),linestyle='--',color='gray')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.tight_layout()


# acf and pacf test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# acf test
acf = plot_acf(lh,lags=40, ax=plt.subplot(1, 2, 1) )
plt.title("ACF plot of Lean Hogs Future")
acf.show()

# pacf test
pacf = plot_pacf(lh,lags=40, ax=plt.subplot(1, 2, 2))
plt.title("PACF of Lean Hogs Future")
pacf.show()

#To double check the time-series has seasonality pattern.
from pylab import rcParams
decomposition = sm.tsa.seasonal_decompose(lh, model='additive', filt=None, freq=12)
fig = decomposition.plot()
plt.show()



#find the optimal parameters and then we implement the parameters into the formula
mod = sm.tsa.statespace.SARIMAX(lh,
                                order=(2, 0, 1),
                                seasonal_order=(0, 0, 2, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False) 
results = mod.fit()
print(results.summary().tables[1])

#check the residues after formula is stationary or not. If stationary, the model is perfect
results.plot_diagnostics(figsize=(18, 8))
plt.show()

#one-step ahead forecast, we also compare the true values with the forecast predictions.
pred = results.get_prediction(start=pd.to_datetime('2003-12-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = lh['2000':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Retail_sold')
plt.title('Seasonal ARIMA model One-step ahead Forecast')
plt.legend()
plt.show()


#calculate the Mean squared error between predictions and reality
y_forecasted = pd.Series(list(pred.predicted_mean))
y_truth = pd.Series(list(lh['Close']['2004':]) )
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))

from sklearn.metrics import  mean_squared_error
print(np.sqrt(mean_squared_error(lh[-13:], pred.predicted_mean )))


###Garch Test
resid = results.resid
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(resid)
plt.title('Standardized residuals')
plot_acf(resid**2)
plt.title('Squared residuals')



