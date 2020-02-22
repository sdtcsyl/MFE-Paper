# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:11:28 2020

@author: Eric Su
"""
# reference
#https://www.statsmodels.org/dev/vector_ar.html
#https://towardsdatascience.com/vector-autoregressions-vector-error-correction-multivariate-model-a69daf6ab618
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import datetime as dt
import matplotlib.pyplot as plt

import files

#read CME lean hogs futures' data from excel
lh = files.read_data(r'LeanHogsFutures.xlsx')
#read CME corn futures' data from excel
cn = files.read_data(r'EEEHogFuture.xlsx')
#read CME live cattle futures' data from excel
lc = files.read_data(r'KRLeanHogsFutures.xlsx')
start_time = dt.datetime(2000, 1, 1)
data = pd.DataFrame({'LeanHogs':lh['Close'][lh.index > start_time]})
data = data.join(pd.DataFrame({'Ehog' : cn['Close'][cn.index > start_time]}))
data = data.join(pd.DataFrame({'Khog' : lc['Close'][lc.index > start_time]}))
#data = data.resample('M').mean() 
data = np.log(data / data.shift(1)).dropna() # d 1


#forward fill the NA
data = data.fillna(method='ffill')


#Stationarity check
from arch.unitroot import ADF, KPSS, DFGLS, PhillipsPerron, ZivotAndrews, VarianceRatio
adf = ADF(data['LeanHogs'])
print(adf.summary().as_text())
adf = ADF(data['Ehog'])
print(adf.summary().as_text())
adf = ADF(data['Khog'])
print(adf.summary().as_text())


#Split the Series into Training and Testing Data
n_obs = 10
x_train, x_test = data[0:-n_obs], data[-n_obs:]

#Granger Causality test
from statsmodels.tsa.stattools import grangercausalitytests
print(grangercausalitytests(x_train[['LeanHogs','Ehog']], maxlag=15, addconst=True, verbose=True))
print(grangercausalitytests(x_train[['LeanHogs','Khog']], maxlag=15, addconst=True, verbose=True))
print(grangercausalitytests(x_train[['Ehog','Khog']], maxlag=15, addconst=True, verbose=True))


# make a VAR model
model = VAR(data)
#Lag order selection
results = model.fit(maxlags=15, ic='aic')
results.summary()
results.plot()

#Plotting time series autocorrelation
results.plot_acorr()
lag_order = results.k_ar
pred = pd.DataFrame(results.forecast(data.values[-11:], n_obs), index=x_test.index, columns=x_test.columns)
#plot forecast data
results.plot_forecast(10)
#plot forecast leanhogs and actual data
plt.figure(figsize=(12,5))
plt.xlabel('Date')
ax1 = x_test['LeanHogs'].plot(color='blue', grid=True, label='Actual LeanHogs Price')
ax2 = pred.LeanHogs.plot(color='red', grid=True, secondary_y = True, label='Predicted LeanHogs Price')
ax1.legend(loc=1)
ax2.legend(loc=2)
plt.title('Predicted Vs Actual LeanHogs Price')
plt.show()

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
#VAR Forecast evaluation
forecast_error = [x_test.LeanHogs[i]-pred.LeanHogs[i] for i in range(len(x_test))]
bias = sum(forecast_error)*1.0/len(x_test)
print('Bias: %f'%bias)
mae = mean_absolute_error(x_test.LeanHogs, pred.LeanHogs)
print('MAE: %f'%mae)
mse = mean_squared_error(x_test.LeanHogs, pred.LeanHogs)
print('MSE: %f'%mse)
rmse = np.sqrt(mse)
print('RMSE: %f'%rmse)
