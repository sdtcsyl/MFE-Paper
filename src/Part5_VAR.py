# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:11:28 2020

@author: Eric Su
"""
# reference
#https://www.statsmodels.org/dev/vector_ar.html
#https://towardsdatascience.com/vector-autoregressions-vector-error-correction-multivariate-model-a69daf6ab618
#https://towardsdatascience.com/prediction-task-with-multivariate-timeseries-and-var-model-47003f629f9
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
cn = files.read_data(r'CornFutures.xlsx')
#read CME live cattle futures' data from excel
lc = files.read_data(r'LivecCattleFutures.xlsx')

data = pd.DataFrame({'LeanHogs':lh['Close']})
data = data.join(pd.DataFrame({'Corn' : cn['Close']}))
data = data.join(pd.DataFrame({'Livecattle' : lc['Close']}))
data = data.resample('M').mean() # resample daily data to monthly data
data = np.log(data / data.shift(1)).dropna() # d 1

data = 100*data['1992':'2004']

#forward fill the NA
data = data.fillna(method='ffill')

data.plot(subplots=True, grid=True, figsize=(8, 6), title=['Lean hogs futures monthly log return * 100','Corn futures monthly log return * 100','Live cattle futures monthly log return * 100'])


#Stationarity check
from arch.unitroot import ADF, KPSS, DFGLS, PhillipsPerron, ZivotAndrews, VarianceRatio
adf = ADF(data['LeanHogs'])
print(adf.summary().as_text())
adf = ADF(data['Corn'])
print(adf.summary().as_text())
adf = ADF(data['Livecattle'])
print(adf.summary().as_text())


#Split the Series into Training and Testing Data
n_obs = 12
x_train, x_test = data[0:-n_obs], data[-n_obs:]

for i in range(5):
    i += 1
    model = sm.tsa.VARMAX(x_train, order=(i,0))
    model_result = model.fit(maxiter=1000, disp=False)
    print('Order = ', i)
    print('AIC: ', model_result.aic)
    print('BIC: ', model_result.bic)
    print('HQIC: ', model_result.hqic)
    
model = sm.tsa.VARMAX(x_train, order=(1, 0), trend='c')
model_result = model.fit(maxiter=1000, disp=False)
model_result.summary()

model_result.plot_diagnostics()
plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error
z = model_result.forecast(steps=n_obs)
print(np.sqrt(mean_squared_error(x_train['LeanHogs'][-12:], z['LeanHogs'])))
print(np.sqrt(mean_squared_error(x_train['Corn'][-12:], z['Corn'])))
print(np.sqrt(mean_squared_error(x_train['Livecattle'][-12:], z['Livecattle'])))

plt.plot(data['LeanHogs']['2000':])
plt.plot(z['LeanHogs'])
plt.title('VAR model 12 months predications')


