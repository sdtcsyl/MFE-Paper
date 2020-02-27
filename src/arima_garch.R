#### Data Preprocessing ####
library("MTS")
data = read.csv("C:/Users/Eric Su/Documents/GitHub/MFE-Paper/We R Best/data.csv",header=T)
#data = tail(data, 500)
log_price=log(data[,c(2,4)])

return=log_price[2:nrow(log_price),]-log_price[1:(nrow(log_price)-1),] ## Growth rate
return=return*100 ## Percentage growth rates

return_train = return[1:(nrow(return)-7) ,]
return_test = return[(nrow(return)-7):nrow(return) ,]

rm(data, log_price, return)



####ARIMA Model####

## forecast oil
library(tseries)
data = return_train[, 1]
par(mfrow=c(2,3))
plot(data, pch = 20)
acf(data)
pacf(data)
adf.test(data)

library(forecast)
ARMA = auto.arima( data,max.p=20,max.q=20,ic="aic")
print(ARMA)
plot(residuals(ARMA))
acf(residuals(ARMA) )
acf(residuals(ARMA)^2)
Box.test(residuals(ARMA)^2, lag = 12, type="Ljung")
forecasts = predict(ARMA, 8)

## forecast sp500
library(tseries)
data = return_train[, 2]
par(mfrow=c(2,3))
plot(data, pch = 20)
adf.test(data)
acf(data)
pacf(data)


library(forecast)
ARMA = auto.arima( data,max.p=20,max.q=20,ic="aic")
print(ARMA)
plot(residuals(ARMA))
acf(residuals(ARMA) )
acf(residuals(ARMA)^2 )


forecasts = predict(ARMA, 8)





####Garch Model####

## forecast oil
data = return_train[, 1]

library(tseries)
par(mfrow=c(2,3))
plot(data, pch = 20)
adf.test(data)
acf(data)
pacf(data)


library(forecast)
ARMA = auto.arima( data,max.p=20,max.q=20,ic="aic")
print(ARMA)
acf(residuals(ARMA) )
acf(residuals(ARMA)^2 )

library(fGarch)
garch = garchFit(formula= ~arma(0,0) + garch(1,1), data)
res = residuals(garch)
res_std = res / garch@sigma.t
par(mfrow=c(2,3))
plot(res)
acf(res)
acf(res^2)
plot(res_std)
acf(res_std)
acf(res_std^2)

options(digits=4)
summary(garch)
forecasts = predict(garch, 8)

## forecast sp500
data = return_train[, 2]

library(tseries)
par(mfrow=c(2,3))
plot(data, pch = 20)
adf.test(data)
acf(data)
pacf(data)


library(forecast)
ARMA = auto.arima( data,max.p=20,max.q=20,ic="aic")
print(ARMA)
acf(residuals(ARMA) )
acf(residuals(ARMA)^2 )

library(fGarch)
garch = garchFit(formula= ~arma(0,0) + garch(1,1), data)
res = residuals(garch)
res_std = res / garch@sigma.t
par(mfrow=c(2,3))
plot(res)
acf(res)
acf(res^2)
plot(res_std)
acf(res_std)
acf(res_std^2)

options(digits=4)
summary(garch)
forecasts = predict(garch, 8)








#### VAR Model ####
dim(return_train)
par(mfrow=c(4,1)) 
MTSplot(return_train)


ccm(return_train)


m0=VARorder(return_train)
names(m0)
m0$Mstat

m1=VAR(return_train,3) 
m2=refVAR(m1,thres=1.96)

pred_mean = VARpred(m2,8)$pred
pred_std = VARpred(m2,8)[[2]]
pred_upper = pred_mean + 1.96*pred_std
pred_lower = pred_mean - 1.96*pred_std

# plot oil test performance
data = return_test[1]
oil_test = cbind( return_test[, 1], pred_mean[, 1],pred_upper[, 1],  pred_lower[, 1])
oil_test = data.frame(oil_test)
oil_test = cbind(data[(nrow(data)-7):nrow(data), 1], oil_test)
names(oil_test) = c('Date','return','prediction','upper','lower')

library("ggplot2")
ggplot(oil_test, aes(Date, return,group = 1)) +
  geom_point() +
  geom_line(aes(y = prediction), color = "red", linetype = "dashed")+
  geom_line(aes(y = lower), color = "red", linetype = "dashed")+
  geom_line(aes(y = upper), color = "red", linetype = "dashed")+
  theme_minimal()

RMSE = sqrt( mean( (oil_test[,2] - oil_test[,3])^2) )






# plot sp500 test performance
data = return_test[2]
sp500_test = cbind( return_test[, 2], pred_mean[, 2],pred_upper[, 2],  pred_lower[, 2])
sp500_test = data.frame(sp500_test)
sp500_test = cbind(data[(nrow(data)-7):nrow(data), 1], sp500_test)
names(sp500_test) = c('Date','return','prediction','upper','lower')

library("ggplot2")
ggplot(sp500_test, aes(Date, return,group = 1)) +
  geom_point() +
  geom_line(aes(y = prediction), color = "red", linetype = "dashed")+
  geom_line(aes(y = lower), color = "red", linetype = "dashed")+
  geom_line(aes(y = upper), color = "red", linetype = "dashed")+
  theme_minimal()  

RMSE = sqrt( mean( (sp500_test[,2] - sp500_test[,3])^2) )



