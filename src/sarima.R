install.packages("tseries")
install.packages("fGarch")
install.packages("forecast")
install.packages("hutils")
library(forecast)
library(tseries)
library(fGarch)
library(hutils)

# xls files
my_data <- read.csv("C:/Users/Eric Su/Documents/GitHub/MFE-Paper/data/leanhogsfudif2.csv")

data = my_data
data <- data[rowSums(is.na(data))==0,]
par(mfrow=c(2,3))
plot(data[,2], pch = 20)
acf(data[,2])
pacf(data[,2])
adf.test(data[,2])

d2 <- ts(data[,2],frequency=12)
auto.arima(d2, ic="aic") #,  seasonal = T)
auto.arima(d2,max.P=0,max.Q=0,ic="bic", seasonal = T)

fit1 = arima(d2,order=c(2,0,1),seasonal=list(order=c(0, 0, 2), period=12)) # aic standard 1,0,2
acf(residuals(fit1))
Box.test(residuals(fit1), lag = 10, type="Ljung")
#X-squared = 5.2855, df = 10, p-value = 0.1655 it shows residue is stationary
resid2 = residuals(fit1)^2
acf(resid2)
Box.test(resid2, lag = 10, type="Ljung")
#X-squared = 5.1337, df = 10, p-value = 0.001099 it shows residue is stationary

garch.model.Tbill = garchFit(formula= ~arma(0,1) + garch(1,0),cond.dist="std",data = d2) # arch model
summary(garch.model.Tbill)
garch.model.Tbill@fit$matcoef

res = residuals(garch.model.Tbill)
res_std = res / garch.model.Tbill@sigma.t
par(mfrow=c(2,3))
plot(res)
acf(res)
Box.test(res, lag = 10, type="Ljung")
acf(res^2)
Box.test(res^2, lag = 10, type="Ljung")
plot(res_std)
acf(res_std)
acf(res_std^2)


garch.model.Tbill = garchFit(formula= ~arma(0,1) + garch(1,1),cond.dist="std",data = d2) # Garch model
summary(garch.model.Tbill)
garch.model.Tbill@fit$matcoef

res = residuals(garch.model.Tbill)
res_std = res / garch.model.Tbill@sigma.t
par(mfrow=c(2,3))
plot(res)
acf(res)
Box.test(res, lag = 10, type="Ljung")
acf(res^2)
Box.test(res^2, lag = 10, type="Ljung")
plot(res_std)
acf(res_std)
acf(res_std^2)






##
my_data <- read.csv("C:/Users/Eric Su/Documents/GitHub/MFE-Paper/data/LeanHogsFu2.csv")

data = my_data
data <- data[rowSums(is.na(data))==0,]
par(mfrow=c(2,3))
plot(data[,2], pch = 20)
acf(data[,2])
pacf(data[,2])
adf.test(data[,2])

d2 <- ts(data[,2],frequency=12)
auto.arima(d2, ic="aic",  seasonal = T)
auto.arima(d2,max.P=0,max.Q=0,ic="bic", seasonal = T)

fit1 = arima(d2,order=c(1,1,2),seasonal=list(order=c(0, 0, 2), period=12)) #aic standard
acf(residuals(fit1))
Box.test(residuals(fit1), lag = 12, type="Ljung")
#X-squared = 7.4121, df = 10, p-value = 0.6861 it shows residue is stationary
resid2 = residuals(fit1)^2
acf(resid2)
Box.test(resid2, lag = 12, type="Ljung")
#X-squared = 11.851, df = 10, p-value = 0.2952 it shows residue is stationary no garch effect
Del.d2 = diff(d2)
garch.model.Tbill = garchFit(formula= ~arma(1,2) + garch(1,0),Del.d2)
summary(garch.model.Tbill)
garch.model.Tbill@fit$matcoef

res = residuals(garch.model.Tbill)
res_std = res / garch.model.Tbill@sigma.t
par(mfrow=c(2,3))
plot(res)
acf(res)
acf(res^2)
plot(res_std)
acf(res_std)
acf(res_std^2)