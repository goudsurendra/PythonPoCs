
# coding: utf-8

# In[183]:

import matplotlib as mplt


mplt.use('TkAgg')
# coding: utf-8
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

mplt.use('TkAgg')

data_folder  ='C:\\Users\\surendra\\Box Sync\\Learn\\Master Data Science'

filetoanalyse='UC_claims_made.xlsx'

df= pd.read_excel('C:\\Users\\surendra\\Box Sync\\Learn\\Master Data Science\\UC_claims_made.xlsx')

df.reset_index(inplace=True)

df1=df
df1['date']=pd.to_datetime(df.date,infer_datetime_format=True)


df=df1.loc[:,['count','date']]
df.set_index('date')

df.reset_index(inplace=True)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

data = df['count']
df1= pd.DataFrame(data, pd.DatetimeIndex(start='4/1/2013',end='10/1/2017', freq='D'))
df1['count']=df1['count'].fillna(0)
df1=df1.resample("1W").sum()
decompose = sm.tsa.seasonal_decompose(df1, model='additive', filt=None)
fig = decompose.plot()
plt.ion()
fig.show()


# In[184]:

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=15,center=False).mean()    
    rolstd = timeseries.rolling(window=15,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    print(dftest)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[185]:

ts=df1[3:228]


# In[186]:

ts


# In[187]:

ts_log = np.log(ts)
ts_log
#plt.plot(ts_log)


# In[188]:

moving_avg = ts_log.rolling(window=4,center=False).mean()
moving_avg.fillna(0)
#plt.plot(ts_log)
#plt.plot(moving_avg, color='red')


# In[189]:

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(15).fillna(0)


# In[190]:

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)


# In[191]:

expwighted_avg = ts_log.ewm(halflife=4,min_periods=0,adjust=True,ignore_na=False).mean()
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')


# In[192]:

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)


# In[182]:

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)


# In[193]:

ts_log_diff.dropna(inplace=True)
ts_log_diff
test_stationarity(ts_log_diff)


# In[194]:

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# In[76]:

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
#test_stationarity(ts_log_decompose)


# In[195]:

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')


# In[196]:

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


# In[119]:

#AR Model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(1, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))



# In[120]:

#MA Model
model = ARIMA(ts_log, order=(1, 1, 0))  
results_MA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='blue')
#plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))


# In[220]:

#Combined Model
model = ARIMA(ts_log, order=(2,1,3))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='black')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))


# In[221]:

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()


# In[141]:

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()


# In[224]:

predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0).fillna(0)
predictions_ARIMA_log.head()


# In[225]:

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA,color='green')
#plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))


# In[227]:

predictions_ARIMA


# In[228]:

ts


# In[ ]:



