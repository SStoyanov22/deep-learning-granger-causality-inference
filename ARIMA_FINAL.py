#import libraties
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels import api as sm
#conda install yahoofinancials
from yahoofinancials import YahooFinancials

#### import libraries for ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
#pip install pmdarima
from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

"""Constants"""
TICKER_BTC = "BTC-USD"
TICKER_NASDAQ = "^IXIC"
TICKER_SNP = "^GSPC"
START_DATE = "2014-09-17"
END_DATE = "2022-03-31"
pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 100)

# ### Get data for BTC

btc_financials = YahooFinancials(TICKER_BTC)
data=btc_financials.get_historical_price_data(START_DATE, END_DATE, "daily")
btc_df = pd.DataFrame(data[TICKER_BTC]['prices'])


# ### Get data for S&P 500


snp_financials = YahooFinancials(TICKER_SNP)
data=snp_financials.get_historical_price_data(START_DATE, END_DATE, "daily")
snp_df = pd.DataFrame(data[TICKER_SNP]['prices'])


# ## Data preparation

#Put all unecessarry columns in an array
to_drop = ['high', 'low', 'open', 'volume', 'adjclose', 
       'formatted_date']


# ### Prepare BTC data

#rename close column to price
btc_df = btc_df.rename(columns = {"close":"BTC"})

#check for date duplicates
len(btc_df['formatted_date'].unique()) #2759
len(btc_df.index) #2759

#drop unecessary columns, create date column and set as index
btc_df.date = pd.to_datetime(btc_df.formatted_date)
btc_df = btc_df.drop(to_drop, axis=1).set_index("date")

#Remove duplicated indexes/dates for BTC
if btc_df.index.duplicated().any():
    btc_df = btc_df.loc[~btc_df.index.duplicated(), :]

# ### Prepare S&P 500 data

#rename close column to price
snp_df = snp_df.rename(columns = {"close":"SP500"})

#check for date duplicates
len(snp_df['formatted_date'].unique()) #1902
len(snp_df.index) #1902

#drop unecessary columns, create date column and set as index
snp_df.date = pd.to_datetime(snp_df.formatted_date)
snp_df = snp_df.drop(to_drop, axis=1).set_index("date")

#Remove duplicated indexes/dates for BTC
if snp_df.index.duplicated().any():
    snp_df = snp_df.loc[~snp_df.index.duplicated(), :]

snp_df.head(10)


# ### Merge both dataframes into one

#Data frame with NaN values that is going to be used for imputation and modelling
df = btc_df.merge(snp_df, how="outer", left_index=True, right_index=True)

#Reindex to get all dates
df.reindex(pd.date_range(start=min(df.index), end=max(df.index), freq='D'))

#using forward propagation to fill all missing date values for SNP500
df.fillna(method='ffill', inplace=True)

###############################ARIMA MODEL###############################

#check data for stationarity
plt.plot('BTC',data=df)
plt.title("Price BTC")
adfuller(df["BTC"]) # p value 0.79
#value is not stationary and has to be normalized

#1st order differencing
plt.plot(df['BTC'].diff())
plt.title("1st Order Differencing")

#performing ADF test to check if the data after 1st differencing is stationary
adfuller(df['BTC'].diff().dropna())
#p value is < 0.05 therefore data is normalized after 1 period

#acf after 1 differencing
plot_acf(df['BTC'].diff().dropna())

#pacf after 1 differencing
sm.graphics.tsa.plot_pacf(df["BTC"].diff().dropna())

# Create Training and Test data sets; 75% of the data is for training purposes
train = btc_df.BTC[:2202]
test = btc_df.BTC[2202:]

# Auto Arima for best possible combination of p,d,q
warnings.filterwarnings("ignore")
auto_model = auto_arima(train,suppress_warnings=True)           
auto_model.summary() # model to be 0,1,0

# Build Model

#checking data for seasonality
result=seasonal_decompose(btc_df['BTC'], model='multiplicable', period=365)
result.seasonal.plot()
result.trend.plot()
result.plot()
#after decomposing the data, it looks seasonal

#showing our model that there is seasonality
model = ARIMA(train, order=(0, 1, 0), seasonal_order=(0,1,0,365))
res=model.fit()
print(res.summary())

# Forecast
forecast = res.forecast(551,alpha=0.05)

forecast_series = pd.Series(forecast, index=test.index)
lower_series = pd.Series(index=test.index)
upper_series = pd.Series(index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(forecast_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()