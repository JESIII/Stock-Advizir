import yfinance as yf
from datetime import datetime
from datetime import timedelta
import stockstats as ss
from joblib import dump, load
import pandas_datareader.data as web
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
# Dates
end = datetime.today().strftime('%Y-%m-%d')
start = datetime.today() - timedelta(days=(120))
start = start.strftime('%Y-%m-%d')
#Import historical data
ticker = yf.Ticker('AAPL')
historicalPrice = web.get_data_yahoo("AAPL", start=start, end=end)
#print(historicalPrice)
# Window length for moving average
window_length = 14
#print(start)
#print(end)
# Get data
data = web.DataReader('AAPL', 'yahoo', start, end)
#get macd signal
stockDF = data.copy()
stockDF = ss.StockDataFrame.retype(stockDF)
macd = stockDF['macdh']
boll = stockDF['boll']
kdj = stockDF['kdjk']
trix = stockDF['trix']
#print(macd)
# Get just the adjusted close
close = data['Adj Close']
# Get the difference in price from previous step
delta = close.diff()
# Get rid of the first row, which is NaN since it did not have a previous 
# row to calculate the differences
delta = delta[1:] 
# Make the positive gains (up) and negative gains (down) Series
up, down = delta.copy(), delta.copy()
up[up < 0] = 0
down[down > 0] = 0

# Calculate the SMA
roll_up = up.rolling(window_length).mean()
roll_down = down.abs().rolling(window_length).mean()

# Calculate the RSI based on SMA
RS2 = roll_up / roll_down
historicalRSI = 100.0 - (100.0 / (1.0 + RS2))
#print(historicalPrice)
historicalRSI = historicalRSI.to_frame()
# print(type(historicalPrice))
# print(type(historicalRSI))
# print(historicalPrice)
# print(historicalRSI)
#history = historicalPrice.join(historicalRSI, how='left')
historicalRSI.columns = ['RSI']
historicalPrice.columns = ['High','Low','Open','Close','Volume','AdjClose']
historicalPrice['RSI'] = historicalRSI['RSI']
# print(historicalPrice)
historicalPrice['RollingAvg'] = historicalPrice.Volume.rolling(20).mean()
historicalPrice['MACDh'] = macd
historicalPrice['BOLL'] = boll
historicalPrice['kdj'] = kdj
historicalPrice['TRIX'] = trix
historicalPrice = historicalPrice.dropna(axis=0, how='any')
clf = load('AMDTrainedModel.joblib')
historicalPrice["Open"]=((historicalPrice["Open"]-historicalPrice["Open"].min())/(historicalPrice["Open"].max()-historicalPrice["Open"].min()))*20
historicalPrice["Close"]=((historicalPrice["Close"]-historicalPrice["Close"].min())/(historicalPrice["Close"].max()-historicalPrice["Close"].min()))*20
historicalPrice["High"]=((historicalPrice["High"]-historicalPrice["High"].min())/(historicalPrice["High"].max()-historicalPrice["High"].min()))*20
historicalPrice["Low"]=((historicalPrice["Low"]-historicalPrice["Low"].min())/(historicalPrice["Low"].max()-historicalPrice["Low"].min()))*20
historicalPrice["TRIX"]=((historicalPrice["TRIX"]-historicalPrice["TRIX"].min())/(historicalPrice["TRIX"].max()-historicalPrice["TRIX"].min()))*20
historicalPrice["BOLL"]=((historicalPrice["BOLL"]-historicalPrice["BOLL"].min())/(historicalPrice["BOLL"].max()-historicalPrice["BOLL"].min()))*20
historicalPrice["RollingAvg"]=((historicalPrice["RollingAvg"]-historicalPrice["RollingAvg"].min())/(historicalPrice["RollingAvg"].max()-historicalPrice["RollingAvg"].min()))*20
historicalPrice["Volume"]=((historicalPrice["Volume"]-historicalPrice["Volume"].min())/(historicalPrice["Volume"].max()-historicalPrice["Volume"].min()))*20
historicalPrice["kdj"]=((historicalPrice["kdj"]-historicalPrice["kdj"].min())/(historicalPrice["kdj"].max()-historicalPrice["kdj"].min()))*20
y = historicalPrice
print(y)
y_pred = clf.predict(y)
print(y_pred)