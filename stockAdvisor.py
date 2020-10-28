# #Stock Advisor
# import numpy as np
# from scipy import optimize
# import json
# import pandas as pd
# historicalStock = pd.read_csv("C:/Users/tinco/Downloads/archive/Stocks/amd.us.txt")
# historicalRSI = pd.DataFrame
# for row in historicalAMD:
import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import stockstats as ss
from fastai.tabular.all import *
def calcRating(row):
    if row['RSI'] <= 45 and row['Close'] > row['Open'] and row['RollingAvg'] > row['Volume'] and row['MACDh'] > -0.10:
        return 'Buy'
    elif row['RSI'] > 65 and row['Close'] < row['Open'] and row['RollingAvg'] > row['Volume'] and row['MACDh'] < 0.15:
        return 'Sell'
    else:
        return 'Hold'
# Dates
end = datetime.today().strftime('%Y-%m-%d')
start = datetime.today() - timedelta(days=(365*5))
start = start.strftime('%Y-%m-%d')
#Import historical data
ticker = yf.Ticker('AMD')
historicalPrice = web.get_data_yahoo("AMD", start=start, end=end)
#print(historicalPrice)
# Window length for moving average
window_length = 14


#print(start)
#print(end)
# Get data
data = web.DataReader('AMD', 'yahoo', start, end)
#get macd signal
stockDF = data.copy()
stockDF = ss.StockDataFrame.retype(stockDF)
macd = stockDF['macdh']
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
historicalPrice = historicalPrice.dropna(axis=0, how='any')
historicalPrice['Rating'] = historicalPrice.apply(lambda row: calcRating(row), axis=1)
print(historicalPrice)
lines = historicalPrice.plot.line(y='Close')
# print(historicalPrice.query("Rating == 'Buy'"))
buyRating = historicalPrice.query("Rating == 'Buy'")
# buyRating.plot(x=buyRating.index, y='Close', c='green', kind='scatter', sharex=True)
sellRating = historicalPrice.query("Rating == 'Sell'")
# sellRating.plot(x=sellRating.index, y='Close', c='red', kind='scatter', sharex=True)
# plt.xticks(historicalPrice.index,historicalPrice['index'])
plt.plot(buyRating.index, buyRating.Close, 'g*')
plt.plot(sellRating.index, sellRating.Close, 'r*')
#plt.show()
#AI STUFF
splits = RandomSplitter(valid_pct=0.2)(range_of(historicalPrice))
dls = TabularPandas(historicalPrice, cont_names = ['RSI', 'MACDh', 'RollingAvg', 'Volume', 'High', 'Low', 'Open', 'Close'], y_names='Rating')
to = TabularPandas(historicalPrice, cont_names = ['RSI', 'MACDh', 'RollingAvg', 'Volume', 'High', 'Low', 'Open', 'Close'], y_names='Rating', splits=splits)
dls = to.dataloaders(bs=500, val_bs=100)
dls.show_batch()
learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(7)
learn.show_results()