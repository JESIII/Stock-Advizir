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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import tree
from joblib import dump, load
from sklearn import preprocessing
import numpy as np
def calcRating(row):
    if row['RSI'] <= 45 and row['Close'] > row['Open'] and row['RollingAvg'] > row['Volume'] and row['MACDh'] > 0:
        return 'Buy'
    elif row['RSI'] > 65 and row['Close'] < row['Open'] and row['RollingAvg'] > row['Volume'] and row['MACDh'] < 0:
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
historicalPrice['Rating'] = historicalPrice.apply(lambda row: calcRating(row), axis=1)
historicalPrice.loc[['2020-03-16'],'Rating'] = 'Buy'
historicalPrice.loc[['2020-03-17'],'Rating'] = 'Buy'
historicalPrice.loc[['2020-03-18'],'Rating'] = 'Buy'
historicalPrice.loc[['2020-03-19'],'Rating'] = 'Buy'
historicalPrice.loc[['2020-06-26'],'Rating'] = 'Buy'
historicalPrice.loc[['2020-06-29'],'Rating'] = 'Buy'
historicalPrice.loc[['2020-05-01'],'Rating'] = 'Buy'
historicalPrice.loc[['2019-10-07'],'Rating'] = 'Buy'
historicalPrice.loc[['2019-10-01'],'Rating'] = 'Buy'
historicalPrice.loc[['2019-06-24'],'Rating'] = 'Buy'
historicalPrice.loc[['2019-06-25'],'Rating'] = 'Buy'
historicalPrice.loc[['2020-02-19'],'Rating'] = 'Sell'
historicalPrice.loc[['2020-04-20'],'Rating'] = 'Sell'
historicalPrice.loc[['2020-01-22'],'Rating'] = 'Sell'
historicalPrice.loc[['2020-01-21'],'Rating'] = 'Sell'
historicalPrice.loc[['2019-07-24'],'Rating'] = 'Sell'
historicalPrice.loc[['2020-10-07'],'Rating'] = 'Sell'
historicalPrice.loc[['2020-10-08'],'Rating'] = 'Sell'
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
#AI STUFF
historicalPrice["Open"]=((historicalPrice["Open"]-historicalPrice["Open"].min())/(historicalPrice["Open"].max()-historicalPrice["Open"].min()))*20
historicalPrice["Close"]=((historicalPrice["Close"]-historicalPrice["Close"].min())/(historicalPrice["Close"].max()-historicalPrice["Close"].min()))*20
historicalPrice["High"]=((historicalPrice["High"]-historicalPrice["High"].min())/(historicalPrice["High"].max()-historicalPrice["High"].min()))*20
historicalPrice["Low"]=((historicalPrice["Low"]-historicalPrice["Low"].min())/(historicalPrice["Low"].max()-historicalPrice["Low"].min()))*20
historicalPrice["TRIX"]=((historicalPrice["TRIX"]-historicalPrice["TRIX"].min())/(historicalPrice["TRIX"].max()-historicalPrice["TRIX"].min()))*20
historicalPrice["BOLL"]=((historicalPrice["BOLL"]-historicalPrice["BOLL"].min())/(historicalPrice["BOLL"].max()-historicalPrice["BOLL"].min()))*20
historicalPrice["RollingAvg"]=((historicalPrice["RollingAvg"]-historicalPrice["RollingAvg"].min())/(historicalPrice["RollingAvg"].max()-historicalPrice["RollingAvg"].min()))*20
historicalPrice["Volume"]=((historicalPrice["Volume"]-historicalPrice["Volume"].min())/(historicalPrice["Volume"].max()-historicalPrice["Volume"].min()))*20
historicalPrice["kdj"]=((historicalPrice["kdj"]-historicalPrice["kdj"].min())/(historicalPrice["kdj"].max()-historicalPrice["kdj"].min()))*20
x = historicalPrice.drop('Rating', axis=1)
print(x)
y = historicalPrice['Rating']
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)*100
print(accuracy)
dump(clf, 'AMDTrainedModel.joblib')
#plt.show()