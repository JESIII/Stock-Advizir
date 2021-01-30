# #Stock Advisor
import datetime
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import stockstats as ss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from joblib import dump
#import requests
pd.set_option('display.max_columns',10)
#Set stock to train on
stock = 'MSFT'
#Finnhub Info

#Clean data for training
def calcRating(row):
    index = row.name
    index = index + pd.DateOffset(days=20)
    index_offset = index.strftime('%Y-%m-%d')
    row_pos_shift = None
    tries = 0
    while (tries<10) & (index < (datetime.today() - pd.DateOffset(days=21))):
        try:
            row_pos_shift = stock_dataset.loc[[str(index_offset)]]
        except: 
            index = index + pd.DateOffset(days=1)
            index_offset = index.strftime('%Y-%m-%d')
            continue
        else:
            tries = 10
        tries += 1
    if index > (datetime.today() - pd.DateOffset(days=21)):
        return None
    price_increased = True if row_pos_shift.Open[0] > row.Open else False
    good_RSI = True if row.RSI < 45 else False
    good_TRIX = True if row.TRIX < 50 else False
    bad_MACD = True if row.MACDh > 0.1 else False
    good_MACD = True if row.MACDh < -0.1 else False
    good_KDJ = True if row.KDJ < 50 else False
    bad_RSI = True if row.RSI > 65 else False
    if good_MACD & good_KDJ & good_RSI & good_TRIX & price_increased:
        return 'Buy'
    elif (not good_KDJ) & (not good_MACD) & (bad_RSI) & (not good_TRIX) & bad_MACD & (not price_increased):
        return 'Sell'
    else:
        return 'Hold'
#Load data into dataframe and stock dataframe
stock_data = yf.Ticker(stock)
stock_dataset = stock_data.history(period='10y')
stock_DF = stock_dataset.copy()
#Det indicators
stock_DF = ss.StockDataFrame.retype(stock_DF)
macdh = stock_DF['macdh']
boll = stock_DF['boll']
kdjk = stock_DF['kdjk']
trix = stock_DF['trix']
rsi = stock_DF['rsi_6']
#add indicators to dataframe
stock_dataset['RollingAvg'] = stock_dataset.Volume.rolling(20).mean()
stock_dataset['RSI'] = rsi
stock_dataset['MACDh'] = macdh
stock_dataset['KDJ'] = kdjk
stock_dataset['TRIX'] = trix
stock_dataset = stock_dataset.dropna()
#print(stock_dataset)

#Normalize Data
stock_dataset["TRIX"]=((stock_dataset["TRIX"]-stock_dataset["TRIX"].min())/(stock_dataset["TRIX"].max()-stock_dataset["TRIX"].min()))*100
stock_dataset["RollingAvg"]=((stock_dataset["RollingAvg"]-stock_dataset["RollingAvg"].min())/(stock_dataset["RollingAvg"].max()-stock_dataset["RollingAvg"].min()))*20
stock_dataset["Volume"]=((stock_dataset["Volume"]-stock_dataset["Volume"].min())/(stock_dataset["Volume"].max()-stock_dataset["Volume"].min()))*100
stock_dataset["KDJ"]=((stock_dataset["KDJ"]-stock_dataset["KDJ"].min())/(stock_dataset["KDJ"].max()-stock_dataset["KDJ"].min()))*100

#calculate good buy times and add column to dataframe
stock_dataset['Rating'] = stock_dataset.apply(lambda row: calcRating(row), axis=1)

#Graph Data
stock_dataset.plot.line(y='Close')
#lines = stock_dataset.plot.line(y='Close')
buyRating = stock_dataset.query("Rating == 'Buy'")
#buyRating.plot(x=buyRating.index, y='Close', c='green', kind='scatter', sharex=True)
sellRating = stock_dataset.query("Rating == 'Sell'")
#sellRating.plot(x=sellRating.index, y='Close', c='red', kind='scatter', sharex=True)
#plt.xticks(stock_dataset.index,stock_dataset['index'])
plt.plot(buyRating.index, buyRating.Close, 'g*')
plt.plot(sellRating.index, sellRating.Close, 'r*')
plt.show()

#Machine Learning
stock_dataset = stock_dataset.dropna()
y = stock_dataset['Rating']
x = stock_dataset.drop(['Rating','Open','Close','High','Low'], axis=1)
#print(x)

#print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
#print(y_pred)
accuracy = accuracy_score(y_test, y_pred)*100
#print(accuracy)
dump(clf, 'TrainedModel.joblib')
