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
#show more columns
pd.set_option('display.max_columns',10)
def calcRating(row):
    index = row.name
    index = index + pd.DateOffset(days=20)
    indexOffset = index.strftime('%Y-%m-%d')
    rowPosShift = None
    tries = 0
    while (tries<10) & (index < (datetime.today() - pd.DateOffset(days=21))):
        #print(tries)
        try:
            rowPosShift = stockDataset.loc[[str(indexOffset)]]
        except: 
            index = index + pd.DateOffset(days=1)
            #print(index)
            indexOffset = index.strftime('%Y-%m-%d')
            continue
        else:
            tries = 10
        tries += 1
    if index > (datetime.today() - pd.DateOffset(days=21)):
        return None
    priceIncreased = True if rowPosShift.Open[0] > row.Open else False
    goodRSI = True if row.RSI < 45 else False
    goodTRIX = True if row.TRIX < 50 else False
    badMACD = True if row.MACDh > 0.1 else False
    goodMACD = True if row.MACDh < -0.1 else False
    goodKDJ = True if row.KDJ < 50 else False
    badRSI = True if row.RSI > 65 else False
    if goodMACD & goodKDJ & goodRSI & goodTRIX & priceIncreased:
        return 'Buy'
    elif (not goodKDJ) & (not goodMACD) & (badRSI) & (not goodTRIX) & badMACD & (not priceIncreased):
        return 'Sell'
    else:
        return 'Hold'
#load data into dataframe and stock dataframe
# stockDataset = pd.read_csv('D:/Dropbox/Dropbox (CSU Fullerton)/481/zStock-AI/Datasets/AMD.csv')
# stockDataset.columns = ['Date','High','Low','Open','Close','Volume','AdjClose']
msft = yf.Ticker("MSFT")
stockDataset = msft.history(period='10y')
print(stockDataset)
stockDF = stockDataset.copy()
#get data indicators
stockDF = ss.StockDataFrame.retype(stockDF)
macdh = stockDF['macdh']
boll = stockDF['boll']
kdjk = stockDF['kdjk']
trix = stockDF['trix']
rsi = stockDF['rsi_6']
#add indicators to dataframe
stockDataset['RollingAvg'] = stockDataset.Volume.rolling(20).mean()
stockDataset['RSI'] = rsi
stockDataset['MACDh'] = macdh
stockDataset['KDJ'] = kdjk
stockDataset['TRIX'] = trix
stockDataset = stockDataset.dropna()
#print(stockDataset)

#Normalize Data
stockDataset["TRIX"]=((stockDataset["TRIX"]-stockDataset["TRIX"].min())/(stockDataset["TRIX"].max()-stockDataset["TRIX"].min()))*100
stockDataset["RollingAvg"]=((stockDataset["RollingAvg"]-stockDataset["RollingAvg"].min())/(stockDataset["RollingAvg"].max()-stockDataset["RollingAvg"].min()))*20
stockDataset["Volume"]=((stockDataset["Volume"]-stockDataset["Volume"].min())/(stockDataset["Volume"].max()-stockDataset["Volume"].min()))*100
stockDataset["KDJ"]=((stockDataset["KDJ"]-stockDataset["KDJ"].min())/(stockDataset["KDJ"].max()-stockDataset["KDJ"].min()))*100

#calculate good buy times and add column to dataframe
stockDataset['Rating'] = stockDataset.apply(lambda row: calcRating(row), axis=1)

#Graph Data
stockDataset.plot.line(y='Close')
#lines = stockDataset.plot.line(y='Close')
buyRating = stockDataset.query("Rating == 'Buy'")
#buyRating.plot(x=buyRating.index, y='Close', c='green', kind='scatter', sharex=True)
sellRating = stockDataset.query("Rating == 'Sell'")
#sellRating.plot(x=sellRating.index, y='Close', c='red', kind='scatter', sharex=True)
#plt.xticks(stockDataset.index,stockDataset['index'])
plt.plot(buyRating.index, buyRating.Close, 'g*')
plt.plot(sellRating.index, sellRating.Close, 'r*')
plt.show()

#Machine Learning
stockDataset = stockDataset.dropna()
y = stockDataset['Rating']
x = stockDataset.drop(['Rating','Open','Close','High','Low'], axis=1)
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
