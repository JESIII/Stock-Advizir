import yfinance as yf
from datetime import datetime
from datetime import timedelta
import stockstats as ss
from joblib import dump, load
import pandas_datareader.data as web
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
def getRating(tickr):
    #load data into dataframe and stock dataframe
    # stockDataset = pd.read_csv('D:/Dropbox/Dropbox (CSU Fullerton)/481/zStock-AI/Datasets/AMD.csv')
    # stockDataset.columns = ['Date','High','Low','Open','Close','Volume','AdjClose']
    msft = yf.Ticker(tickr)
    stockDataset = msft.history(period='3mo')
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
    y = stockDataset.drop(['Open','Close','High','Low'], axis=1)
    print(y)
    clf = load('TrainedModel.joblib')
    y_pred = clf.predict(y)
    print(y_pred[-10:])
if __name__ == "__main__":
    print('Enter a ticker: ')
    tickr = input()
    getRating(tickr)