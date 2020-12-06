import yfinance as yf
import stockstats as ss
from joblib import load
import matplotlib.pyplot as plt
def getRating(tickr, period):
    #load data into dataframe and stock dataframe
    msft = yf.Ticker(tickr)
    stockDataset = msft.history(period=period)
    #print(stockDataset)
    stockDF = stockDataset.copy()
    #get data indicators
    stockDF = ss.StockDataFrame.retype(stockDF)
    macdh = stockDF['macdh']
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
    dataFrameCopy = stockDataset.copy()
    dataFrameCopy = dataFrameCopy[-90:]
    #Normalize Data
    stockDataset["TRIX"]=((stockDataset["TRIX"]-stockDataset["TRIX"].min())/(stockDataset["TRIX"].max()-stockDataset["TRIX"].min()))*100
    stockDataset["RollingAvg"]=((stockDataset["RollingAvg"]-stockDataset["RollingAvg"].min())/(stockDataset["RollingAvg"].max()-stockDataset["RollingAvg"].min()))*20
    stockDataset["Volume"]=((stockDataset["Volume"]-stockDataset["Volume"].min())/(stockDataset["Volume"].max()-stockDataset["Volume"].min()))*100
    stockDataset["KDJ"]=((stockDataset["KDJ"]-stockDataset["KDJ"].min())/(stockDataset["KDJ"].max()-stockDataset["KDJ"].min()))*100
    y = stockDataset.drop(['Open','Close','High','Low'], axis=1)
    #print(y)
    clf = load('TrainedModel.joblib')
    y_pred = clf.predict(y)
    dataFrameCopy['Rating'] = y_pred[-90:]
    dataFrameCopy = dataFrameCopy.drop(['High','Low', 'Stock Splits', 'Dividends'], axis=1)
    dataFrameCopy.plot.line(y='Close')
    buyRating = dataFrameCopy.query("Rating == 'Buy'")
    sellRating = dataFrameCopy.query("Rating == 'Sell'")
    plt.plot(buyRating.index, buyRating.Close, 'g*')
    plt.plot(sellRating.index, sellRating.Close, 'r*')
    print(dataFrameCopy)
    plt.show()
    #print(y_pred[-30:])
    
if __name__ == "__main__":
    print('Enter a ticker: ')
    tickr = input()
    print('Enter a period: 3mo,6mo,1y,2y,5y,10y,ytd')
    period = input()
    getRating(tickr, period)