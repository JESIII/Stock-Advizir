import yfinance as yf
import stockstats as ss
from joblib import load
import matplotlib.pyplot as plt
def get_rating(tickr, period):
    #load data into dataframe and stock dataframe
    stock = yf.Ticker(tickr)
    stock_dataset = stock.history(period=period)
    stock_DF = stock_dataset.copy()
    #get data indicators
    stock_DF = ss.StockDataFrame.retype(stock_DF)
    macdh = stock_DF['macdh']
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
    df_copy = stock_dataset.copy()
    df_copy = df_copy[-90:]
    #Normalize Data
    stock_dataset["TRIX"]=((stock_dataset["TRIX"]-stock_dataset["TRIX"].min())/(stock_dataset["TRIX"].max()-stock_dataset["TRIX"].min()))*100
    stock_dataset["RollingAvg"]=((stock_dataset["RollingAvg"]-stock_dataset["RollingAvg"].min())/(stock_dataset["RollingAvg"].max()-stock_dataset["RollingAvg"].min()))*20
    stock_dataset["Volume"]=((stock_dataset["Volume"]-stock_dataset["Volume"].min())/(stock_dataset["Volume"].max()-stock_dataset["Volume"].min()))*100
    stock_dataset["KDJ"]=((stock_dataset["KDJ"]-stock_dataset["KDJ"].min())/(stock_dataset["KDJ"].max()-stock_dataset["KDJ"].min()))*100
    y = stock_dataset.drop(['Open','Close','High','Low'], axis=1)
    #print(y)
    clf = load('TrainedModel.joblib')
    y_pred = clf.predict(y)
    df_copy['Rating'] = y_pred[-90:]
    df_copy = df_copy.drop(['High','Low', 'Stock Splits', 'Dividends'], axis=1)
    df_copy.plot.line(y='Close')
    buy_rating = df_copy.query("Rating == 'Buy'")
    sell_rating = df_copy.query("Rating == 'Sell'")
    plt.plot(buy_rating.index, buy_rating.Close, 'g*')
    plt.plot(sell_rating.index, sell_rating.Close, 'r*')
    print(df_copy)
    plt.show()
    
if __name__ == "__main__":
    print('Enter a ticker: ')
    tickr = input()
    print('Enter a period: 3mo,6mo,1y,2y,5y,10y,ytd')
    period = input()
    get_rating(tickr, period)