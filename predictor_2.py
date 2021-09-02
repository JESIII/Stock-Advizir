import os
import fnmatch
from joblib import dump, load
import requests
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
alpha_vantage_key = 'demo'

def predict(stock):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&datatype=csv&adjusted=true&symbol={stock}&interval=30min&outputsize=compact&apikey={alpha_vantage_key}'
    data_stock = pd.read_csv(url)
    url = f'https://www.alphavantage.co/query?function=RSI&datatype=csv&symbol={stock}&interval=30min&apikey={alpha_vantage_key}'
    data_rsi = pd.read_csv(url)
    url = f'https://www.alphavantage.co/query?function=MACD&datatype=csv&symbol={stock}&interval=30min&apikey={alpha_vantage_key}'
    data_macd = pd.read_csv(url)
    url = f'https://www.alphavantage.co/query?function=VWAP&datatype=csv&symbol={stock}&interval=30min&apikey={alpha_vantage_key}'
    data_vwap = pd.read_csv(url)
    url = f'https://www.alphavantage.co/query?function=STOCH&datatype=csv&symbol={stock}&interval=30min&apikey={alpha_vantage_key}'
    data_stoch = pd.read_csv(url)
    url = f'https://www.alphavantage.co/query?function=CPI&datatype=csv&interval=monthly&apikey={alpha_vantage_key}'
    data_cpi = pd.read_csv(url)
    url = f'https://www.alphavantage.co/query?function=CONSUMER_SENTIMENT&datatype=csv&apikey={alpha_vantage_key}'
    data_sent = pd.read_csv(url)
    url = f'https://www.alphavantage.co/query?function=UNEMPLOYMENT&datatype=csv&apikey={alpha_vantage_key}'
    data_unemployment = pd.read_csv(url)
    data = data_stock.join(other=[data_stock, data_rsi, data_macd, data_vwap, data_stoch, data_cpi, data_sent, data_unemployment]) 
    data.fillna(method='ffill', axis='columns', inplace=True)
    data.dropna(inplace=True)
    sector = None # get sector from api
    market_cap_size = None # get market cap size from api
    ratings = {
        0:0,
        1:0,
        2:0
    }
    for filename in os.listdir('./classifiers'):
        if fnmatch.fnmatch(filename, f'{sector}*'):
            neigh : KNeighborsClassifier = load(filename)
            prediction = neigh.predict(data)
            ratings[prediction['rating']] += 1
    if list(ratings.values).count(max(ratings, key=ratings.get)) > 1:
        return 1 # 1 == Hold
    else:
        return max(ratings, key=ratings.get)