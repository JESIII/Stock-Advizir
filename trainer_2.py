import os
import fnmatch
import datetime
from datetime import datetime
from pandas.core.frame import DataFrame
from pandas.core.indexes import category
import pandas as pd
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def add_ratings_to_data(df: DataFrame):
    shift = 30 * 6.5 * 2 #30 days, 6.5 hrs/day, 2 half-hrs/hr
    if (df['Open'] < df.shift(shift)['Open']) and ((df.shift(shift)['Open'] - df['Open']) / df['Open']) * 100 > 10:
        df['rating'] = 'Mega Buy'
    elif (df['Open'] < df.shift(shift)['Open']) and ((df.shift(shift)['Open'] - df['Open']) / df['Open']) * 100 > 2:
        df['rating'] = 'Buy'
    elif (df['Open'] > df.shift(shift)['Open']) and ((df['Open'] - df.shift(shift)['Open']) / df['Open']) * 100 > 10:
        df['rating'] = 'Mega Sell'
    elif (df['Open'] > df.shift(shift)['Open']) and ((df['Open'] - df.shift(shift)['Open']) / df['Open']) * 100 > 2:
        df['rating'] = 'Sell'
    else:
        df['rating'] = 'Hold'
    return df


def train():
    # screen stocks into sectors, market cap sizes

    # find api that can screen by those

    # https://api-v2.intrinio.com/securities/screen
    # need to have top 3 of each sector(aka GICS group)/market cap pair
    training_stocks = {
        'materials_small':[],
        'materials_mid':[],
        'materials_large':[],
        'industrials_small':[],
        'industrials_small':[],
        'industrials_small':[],
        'utilities_small':[],
        'utilities_small':[],
        'utilities_small':[],
        'energy_small':[],
        'energy_small':[],
        'energy_small':[],
        'financials_small':[],
        'financials_small':[],
        'financials_small':[],
        'consumerdiscretionary_small':[],
        'consumerdiscretionary_small':[],
        'consumerdiscretionary_small':[],
        'consumerstaples_small':[],
        'consumerstaples_small':[],
        'consumerstaples_small':[],
    }

    # alpha vantage for stock data

    # we need take into account: 
    # sentiment, country, rsi, macd, 
    # earnings date, price, volume, fear index of overall market

    for category in training_stocks:
        stocks = training_stocks[category]
        for stock in stocks:
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
            data = add_ratings_to_data(data)
            data.dropna(inplace=True)
            category = get_stock_category(stock)
            x = data.drop('rating', axis=1)
            y = data['rating']
            X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1337)
            classifier = MLPClassifier(random_state=1337)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            stats_str = f'Classifier: {category}_{stock}\nAccuracy: {accuracy}\nMSE: {mse}\n----------------------\n'
            file_name = f'{category}_{stock}'
            print(stats_str)
            open(f'./stats/{file_name}.txt').write(stats_str)
            dump(classifier, f'./classifiers/{file_name}.classifier', compress=True)
            # when in predictor, run all 3 for each sector/cap and do majority voting to choose the result

def get_stock_category(stock):
    sector = None # get sector via api
    cap = None # get cap size via api
    category = f'{sector}_{cap}'
    return category
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
    category = get_stock_category(stock)
    ratings = {
        'Mega Buy':0,
        'Buy':0,
        'Hold':0,
        'Mega Sell':0,
        'Sell':0
    }
    for filename in os.listdir('./classifiers'):
        if fnmatch.fnmatch(filename, f'{category}*'):
            classifier : MLPClassifier = load(filename)
            prediction = classifier.predict(data)
            ratings[prediction['rating']] += 1
    if list(ratings.values).count(max(ratings, key=ratings.get)) > 1:
        return 1 # 1 == Hold
    else:
        return max(ratings, key=ratings.get)
        


    