import os
import fnmatch
import datetime
from datetime import datetime
import json
import pandas as pd
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
from finta import TA
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import requests
alpha_vantage_key = '0KLB6EE5V872NI82'
fmp_key = 'ab95806044be10a3e4f5868c4d8f4ac7'
def add_ratings_to_data(df: pd.DataFrame):
    shift = 30 * 6.5 * 2 #30 days, 6.5 hrs/day, 2 half-hrs/hr
    df['rating'] = 'hold'
    df.loc[((df['open'] < df.shift(shift)['open']) and ((df.shift(shift)['open'] - df['open']) / df['open']) * 100 > 10), df['rating']] = 'mega buy'
    df.loc[((df['open'] < df.shift(shift)['open']) and ((df.shift(shift)['open'] - df['open']) / df['open']) * 100 > 2), df['rating']] = 'buy'
    df.loc[((df['open'] > df.shift(shift)['open']) and ((df['open'] - df.shift(shift)['open']) / df['open']) * 100 > 10), df['rating']] = 'mega sell'
    df.loc[((df['open'] > df.shift(shift)['open']) and ((df['open'] - df.shift(shift)['open']) / df['open']) * 100 > 2), df['rating']] = 'sell'

def train():
    # screen stocks into sectors, market cap sizes
    small_cap_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=300000000&marketCapLowerThan=2000000001&limit=3&apikey={fmp_key}'
    mid_cap_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=2000000000&marketCapLowerThan=10000000001&limit=3&apikey={fmp_key}'
    large_cap_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=10000000000&limit=3&apikey={fmp_key}'
    # find api that can screen by those

    # https://api-v2.intrinio.com/securities/screen
    # need to have top 3 of each sector(aka GICS group)/market cap pair
    training_stocks = {
        'Consumer Cyclical':[],
        'Energy':[],
        'Technology':[],
        'Industrials':[],
        'Financial Services':[],
        'Basic Materials':[],
        'Communication Services':[],
        'Consumer Defensive':[],
        'Healthcare':[],
        'Real Estate':[],
        'Utilities':[],
        'Industrial Goods':[],
        'Financial':[],
        'Services':[],
        'Conglomerates':[]
    }
    for sector in training_stocks:
        url = small_cap_url + f'&sector={sector}'
        r = requests.get(url)
        data = r.json()
        for stock in data:
            training_stocks[sector].append(stock['symbol'])
        url = mid_cap_url + f'&sector={sector}'
        r = requests.get(url)
        data = r.json()
        for stock in data:
            training_stocks[sector].append(stock['symbol'])
        url = large_cap_url + f'&sector={sector}'
        r = requests.get(url)
        data = r.json()
        for stock in data:
            training_stocks[sector].append(stock['symbol'])
        time.sleep(1)
    # alpha vantage for stock data

    # we need take into account: 
    # sentiment, country, rsi, macd, 
    # earnings date, price, volume, fear index of overall market

    for sector in training_stocks:
        stocks = training_stocks[sector]
        for stock in stocks:
            data = get_and_clean_data(stock)
            data = add_ratings_to_data(data)
            data.dropna(inplace=True)
            category = get_stock_category(stock)
            x = data.drop(['open', 'high', 'low', 'close', 'volume', 'rating'], axis=1)
            y = data['rating']
            X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1337)
            classifier = KNeighborsClassifier()
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            stats_str = f'Classifier: {category}_{stock}\nAccuracy: {accuracy}\nMSE: {mse}\n----------------------\n'
            file_name = f'{category}_{stock}'
            print(stats_str)
            open(f'./stats/{file_name}.txt').write(stats_str)
            dump(classifier, f'./classifiers/{file_name}.classifier', compress=True)
            pass
            # when in predictor, run all 3 for each sector/cap and do majority voting to choose the result
def get_and_clean_data(stock):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&datatype=csv&adjusted=true&symbol={stock}&interval=30min&outputsize=compact&apikey={alpha_vantage_key}'
    data = pd.read_csv(url, index_col='timestamp')
    data_rsi = TA.STOCHRSI(data)
    data_macd = TA.MACD(data)
    data_stoch = TA.STOCH(data)
    data_vzo = TA.VZO(data)
    url = f'https://www.alphavantage.co/query?function=CCI&time_period=60&datatype=csv&symbol={stock}&interval=30min&apikey={alpha_vantage_key}'
    data_cci = pd.read_csv(url)
    url = f'https://www.alphavantage.co/query?function=CONSUMER_SENTIMENT&datatype=csv&apikey={alpha_vantage_key}'
    data_sent = pd.read_csv(url)
    url = f'https://www.alphavantage.co/query?function=UNEMPLOYMENT&datatype=csv&apikey={alpha_vantage_key}'
    data_unemployment = pd.read_csv(url)
    data = data.merge(data_cci, how='left', left_index=True, right_index=True) 
    data = data.merge(data_sent, how='left', left_index=True, right_index=True)
    data = data.merge(data_unemployment, how='left', left_index=True, right_index=True)
    data['vzo'] = data_vzo
    data['stoch'] = data_stoch
    data['macd'] = data_macd
    data['rsi'] = data_rsi
    x = data.drop(['open', 'high', 'low', 'close', 'volume']).values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)
    data.interpolate(axis='columns', inplace=True)
    data.dropna(inplace=True)
    return data
def get_stock_category(stock):
    url = f'https://financialmodelingprep.com/api/v3/profile/{stock}&apikey={fmp_key}'
    data = requests.get(url).json()
    sector = data['sector']
    cap = data['mktCap']
    if cap > 10_000_000_000:
        cap = 'large'
    elif cap < 10_000_000_000 and cap > 2_000_000_000:
        cap = 'mid'
    else:
        cap = 'small'
    category = f'{sector}_{cap}'
    return category
def predict(stock):
    data = get_and_clean_data(stock)
    category = get_stock_category(stock)
    ratings = {
        'mega buy':0,
        'buy':0,
        'hold':0,
        'mega sell':0,
        'sell':0
    }
    for filename in os.listdir('./classifiers'):
        if fnmatch.fnmatch(filename, f'{category}*'):
            classifier : KNeighborsClassifier = load(filename)
            prediction = classifier.predict(data)
            ratings[prediction['rating']] += 1
    if list(ratings.values).count(max(ratings, key=ratings.get)) > 1:
        return 1 # 1 == Hold
    else:
        return max(ratings, key=ratings.get)
# alpha_vantage_key = '0KLB6EE5V872NI82'
# import pandas as pd
# from finta import TA
# url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&datatype=csv&adjusted=true&symbol=AMD&interval=30min&outputsize=compact&apikey={alpha_vantage_key}'
# data_stock = pd.read_csv(url, index_col='timestamp')
# data_stock['STOCHRSI'] = TA.STOCHRSI(data_stock)
# data_stock.tail()
train()