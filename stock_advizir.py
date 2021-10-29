from math import nan
from operator import index
import os
import fnmatch
import datetime
from datetime import datetime
import json
from numpy import NaN, greater
import pandas as pd
from matplotlib import dates
import matplotlib.pyplot as plt
from joblib import dump, load
from pandas.core.frame import DataFrame
from finta import TA
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.signal import argrelextrema
import time
import numpy as np
import requests
keys = open('./keys.txt', 'r').readlines()
alpha_vantage_key = keys[0].split(':')[1].strip()
fmp_key = keys[1].split(':')[1].strip()
finnhub_key = keys[3].split(':')[1].strip()
def add_ratings_to_data(df: pd.DataFrame):
    df['rating'] = 'hold'
    df['local_max'] = df.iloc[argrelextrema(df['Open'].values, np.greater, order=50)[0]]['Open']
    df['local_max'] = df['local_max'].notnull().astype('bool')
    df['local_min'] = df.iloc[argrelextrema(df['Open'].values, np.less, order=50)[0]]['Open']
    df['local_min'] = df['local_min'].notnull().astype('bool')
    df.loc[(((df['Open'] < df.shift(-30)['Open']) & (df['Open'] < df.shift(-100)['Open']) & ((df.shift(-50)['Open'] - df['Open']) / df['Open'] * 100 > 3) & ~df['local_max']) | (df['local_min'] & ((df.shift(-50)['Open'] - df['Open']) / df['Open'] * 100 > 3))), 'rating'] = 'buy'
    df.loc[((df['Open'] < df.shift(-30)['Open']) & (df['Open'] < df.shift(-100)['Open']) & ((df.shift(-50)['Open'] - df['Open']) / df['Open'] * 100 > 10) & ~df['local_max']), 'rating'] = 'mega buy'
    df.loc[((df['Open'] > df.shift(-30)['Open']) & (df['Open'] > df.shift(-100)['Open']) & ((df['Open'] - df.shift(-50)['Open']) / df['Open'] * 100 > 5) & ~df['local_min']), 'rating'] = 'sell'
    df.loc[(((df['Open'] > df.shift(-30)['Open']) & (df['Open'] > df.shift(-100)['Open']) & ((df['Open'] - df.shift(-50)['Open']) / df['Open'] * 100 > 10)  & ~df['local_min']) | (df['local_max'] & ((df['Open'] - df.shift(-50)['Open']) / df['Open'] * 100 > 5))), 'rating'] = 'mega sell'
    df.drop(labels=['local_max','local_min'], axis=1, inplace=True)
    df.drop(df.tail(100).index, inplace=True)
    return df

def train():
    # screen stocks into sectors, market cap sizes
    small_cap_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=300000000&marketCapLowerThan=2000000001&limit=3&apikey={fmp_key}'
    mid_cap_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=2000000000&marketCapLowerThan=10000000001&limit=3&apikey={fmp_key}'
    large_cap_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=10000000000&limit=3&apikey={fmp_key}'

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
    # url = f'https://www.alphavantage.co/query?function=CONSUMER_SENTIMENT&datatype=csv&apikey={alpha_vantage_key}'
    # data_sent = pd.read_csv(url, index_col='timestamp', parse_dates=True)

    # data_sent = pd.read_csv('./data/sentiment.csv', index_col='timestamp', parse_dates=True)

    # for sector in training_stocks:
    #     url = small_cap_url + f'&sector={sector}'
    #     r = requests.get(url)
    #     data = r.json()
    #     for stock in data:
    #         training_stocks[sector].append(stock['symbol'])
    #     url = mid_cap_url + f'&sector={sector}'
    #     r = requests.get(url)
    #     data = r.json()
    #     for stock in data:
    #         training_stocks[sector].append(stock['symbol'])
    #     url = large_cap_url + f'&sector={sector}'
    #     r = requests.get(url)
    #     data = r.json()
    #     for stock in data:
    #         training_stocks[sector].append(stock['symbol'])
    #     time.sleep(1)

    training_stocks = load('./training_stocks_dict.dict')
    data_cci = None
    for f in os.listdir('./data/'):
        if f.startswith('cci_data'):
            todays_date = datetime.now().strftime("%Y-%m-%d")
            date_in_dir = f.split('_')[2]
            if date_in_dir != todays_date:
                os.remove(f'./data/{f}')
    if os.path.exists('./data/cci_data_{datetime.now().strftime("%Y-%m-%d")}'):
        data_cci = load('./data/cci_data_{datetime.now().strftime("%Y-%m-%d")}')
    else:
        data_cci = pd.read_csv('https://stats.oecd.org/sdmx-json/data/DP_LIVE/.CCI.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en', index_col='TIME', usecols=['TIME', 'Value'], parse_dates=True)
        data_cci.index.names = ['Date']
        dump(data_cci, f'./data/cci_data_{datetime.now().strftime("%Y-%m-%d")}')
    for sector in training_stocks:
        stocks = training_stocks[sector]
        for stock in stocks:
            time.sleep(5)
            try:
                data = get_and_clean_data(stock, data_cci)
                data = add_ratings_to_data(data)
                data.dropna(inplace=True)
                data.to_csv(f'./data/{stock}.csv')
                category = get_stock_category(stock)
                plot_data(data, stock, mode='training')
                x = data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'rating'], axis=1)
                y = data['rating']
                X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1337)
                classifier = RandomForestClassifier(random_state=1337, min_samples_split=4, bootstrap=False, n_jobs=-1, n_estimators=50, max_depth=20)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                accuracy = metrics.accuracy_score(y_test, y_pred)
                y_test_adj = y_test.copy()
                y_pred_adj = y_pred.copy()
                for i, each in enumerate(y_test_adj):
                    if each == 'mega buy':
                        y_test_adj[i] = 'buy'
                    elif each == 'mega sell':
                        y_test_adj[i] = 'sell'
                for i, each in enumerate(y_pred_adj):
                    if each == 'mega buy':
                        y_pred_adj[i] = 'buy'
                    elif each == 'mega sell':
                        y_pred_adj[i] = 'sell'
                adjusted_accuracy = metrics.accuracy_score(y_test_adj, y_pred_adj)
                plot_cm(classifier, X_test, y_test, stock)
                stats_str = f'Classifier:{category}_{stock}\nAccuracy:{accuracy}\nAdjusted Accuracy Score:{adjusted_accuracy}'
                file_name = f'{category}_{stock}'
                print(stats_str + '\n----------------------')
                open(f'./stats/{file_name}.txt', "w").write(stats_str)
                dump(classifier, f'./classifiers/{file_name}.classifier', compress=3)
            except Exception as e:
                print(f'Something broke with {stock}:\n{e}')
            # when in predictor, run all 3 for each sector/cap and do majority voting to choose the result
def plot_cm(clf, X_test, y_test, stock):
    plot_confusion_matrix(clf, X_test, y_test)
    plt.savefig(f'./figs/confusion_matrix/{stock}.png')
    plt.close()
def get_and_clean_data(stock, data_cci, period='max'):
    time.sleep(30)
    yf_obj = yf.Ticker(stock)
    data = yf_obj.history(period=period, interval='1d')
    data.index = pd.to_datetime(data.index)
    data_price = data.copy()
    data_rsi = TA.STOCHRSI(data)
    data_macd = TA.MACD(data)
    data_stoch = TA.STOCH(data)
    data_vzo = TA.VZO(data)
    data = data.join(data_cci, how='left', on='Date') 
    data = data.join(data_rsi, how='left', on='Date') 
    data = data.join(data_macd, how='left', on='Date') 
    data = data.join(data_stoch, how='left', on='Date') 
    data = data.join(data_vzo, how='left', on='Date')
    data = data.drop(labels=['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    scaler = preprocessing.MinMaxScaler() 
    data.interpolate(axis=0, inplace=True)
    scaled_values = scaler.fit_transform(data) 
    data.loc[:,:] = scaled_values
    data = data.join(data_price.drop(labels=['Dividends', 'Stock Splits'], axis=1), how='left')
    data.interpolate(axis='columns', inplace=True)
    data.interpolate(axis=0, inplace=True)
    return data
def get_stock_category(stock):
    time.sleep(30)
    try:
        ticker = yf.Ticker(stock)
        sector = ticker.info['sector']
        cap = ticker.info['marketCap']
        if cap > 10_000_000_000:
            cap = 'large'
        elif cap < 10_000_000_000 and cap > 2_000_000_000:
            cap = 'mid'
        else:
            cap = 'small'
        sector = sector.replace(' ', '')
        category = f'{sector}_{cap}'
    except:
        return 'ConsumerCyclical_mid'
    return category
def predict(stock):
    data_cci = None
    for f in os.listdir('./data/'):
        if f.startswith('cci_data'):
            todays_date = datetime.now().strftime("%Y-%m-%d")
            date_in_dir = f.split('_')[2]
            if date_in_dir != todays_date:
                os.remove(f'./data/{f}')
    if os.path.exists('./data/cci_data_{datetime.now().strftime("%Y-%m-%d")}'):
        data_cci = load('./data/cci_data_{datetime.now().strftime("%Y-%m-%d")}')
    else:
        data_cci = pd.read_csv('https://stats.oecd.org/sdmx-json/data/DP_LIVE/.CCI.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en', index_col='TIME', usecols=['TIME', 'Value'], parse_dates=True)
        data_cci.index.names = ['Date']
        dump(data_cci, f'./data/cci_data_{datetime.now().strftime("%Y-%m-%d")}')
    data : DataFrame = get_and_clean_data(stock, data_cci, '1y')
    data_copy = data.copy()
    category = get_stock_category(stock)
    classifiers = []
    for filename in os.listdir('./classifiers'):
        if fnmatch.fnmatch(filename, f'{category}*'):
            classifier : RandomForestClassifier = load(f'./classifiers/{filename}')
            classifiers.append(classifier)
    data = data.drop(labels=['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    predictions = []
    for classifier in classifiers:
        predictions.append(classifier.predict(data)) 
    prediction = []
    for i in range(len(predictions[0])):
        ratings = {
        'buy':0,
        'hold':0,
        'mega buy':0,
        'mega sell':0,
        'sell':0
        }
        if len(predictions) >= 3:
            ratings[predictions[0][i]] += 1
            ratings[predictions[1][i]] += 1
            ratings[predictions[2][i]] += 1
            prediction.append(max(ratings, key=ratings.get))
        elif len(predictions) == 2:
            if ratings[predictions[0][i]] == ratings[predictions[1][i]]:
                prediction.append(ratings[predictions[0][i]])
            else:
                prediction.append('hold')
        elif len(predictions) == 1:
            prediction.append(ratings[predictions[0][i]])
        else:
            prediction.append('hold')
    data_copy['rating'] = prediction
    #plot_data(data_copy, stock, mode='predicting')
    return json.dumps(json.loads(data_copy.reset_index().to_json(orient='records')), indent=2)
def predict10y(stock):
    data_cci = None
    for f in os.listdir('./data/'):
        if f.startswith('cci_data'):
            todays_date = datetime.now().strftime("%Y-%m-%d")
            date_in_dir = f.split('_')[2]
            if date_in_dir != todays_date:
                os.remove(f'./data/{f}')
    if os.path.exists('./data/cci_data_{datetime.now().strftime("%Y-%m-%d")}'):
        data_cci = load('./data/cci_data_{datetime.now().strftime("%Y-%m-%d")}')
    else:
        data_cci = pd.read_csv('https://stats.oecd.org/sdmx-json/data/DP_LIVE/.CCI.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en', index_col='TIME', usecols=['TIME', 'Value'], parse_dates=True)
        data_cci.index.names = ['Date']
        dump(data_cci, f'./data/cci_data_{datetime.now().strftime("%Y-%m-%d")}')
    data : DataFrame = get_and_clean_data(stock, data_cci, '10y')
    data_copy = data.copy()
    category = get_stock_category(stock)
    classifiers = []
    for filename in os.listdir('./classifiers'):
        if fnmatch.fnmatch(filename, f'{category}*'):
            classifier : RandomForestClassifier = load(f'./classifiers/{filename}')
            classifiers.append(classifier)
    data = data.drop(labels=['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    predictions = []
    for classifier in classifiers:
        predictions.append(classifier.predict(data)) 
    prediction = []
    for i in range(len(predictions[0])):
        ratings = {
        'buy':0,
        'hold':0,
        'mega buy':0,
        'mega sell':0,
        'sell':0
        }
        if len(predictions) >= 3:
            ratings[predictions[0][i]] += 1
            ratings[predictions[1][i]] += 1
            ratings[predictions[2][i]] += 1
            prediction.append(max(ratings, key=ratings.get))
        elif len(predictions) == 2:
            if ratings[predictions[0][i]] == ratings[predictions[1][i]]:
                prediction.append(ratings[predictions[0][i]])
            else:
                prediction.append('hold')
        elif len(predictions) == 1:
            prediction.append(ratings[predictions[0][i]])
        else:
            prediction.append('hold')
    data_copy['rating'] = prediction
    #plot_data(data_copy, stock, mode='predicting')
    return json.dumps(json.loads(data_copy.reset_index().to_json(orient='records')), indent=2)

def plot_data(data: pd.DataFrame, stock, mode):
    buys = data.loc[lambda data: data['rating'] == 'buy']
    mega_buys = data.loc[lambda data: data['rating'] == 'mega buy']
    mega_sells = data.loc[lambda data: data['rating'] == 'mega sell']
    sells = data.loc[lambda data: data['rating'] == 'sell']
    plt.figure(figsize=(30, 14))
    plt.plot(data.index, data.Open)
    plt.plot(mega_buys.index, mega_buys.Open, 'g*')
    plt.plot(buys.index, buys.Open, 'go')
    plt.plot(mega_sells.index, mega_sells.Open, 'r*')
    plt.plot(sells.index, sells.Open, 'ro')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.title(stock)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(dates.MonthLocator([1, 5, 9]))
    ax.xaxis.set_minor_formatter(dates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(dates.YearLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    if mode == 'training':
        plt.savefig(f'./figs/training_data/{stock}.png')
    elif mode == 'predicting':
        plt.savefig(f'./figs/prediction_data/{stock}.png')
    plt.close()