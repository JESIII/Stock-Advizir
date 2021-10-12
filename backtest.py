from genericpath import exists
from numpy import average, typeDict
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.io.pytables import IndexCol
from stock_advizir import predict
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import math, time, schedule
from joblib import dump, load
def auto_trade(stocks):
    positions, trades = DataFrame, DataFrame
    balance = float(100000)
    if exists('./backtesting/trades.csv'):
        trades = pd.read_csv('./backtesting/trades.csv', parse_dates=True, index_col='date')
    else:
        trades = DataFrame({'stock':str, 'date':type(datetime.today().date()), 'price':float, 'qty':int, 'side':str, 'effect':float}, index=[])
        trades.set_index('date', inplace=True)
    if exists('./backtesting/positions.csv'):
        positions = pd.read_csv('./backtesting/positions.csv', index_col='stock')
    else:
        positions = DataFrame({'stock':str, 'avg_price':float, 'price':float, 'qty':int, 'P/L%':float}, index=[])
    if exists('./backtesting/balance.dump'):
        balance = load('./backtesting/balance.dump')
    else:
        balance = float(100000)
    for stock in stocks:
        time.sleep(60)
        try:
            ticker = yf.Ticker(stock)
            ticker_info = ticker.info
            prediction = None
        except Exception as e:
            print(e)
        try:
            prediction = pd.read_json(predict(stock), keep_default_dates=True)
        except Exception as e:
            print(f'Error with prediction: {e}')
            time.sleep(60)
            continue
        prediction.set_index('Date', inplace=True)
        last_row = prediction.iloc[-1]
        if last_row['rating'] == 'buy':
            try:
                qty = 2500 / last_row['Open']
                effect = -qty*ticker_info['regularMarketPrice']
                if effect + balance < 0:
                    time.sleep(60)
                    continue
                order = {'date':datetime.today().date(), 'stock':stock, 'price':ticker_info['regularMarketPrice'], 'qty':qty,'side':'buy', 'effect':effect}
                balance = balance + effect
                trades = trades.append(order, ignore_index=True)
                try:
                    positions.at[stock, 'avg_price'] = (positions.at[stock, 'avg_price'] + ticker_info['regularMarketPrice']) / 2
                    positions.at[stock, 'qty'] += qty
                    positions.at[stock, 'P/L%'] = ((ticker_info['regularMarketPrice']/positions.at[stock, 'avg_price'])-1) * 100
                except:
                    try:
                        new_pos = {'stock':stock, 'avg_price':ticker_info['regularMarketPrice'], 'price':ticker_info['regularMarketPrice'], 'qty':qty, 'P/L%':0}
                        positions = positions.append(new_pos, ignore_index=True)
                    except Exception as e:
                        print(e)
                print(order)
            except Exception as e:
                print(e)
        elif last_row['rating'] == 'mega buy':
            try:
                qty = 5000 / last_row['Open']
                effect = -qty*ticker_info['regularMarketPrice']
                if effect + balance < 0:
                    time.sleep(60)
                    continue
                order = {'date':datetime.today().date(), 'stock':stock, 'price':ticker_info['regularMarketPrice'], 'qty':qty,'side':'buy', 'effect':effect}
                balance = balance + effect
                trades = trades.append(order, ignore_index=True)
                try:
                    positions.at[stock, 'avg_price'] = (positions.at[stock, 'avg_price'] + ticker_info['regularMarketPrice']) / 2
                    positions.at[stock, 'qty'] += qty
                    positions.at[stock, 'P/L%'] = ((ticker_info['regularMarketPrice']/positions.at[stock, 'avg_price'])-1) * 100
                except:
                    try:
                        new_pos = {'stock':stock, 'avg_price':ticker_info['regularMarketPrice'], 'price':ticker_info['regularMarketPrice'], 'qty':qty, 'P/L%':0}
                        positions = positions.append(new_pos, ignore_index=True)
                    except Exception as e:
                        print(e)
                print(order)
            except Exception as e:
                print(e)
        elif last_row['rating'] == 'sell':
            try:
                qty = math.floor(positions.at[stock, 'qty']*.75)
                if qty == 0:
                    time.sleep(60)
                    continue
                effect = qty*ticker_info['regularMarketPrice']
                order = {'date':datetime.today().date(), 'stock':stock, 'price':ticker_info['regularMarketPrice'], 'qty':qty,'side':'sell', 'effect':effect}
                balance = balance + effect
                trades = trades.append(order, ignore_index=True)
                positions.at[stock, 'qty'] -= qty
                print(order)
            except Exception as e:
                print(e)
        elif last_row['rating'] == 'mega sell':
            try:
                qty = math.floor(positions.at[stock, 'qty'])
                if qty == 0:
                    time.sleep(60)
                    continue
                effect = qty*ticker_info['regularMarketPrice']
                order = {'date':datetime.today().date(), 'stock':stock, 'price':ticker_info['regularMarketPrice'], 'qty':qty,'side':'sell', 'effect':effect}
                balance = balance + effect
                trades = trades.append(order, ignore_index=True)
                positions.at[stock, 'qty'] -= qty
                print(order)
            except Exception as e:
                print(e)
        else:
            print(f'Hold on {stock}')
        time.sleep(60)
    dump(balance, './backtesting/balance.dump')
    positions.to_csv('./backtesting/positions.csv')
    trades.to_csv('./backtesting/trades.csv')
def update_PL():
    print('Updating P/L...')
    if exists('./backtesting/positions.csv'):
        positions = pd.read_csv('./backtesting/positions.csv', index_col='stock')
        positions['P/L%'] = positions.apply(lambda x: calc_PL(x), axis=1)
        positions['price'] = positions.apply(lambda x: x['avg_price']*(1+(x['P/L%']/100)), axis=1)
        positions.to_csv('./backtesting/positions.csv')
    print('Finished Updating P/L...')
def calc_PL(x):
    PL = ((yf.Ticker(x.name).info['regularMarketPrice']/x['avg_price'])-1) * 100
    time.sleep(30)
    return PL
def backtest():
    stocks = [ 
    'amd', 'msft', 'amzn', 'spce', 'msft', 
    'mu', 'nvda', 'intc', 'dkng', 'bac', 
    'v', 'coin', 'gme', 'amc', 'hood', 'aal', 
    'mgm', 'hd','logi', 'wmt', 'spot', 'fb', 
    'ge', 'cgc', 'tlry', 't', 'pltr', 'nclh', 
    'pfe', 'f', 'gm', 'xom']
    
    schedule.every().monday.at("15:00").do(auto_trade(stocks),'Monday Trade')
    schedule.every().tuesday.at("15:00").do(auto_trade(stocks),'Tuesday Trade')
    schedule.every().wednesday.at("15:00").do(auto_trade(stocks),'Wednesday Trade')
    schedule.every().thursday.at("15:00").do(auto_trade(stocks),'Thursday Trade')
    schedule.every().friday.at("15:00").do(auto_trade(stocks),'Friday Trade')
    schedule.every().monday.hour.at("18:00").until("22:00").do(update_PL,'Update Profit/loss')
    schedule.every().tuesday.hour.at("18:00").until("22:00").do(update_PL,'Update Profit/loss')
    schedule.every().wednesday.hour.at("18:00").until("22:00").do(update_PL,'Update Profit/loss')
    schedule.every().thursday.hour.at("18:00").until("22:00").do(update_PL,'Update Profit/loss')
    schedule.every().friday.hour.at("18:00").until("22:00").do(update_PL,'Update Profit/loss')
    while True:
        print('Pending tasks: ', schedule.get_jobs())
        print('Running pending tasks...')
        schedule.run_pending()
        time.sleep(600) # wait 10 mins
        print('Finished pending tasks...')
backtest()