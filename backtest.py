from genericpath import exists
from numpy import average, typeDict
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.io.pytables import IndexCol
from stock_advizir import predict
import traceback
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
        trades = pd.read_csv('./backtesting/trades.csv', parse_dates=True, index_col='trade')
    else:
        print('Can\'t find ./backtesting/trades.csv')
        return
    if exists('./backtesting/positions.csv'):
        positions = pd.read_csv('./backtesting/positions.csv', index_col='stock')
    else:
        print('Can\'t find ./backtesting/positions.csv')
        return
    if exists('./backtesting/balance.dump'):
        balance = load('./backtesting/balance.dump')
    else:
        balance = float(100000)
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock)
            ticker_info = ticker.info
            prediction = None
        except Exception:
            traceback.print_exc()
            time.sleep(60)
            continue
        try:
            prediction = pd.read_json(predict(stock), keep_default_dates=True)
        except Exception:
            traceback.print_exc()
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
                order = pd.DataFrame({'stock':stock, 'price':ticker_info['regularMarketPrice'], 'qty':qty,'side':'buy', 'effect':effect, 'date':datetime.today().date()}, index=[trades.index.values[-1]+1])
                order.index.rename('trade')
                balance = balance + effect
                trades = trades.append(order)
                try:
                    positions.at[stock, 'avg_price'] = (positions.at[stock, 'avg_price'] + ticker_info['regularMarketPrice']) / 2
                    positions.at[stock, 'qty'] += qty
                    positions.at[stock, 'P/L%'] = ((ticker_info['regularMarketPrice']/positions.at[stock, 'avg_price'])-1) * 100
                except:
                    try:
                        new_pos = pd.DataFrame({'avg_price':ticker_info['regularMarketPrice'], 'price':ticker_info['regularMarketPrice'], 'qty':qty, 'P/L%':0}, index=[stock])
                        new_pos.index.rename('stock')
                        positions = positions.append(new_pos)
                    except Exception:
                        traceback.print_exc()
                        time.sleep(60)
                        continue
                print(order)
            except Exception:
                traceback.print_exc()
                time.sleep(60)
                continue
        elif last_row['rating'] == 'mega buy':
            try:
                qty = 5000 / last_row['Open']
                effect = -qty*ticker_info['regularMarketPrice']
                if effect + balance < 0:
                    continue
                order = pd.DataFrame({'stock':stock, 'price':ticker_info['regularMarketPrice'], 'qty':qty,'side':'buy', 'effect':effect, 'date':datetime.today().date()}, index=[trades.index.values[-1]+1])
                order.index.rename('trade')
                balance = balance + effect
                trades = trades.append(order)
                try:
                    positions.at[stock, 'avg_price'] = (positions.at[stock, 'avg_price'] + ticker_info['regularMarketPrice']) / 2
                    positions.at[stock, 'qty'] += qty
                    positions.at[stock, 'P/L%'] = ((ticker_info['regularMarketPrice']/positions.at[stock, 'avg_price'])-1) * 100
                except:
                    try:
                        new_pos = pd.DataFrame({'avg_price':ticker_info['regularMarketPrice'], 'price':ticker_info['regularMarketPrice'], 'qty':qty, 'P/L%':0}, index=[stock])
                        new_pos.index.rename('stock')
                        positions = positions.append(new_pos)
                    except Exception:
                        traceback.print_exc()
                        time.sleep(60)
                        continue
                print(order)
            except Exception:
                traceback.print_exc()
                time.sleep(60)
                continue
        elif last_row['rating'] == 'sell':
            try:
                qty = math.floor(positions.at[stock, 'qty']*.75)
                if qty == 0:
                    time.sleep(60)
                    continue
                effect = qty*ticker_info['regularMarketPrice']
                order = pd.DataFrame({'stock':stock, 'price':ticker_info['regularMarketPrice'], 'qty':qty,'side':'buy', 'effect':effect, 'date':datetime.today().date()}, index=[trades.index.values[-1]+1])
                order.index.rename('trade')
                balance = balance + effect
                trades = trades.append(order)
                positions.at[stock, 'qty'] -= qty
                print(order)
            except Exception:
                traceback.print_exc()
                time.sleep(60)
                continue
        elif last_row['rating'] == 'mega sell':
            try:
                qty = math.floor(positions.at[stock, 'qty'])
                if qty == 0:
                    time.sleep(60)
                    continue
                effect = qty*ticker_info['regularMarketPrice']
                order = pd.DataFrame({'stock':stock, 'price':ticker_info['regularMarketPrice'], 'qty':qty,'side':'buy', 'effect':effect, 'date':datetime.today().date()}, index=[trades.index.values[-1]+1])
                order.index.rename('trade')
                balance = balance + effect
                trades = trades.append(order)
                positions.at[stock, 'qty'] -= qty
                print(order)
            except Exception:
                traceback.print_exc()
                time.sleep(60)
                continue
        else:
            print(f'Hold on {stock}')
        time.sleep(60)
    dump(balance, './backtesting/balance.dump')
    positions.to_csv('./backtesting/positions.csv', index_label='stock')
    trades.to_csv('./backtesting/trades.csv', index_label='trade')
def update_PL():
    print('Updating P/L...')
    if exists('./backtesting/positions.csv'):
        positions = pd.read_csv('./backtesting/positions.csv', index_col='stock')
        positions['P/L%'] = positions.apply(calc_PL, axis=1)
        positions['price'] = positions['avg_price']*(1+(positions['P/L%']/100))
        positions.to_csv('./backtesting/positions.csv', index_label='stock')
    print('Finished Updating P/L...')
def calc_PL(x):
    ret_val = ((yf.Ticker(x.name).info['regularMarketPrice']/x['avg_price'])-1) * 100
    time.sleep(30)
    return ret_val
def backtest():
    stocks = [ 
    'amd', 'msft', 'amzn', 'spce', 'msft', 
    'mu', 'nvda', 'intc', 'dkng', 'bac', 
    'v', 'coin', 'gme', 'amc', 'hood', 'aal', 
    'mgm', 'hd','logi', 'wmt', 'spot', 'fb', 
    'ge', 'cgc', 'tlry', 't', 'pltr', 'nclh', 
    'pfe', 'f', 'gm', 'xom']
    
    schedule.every().monday.at("15:00").do(auto_trade, stocks=stocks)
    schedule.every().tuesday.at("15:00").do(auto_trade, stocks=stocks)
    schedule.every().wednesday.at("15:00").do(auto_trade, stocks=stocks)
    schedule.every().thursday.at("15:00").do(auto_trade, stocks=stocks)
    schedule.every().friday.at("15:00").do(auto_trade, stocks=stocks)
    schedule.every().monday.at("18:00").do(update_PL)
    schedule.every().tuesday.at("18:00").do(update_PL)
    schedule.every().wednesday.at("18:00").do(update_PL)
    schedule.every().thursday.at("18:00").do(update_PL)
    schedule.every().friday.at("18:00").do(update_PL)
    while True:
        print('Running pending tasks...')
        schedule.run_pending()
        print('Finished pending tasks...')
        time.sleep(1800) # wait 10 mins
stocks = [ 
    'amd', 'msft', 'amzn', 'spce', 'msft', 
    'mu', 'nvda', 'intc', 'dkng', 'bac', 
    'v', 'coin', 'gme', 'amc', 'hood', 'aal', 
    'mgm', 'hd','logi', 'wmt', 'spot', 'fb', 
    'ge', 'cgc', 'tlry', 't', 'pltr', 'nclh', 
    'pfe', 'f', 'gm', 'xom']
backtest()