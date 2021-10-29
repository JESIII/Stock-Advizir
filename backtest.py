from genericpath import exists
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.io.pytables import IndexCol
from stock_advizir import predict, predict10y
import traceback, math, time
import yfinance as yf
import pandas as pd
from datetime import datetime
from joblib import dump, load
def make_prediction_files():
    stocks = [ 
        'amd', 'msft', 'amzn', 'spce', 'msft', 
        'mu', 'nvda', 'intc', 'dkng', 'bac', 
        'v', 'coin', 'gme', 'amc', 'hood', 'aal', 
        'mgm', 'hd','logi', 'wmt', 'spot', 'fb', 
        'ge', 'cgc', 'tlry', 't', 'pltr', 'nclh', 
        'pfe', 'f', 'gm', 'xom']
    for stock in stocks:
        data = pd.read_json(predict10y(stock))
        data.to_csv(f'./data/backtest_predictions/{stock}_10y_preictions.csv')
        print(f'Done with {stock}')
        time.sleep(100)
trades, positions = DataFrame, DataFrame
balance = float(100000)
def calculate_returns(stocks):
    global trades, positions, balance
    for stock in stocks:
        balance = 100000
        if exists(f'./data/backtest_predictions/{stock}_10y_preictions.csv'):
            trades_for_stock = pd.read_csv(f'./data/backtest_predictions/{stock}_10y_preictions.csv', parse_dates=True, index_col='Date').drop('Unnamed: 0', axis=1)
            trades_for_stock = trades_for_stock[trades_for_stock['rating'] != 'hold']
            trades_for_stock = trades_for_stock[~trades_for_stock.index.duplicated(keep='first')]
        else:
            print(f'Can\'t find ./data/backtest_predictions/{stock}_10y_preictions.csv')
            continue
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
        trades_for_stock.apply(lambda x: add_row_to_trades(x, stock), axis=1)
        try:
            qty = positions.iloc[-1]['qty']
            effect = positions.iloc[-1]['qty']*trades_for_stock.iloc[-1]['Open']
            order = pd.DataFrame({'stock':stock, 'price':trades_for_stock.iloc[-1]['Open'], 'qty':qty,'side':'sell', 'effect':effect, 'date':trades_for_stock.iloc[-1].name}, index=[trades.index.values[-1]+1])
            order.index.rename('trade')
            trades = trades.append(order)
            positions.at[stock, 'qty'] -= qty
        except Exception:
            traceback.print_exc()
        positions.to_csv('./backtesting/positions.csv', index_label='stock')
        trades.to_csv('./backtesting/trades.csv', index_label='trade')
def add_row_to_trades(row, stock):
    global trades, positions, balance
    if row['rating'] == 'buy':
        try:
            qty = 2500 / row['Open']
            effect = -qty*row['Open']
            if effect + balance < 0:
                return
            order = pd.DataFrame({'stock':stock, 'price':row['Open'], 'qty':qty,'side':'buy', 'effect':effect, 'date':row.name}, index=[trades.index.values[-1]+1])
            order.index.rename('trade')
            balance = balance + effect
            trades = trades.append(order)
            try:
                positions.at[stock, 'avg_price'] = (positions.at[stock, 'avg_price'] + row['Open']) / 2
                positions.at[stock, 'qty'] += qty
                positions.at[stock, 'P/L%'] = ((row['Open']/positions.at[stock, 'avg_price'])-1) * 100
            except:
                try:
                    new_pos = pd.DataFrame({'avg_price':row['Open'], 'price':row['Open'], 'qty':qty, 'P/L%':0}, index=[stock])
                    new_pos.index.rename('stock')
                    positions = positions.append(new_pos)
                except Exception:
                    traceback.print_exc()
                    return
        except Exception:
            traceback.print_exc()
    elif row['rating'] == 'mega buy':
        try:
            qty = 5000 / row['Open']
            effect = -qty*row['Open']
            if effect + balance < 0:
                return
            order = pd.DataFrame({'stock':stock, 'price':row['Open'], 'qty':qty,'side':'buy', 'effect':effect, 'date':row.name}, index=[trades.index.values[-1]+1])
            order.index.rename('trade')
            balance = balance + effect
            trades = trades.append(order)
            try:
                positions.at[stock, 'avg_price'] = (positions.at[stock, 'avg_price'] + row['Open']) / 2
                positions.at[stock, 'qty'] += qty
                positions.at[stock, 'P/L%'] = ((row['Open']/positions.at[stock, 'avg_price'])-1) * 100
            except:
                try:
                    new_pos = pd.DataFrame({'avg_price':row['Open'], 'price':row['Open'], 'qty':qty, 'P/L%':0}, index=[stock])
                    new_pos.index.rename('stock')
                    positions = positions.append(new_pos)
                except Exception:
                    traceback.print_exc()
                    return
        except Exception:
            traceback.print_exc()
    elif row['rating'] == 'sell':
        try:
            qty = math.floor(positions.at[stock, 'qty']*.75)
            if qty == 0:
                return
            effect = qty*row['Open']
            order = pd.DataFrame({'stock':stock, 'price':row['Open'], 'qty':qty,'side':'sell', 'effect':effect, 'date':row.name}, index=[trades.index.values[-1]+1])
            order.index.rename('trade')
            balance = balance + effect
            trades = trades.append(order)
            positions.at[stock, 'qty'] -= qty
        except Exception:
            traceback.print_exc()
    elif row['rating'] == 'mega sell':
        try:
            qty = math.floor(positions.at[stock, 'qty'])
            if qty == 0:
                return
            effect = qty*row['Open']
            order = pd.DataFrame({'stock':stock, 'price':row['Open'], 'qty':qty,'side':'sell', 'effect':effect, 'date':row.name}, index=[trades.index.values[-1]+1])
            order.index.rename('trade')
            balance = balance + effect
            trades = trades.append(order)
            positions.at[stock, 'qty'] -= qty
        except Exception:
            traceback.print_exc()
stocks = [ 
        'amd', 'msft', 'amzn', 'spce', 'msft', 
        'mu', 'nvda', 'intc', 'dkng', 'bac', 
        'v', 'coin', 'gme', 'amc', 'hood', 'aal', 
        'mgm', 'hd','logi', 'wmt', 'spot', 'fb', 
        'ge', 'cgc', 'tlry', 't', 'pltr', 'nclh', 
        'pfe', 'f', 'gm', 'xom']
calculate_returns(stocks)