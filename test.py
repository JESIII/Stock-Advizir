from pandas.io.pytables import IndexCol
from stock_advizir import predict
import pandas as pd
import time
import alpaca_trade_api as ata
from datetime import datetime
import math
import pytz
import json
from alpaca_trade_api.rest import REST, TimeFrame
keys = open('./keys.txt', 'r').readlines()
paca_key_id = keys[4].split(':')[1].strip()
paca_secret = keys[5].split(':')[1].strip()
api = REST(key_id=paca_key_id, secret_key=paca_secret, base_url='https://paper-api.alpaca.markets')
def auto_trade(stocks):
    for stock in stocks:
        prediction = None
        try:
            prediction = pd.read_json(predict(stock), keep_default_dates=True)
        except Exception as e:
            print(f'Error with prediction: {e}')
            continue
        prediction.set_index('Date', inplace=True)
        last_row = prediction.iloc[-1]
        if last_row['rating'] == 'buy':
            try:
                qty = 2500 / last_row['Open']
                order = api.submit_order(symbol=stock.upper(), qty=int(qty), side='buy', type='market', time_in_force='day')
                if order != None:
                    print(order)
            except Exception as e:
                print(f'Buy order didn\'t go through for {stock}: {e}')
        elif last_row['rating'] == 'mega buy':
            try:
                qty = 5000 / last_row['Open']
                order = api.submit_order(symbol=stock.upper(), qty=int(qty), side='buy', type='market', time_in_force='day')
                if order != None:
                    print(order)
            except Exception as e:
                print(f'Mega buy order didn\'t go through for {stock}: {e}')
        elif last_row['rating'] == 'sell':
            try:
                position = api.get_position(stock.upper())
                order = api.submit_order(symbol=stock.upper(), side='sell', type='market', qty=math.floor(position*.75), time_in_force='day')
                if order != None:
                    print(order)
            except Exception as e:
                print(f'Sell order didn\'t go through for {stock}: {e}')
        elif last_row['rating'] == 'mega sell':
            try:
                position = api.get_position(stock.upper())
                order = api.submit_order(symbol=stock.upper(), side='sell', type='market', qty=int(position), time_in_force='day')
                if order != None:
                    print(order)       
            except Exception as e:
                print(f'Mega sell order didn\'t go through for {stock}: {e}')
        else:
            print(f'Hold on {stock}')
        time.sleep(20)

def test():
    done = False
    stocks = [ 
    'amd', 'msft', 'amzn', 'spce', 'msft', 
    'mu', 'nvda', 'intc', 'dkng', 'bac', 
    'v', 'coin', 'gme', 'amc', 'hood', 'aal', 
    'mgm', 'hd','logi', 'wmt', 'spot', 'fb', 
    'ge', 'cgc', 'tlry', 't', 'pltr', 'nclh', 
    'pfe']
    while True:
        print('Stock advizir autotrade running...')
        if not done and datetime.now(tz=pytz.utc).hour == 15 and datetime.today().weekday() < 5:
            auto_trade(stocks)
            done = True
        if datetime.now(tz=pytz.utc).hour != 15:
            done = False
        print('Done autotrading for the next hour.')
        time.sleep(3600)
test()
# stocks = ['amd', 'msft', 'amzn', 'msft', 'spce', 'mu', 'nvda', 'intc', 
#     'dkng', 'bac', 'v', 'coin', 'gme', 'amc', 'hood', 'aal', 'mgm', 'hd', 
#     'logi', 'wmt', 'spot', 'fb', 'ge', 'cgc', 't', 'pltr', 'nclh', 
#     'pfe']
# for stock in stocks:
#     time.sleep(10)
#     try:
#         predict(stock)
#     except:
#         print(f'{stock} didn\'t work.')
