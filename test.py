from pandas.io.pytables import IndexCol
from stock_advizir import predict
import pandas as pd
import time
import alpaca_trade_api as ata
from datetime import datetime
import pytz
import json
from alpaca_trade_api.rest import REST, TimeFrame
keys = open('./keys.txt', 'r').readlines()
paca_key_id = keys[4].split(':')[1].strip()
paca_secret = keys[5].split(':')[1].strip()
api = REST(key_id=paca_key_id, secret_key=paca_secret)
def auto_trade(stocks):
    for stock in stocks:
        prediction = pd.read_json(predict(stock), IndexCol='timestamp', keep_default_dates=True)
        if prediction.tail(1)['rating'] == 'buy':
            order = REST.submit_order(symbol=stock, side='buy', type='market', notional=2500)
            if order != None:
                print(order)
            else:
                print(f'Buy order didn\'t go through for {stock}')
        elif prediction.tail(1)['rating'] == 'mega buy':
            order = REST.submit_order(symbol=stock, side='buy', type='market', notional=5000)
            if order != None:
                print(order)
            else:
                print(f'Mega buy order didn\'t go through for {stock}')
        elif prediction.tail(1)['rating'] == 'sell':
            positions = REST.list_positions()
            qty = positions['qty']
            order = REST.submit_order(symbol=stock, side='sell', type='market', qty=qty*.75)
            if order != None:
                print(order)
            else:
                print(f'Sell order didn\'t go through for {stock}')
        elif prediction.tail(1)['rating'] == 'mega sell':
            positions = REST.list_positions()
            qty = positions['qty']
            order = REST.submit_order(symbol=stock, side='sell', type='market', qty=qty)
            if order != None:
                print(order)
            else:
                print(f'Mega sell order didn\'t go through for {stock}')
        time.sleep(10)
    print('Done autotrading for the day')

if __name__ == '__main__':
    done = False
    stocks = ['amd', 'msft', 'amzn', 'msft', 'spce', 'mu', 'nvda', 'intc', 
    'dkng', 'bac', 'v', 'coin', 'gme', 'amc', 'hood', 'aal', 'mgm', 'hd', 
    'logi', 'wmt', 'spot', 'fb', 'ge', 'gcg', 'tlry', 't', 'pltr', 'nclh', 
    'pfe']
    while True:
        if not done and datetime.datetime.now(tz=pytz.utc).hour == 15 and datetime.weekday < 5:
            auto_trade(stocks)
            done = True
        if datetime.datetime.now(tz=pytz.utc).hour != 15:
            done = False
        time.sleep(86400)

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
