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
            pass
        elif prediction.tail(1)['rating'] == 'mega buy':
            pass
        elif prediction.tail(1)['rating'] == 'sell':
            pass
        elif prediction.tail(1)['rating'] == 'mega sell':
            pass
        time.sleep(10)
    pass
def __main__():
    done = False
    stocks = ['amd', 'msft', 'amzn', 'msft', 'spce', 'mu', 'nvda', 'intc', 
    'dkng', 'bac', 'v', 'coin', 'gme', 'amc', 'hood', 'aal', 'mgm', 'hd', 
    'logi', 'wmt', 'spot', 'fb', 'ge', 'gcg', 'tlry', 't', 'pltr', 'nclh', 
    'pfe']
    while True:
        if not done and datetime.datetime.now(tz=pytz.utc).hour == 15:
            auto_trade(stocks)
            done = True
        if datetime.datetime.now(tz=pytz.utc).hour != 15:
            done = False
        time.sleep(86400)

predict('amc')
time.sleep(10)
predict('hood')
time.sleep(10)
predict('fb')
time.sleep(10)
predict('tlry')
time.sleep(10)
predict('pltr')
time.sleep(10)
predict('nclh')
time.sleep(10)
predict('pfe')
time.sleep(10)
predict('wmt')
time.sleep(10)
predict('intc')
time.sleep(10)