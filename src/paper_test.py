from sklearn import preprocessing
import yfinance as yf
import pandas as pd
import xgboost as xgb
import joblib
from data_prep import add_technical_indicators
import os
from datetime import datetime
import keras
import numpy as np

# Paper trading CSV file path
paper_trade_csv = 'G:/Repos/Stock-Advizir/data/paper_trading/paper_trading.csv'
paper_trade_log = 'G:/Repos/Stock-Advizir/data/paper_trading/paper_trading_log.txt'
time_initiated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def print_and_log(text):
    text = f"{time_initiated} - {text}"
    print(text)
    write_to_log(text)

# Function to initialize the paper trading CSV file
def initialize_paper_trading_csv(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(columns=['date', 'balance', 'amount', 'quantity', 'current_price'])
        df.loc[0] = [datetime.now().strftime('%Y-%m-%d'), 10000, 0, 0, 0]
        df.to_csv(file_path, index=False)
        print(f'Initialized paper trading CSV: {file_path}')

def initialize_paper_trading_log(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

def write_to_log(text):
    with open(paper_trade_log, 'a') as file:
        file.write(f"{text}\n")

# Function to check if a trade has already been made today
def check_last_trade_date(file_path):
    df = pd.read_csv(file_path)
    last_trade_date = df.iloc[-1]['date']
    return last_trade_date == datetime.now().strftime('%Y-%m-%d')

# Function to record a trade in the CSV file
def record_trade(file_path, balance, amount, quantity, current_price):
    df = pd.read_csv(file_path)
    new_row = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'balance': balance,
        'amount': amount,
        'quantity': quantity,
        'current_price': current_price
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(file_path, index=False)
    print_and_log(f'Recorded trade in CSV: {file_path}')

# Function to execute a trade based on prediction
def make_trade(stock: str, move: str, quote):
    # Load the CSV file
    df = pd.read_csv(paper_trade_csv)
    current_balance = df.iloc[-1]['balance']
    current_quantity = df.iloc[-1]['quantity']

    spend_amount = 0
    new_quantity = current_quantity
    new_balance = current_balance

    if move == 'buy':
        spend_amount = 100
        new_quantity += spend_amount / quote
        new_balance -= spend_amount
    elif move == 'mega buy':
        spend_amount = 200
        new_quantity += spend_amount / quote
        new_balance -= spend_amount
    elif move == 'sell':
        sell_amount = current_quantity / 6
        spend_amount = sell_amount * quote
        new_quantity -= sell_amount
        new_balance += spend_amount
    elif move == 'mega sell':
        sell_amount = current_quantity / 2
        spend_amount = sell_amount * quote
        new_quantity -= sell_amount
        new_balance += spend_amount
    else:
        print_and_log("Holding. No trade to be made.")
        return

    print_and_log(f"Placing trade for {stock} with move {move}")
    print_and_log(f"New Balance: {new_balance}")
    print_and_log(f"New Quantity: {new_quantity}")

    # Record the trade in the CSV file
    record_trade(paper_trade_csv, new_balance, spend_amount, new_quantity, quote)

print_and_log(f"-----Trading--------Begin------")
# Initialize the paper trading CSV
initialize_paper_trading_csv(paper_trade_csv)
initialize_paper_trading_log(paper_trade_log)

def do_trade_xgboost(ticker):
    try:
        # Fetching historical data for the ticker
        dat = yf.Ticker(ticker)
        last_30_days = dat.history(period='1mo')
        last_30_days.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}, inplace=True)
        quote = last_30_days.iloc[-1].get('open')
        print_and_log(f'Ticker: {ticker}')
        print_and_log(f'Quote: {quote}')

        # Adding indicators
        last_30_days = add_technical_indicators(last_30_days)
        print(last_30_days.tail(4))

        # Loading the trained model
        model: xgb.XGBClassifier = joblib.load('./models/xgboost_model.pkl')
        label_encoder: preprocessing.LabelEncoder = joblib.load('./models/label_encoder.pkl')

        # Preparing the row for prediction
        predict_row = last_30_days.iloc[-2]

        print_and_log('Predicting for row:')
        print_and_log(f"{predict_row}")

        # Exclude the index and select only the required features
        predict_row = predict_row[['rsi', 'trix', 'stoch', 'ppo', 'vzo']].to_frame().T
        # print('Cleaned row:')
        # print(predict_row)

        # Making the prediction
        y_pred = model.predict(predict_row)
        prediction = label_encoder.inverse_transform(y_pred)
        print_and_log(prediction[0])

        # Check if a trade has already been made today
        if not check_last_trade_date(paper_trade_csv):
            make_trade(ticker, prediction[0], quote)
            print_and_log("Trading complete.")
        else:
            print_and_log("Trade already made today. No action taken.")

    except Exception as ex:
        print_and_log(f"Error trading: [{ex}]")
        
def do_trade_nn(ticker):
    try:
        # Fetching historical data for the ticker
        dat = yf.Ticker(ticker)
        last_30_days = dat.history(period='1mo')
        last_30_days.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}, inplace=True)
        quote = last_30_days.iloc[-1].get('open')
        print_and_log(f'Ticker: {ticker}')
        print_and_log(f'Quote: {quote}')

        # Adding indicators
        last_30_days = add_technical_indicators(last_30_days)
        print(last_30_days.tail(4))

        # Loading the trained model
        model: keras.Sequential = joblib.load('./models/nn_model.pkl')
        label_encoder: preprocessing.LabelEncoder = joblib.load('./models/nn_label_encoder.pkl')

        # Preparing the row for prediction
        predict_row = last_30_days.iloc[-2]

        print_and_log('Predicting for row:')
        print_and_log(f"{predict_row}")

        # Exclude the index and select only the required features
        predict_row = predict_row[['rsi', 'trix', 'stoch', 'ppo', 'vzo']].to_frame().T
        # print('Cleaned row:')
        # print(predict_row)

        # Making the prediction
        prediction = model.predict(predict_row)
        print_and_log(f"{prediction}")
        # le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        # print(le_name_mapping)
        max_index = np.argmax(prediction[0])
        # print("Max Index: ", max_index)
        # print("le classes:", label_encoder.classes_)
        # print("le encodings:", label_encoder.transform(label_encoder.classes_))
        prediction_label = label_encoder.inverse_transform([max_index])
        print_and_log(f"Prediction: {prediction_label[0]}")

        # Check if a trade has already been made today
        if not check_last_trade_date(paper_trade_csv):
            make_trade(ticker, prediction_label, quote)
            print_and_log("Trading complete.")
        else:
            print_and_log("Trade already made today. No action taken.")

    except Exception as ex:
        print_and_log(f"Error trading: [{ex}]")
        
do_trade_nn("AMD")