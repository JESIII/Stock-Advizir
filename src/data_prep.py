import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from finta import TA
import matplotlib.pyplot as plt
from pathlib import Path

def download_dataset_from_hf():
    current_file_path = Path(__file__).resolve()
    splits = {'train': 'train-stocks.parquet', 'test': 'test-stocks.parquet'}
    df = pd.read_parquet("hf://datasets/Kicel/daily-stocks/" + splits["train"])
    current_dir_path = current_file_path.parent
    relative_path = current_dir_path.parent / "data" / "training" / "daily-stocks.parquet"
    df.to_parquet(relative_path)

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df['rating'] = 'hold'
    df['local_max'] = df.iloc[argrelextrema(df['open'].values, np.greater, order=10)[0]]['open']
    df['local_max'] = df['local_max'].notnull().astype('bool')
    df['local_min'] = df.iloc[argrelextrema(df['open'].values, np.less, order=10)[0]]['open']
    df['local_min'] = df['local_min'].notnull().astype('bool')
    
    buy_condition = (df['open'] < df.shift(-5)['open']) & (df['open'] < df.shift(5)['open']) & ((abs(df.shift(-10)['open'] - df['open']) / df['open'] * 100) > 2) & ~df['local_max']
    mega_buy_condition = (df['open'] < df.shift(-5)['open']) & (df['open'] < df.shift(5)['open']) & ((abs(df.shift(-10)['open'] - df['open']) / df['open'] * 100) > 5) & ~df['local_max']
    sell_condition = (df['open'] > df.shift(-5)['open']) & (df['open'] > df.shift(5)['open']) & ((abs(df['open'] - df.shift(5)['open']) / df['open'] * 100) > 5) & ~df['local_min']
    mega_sell_condition = (df['open'] > df.shift(-5)['open']) & (df['open'] > df.shift(5)['open']) & ((abs(df['open'] - df.shift(5)['open']) / df['open'] * 100) > 10) & ~df['local_min']

    df.loc[buy_condition, 'rating'] = 'buy'
    df.loc[mega_buy_condition, 'rating'] = 'mega buy'
    df.loc[sell_condition, 'rating'] = 'sell'
    df.loc[mega_sell_condition, 'rating'] = 'mega sell'
    df.drop(labels=['local_max', 'local_min'], axis=1, inplace=True)
    df.drop(df.tail(100).index, inplace=True)
    print('Signals added based on extrema.')
    return df

def process_single_stock(stock_data):
    stock_data = add_technical_indicators(stock_data)
    stock_data = add_labels(stock_data)
    return stock_data

def load_and_process_single_parquet_file(file_path: str, output_dir: str) -> None:
    pd.options.mode.copy_on_write = True
    try:
        all_data = pd.read_parquet(file_path)
        new_data = None
        
        stocks = all_data['symbol'].unique()
        stock_count = len(stocks)
        print("Stock count:", stock_count)
        curr_stock = 0
        for i in range(len(stocks)):#range(len(stocks)):
            stock_data = all_data[all_data['symbol'] == stocks[i]]
            stock_data = process_single_stock(stock_data)
            if i == 0:
                new_data = pd.DataFrame(columns=stock_data.columns)
            new_data = pd.concat([new_data, stock_data])
            curr_stock += 1
            print(f'Processed {curr_stock}/{stock_count} stocks', end='\r')
        
        # Save the processed dataset with _processed appended to the file name
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        processed_file_name = f"{name}_processed{ext}"
        output_file_path = os.path.join(output_dir, processed_file_name)
        new_data.to_parquet(output_file_path)
        print(f'Saved processed data to {output_file_path}')
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def load_and_process_single_csv_file(file_path: str, output_dir: str) -> None:
    try:
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        df['Close/Last'] = df['Close/Last'].replace({r'\$': ''}, regex=True).astype(np.float32)
        df['Open'] = df['Open'].replace({r'\$': ''}, regex=True).astype(np.float32)
        df['High'] = df['High'].replace({r'\$': ''}, regex=True).astype(np.float32)
        df['Low'] = df['Low'].replace({r'\$': ''}, regex=True).astype(np.float32)
        df.rename(columns={'Close/Last': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low'}, inplace=True)
        df = add_technical_indicators(df)
        df = add_labels(df)
        
        # Save the processed dataset with _processed appended to the file name
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        processed_file_name = f"{name}_processed{ext}"
        output_file_path = os.path.join(output_dir, processed_file_name)
        df.to_csv(output_file_path, float_format='%.3f')
        print(f'Saved processed data to {output_file_path}')
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data.index = pd.to_datetime(data.index)
    data_rsi = TA.STOCHRSI(data)
    data_trix = TA.TRIX(data)
    data_stoch = TA.STOCH(data)
    data_ppo = TA.PPO(data)
    data_vzo = TA.VZO(data)
    
    data.loc[:,'rsi'] = (data_rsi - np.min(data_rsi)) / (np.max(data_rsi) - np.min(data_rsi))
    data.loc[:,'trix'] = (data_trix - np.min(data_trix)) / (np.max(data_trix) - np.min(data_trix))
    data.loc[:,'stoch'] = (data_stoch - np.min(data_stoch)) / (np.max(data_stoch) - np.min(data_stoch))
    data.loc[:,'ppo'] = (data_ppo['PPO'] - data_ppo['PPO'].min()) / (data_ppo['PPO'].max() - data_ppo['PPO'].min())
    data.loc[:,'vzo'] = (data_vzo - np.min(data_vzo)) / (np.max(data_vzo) - np.min(data_vzo))
    data = data.dropna()
    # print('Oscillator indicators added.')
    # print(data.head(5))
    return data

def load_process_and_save_data(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for root, _, files in os.walk('./data/training/'):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                load_and_process_single_csv_file(file_path, output_dir)

def plot_signals(data: pd.DataFrame, title: str):
    # Plotting the open price
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['open'], label='Open Price', color='blue')

    # Plotting the signals
    buy_signals = data[data['rating'] == 'buy']
    mega_buy_signals = data[data['rating'] == 'mega buy']
    sell_signals = data[data['rating'] == 'sell']
    mega_sell_signals = data[data['rating'] == 'mega sell']

    plt.scatter(buy_signals.index, buy_signals['open'], label='Buy', color='yellowgreen', marker='^', alpha=1)
    plt.scatter(mega_buy_signals.index, mega_buy_signals['open'], label='Mega Buy', color='green', marker='^', alpha=1)
    plt.scatter(sell_signals.index, sell_signals['open'], label='Sell', color='orange', marker='v', alpha=1)
    plt.scatter(mega_sell_signals.index, mega_sell_signals['open'], label='Mega Sell', color='red', marker='v', alpha=1)

    plt.title('Stock Open Price with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Open Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'./charts/test_signals_{title}.png', dpi=300, format='png')
    plt.show()
    
def plot_test_data(file: str = 'HistoricalData_1738966591583_processed.csv', output_dir: str = './data/output/'):
    test_data = os.path.join(output_dir, file)
    if os.path.exists(test_data):
        sample_data = pd.read_csv(test_data, index_col="Date", parse_dates=True)
        plot_signals(sample_data, file)
    else:
        print(f"Test data not found: {test_data}")