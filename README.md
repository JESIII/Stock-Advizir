# Stock Advizir

Stock Advizir is an app for deciding if it's a good time to Buy, Sell, or Hold a stock.
This app was trained using data gathered from Yahoo Finance with indicators added which include:
MACD, Stochastic Oscillator, Triple Exponential Moving, Average, RSI, and 20 Day Rolling Average.
The data is then passed into my own algorithm which is meant to categorize the
days into Buy/Sell/Hold the way I do for my own stocks. It uses two indicators: RSI & MACD. It
also rejects a buy if the price is lower 20 days later so it guarantees the input data is profitable
which is what we want the model to recognize without seeing the future data.
This is what the input data looks like graphed. Green is “buy” and red is “sell”:
![Training Data Graph](https://i.imgur.com/HBPMrK9.png)

## Usage
```cmd
python ./predictor.py
Enter a ticker:
<Your Entry>
Enter a Period: 3mo,6mo,1y,2y,5y,10y,ytd
<Your Period>
```
Output (With input AAPL, 3mo):  
![Terminal Output](https://i.imgur.com/Z5R6A6i.png)  
![Graph Output](https://i.imgur.com/481mwAh.png)  
  
## License
[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)
