# Stock Advizir

This app trains a machine learning model to infer if a user should buy, sell, or hold a stock. It is accompanied by a script to "paper-trade" a stock to test the model.

## Data  
The data was partially gathered from [Yahoo Finance](https://finance.yahoo.com/) and [the Nasdaq](https://www.nasdaq.com/).
  
The maximum amount of historical data (up to 10 yrs) was gathered for 11 popular stocks from various sectors.
  
I made a simple algorithm to set 'buy', 'mega buy', 'sell', 'mega sell', and 'hold' ratings for the historical data based on local extrema.
  
The following indicators were also added for each stock to the historic data: Stochastic RSI, Pecentage Price Oscillator, Stochastic Oscillator, TRIX, and Volume Zone Oscillator and they were all normalized equally.  
  
### Data After Processing:    
Need to update.
  
## Training  
The algorithm used currently is the Scikit Learn Random Forest Classifier.
The accuracy of the model after testing was 91%.
  
## Prediction
The stock data is gathered and cleaned the same as during training. Indicators are added as well.  
   
### Example of predictions made on test datasets:  
![Example 1](https://github.com/JESIII/Stock-Advizir/blob/main/charts/test_signals_HistoricalData_1738966591583_processed.csv.png)
![Example 2](https://github.com/JESIII/Stock-Advizir/blob/main/charts/test_signals_HistoricalData_1738966623767_processed.csv.png)
  
## Android App
Currently in development: 
[Download on Google Play Store](https://play.google.com/store/apps/details?id=com.jesiii.stockadvizir)  
![Play Store Page](https://i.imgur.com/x2uDZBy.png)
  
## Backtesting  
Backtesting results: in prog
  
## Credit
Thanks to the developers of [FinTa](https://github.com/peerchemist/finta) and [YFinance](https://github.com/ranaroussi/yfinance) for their useful tools that are used in this program.
<!-- This is what the input data looks like graphed. Green is “buy” and red is “sell”:  
![Training Data Graph](https://i.imgur.com/HBPMrK9.png)  

Output (With input AAPL, 3mo):    
  
Terminal Output. Ratings for each day  
![Terminal Output](https://i.imgur.com/Z5R6A6i.png)  
Graph Output.  
Y-axis: Open Price, X-axis: Date.  
Green*: Buy, Red*: Sell.  
![Graph Output](https://i.imgur.com/481mwAh.png)  
     -->
  
## License
[GPL-3.0-or-later](https://choosealicense.com/licenses/gpl-3.0/)
