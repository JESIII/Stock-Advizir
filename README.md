# Stock Advizir

This app uses a machine learning model to infer if a user should buy, sell, or hold a stock.

## Data  
The data was gathered from [Yahoo Finance](https://finance.yahoo.com/), [Organisation for Economic Co-operation and Development](https://www.oecd.org/unitedstates/), and [Financial Modeling Prep](https://financialmodelingprep.com)  
  
Data was gathered for 3 stocks per sector per market cap size. This results in a total of 9 stocks per sector, 3 small, 3 mid, 3 large.  
There are 15 total sectors for a total of 9 * 15 =  135 stocks that data was gathered for and models trained on.
  
The data was cleaned and ratings of 'buy', 'mega buy', 'sell', 'mega sell', and 'hold' were added to the historic data for the model to be trained on.  
  
The following indicators were also added for each stock to the historic data: RSI, MACD, Stochastic Oscillator, & Volume Zone Oscillator.  
  
The Consumer Confidence Index was also added to the historic data, but is not specific to any stock.  
  
## Training  
The indicators above are what the model is trained on. Volume, and price are not used for training, just the oscillators that are calculated based on price and volume because they are easily normalized.  
  
The accuracy of the models was around the mid 60s, with some stocks in the 70s. I may try AdaBoost to improve this.  
  
The algorithm used currently is the Scikit Learn Random Forest Ensemble  
  
## Prediction
The stock data is gathered and cleaned the same as during training. Indicators are added as well.  
  
The models for the stocks sector and market cap size are loaded.  
  
The prediction for each day is a result of majority voting, which is why we chose an odd number of stocks per sector.
   
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
## Android App
Currently in development: 
[Download on Google Play Store](https://play.google.com/store/apps/details?id=com.jesiii.stockadvizir)
  
## License
[GPL-3.0-or-later](https://choosealicense.com/licenses/gpl-3.0/)
