# Stock Advizir

This app uses a trained machine learning model to aid in the decision of Buying, Selling, or Holding a stock. 
  
It was trained using the scikit learn Decision Tree Classifier algorithm.  
  
Training data was gathered from Yahoo Finance with indicators added which include:  
MACD, Stochastic Oscillator, Triple Exponential Moving, Average, RSI, and 20 Day Rolling Average.  
  
The data is then passed into my own algorithm which is meant to categorize the  
days into Buy/Sell/Hold using two indicators: RSI and MACD.  
It also rejects a buy if the price is lower 20 days later so it guarantees the input data is profitable so that the model only learns from profitable purchases.  
   
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
  
Terminal Output. Ratings for each day  
![Terminal Output](https://i.imgur.com/Z5R6A6i.png)  
Graph Output.  
Y-axis: Open Price, X-axis: Date.  
Green*: Buy, Red*: Sell.  
![Graph Output](https://i.imgur.com/481mwAh.png)  
  
## License
[GPL-3.0-or-later](https://choosealicense.com/licenses/gpl-3.0/)
