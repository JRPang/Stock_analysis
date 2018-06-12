# Stock_analysis: Kuala Lumpur Stock Exchange(Malaysia Stock Market) 

Screen out potential stocks listed on Malaysia stock market using quantitative analysis. The stock list and historical stock prices scrapped from <https://www.investing.com/>, used to find out the stocks with high growth potential and low risk. 

In order to balance the good historical performance and investment diversification, correlation among the potential stocks are assessed in this analysis. A diversification strategy can help you achieve more consistent returns over time and reduce your overall investment risk.

Additionally, two additive time series model are built to predict and compare the future values of potential stock pairs. Prophet package developed by Facebook Data Science team are used to build the additive time series model, more information about this package can be found at <https://github.com/facebook/prophet/tree/v0.3/python/fbprophet>.


## Important Links for web scrapping:
* Investing.com : <https://www.investing.com/stock-screener/>
* KLSE : <http://klse.i3investor.com/jsp/quote.jsp>

## Import Custom User-defined functions:
You can find the custom modules <font color='green'>stock_data_download</font> and stock_calculation in this repository. The python utility function are used to scrap data from webpages and compute the important performance metrics used in this project. 

* Web-scrapping
```
from stock_data_download import get_stocks_info
from stock_data_download import historical_extract
```

* Metrics Calculation
```
from stock_calculation import get_log_returns
from stock_calculation import metrics_calculation
```

## References:
1. Cluster map in searborn package: <https://seaborn.pydata.org/generated/seaborn.clustermap.html>
2. Prophet package: <https://github.com/facebook/prophet/tree/v0.3/python/fbprophet>
3. Stock prediction in python: <https://towardsdatascience.com/stock-prediction-in-python-b66555171a2>
4. Time series analysis in python: <https://towardsdatascience.com/time-series-analysis-in-python-an-introduction-70d5a5b1d52a>

Key words: Web scrapping, growth analysis, correlation analysis, time series analysis
