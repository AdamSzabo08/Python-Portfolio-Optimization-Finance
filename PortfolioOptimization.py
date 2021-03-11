import pandas as pd
from pandas_datareader import data as web
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#('AAPL','GOOGL', 'FB', 'AMZN', 'MSFT', 'NOW', 'BABA', 'TSM', 'AVGO', 'ADI', 'QCOM', 'SNE', 'INTU', 'V','PYPL', 'ADBE', 'NVDA', 'DIS', 'NKE', 'TMUS','FVRR','SHOP','NIO','TSLA','NVDA')
assets = ['GOOGL', 'FB', 'AMZN', 'MSFT', 'V','ADBE', 'NKE', 'TMUS','TM','NVS','PYPL','ABT','MA','v','UNH','AAPL','NIO','TSLA','SHOP']
#,'NIO','FVRR','SHOP','NOW','PYPL'

stockStartDate = '2020-01-01'
today = datetime.today().strftime('%Y-%m-%d')

df = pd.DataFrame()
for stock in assets:
    df[stock] = web.DataReader(stock, data_source='yahoo', start=stockStartDate, end=today)['Adj Close']

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

latest_prices = get_latest_prices(df)
weights = cleaned_weights
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=99999)

allocation, leftover = da.lp_portfolio()
print('Discrete allocation: ', allocation)
print('Funds remaining: ${:.2f}'.format(leftover))
