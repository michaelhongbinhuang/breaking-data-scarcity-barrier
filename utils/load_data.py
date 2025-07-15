import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
# from pandas_datareader import data as pdr
# yf.pdr_override()
import torch
torch.set_default_dtype(torch.float)

data_path = 'data/'



#########################
# Load time series data #
#########################

def get_fts(
    ticker='^GSPC',
    fts_name='sp500',
    start_date='1957-01-01', 
    end_date='2024-12-31',
    store_fts=False,
    return_close=False,
    return_returns=False,
    plot_fts=None,
    ):
  """ Get financial time series from yfinance """
  df = yf.download(ticker, start=start_date, end=end_date)
  # df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
  df['Return'] = df.loc[:,('Close')].pct_change()
  df['Return'] = df['Return'].fillna(0)
  if plot_fts is not None:
    datatype = plot_fts['data']
    periods = plot_fts['periods']
    print(f"Plotting {fts_name.upper()} {datatype} spanning {periods}")
    plot_fts_series(df.loc[:, (datatype)], dataname=fts_name, datatype=datatype, periods=periods)
  if store_fts:
    df['Close'].to_csv(data_path + fts_name + '_timeseries.csv', index=True)
    df['Return'].to_csv(data_path + fts_name + '-returns_timeseries.csv', index=True)
    print(f"{fts_name.upper()} series stored.")
  if return_close:
    return df['Close']
  if return_returns:
    return df['Return']


def load_historical_fts(dataname):
  """ Load historical fts """
  filename = f"{dataname}"
  series = pd.read_csv(data_path + f"{filename}_timeseries.csv")
  series.index = pd.to_datetime(series['Date'])
  series.drop('Date', axis=1, inplace=True)
  return series


def plot_fts_series(df, dataname, datatype, periods=['1957-2024']):
  """ Plot fts series """
  plt.figure(figsize=(20, 5))
  colors = ['blue', 'orange', 'red']
  date_range = periods[0].split('-')[0] + '-' + periods[-1].split('-')[1]
  for i, p in enumerate(periods):
    period_str = p.split('-')
    period_start, period_end = int(period_str[0]), int(period_str[1])
    period = df.loc[f"{period_start}-01-01":f"{period_end}-12-31"]
    plt.plot(period.index, period.values, label=p, color=colors[i])
  plt.legend(loc='upper left', fontsize=15)
  plt.title(f"{dataname.upper()} {datatype} ({date_range})", fontsize=17)
  plt.xlabel('Date', fontsize=15)
  plt.ylabel(f"{datatype}", fontsize=15)
  plt.grid(True, linestyle='--', alpha=0.7)
  plt.show()









