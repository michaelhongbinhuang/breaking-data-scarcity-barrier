from utils.load_data import get_fts, get_data_split
from experiments.utils_tatr import tatr_init_data, tatr_test_data



def data_download_demo():
  dataname = 'sp500'
  tickers = ['^GSPC', '^SPX']
  start_date = get_data_split(dataname)[0].split('-')[0] + '-01-01'
  successful_download = False
  for ticker in tickers:
    try:
      data = get_fts(ticker=ticker, fts_name=dataname, start_date=start_date, end_date='2024-12-31', store_fts=True)
      if not data.empty:
        successful_download = True
        break
    except Exception as e:
      continue
  if not successful_download:
    raise Exception("Failed to download data. Please check the network connection and Yahoo Finance status")
  tatr_init_data(dataname=dataname, store_data=True)
  tatr_test_data(dataname=dataname, store_data=True)
  print("-- Data download completed")









