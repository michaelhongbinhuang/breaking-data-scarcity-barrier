import argparse

from utils.load_data import get_fts
from experiments.utils_tatr import tatr_init_data, tatr_test_data, get_data_split



########################
# Download Market Data #
########################

if __name__ == "__main__":
  # Set up inputs
  parser = argparse.ArgumentParser(description="Download Market Data")
  parser.add_argument("--dataname", type=str, default='sp500', help="Data name")
  args = parser.parse_args()

  # Download data
  dataname = args.dataname
  mapping_dataname_ticker = {'sp500': '^GSPC', 'ftse': '^FTSE', 'corn': 'ZC=F'}
  ticker = mapping_dataname_ticker[dataname]
  start_date = get_data_split(dataname)[0].split('-')[0] + '-01-01'
  data = get_fts(ticker=ticker, fts_name=dataname, start_date=start_date, end_date='2024-12-31', store_fts=True)

  # Split data
  tatr_init_data(dataname=dataname, store_data=True)
  tatr_test_data(dataname=dataname, store_data=True)









