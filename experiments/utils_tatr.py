import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE

from experiments.run_downstream_model import *
from experiments.predictors.predictor_benchmark import *
from utils.load_data import load_historical_fts
from utils.series_processing import convert_datatype



#################################
# Load Historical Data for TATR #
#################################

def tatr_hist_data(mode, dataname='sp500', splits=None):
  """ Load historical data for training, validation, or testing """
  hist_fts = load_historical_fts(dataname)
  splits = get_data_split(dataname) if splits is None else splits
  modes = {'train': 0, 'val': 1, 'test': 2}
  period = splits[modes[mode]]
  timeseries = get_df_period(hist_fts, period)
  return timeseries.values.ravel()

def get_df_period(df, period):
  """ Get the dataframe in the given period """
  if period is None:
    return df
  period_str = period.split('-')
  period_start, period_end = int(period_str[0]), int(period_str[1])
  df = df.loc[f"{period_start}-01-01":f"{period_end}-12-31"]
  return df

def get_data_split(dataname):
  """ Get the split for diverse data """
  splits = {
    'sp500': ['1957-1996', '1997-2009', '2010-2024'], 
    'ftse': ['1984-2005', '2006-2015', '2016-2024'], 
    'corn': ['2000-2012', '2013-2017', '2018-2024'], 
    }
  return splits[dataname]



##################################
# Initial Training Data for TATR #
##################################

def tatr_init_data(dataname='sp500', return_data=True, store_data=False):
  """ Prepare the historical inital training data for TATR """
  print(f"TATR - Historical initial {dataname.upper()}.")
  data_tatr_path = f"data/augmentations_{dataname}/"
  init_timeseries = tatr_hist_data('train', dataname=dataname)
  if store_data:
    filename = f"tatr_init_{dataname}.txt"
    with open(data_tatr_path + filename, 'w') as file:
      file.write(','.join(map(str, init_timeseries)) + '\n')
    print(f"{filename} stored.")
  if return_data:
    return init_timeseries

def load_tatr_init(dataname='sp500'):
  """ Load the stored historical initial training data for TATR """
  data_tatr_path = f"data/augmentations_{dataname}/"
  filename = data_tatr_path + f"tatr_init_{dataname}.txt"
  init_timeseries = np.loadtxt(filename, delimiter=',')
  return init_timeseries



######################
# Test Data for TATR #
######################

def tatr_test_data(dataname='sp500', return_data=False, store_data=False):
  """ Prepare the test data for TATR """
  print(f"TATR - Test {dataname.upper()}.")
  data_tatr_path = f"data/augmentations_{dataname}/"
  test_timeseries = tatr_hist_data('test', dataname=dataname)
  if store_data:
    filename = f"tatr_test_{dataname}.txt"
    with open(data_tatr_path + filename, 'w') as file:
      file.write(','.join(map(str, test_timeseries)) + '\n')
    print(f"{filename} stored.")
  if return_data:
    return test_timeseries

def load_tatr_test(dataname='sp500'):
  """ Load the stored initial limited training data for TATR """
  data_tatr_path = f"data/augmentations_{dataname}/"
  filename = data_tatr_path + f"tatr_test_{dataname}.txt"
  test_timeseries = np.loadtxt(filename, delimiter=',')
  return test_timeseries



###############################################
# Load Predictions Trained with Augmentations #
###############################################

def load_tatr_augmentation(
    run,
    dataname='sp500',
    aug_model='',
    n_aug_year=100, 
    **kwargs):
  """ Load the stored augmented synthetic datasets for TATR """
  data_aug_path = f"data/augmentations_{dataname}/{aug_model}/"
  if kwargs.get('start') is not None:
    aug_model_name = f"{aug_model}_{kwargs.get('start')}-{kwargs.get('end')}"
  else:
    aug_model_name = aug_model
  filename = data_aug_path + f"tatr_aug_{dataname}_{aug_model_name}_an{n_aug_year}_{run}.txt"
  aug_timeseries = np.loadtxt(filename, delimiter=',')
  return aug_timeseries



#########################
# Tensorize the dataset #
#########################

def init_tatr_set(timeseries, window_size=61, stride=1, scaler=None, normalize=False):
  """ Tensorize the TATR dataset of real data """
  return Timeseries2Dataset_Downstream(timeseries, window_size, stride=stride, scaler=scaler, normalize=normalize)

def Timeseries2Dataset_Downstream(timeseries, window_size=61, stride=1, scaler=None, normalize=False):
  """ Convert the downstream timeseries to rolling samples """
  if normalize:
    if scaler is None:
      scaler = MinMaxScaler(feature_range=(-1, 1))
      timeseries = scaler.fit_transform(timeseries.reshape(-1, 1))
      timeseries = torch.tensor(timeseries).squeeze(1)
      dataset = timeseries.unfold(0, window_size, stride).float()
      return dataset, scaler
    else:
      timeseries = scaler.transform(timeseries.reshape(-1, 1))
      timeseries = torch.tensor(timeseries).squeeze(1)
      dataset = timeseries.unfold(0, window_size, stride).float()
      return dataset
  else:
    timeseries = torch.tensor(timeseries)
    dataset = timeseries.unfold(0, window_size, stride).float()
    return dataset


def copy_dataset_downstream(dataset):
  """ Copy the dataset """
  dataset_copy = dataset.clone().detach()
  dataset_copy.requires_grad = False
  return dataset_copy


def concat_datasets_downstream(dataset_1, dataset_2):
  """ Concatenate the datasets """
  return torch.cat((dataset_1, dataset_2), dim=0)


def construct_dataloader_downstream(dataset, batch_size=16):
  """ Construct the dataloader for the augmented dataset """
  return DataLoader(dataset, batch_size=batch_size, shuffle=False)



######################
# Evaluation Metrics #
######################

def downstream_test_criterion(y_true, y_pred, y_benchmark=None, criterion='mape'):
  """ Compute the predition error """
  y_true, y_pred = y_true.ravel(), y_pred.ravel()
  if y_benchmark is not None:
    y_benchmark = y_benchmark.ravel()
  if criterion == 'mae':
    return MAE(y_true, y_pred)
  elif criterion == 'mape':
    return MAPE(y_true, y_pred)
  elif criterion == 'mse':
    return MSE(y_true, y_pred)
  elif criterion == 'r2':
    return compute_r_squared(y_true, y_pred, y_benchmark)

def compute_r_squared(y_true, y_pred, y_benchmark=None):
  """ Compute the out-of-sample R-squared for model predictions """
  if y_benchmark is None:
    y_benchmark = np.zeros_like(y_true)
  mse_model = MSE(y_true, y_pred)
  mse_benchmark = MSE(y_true, y_benchmark)
  if mse_benchmark == 0:
    return float('inf') if mse_model == 0 else float('-inf')
  return 1 - (mse_model / mse_benchmark)

def compute_sharpe_ratio(returns, dataname='sp500'):
  """ Compute the Sharpe ratio (daily) of given returns """
  risk_free_rate = get_risk_free_rate(dataname)
  excess_return = returns - risk_free_rate
  sharpe_ratio = excess_return.mean() / excess_return.std()
  sharpe_ratio = sharpe_ratio * np.sqrt(252)
  return sharpe_ratio

def get_risk_free_rate(dataname='sp500', freq='daily'):
  """ Get the risk-free rate for the given dataset """
  df = pd.read_csv(f'experiments/fin/risk_free_rate_{freq}_{dataname}.csv', index_col=0)
  period = get_data_split(dataname)[-1]
  df = get_df_period(df, period)
  risk_free_rate = df['RF'].values[1:]
  return risk_free_rate

def compute_sr_bh(dataname='sp500', test_returns=None):
  """ Compute the Sharpe ratio of the buy-and-hold strategy on testing data """
  if test_returns is None:
    test_timeseries = load_tatr_test(dataname)
    test_returns = convert_datatype(test_timeseries, datatype='returns')
  sr_bh = compute_sharpe_ratio(test_returns, dataname=dataname)
  return sr_bh

def compute_model_sharpe_ratio(r2, dataname='sp500'):
  """ Compute the Sharpe ratio achievable by the prediction model """
  if r2 < 0 or r2 > 1:
    return np.nan
  sr_bh = compute_sr_bh(dataname) / np.sqrt(252)
  sr_squared = sr_bh ** 2
  sr_model = np.sqrt((sr_squared + r2) / (1 - r2))
  return sr_model * np.sqrt(252)

def convert_r_squared_benchmark(r2, dataname='sp500', datatype='returns'):
  """ Convert the R-squared using benchmark from avg. to 0 """
  test_timeseries = load_tatr_test(dataname=dataname)
  test_timeseries = convert_datatype(test_timeseries, datatype=datatype)
  test_dataset = init_tatr_set(test_timeseries, window_size=61)
  y_test = test_dataset[:, -1:].detach().cpu().numpy().ravel()
  y_benchmark_avg = prediction_benchmark(test_dataset, 1, benchmark='avg')
  y_0 = np.zeros_like(y_test)
  factor = (MSE(y_test, y_benchmark_avg) / MSE(y_test, y_0))
  r2_new = 1 - (1 - r2) * factor
  return r2_new









