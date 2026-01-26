import pandas as pd
import time
import multiprocessing
from rich.progress import track
from tqdm.notebook import tqdm
from threadpoolctl import threadpool_limits
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from experiments.utils_tatr import *
from experiments.utils_tatr_extension import *

NN_MODELS = {'mlp', 'itransformer'}

def get_res_tatr_init_only_path(dataname, pred_model):
  res_tatr_init_only_path = f"experiments/res/{dataname}/res_init_only/"
  if 'gbdt' in pred_model:
    return res_tatr_init_only_path + 'gbdt/'
  elif 'rf' in pred_model:
    return res_tatr_init_only_path + 'rf/'
  elif 'mlp' in pred_model:
    return res_tatr_init_only_path + 'mlp/'
  elif 'itransformer' in pred_model:
    return res_tatr_init_only_path + 'itransformer/'
  else:
    NotImplementedError



##############################################
# Train Prediction Models on Historical Data #
##############################################

def tatr_train_test_init_only_single(
    n_runs, 
    init_timeseries, 
    test_timeseries, 
    datatype, 
    pred_model, 
    dataname='sp500', 
    window_size=60, 
    predstep=1, 
    ahead=1, 
    adjust_lr=False, 
    store_res=False, 
    ):
  """ Single experiment of parallel setting for training the prediction models on historical data """
  # Prepare training and test dataset
  init_timeseries = convert_datatype(init_timeseries, datatype=datatype)
  init_dataset = init_tatr_set(init_timeseries, window_size=window_size, predstep=predstep, stride=1)
  test_timeseries = convert_datatype(test_timeseries, datatype=datatype)
  test_dataset = init_tatr_set(test_timeseries, window_size=window_size, predstep=predstep, stride=1)

  # Training and prediction
  predictions = []
  res_tatr_init_only_path = get_res_tatr_init_only_path(dataname=dataname, pred_model=pred_model)
  filename = f"tatr_initonly_{dataname}-{datatype}_{pred_model}.txt"
  
  track_description = f"{dataname.upper()}-{datatype[0].upper()}-Pred.{pred_model}"
  for run in track(range(n_runs), description=track_description):
  # for run in range(n_runs):
    # Train on hist. set
    with threadpool_limits(limits=1, user_api='blas'):
      model = separate_train_downstream_model(init_dataset, ahead, datatype=datatype, modelname=pred_model, adjust_lr=adjust_lr)
    # Test the prediction model
    _, y_pred = prediction_downstream_model(model, test_dataset, ahead, modelname=pred_model)
    predictions.append(y_pred)

    # Store the model predictions
    if store_res:
      with open(res_tatr_init_only_path + filename, 'w') as file:
        for pred in predictions:
          file.write(','.join(map(str, pred)) + '\n')


def tatr_train_test_init_only_parallel(
    n_runs, 
    datatypes, 
    pred_models, 
    dataname='sp500', 
    window_size=60, 
    predstep=1, 
    ahead=1, 
    adjust_lr=False, 
    store_res=False, 
    max_parallel_jobs=9, 
    ):
  """ Parallel setting for training the prediction models on historical data """
  # Load test set
  test_timeseries = load_tatr_test(dataname=dataname)
  init_timeseries = load_tatr_init(dataname=dataname)

  # Parallel training
  active_processes = []
  with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
    datatypes = [datatypes] if not isinstance(datatypes, list) else datatypes
    pred_models = [pred_models] if not isinstance(pred_models, list) else pred_models
    for d_model in pred_models:
      for datatype in datatypes:
        kwargs = {
          'n_runs': n_runs,
          'init_timeseries': init_timeseries,
          'test_timeseries': test_timeseries,
          'datatype': datatype,
          'pred_model': d_model,
          'dataname': dataname,
          'window_size': window_size,
          'predstep': predstep,
          'ahead': ahead,
          'adjust_lr': adjust_lr,
          'store_res': store_res,
        }
        process = pool.apply_async(tatr_train_test_init_only_single, kwds=kwargs)
        active_processes.append((d_model, process))
        while len(active_processes) >= max_parallel_jobs:
          for j, (_, process) in enumerate(active_processes):
            if process.ready():
              active_processes.pop(j)
              break
          else:
            time.sleep(2)
    for _, process in active_processes:
      process.wait()


def return_prediction_init(dataname='sp500', n_runs=100, datatype='returns', adjust_lr=False, store_res=True):
  init_timeseries = load_tatr_init(dataname=dataname)
  test_timeseries = load_tatr_test(dataname=dataname)
  pred_models = ['gbdt', 'rf', 'mlp', 'itransformer-pt-exchange']
  for pred_model in pred_models:
    tatr_train_test_init_only_single(
      n_runs=n_runs, 
      init_timeseries=init_timeseries, 
      test_timeseries=test_timeseries, 
      datatype=datatype, 
      pred_model=pred_model, 
      dataname=dataname, 
      adjust_lr=adjust_lr, 
      store_res=store_res, 
      )
    compute_res_tatr_init_only(datatype=datatype, pred_model=pred_model, n_runs=n_runs, dataname=dataname)


def compute_res_tatr_init_only(
    datatype, 
    pred_model, 
    n_runs=100,
    dataname='sp500',
    window_size=60,
    predstep=1,
    ahead=1,
    benchmark='avg', 
    store_res=False, 
    ):
  """ Compute the TATR results of prediction models on historical data """
  # Load test set
  test_timeseries = load_tatr_test(dataname=dataname)
  test_timeseries = convert_datatype(test_timeseries, datatype=datatype)
  test_dataset = init_tatr_set(test_timeseries, window_size=window_size, predstep=predstep)
  y_test = test_dataset[:, -ahead:].detach().cpu().numpy().ravel()
  y_benchmark_avg = prediction_benchmark(test_dataset, ahead, benchmark=benchmark)

  # Load predictions
  res_tatr_init_only_path = get_res_tatr_init_only_path(dataname=dataname, pred_model=pred_model)
  filename_pred = f"tatr_initonly_{dataname}-{datatype}_{pred_model}.txt"
  predictions = np.loadtxt(res_tatr_init_only_path + filename_pred, delimiter=',')
  
  # Evaluation
  metrics = np.zeros((1, n_runs))
  r2_total = 0
  for run, y_pred in enumerate(predictions):
    r2 = downstream_test_criterion(y_test, y_pred, y_benchmark_avg, criterion='r2')
    metrics[0, run] = r2
    r2_total += r2

  # Store results
  if store_res:
    res_tatr_init_only_path = get_res_tatr_init_only_path(dataname=dataname, pred_model=pred_model)
    res_tatr_init_only_r2_path = res_tatr_init_only_path + "r2/"
    filename_metrics = f"tatr_{dataname}-{datatype}_initonly_{pred_model}_r2avg.csv"
    df = pd.DataFrame(metrics.T, columns=['r2'])
    df.to_csv(res_tatr_init_only_r2_path + filename_metrics, index=False)


def load_res_tatr_init_only(pred_model, datatype, dataname='sp500'):
  """ Load the TATR results of prediction models on historical data """
  res_tatr_init_only_path = get_res_tatr_init_only_path(dataname=dataname, pred_model=pred_model)
  res_tatr_init_only_r2_path = res_tatr_init_only_path + "r2/"
  filename = f"tatr_{dataname}-{datatype}_initonly_{pred_model}_r2avg.csv"
  df = pd.read_csv(res_tatr_init_only_r2_path + filename, index_col=False)
  return df









