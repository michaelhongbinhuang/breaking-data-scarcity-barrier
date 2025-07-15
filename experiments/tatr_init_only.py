import pandas as pd
from rich.progress import track
from threadpoolctl import threadpool_limits
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from experiments.utils_tatr import *
from experiments.utils_tatr_extension import *
from utils.utils_parallel import *

NN_MODELS = {'mlp', 'itransformer'}

def get_res_tatr_init_only_path(dataname, pred_model):
  device_path = 'gpu' if pred_model.split('-')[0] in NN_MODELS else 'cpu'
  res_tatr_init_only_path = f"res/{dataname}/{device_path}/res_init_only/"
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
    window_size=61, 
    ahead=1, 
    adjust_lr=False, 
    store_res=False, 
    ):
  """ Single experiment of parallel setting for training the prediction models on historical data """
  # Prepare training and test dataset
  init_timeseries = convert_datatype(init_timeseries, datatype=datatype)
  init_dataset = init_tatr_set(init_timeseries, window_size=window_size, stride=1)
  test_timeseries = convert_datatype(test_timeseries, datatype=datatype)
  test_dataset = init_tatr_set(test_timeseries, window_size=window_size, stride=1)

  # Training and prediction
  predictions = []
  track_description = f"{dataname.upper()}-{datatype[0].upper()}-Pred.{pred_model}"
  res_tatr_init_only_path = get_res_tatr_init_only_path(dataname=dataname, pred_model=pred_model)
  filename = f"tatr_initonly_{dataname}-{datatype}_{pred_model}.txt"
  for run in track(range(n_runs), description=track_description):
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
  
  # Store prediction R-squared
  compute_res_tatr_init_only(datatype, pred_model, dataname=dataname, store_res=store_res)


def tatr_train_test_init_only_parallel(
    n_runs, 
    datatypes, 
    pred_models, 
    dataname='sp500', 
    window_size=61, 
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
  print(f"Parallel training of multiple prediction models {pred_models} on historical data.")
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


def compute_res_tatr_init_only(
    datatype, 
    pred_model, 
    n_runs=100,
    dataname='sp500',
    window_size=61,
    ahead=1,
    benchmark='avg', 
    store_res=False, 
    ):
  """ Compute the TATR results of prediction models on historical data """
  # Load test set
  test_timeseries = load_tatr_test(dataname=dataname)
  test_timeseries = convert_datatype(test_timeseries, datatype=datatype)
  test_dataset = init_tatr_set(test_timeseries, window_size=window_size)
  y_test = test_dataset[:, -ahead:].detach().cpu().numpy().ravel()
  y_benchmark_avg = prediction_benchmark(test_dataset, ahead, benchmark=benchmark)

  # Load predictions
  res_tatr_init_only_path = get_res_tatr_init_only_path(dataname=dataname, pred_model=pred_model)
  filename_pred = f"tatr_initonly_{dataname}-{datatype}_{pred_model}.txt"
  predictions = np.loadtxt(res_tatr_init_only_path + filename_pred, delimiter=',')
  if predictions.shape[0] < n_runs:
    n_repeats =  n_runs // predictions.shape[0] + 1
    predictions = np.tile(predictions, (n_repeats, 1))[:n_runs, :]
  
  # Evaluation
  metrics = np.zeros((1, n_runs))
  r2_total = 0
  for run, y_pred in enumerate(predictions):
    r2 = downstream_test_criterion(y_test, y_pred, y_benchmark_avg, criterion='r2')
    metrics[0, run] = r2
    r2_total += r2

  # Store results
  if store_res:
    res_tatr_init_only_r2_path = f"res/{dataname}/r2/init/"
    filename_metrics = f"tatr_{dataname}-{datatype}_initonly_{pred_model}_r2avg.csv"
    df = pd.DataFrame(metrics.T, columns=['r2'])
    df.to_csv(res_tatr_init_only_r2_path + filename_metrics, index=False)
    print(f"Results of {res_tatr_init_only_r2_path + filename_metrics} stored.")


def load_res_tatr_init_only(pred_model, datatype, dataname='sp500'):
  """ Load the TATR results of prediction models on historical data """
  res_tatr_init_only_r2_path = f"res/{dataname}/r2/init/"
  filename = f"tatr_{dataname}-{datatype}_initonly_{pred_model}_r2avg.csv"
  df = pd.read_csv(res_tatr_init_only_r2_path + filename, index_col=False)
  return df









