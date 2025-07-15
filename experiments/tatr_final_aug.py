import pandas as pd
from rich.progress import track
import multiprocessing
from threadpoolctl import threadpool_limits
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings('ignore')

from experiments.utils_tatr import *
from experiments.utils_tatr_extension import *
from utils.utils_parallel import *

NN_MODELS = {'mlp', 'itransformer'}

def get_res_tatr_final_aug_path(dataname, aug_model, pred_model):
  device_path = 'gpu' if pred_model.split('-')[0] in NN_MODELS else 'cpu'
  res_tatr_final_aug_path = f"res/{dataname}/{device_path}/res_final_aug/aug_{aug_model}/"
  if 'gbdt' in pred_model:
    return res_tatr_final_aug_path + 'pred_gbdt/'
  elif 'rf' in pred_model:
    return res_tatr_final_aug_path + 'pred_rf/'
  elif 'mlp' in pred_model:
    return res_tatr_final_aug_path + 'pred_mlp/'
  elif 'itransformer' in pred_model:
    return res_tatr_final_aug_path + 'pred_itransformer/'
  else:
    NotImplementedError



#############################################
# Train Prediction Models on Augmented Data #
#############################################

def tatr_train_test_final_aug(
    n_runs, 
    dataname='sp500', 
    datatypes='returns', 
    aug_models='', 
    pred_models='', 
    start_run=0, 
    n_augmentations=100, 
    window_size=61, ahead=1, 
    store_res=True, 
    max_parallel_jobs=9, 
    ):
  """ Parallelly train the prediction models on augmented data """
  # Load test data
  test_timeseries = load_tatr_test(dataname=dataname)

  # Parallel training
  if isinstance(pred_models, list):
    log_string = f"Parallel training of multiple prediction models {pred_models} on {aug_models} augmentations."
    print(log_string)
    tatr_train_test_final_aug_parallel_downmodel(
        max_parallel_jobs=max_parallel_jobs, 
        n_runs=n_runs, 
        test_timeseries=test_timeseries, 
        dataname=dataname, 
        datatypes=datatypes, 
        aug_models=aug_models, 
        pred_models=pred_models,
        start_run=start_run, 
        n_augmentations=n_augmentations, 
        window_size=window_size, ahead=ahead, 
        store_res=store_res,
        )


def tatr_train_test_final_aug_single(
    n_runs, 
    test_dataset, 
    dataname='sp500', 
    datatype='returns', 
    aug_model='', 
    pred_model='', 
    start_run=0, 
    aug_size=100, 
    window_size=61, ahead=1, 
    adjust_lr=False, 
    store_res=True, 
    ):
  """ Train the prediction models on augmented data """
  # Initialize the training set and test set
  init_timeseries = load_tatr_init(dataname=dataname)
  init_timeseries = convert_datatype(init_timeseries, datatype=datatype)
  init_dataset = init_tatr_set(init_timeseries, window_size=window_size, stride=1)
  returns_models = {'armagarch', 'bootstrap', 'timegan'}
  
  # Train and test
  predictions = []
  track_description = f"{dataname.upper()}-{datatype[0].upper()}-Aug.{aug_model}-Size.{aug_size}-Pred.{pred_model}"
  res_tatr_final_aug_path = get_res_tatr_final_aug_path(dataname=dataname, aug_model=aug_model, pred_model=pred_model)
  aug_model_name = f"{aug_model}-an{aug_size}"
  res_filename = f"tatr_finalaug_{dataname}-{datatype}_{aug_model_name}_{pred_model}.txt"
  for run in track(range(start_run, n_runs), description=track_description):
    curr_dataset = copy_dataset_downstream(init_dataset)
    augmentations = load_tatr_augmentation(run, dataname=dataname, aug_model=aug_model)
    TRADING_DAYS_PER_YEAR = 252
    n_aug_times = int(np.ceil(aug_size * TRADING_DAYS_PER_YEAR / window_size))
    for i in range(n_aug_times):
      aug_timeseries = augmentations[i]
      if aug_model not in returns_models:
        aug_timeseries = convert_datatype(aug_timeseries, datatype=datatype)
      else:
        if datatype == 'returns':
          aug_timeseries = aug_timeseries
        else:
          raise ValueError(f"Invalid datatype {datatype} for converting the time series augmented by models generating returns series.")
      aug_dataset = Timeseries2Dataset_Downstream(aug_timeseries, window_size=window_size, stride=1)
      curr_dataset = concat_datasets_downstream(curr_dataset, aug_dataset)
    
    # Train on augmented data
    with threadpool_limits(limits=1, user_api='blas'):
      model = separate_train_downstream_model(curr_dataset, ahead, datatype=datatype, modelname=pred_model, adjust_lr=adjust_lr)

    # Test on real
    _, y_pred = prediction_downstream_model(model, test_dataset, ahead, modelname=pred_model)
    predictions.append(y_pred)

    # Store the model predictions
    if store_res:
      with open(res_tatr_final_aug_path + res_filename, 'w') as file:
        for pred in predictions:
          file.write(','.join(map(str, pred)) + '\n')
  if store_res:
    print(f"Results of {res_tatr_final_aug_path + res_filename} stored.")


def tatr_train_test_final_aug_parallel_downmodel(
    max_parallel_jobs, 
    n_runs, 
    test_timeseries, 
    dataname='sp500', 
    datatypes='returns', 
    aug_models='', 
    pred_models='', 
    start_run=0, 
    n_augmentations=100, 
    window_size=61, ahead=1, 
    store_res=True, 
    ):
  """ Parallel setting for training multiple prediction models on augmented data """
  active_processes = []
  with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
    n_augmentations = [n_augmentations] if not isinstance(n_augmentations, list) else n_augmentations
    datatypes = [datatypes] if not isinstance(datatypes, list) else datatypes
    pred_models = [pred_models] if not isinstance(pred_models, list) else pred_models
    aug_models = [aug_models] if not isinstance(aug_models, list) else aug_models
    for n_aug in n_augmentations:
      for p_model in pred_models:
        for datatype in datatypes:
          test_dataset = init_tatr_set(convert_datatype(test_timeseries, datatype=datatype), window_size=window_size, stride=1)
          for a_model in aug_models:
            kwargs = {
              'n_runs': n_runs, 
              'test_dataset': test_dataset, 
              'dataname': dataname, 
              'datatype': datatype, 
              'aug_model': a_model, 
              'start_run': start_run, 
              'aug_size': n_aug, 
              'window_size': window_size, 
              'ahead': ahead, 
              'pred_model': p_model, 
              'store_res': store_res, 
            }
            process = pool.apply_async(tatr_train_test_final_aug_single, kwds=kwargs)
            active_processes.append((p_model, a_model, process))
            while len(active_processes) >= max_parallel_jobs:
              for j, (_, _, process) in enumerate(active_processes):
                if process.ready():
                  active_processes.pop(j)
                  break
              else:
                time.sleep(2)
    for _, _, process in active_processes:
      process.wait()
  print("Parallel training completed.")


def compute_res_tatr_final_aug(
    datatype, 
    pred_model, 
    dataname='sp500', 
    aug_model='', 
    n_runs=100, 
    n_augmentations=100, 
    ahead=1, 
    benchmark='avg', 
    store_res=False, 
    ):
  """ Compute the results of prediction models on augmented data """
  # Load test dataset
  test_timeseries = load_tatr_test(dataname=dataname)
  test_timeseries = convert_datatype(test_timeseries, datatype=datatype)
  test_dataset = init_tatr_set(test_timeseries, window_size=61)
  y_test = test_dataset[:, -ahead:].detach().cpu().numpy().ravel()
  y_benchmark_avg = prediction_benchmark(test_dataset, ahead, benchmark=benchmark)

  # Evaluation
  n_aug = f"-an{n_augmentations}"
  metrics = np.zeros((1, n_runs))
  res_tatr_final_aug_model_path = get_res_tatr_final_aug_path(dataname=dataname, aug_model=aug_model, pred_model=pred_model)
  filename = f"tatr_finalaug_{dataname}-{datatype}_{aug_model}{n_aug}_{pred_model}.txt"
  predictions = np.loadtxt(res_tatr_final_aug_model_path + filename, delimiter=',')
  if predictions.shape[0] < n_runs:
    n_repeats = n_runs // predictions.shape[0] + 1
    predictions = np.tile(predictions, (n_repeats, 1))[:n_runs, :]
  for run, y_pred in enumerate(predictions):
    if len(y_test) != len(y_pred) or len(y_benchmark_avg) != len(y_pred):
      len_min = min(len(y_test), len(y_pred), len(y_benchmark_avg))
      y_test = y_test[:len_min]
      y_pred = y_pred[:len_min]
      y_benchmark_avg = y_benchmark_avg[:len_min]
    r2 = downstream_test_criterion(y_test, y_pred, y_benchmark_avg, criterion='r2')
    metrics[0, run] = r2
  metrics = np.array(metrics)

  # Store results
  if store_res:
    res_tatr_final_aug_r2_path = f"res/{dataname}/r2/aug_{aug_model}/"
    df = pd.DataFrame(metrics.T, columns=['r2'])
    filename = f"tatr_{dataname}-{datatype}_finalaug_{aug_model}{n_aug}_{pred_model}_r2avg.csv"
    df.to_csv(res_tatr_final_aug_r2_path + filename, index=False)
    print(f"Results of {res_tatr_final_aug_r2_path + filename} stored.")


def load_res_tatr_final_aug(
    pred_model, datatype, aug_model, 
    dataname='sp500', n_augmentations=100, 
    ):
  """ Load the results of prediction models on augmented data """
  res_tatr_final_aug_r2_path = f"res/{dataname}/r2/aug_{aug_model}/"
  n_aug = f"-an{n_augmentations}"
  filename = f"tatr_{dataname}-{datatype}_finalaug_{aug_model}{n_aug}_{pred_model}_r2avg.csv"
  df = pd.read_csv(res_tatr_final_aug_r2_path + filename, index_col=False)
  return df









