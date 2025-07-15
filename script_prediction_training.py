import argparse
import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from experiments.tatr_init_only import tatr_train_test_init_only_single
from experiments.tatr_final_aug import tatr_train_test_final_aug_single, NN_MODELS
from experiments.utils_tatr import *
from utils.sys_config import *



#################################
# Return Prediction Experiments #
#################################

if __name__ == "__main__":
  # Set up inputs
  parser = argparse.ArgumentParser(description="Return Prediction Experiments")
  parser.add_argument("--dataname", type=str, default='sp500', help="Data name")
  parser.add_argument("--datatypes", type=str, nargs='+', default=['returns'], help="Predicting datatypes")
  parser.add_argument("--aug_models", type=str, nargs='+', default=['bootstrap'], help="Augmentation models")
  parser.add_argument("--pred_models", type=str, nargs='+', default=['gbdt'], help="Downstream models")
  parser.add_argument("--n_runs", type=int, default=100, help="Number of runs")
  parser.add_argument("--start_run", type=int, default=0, help="Start run")
  parser.add_argument("--n_augmentation", type=int, default=100, help="Number of augmentations")
  parser.add_argument("--adjust_lr", type=args_str_to_bool, default=False, help="Adjust learning rate")
  parser.add_argument("--store_res", type=args_str_to_bool, default=False, help="Store the results")
  parser.add_argument("--set_gpu", type=int, default=0, help="Set the CUDA device")
  args = parser.parse_args()

  # Get experiment settings
  print("-- Experimental Setting --")
  dataname = args.dataname
  datatype = args.datatypes[0]
  aug_model = args.aug_models[0]
  pred_model = args.pred_models[0]
  n_augmentation = args.n_augmentation
  n_runs = args.n_runs
  start_run = args.start_run
  adjust_lr = args.adjust_lr
  store_res = args.store_res
  print(f"Data name: {dataname}")
  print(f"Data types: {datatype}")
  print(f"Augmentation models: {aug_model}")
  print(f"Downstream models: {pred_model}")
  print(f"Number of augmentations: {n_augmentation}")
  print(f"Number of runs: {n_runs}. Start run: {start_run}")
  print(f"Adjust learning rate: {adjust_lr}")
  print(f"Store results: {store_res}")

  # Get system settings
  print("-- System Configuration --")
  if pred_model in NN_MODELS:
    torch.cuda.set_device(args.set_gpu)
    assign_gpus = get_device()
    print(f"Using GPUs: {assign_gpus}")
  
  # Train on init. data and predict
  print("** Prediction Experiments on Historical Data Started **")
  init_timeseries = load_tatr_init(dataname=dataname)
  test_timeseries = load_tatr_test(dataname=dataname)
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
  print("** Prediction Experiments on Historical Data Completed **", end='\n\n')

  # Train on aug. data and predict
  print("** Prediction Experiments on Augmented Data Started **")
  test_timeseries = load_tatr_test(dataname=dataname)
  test_timeseries = convert_datatype(test_timeseries, datatype=datatype)
  test_dataset = init_tatr_set(test_timeseries)
  tatr_train_test_final_aug_single(
    n_runs=n_runs, 
    test_dataset=test_dataset, 
    dataname=dataname, 
    datatype=datatype, 
    aug_model=aug_model, 
    pred_model=pred_model, 
    aug_size=n_augmentation, 
    start_run=start_run, 
    adjust_lr=adjust_lr, 
    store_res=store_res, 
    )
  print("** Prediction Experiments on Augmented Data Completed **", end='\n\n')

  







