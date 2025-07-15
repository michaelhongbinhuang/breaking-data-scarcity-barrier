import time
import argparse
import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from experiments.data_augmentation import *
from experiments.utils_tatr import *
from utils.sys_config import *



# ###################
# Data Augmentation #
# ###################

def tatr_augment_data_parallel(
    max_parallel_jobs, 
    dataname, 
    aug_models, 
    n_runs, 
    aug_size, aug_length, 
    store_data=True, 
    init_segment_strategy='random', 
    ):
  """ Parallel data augmentation for TATR """
  active_processes = []
  with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
    for aug_model in aug_models:
      kwargs = {
        'dataname': dataname, 
        'aug_model': aug_model,
        'n_runs': n_runs,
        'n_aug_year': aug_size,
        'aug_length': aug_length,
        'store_data': store_data,
        'init_segment_strategy': init_segment_strategy,
      }
      process = pool.apply_async(tatr_augment_data, kwds=kwargs)
      active_processes.append((aug_model, process))
      while len(active_processes) >= max_parallel_jobs:
        for j, (_, process) in enumerate(active_processes):
          if process.ready():
            active_processes.pop(j)
            break
        else:
          time.sleep(2)
    for _, process in active_processes:
      process.wait()
  print("All data augmentation processes completed.")


if __name__ == "__main__":
  # Set up inputs
  parser = argparse.ArgumentParser(description="Data Augmentation")
  parser.add_argument("--dataname", type=str, default='sp500', help="Asset")
  parser.add_argument("--aug_models", type=str, nargs='+', default=['bootstrap'], help="Augmentation models")
  parser.add_argument("--n_runs", type=int, default=100, help="Number of runs")
  parser.add_argument("--aug_size", type=int, default=100, help="Augmentation size")
  parser.add_argument("--aug_length", type=int, default=100, help="Augmentation length")
  parser.add_argument("--init_segment_strategy", type=str, default='random', help="Initial segment strategy")
  parser.add_argument("--idle_cpu", type=int, default=2, help="Number of idle CPUs")
  args = parser.parse_args()

  # Get experiment settings
  print("-- Augmentation Setting --")
  dataname = args.dataname
  aug_models = args.aug_models
  n_runs = args.n_runs
  aug_size = args.aug_size
  aug_length = args.aug_length
  init_segment_strategy = args.init_segment_strategy
  print(f"Asset: {dataname}")
  print(f"Augmentation models: {aug_models}")
  print(f"Number of runs: {n_runs}")
  print(f"Augmentation size: {aug_size}")
  print(f"Augmentation length: {aug_length}")
  print(f"Initial segment strategy: {init_segment_strategy}")
  
  # Get system settings
  print("-- System Configuration --")
  available_cpus = get_cpu_count()
  max_parallel_jobs = min(available_cpus - args.idle_cpu, 8)
  print(f"Available CPUs: {available_cpus}")
  print(f"Idle CPUs: {args.idle_cpu}")
  print(f"Max parallel jobs: {max_parallel_jobs}")

  # Data augmentation
  print("-- Data Augmentation --")
  tatr_augment_data_parallel(
    max_parallel_jobs=max_parallel_jobs, 
    dataname=dataname, 
    aug_models=aug_models,
    n_runs=n_runs, 
    aug_size=aug_size, 
    aug_length=aug_length, 
    store_data=True, 
    init_segment_strategy=init_segment_strategy, 
    )









