import numpy as np
import time
import multiprocessing



###############################
# Parallel training functions #
###############################

def get_available_cpus():
  """ Get the number of CPUs """
  cpu_count = multiprocessing.cpu_count()
  # print(f"available_cpus: {cpu_count}")
  return cpu_count


def display_active_processes(active_processes):
  """ Display the active processes """
  active_tasks = [p.name for p in active_processes if p.is_alive()]
  print(f"{len(active_tasks)} Active Processes: {active_tasks}")


def display_active_processes_pool(active_processes):
  """ Display the running processes """
  running_processes = [run for run, process in active_processes if not process.ready()]
  print(f"{len(running_processes)} running processes: {running_processes}")


def display_parallel_progress_pool(n_total, n_completed, n_parallel_jobs, start_time):
  """ Display the current progress and remaining time to complete all processes """
  if n_completed > 0:
    exec_time = time.time() - start_time
    n_total_parallel = (n_total + n_parallel_jobs - 1) // n_parallel_jobs
    n_completed_parallel = (n_completed + n_parallel_jobs - 1) // n_parallel_jobs
    avg_time_per_process = exec_time / n_completed_parallel
    n_remaining_parallel = n_total_parallel - n_completed_parallel
    remaining_time = n_remaining_parallel * avg_time_per_process
    remaining_time_min = remaining_time / 60
    print(f"Progress: {n_completed}/{n_total} Expected remaining time: {remaining_time_min:.2f} min.")


def set_check_interval(run_list):
  """ Set the check interval during parallel training """
  long_response_list = ['lstm', 'rf']
  short_response_list = ['gbdt']
  if run_list in long_response_list:
    return 1.0
  elif run_list in short_response_list:
    return 0.5
  else:
    return 0.1









