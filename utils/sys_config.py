import torch
import multiprocessing
import argparse


############################
# Get the computing device #
############################

def get_device(set_gpu=0):
  """ Get the computing device """
  if torch.cuda.is_available():
    torch.cuda.set_device(set_gpu)
    return torch.device(f'cuda:{set_gpu}')
  else:
    return torch.device("cpu")


def get_cpu_count():
  """ Get the number of CPUs """
  cpu_count = multiprocessing.cpu_count()
  return cpu_count


def args_str_to_bool(value):
  if value.lower() in ("true", "1"):
    return True
  elif value.lower() in ("false", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Invalid boolean value")









