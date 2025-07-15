import torch
import multiprocessing
import argparse


############################
# Get the computing device #
############################

def get_device(train_nn=False):
  """ Get the computing device """
  if torch.cuda.is_available():
    return torch.cuda.current_device()
  elif torch.backends.mps.is_available() and train_nn:
    return torch.device("mps")
  else:
    return torch.device("cpu")


def get_cpu_count():
  """ Get the number of CPUs """
  cpu_count = multiprocessing.cpu_count()
  return cpu_count



############################
# Get the computing device #
############################

def args_str_to_bool(value):
  if value.lower() in ("true", "1"):
    return True
  elif value.lower() in ("false", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Invalid boolean value")









