import numpy as np
import torch



########################
# Predition Benchmarks #
########################

def benchmark_avg(X):
  """ Test the benchmark of average over the given history as prediction on real test dataset """
  y_ = torch.mean(X, dim=1)
  return y_.detach().cpu().numpy()


def prediction_benchmark(test_dataset, ahead=1, benchmark='avg'):
  """ Prediction error of downstream linear model """
  X_test, y_test = test_dataset[:, :-ahead], test_dataset[:, -ahead:]
  y_test = y_test.detach().cpu().numpy()
  if benchmark == 'avg':
    y_pred = benchmark_avg(X_test)
  elif benchmark == '0':
    y_pred = np.zeros_like(y_test)
  return y_pred.ravel()









