import numpy as np
import pandas as pd

from models.model_params import prm_params



#############################################
# Load SISC-resulting segments and patterns #
#############################################

def load_segments(dataname, params):
  """ Load SISC learned segmentation and clustering results """
  dict_init = {'kmeans++': 'kmpp', 'random_sample': 'rs', 'random_noise': 'rn'}
  n_clusters = params['k']
  l_min, l_max = params['l_min'], params['l_max']
  barycenter = params['barycenter']
  init_strategy = params['init_strategy']
  model_path = f"trained_models/{dataname}/"
  filename = f'prm_{dataname}_k{n_clusters}_l{l_min}-{l_max}_{barycenter[:4]}_{dict_init[init_strategy]}'
  centroids = pd.read_csv(model_path + filename + '_centroids.csv').values[:,1:]
  labels = pd.read_csv(model_path + filename + '_labels.csv').values[:,1]
  segmentation = pd.read_csv(model_path + filename + '_segmentation.csv').values[:,1]
  segmentation = np.array([segmentation[i+1]-segmentation[i] for i in range(len(segmentation)-1)], dtype=int)
  df_subsequences = pd.read_csv(model_path + filename + '_subsequences.csv')
  subsequences = df_subsequences.values[:,1]
  subsequences = np.array([np.float64(subsequences[i].strip('[]').split()) for i in range(len(subsequences))], dtype=object)
  return centroids, subsequences, labels, segmentation


def prepare_segments(dataname='sp500'):
  """ Prepare the training and test data for experiments """
  centroids, segments, labels, lengths = load_segments(dataname, prm_params)
  segment_set = (segments, labels, lengths)
  return segment_set, centroids









