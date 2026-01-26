import numpy as np
import pandas as pd



#############################################
# Load SISC-resulting segments and patterns #
#############################################

def load_segments(dataname):
  model_path = f"models/trained_models/{dataname}/"
  filename = f"prm_{dataname}"
  centroids = pd.read_csv(model_path + filename + '_centroids.csv').values[:,1:]
  labels = pd.read_csv(model_path + filename + '_labels.csv').values[:,1]
  segmentation = pd.read_csv(model_path + filename + '_segmentation.csv').values[:,1]
  segmentation = np.array([segmentation[i+1]-segmentation[i] for i in range(len(segmentation)-1)], dtype=int)
  df_subsequences = pd.read_csv(model_path + filename + '_subsequences.csv')
  subsequences = df_subsequences.values[:,1]
  subsequences = np.array([
    np.array([float(x.strip('[] ')) for x in s.split() if x.strip('[] ')]) 
    for s in subsequences
    ], dtype=object)
  return centroids, subsequences, labels, segmentation


def prepare_segments(dataname='sp500'):
  """ Prepare the training and test data for experiments """
  centroids, segments, labels, lengths = load_segments(dataname)
  segment_set = (segments, labels, lengths)
  return segment_set, centroids









