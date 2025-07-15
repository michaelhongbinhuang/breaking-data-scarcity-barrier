import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from dtaidistance import dtw as dtai_dtw
from tslearn.barycenters import softdtw_barycenter, dtw_barycenter_averaging, dtw_barycenter_averaging_subgradient



##################
# Cluster losses #
##################

def compute_within_cluster_loss(centroids, sequences, labels):
  """ Compute the within-clutser loss using DTW to measure the total discrepancy within clusters  """
  n_clusters = len(centroids)
  sequences = normalize_segments(sequences)
  n_seqs = len(sequences)
  loss_total = 0
  loss_clusters = np.zeros(n_clusters)
  for i in range(n_clusters):
    sequences_i = sequences[labels==i]
    n_seqs_i = len(sequences_i)
    if n_seqs_i==0:
      loss_clusters[i] = 0
    else:
      loss_i = sum([dtai_dtw.distance_fast(seq.astype(np.double), centroids[i].astype(np.double), use_pruning=True) for seq in sequences_i])
      loss_total += loss_i
      loss_clusters[i] = loss_i/n_seqs_i
  return loss_total/n_seqs, np.mean(loss_clusters)



################################
# Load SISC-resulting clusters #
################################

def load_sisc_res(dataname):
  """ Load learned segmentation and clustering results """
  model_path = f"trained_models/{dataname}/"
  filename = f"prm_{dataname}"
  df_centroids = pd.read_csv(model_path + filename + '_centroids.csv')
  df_labels = pd.read_csv(model_path + filename+'_labels.csv')
  df_subsequences = pd.read_csv(model_path + filename + '_subsequences.csv')
  df_segmentation = pd.read_csv(model_path + filename + '_segmentation.csv')
  subsequences = df_subsequences.values[:,1]
  subsequences = np.array([np.float64(subsequences[i].strip('[]').split()) for i in range(len(subsequences))], dtype=object)
  return df_centroids.values[:,1:], subsequences, df_labels.values[:,1], df_segmentation.values[:,1]



########################################
# Cluster-related computation for SISC #
########################################

def normalize_segments(segments):
  """ Normalize the segments into the unit scale in magnitude """
  segments_norm = []
  for seg in segments:
    max_value = max(seg)
    min_value = min(seg)
    seg_norm = (seg - min_value) / (max_value - min_value)
    segments_norm.append(seg_norm)
  return np.array(segments_norm, dtype=object)


def compute_centroids(n_patterns, segments, labels=None, barycenter='dba', gamma=.001, size=None):
  """ Compute the centroids of segments in each cluster """
  segments = np.array(segments.copy(), dtype=object)
  if n_patterns==1:
    if barycenter=='dba':
      return dtw_barycenter_averaging(segments, barycenter_size=size, tol=1e-5).flatten().astype(float)
    elif barycenter == 'softdtw':
      return softdtw_barycenter(segments, gamma=gamma, tol=1e-5).flatten().astype(float)
    elif barycenter=='dbasubgrad':
      return dtw_barycenter_averaging_subgradient(segments, barycenter_size=size, tol=1e-5).flatten().astype(float)
  else:
    centroids = []
    for i in range(n_patterns):
      idx_i = np.where(labels == i)[0]
      segments_i = segments[idx_i]
      if barycenter == 'dba':
        centroid = dtw_barycenter_averaging(segments_i, barycenter_size=size, tol=1e-5).flatten()
      elif barycenter == 'softdtw':
        centroid = softdtw_barycenter(segments_i, gamma=gamma, tol=1e-5).flatten()
      elif barycenter=='dbasubgrad':
        centroid = dtw_barycenter_averaging_subgradient(segments_i, barycenter_size=size, tol=1e-5).flatten()
      centroids.append(centroid.astype(float))
    return np.array(centroids)


def compute_label_alignment(real, pred):
  """ Compute the label aligment between learned clusters and the ground-truth (if applicable) """
  K = len(real)
  alignment = np.zeros(K)
  candidate = np.arange(K)
  # Greedily find the nearest learned centroid for each ground-truth centroid
  for i in range(K):
    distances = [dtai_dtw.distance_fast(real[i].astype(np.double), pred[j].astype(np.double), use_pruning=True) for j in candidate]
    select = np.argmin(distances)
    alignment[i] = candidate[select]
    candidate = np.delete(candidate, select)
  return alignment.astype(int)


def compute_label_alignment_hungarian(real, pred):
  """ Compute the label aligment between learned clusters and the ground-truth (if applicable) using hungarian algorithm """
  K = len(real)
  distance_matrix = np.zeros((K,K))
  for i in range(K):
    for j in range(K):
      distance_matrix[i,j] = dtai_dtw.distance_fast(real[i].astype(np.double), pred[j].astype(np.double), use_pruning=True)
  row_ind, col_ind = linear_sum_assignment(distance_matrix)
  alignment = col_ind
  return alignment.astype(int)


def align_labels(labels, align):
  """ Align the labels of learned clusters with the ground-truth (if applicable) """
  labels_aligned = [np.where(align==label) for label in labels]
  return np.array(labels_aligned, dtype=object).flatten()


def label_series_from_seg(segmentation, labels):
  """ Get the label series from segmentation """
  N = len(labels)
  label_series = []
  for i in range(N):
    label_series.extend([labels[i]] * (segmentation[i+1]-segmentation[i]))
  return np.array(label_series)


def get_init_strategy_abbr(init_strategy):
  """ Get the abbr. of initial strategies """
  init_strategies = {
      'kmeans++': 'kmpp',
      'random_sample': 'rs',
      'random_noise': 'rn',
      'reference':'ref',
      }
  return init_strategies[init_strategy]









