import numpy as np
import torch
torch.set_default_dtype(torch.float)



##########################
# Construct Markov Chain #
##########################

def construct_markov_chain(subsequences, labels, segmentation):
  """ Construct the Markov chain of states (pattern, length, magnitude) """
  series_pattern = labels.astype(int)
  series_length = segmentation.astype(int)
  series_magnitude = np.array([max(seg)-min(seg) for seg in subsequences], dtype=float)
  return series_pattern, series_length, series_magnitude


def construct_states_pairs(chain):
  """ Construct the (current states, next states) pairs """
  return np.array([[curr, next] for curr, next in zip(chain[:-1], chain[1:])])


def construct_markov_states_pairs(chain_pattern, chain_length, chain_magnitude, l_min=10, catlen=True):
  """ Construct the Markov chain of states pairs of (current states, next states) """
  if not catlen:
    chain_length = chain_length/l_min    # Scale the range of values for regression
  chain = np.stack((chain_pattern, chain_length, chain_magnitude), axis=1)
  states_pairs = construct_states_pairs(chain)
  return states_pairs









