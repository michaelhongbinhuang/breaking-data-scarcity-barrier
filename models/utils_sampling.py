import torch
torch.set_default_dtype(torch.float)

from utils.prepare_segments import *
from utils.markov_processing import *



###########################
# Prepare sampling inputs #
###########################

def sampling_inputs(dataname='sp500', split_raito=0.8):
  """ Prepare the inputs for sampling module """
  _, test_set, patterns = prepare_segments(dataname=dataname, split_ratio=split_raito)
  segments, labels, lengths = test_set
  chain_pattern, chain_length, chain_magnitude = construct_markov_chain(segments, labels, lengths)
  chain_states = np.stack((chain_pattern, chain_length, chain_magnitude), axis=1)
  return segments, chain_states, patterns


def get_init_state_by_index(sample_idx=None):
  """ Initialize the state of first segment for sampling by given index """
  subseqs_real, states_real, _ = sampling_inputs()
  if sample_idx is None:
    sample_idx = np.random.randint(0, len(states_real))
  init_state = torch.tensor(states_real[sample_idx]).view(1, -1)
  init_segment = subseqs_real[sample_idx]
  return init_state, init_segment, sample_idx


def get_init_state(init_state=None, init_segment=None):
  """ Initialize the states """
  init_state, init_segment, _ = get_init_state_by_index()
  return init_state, init_segment


def segments2timeseries(syn_segments, first_segment, lengths, datatype='prices'):
  """ Combine the generated segments together to construct the entire time series """
  syn_segments = syn_segments.detach().cpu().numpy()
  # lengths = lengths.detach().cpu().numpy()
  timeseries = np.array(first_segment)
  for i in range(len(lengths)):
    segment = syn_segments[i, :lengths[i]]
    segment = segment - segment[0] + timeseries[-1]
    timeseries = np.concatenate((timeseries, segment[1:])) if datatype == 'prices' else np.concatenate((timeseries, segment))
  return timeseries









