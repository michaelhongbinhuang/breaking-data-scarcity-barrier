import numpy as np
import torch
from torch.nn import functional as F
torch.set_default_dtype(torch.float)

from models.utils_sampling import *
from models.pattern_generation_module import *
from models.pattern_evolution_module_crf import *
from utils.sys_config import get_device



#############################
# Generate the next segment #
#############################

def segment_generation_fast(model, states, patterns):
  """ Generate segments (fast version) """
  device = get_device(train_nn=True)
  n_segments = len(states)
  patterns = torch.tensor(patterns).float().to(device)
  p = states[:, 0].long().view(-1, 1)
  lengths = states[:, 1].long().view(n_segments).to(device)
  magnitudes = states[:, -1].view(-1, 1).to(device)
  ref_patterns = patterns[p].view(n_segments, patterns.shape[1]).to(device)
  with torch.no_grad():
    x_, _ = model.generate(ref_patterns, lengths)
  new_segments = x_ * magnitudes.to(device)
  return new_segments, lengths



######################################
# Generate new synthetic time series #
######################################

def generate_timeseries_fast(
    T, 
    init_state=None, 
    init_segment=None, 
    patterns=None, 
    datatype='prices', 
    version='', 
    dataname='sp500', 
    ):
  """ Generate the synthetic time series (fast version) """
  model_pgm = load_pattern_generation_module(dataname)
  if patterns is None:
    _, _, patterns = sampling_inputs()
  while True:
    state, first_segment = get_init_state(init_state, init_segment) if init_state is None else init_state, init_segment
    states = []
    curr_T = len(first_segment)
    while curr_T < T:
      # Pattern transition with conditional random fields
      if 'crf' in version:
        state = state_evolution_crf(state)
      else:
        raise ValueError(f"Invalid pattern evolution module {version}.")
      
      states.append(state.detach().cpu().numpy())
      curr_T += state[:, 1].detach().cpu().numpy() - 1

    states =  np.array(states)
    states = torch.tensor(states).view(-1, 3)
    segments, lengths = segment_generation_fast(model_pgm, states, patterns)
    timeseries = segments2timeseries(segments, first_segment, lengths, datatype)
    timeseries = timeseries[:T]
    return timeseries









