from rich.progress import track

from models.sampling import generate_timeseries_fast, prepare_segments
from experiments.utils_tatr import *
from utils.series_processing import convert_datatype



##################################
# Generate synthetic time series #
##################################

def generate_syn_timeseries_downstream(
    dataname='sp500', 
    aug_model='', 
    aug_length=100, 
    init_segment_strategy='random', 
    ):
  """ Provide synthetic timeseries to augment the dataset """
  if 'ftsdiffusion' in aug_model.lower():
    while True:
      init_state, init_segment = get_first_segment(dataname=dataname, strategy=init_segment_strategy)
      syn_ts = generate_timeseries_fast(aug_length, init_state, init_segment, version=aug_model, dataname=dataname)
      syn_ts_returns = convert_datatype(syn_ts, 'returns')
      if not np.isnan(syn_ts_returns).any():
        return syn_ts
  else:
    raise AttributeError(f"The input augmentation model {aug_model} is invalid.")


def get_first_segment(dataname='sp500', strategy='random'):
  """ Get the initial state and segment """
  segment_set, _ = prepare_segments(dataname=dataname)
  segments, labels, lengths = segment_set
  if strategy == 'fixed':
    init_state, init_segment = init_first_segment(segments, labels, lengths)
  elif strategy == 'random':
    idx = np.random.randint(0, int(len(segments) * 0.8))
    init_state, init_segment = init_first_segment(segments, labels, lengths, idx=idx)
  else:
    raise ValueError(f"Strategy {strategy} for initializing the first segment is invalid.")
  return init_state, init_segment


def init_first_segment(segments, labels, lengths, idx=None):
  """ Get the initial state of the first segment """
  idx = 0 if idx is None else idx
  init_segment = segments[idx]
  init_pattern = labels[idx]
  init_length = lengths[idx]
  init_magnitude = max(init_segment) - min(init_segment)
  init_state = np.stack((init_pattern, init_length, init_magnitude), axis=0)
  init_state = torch.tensor(init_state).view(1, -1)
  return init_state, init_segment


def create_syn_dataset(syn_timeseries, window_size=61, stride=1, datatype='prices', scaler=None, normalize=False):
  """ Convert synthetic data to dataset """
  if datatype == 'prices':
    return Timeseries2Dataset_Downstream(syn_timeseries, window_size, stride=stride, scaler=scaler, normalize=True)
  elif datatype == 'returns':
    syn_timeseries = syn_timeseries[1:] / syn_timeseries[:-1] - 1
    return Timeseries2Dataset_Downstream(syn_timeseries, window_size, stride=stride, scaler=scaler, normalize=normalize)
  else:
    raise AttributeError(f"Datatype {datatype} for synthetic time series in TATR is invalid.")



##############################
# Data Augmentation for TATR #
##############################

def tatr_augment_data(
    dataname='sp500', 
    aug_model='',
    n_runs=100, n_aug_year=100, aug_length=100,
    window_size=61, 
    store_data=True, 
    **kwargs, 
    ):
  """ Prepare the augmented synthetic time series for TATR """
  data_aug_path = f"data/augmentations_{dataname}/{aug_model}/"
  TRADING_DAYS_PER_YEAR = 252
  n_augmentations = int(np.ceil(n_aug_year * TRADING_DAYS_PER_YEAR / window_size))
  aug_model_name = aug_model
  log_augmentation = f"TATR - Augmented {dataname.upper()} generation by {aug_model_name}"
  log_augmentation == f" | #Runs: {n_runs} Aug. Length:{aug_length} #Aug. {n_aug_year}."
  if kwargs.get('init_segment_strategy') is not None:
    log_augmentation += f" | Init. Segment Strategy: {kwargs.get('init_segment_strategy')}"
  print(log_augmentation)
  for run in track(range(n_runs), description=f"D.{dataname.upper()}-Aug.{aug_model}"):
    augmentations = []
    for _ in range(n_augmentations):
      syn_timeseries = generate_syn_timeseries_downstream(
        dataname=dataname, aug_model=aug_model, aug_length=aug_length, 
        init_segment_strategy=kwargs.get('init_segment_strategy'), 
        )
      augmentations.append(syn_timeseries)

    # Store the augmented synthetic time series in this run
    if store_data:
      filename = f"tatr_aug_{dataname}_{aug_model_name}_an{n_aug_year}_{run}.txt"
      with open(data_aug_path + filename, 'w') as file:
        for aug_timeseries in augmentations:
          file.write(','.join(map(str, aug_timeseries)) + '\n')
  if store_data:
    print(f"{n_runs} runs of tatr_aug_{dataname}_{aug_model_name}_an{n_aug_year} stored.")


def load_tatr_augmentation(
    run,
    dataname='sp500',
    aug_model='',
    n_aug_year=100, 
    **kwargs):
  """ Load the stored augmented synthetic datasets for TATR """
  data_aug_path = f"data/augmentations_{dataname}/{aug_model}/"
  if kwargs.get('start') is not None:
    aug_model_name = f"{aug_model}_{kwargs.get('start')}-{kwargs.get('end')}"
  else:
    aug_model_name = aug_model
  filename = data_aug_path + f"tatr_aug_{dataname}_{aug_model_name}_an{n_aug_year}_{run}.txt"
  aug_timeseries = np.loadtxt(filename, delimiter=',')
  return aug_timeseries









