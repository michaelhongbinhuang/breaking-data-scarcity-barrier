import sklearn_crfsuite
import pickle

from models.utils_evolution import *
from utils.prepare_segments import prepare_segments



#####################################
# Pattern evolution module with CRF #
#####################################

def train_evolution_module_crf(
    algo='lbfgs',
    c1=0.1, c2=0.1, 
    n_iter=10, 
    store_model=False, 
    dataname='sp500', 
    ):
  """ Training phase of the pattern evolution module using CRF """
  model_path = f"trained_models/{dataname}/"
  X_train, y_train = construct_train_data_evolution_crf(dataname=dataname)
  if algo == 'lbfgs':
    crf = sklearn_crfsuite.CRF(
      algorithm=algo,
      c1=c1,
      c2=c2,
      max_iterations=n_iter,
      all_possible_transitions=True
    )
  elif algo == 'l2sgd':
    crf = sklearn_crfsuite.CRF(
      algorithm=algo,
      c2=c2,
      max_iterations=n_iter,
      all_possible_transitions=True
    )
  else:
    crf = sklearn_crfsuite.CRF(
      algorithm=algo,
      max_iterations=n_iter,
      all_possible_transitions=True
    )
  crf.fit([X_train], [y_train])
  
  # Store the learned model
  if store_model:
    modelname = f"pem-crf_{dataname}"
    with open(model_path + modelname + ".pkl", 'wb') as file:
      pickle.dump(crf, file)
  return crf


def state_evolution_crf(state, crf=None, dataname='sp500'):
  """ Predict the next state by employing pattern evolution module using CRF """
  model_path = f"trained_models/{dataname}/"
  curr_pattern, curr_length, curr_magnitude = state[0]
  curr_state = [{'pattern': curr_pattern, 'length': curr_length, 'magnitude': curr_magnitude}]
  # Load the trained model
  if crf is None:
    modelname = f"pem-crf_{dataname}"
    with open(model_path + modelname + ".pkl", 'rb') as file:
      crf = pickle.load(file)
  # Predict the next state
  pred_state = crf.predict_single(curr_state)
  pred_pattern, pred_length, pred_magnitude = pred_state[0].split('_')
  pred_pattern, pred_length, pred_magnitude = float(pred_pattern), float(pred_length), float(pred_magnitude)
  pred_pattern = torch.tensor([pred_pattern]).unsqueeze(0)
  pred_length = torch.tensor([pred_length]).unsqueeze(0)
  pred_magnitude = torch.tensor([pred_magnitude]).unsqueeze(0)
  next_state = torch.cat((pred_pattern, pred_length, pred_magnitude), dim=1)
  return next_state


def construct_train_data_evolution_crf(dataname='sp500'):
  """ Contruct data for training evolution module with conditional random fields """
  train_set, _, _ = prepare_segments(dataname=dataname)
  chain_pattern, chain_length, chain_magnitude = Dataset2Chain(train_set)
  states_pairs = construct_markov_states_pairs(chain_pattern, chain_length, chain_magnitude)
  X_train = [{'pattern': pair[0][0], 'length': pair[0][1], 'magnitude': pair[0][2]} for pair in states_pairs]
  y_train = [f"{pair[1][0]}_{pair[1][1]}_{pair[1][2]}" for pair in states_pairs]
  return X_train, y_train









