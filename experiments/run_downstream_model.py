import numpy as np
import random
from experiments.predictors.predictor_randomforest import *
from experiments.predictors.predictor_gbdt import *
from experiments.predictors.predictor_mlp import *
from experiments.predictors.predictor_itransformer import *
from experiments.utils_tatr_extension import *



###########################
# Train Prediction Models #
###########################

def separate_train_downstream_model(dataset, ahead, datatype, modelname='linear', adjust_lr=False):
  """ Separately build and train the prediction model on given the dataset """
  # Fix the random seed
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.cuda.empty_cache()

  # GBDT
  if 'gbdt' in modelname:
    gbdt_n_estimators, gbdt_max_depth, gbdt_max_features = get_default_downstream_model_params(modelname, datatype) \
      if '-' not in modelname else get_investigate_downstream_model_params(modelname)
    return separate_train_gbdt_predictor(
      dataset=dataset, ahead=ahead, 
      n_estimators=gbdt_n_estimators, max_depth=gbdt_max_depth, max_features=gbdt_max_features)
  
  # RF
  elif 'rf' in modelname:
    rf_n_estimators, rf_max_depth, rf_max_features = get_default_downstream_model_params(modelname, datatype) \
      if '-' not in modelname else get_investigate_downstream_model_params(modelname)
    return separate_train_rf_predictor(
      dataset=dataset, ahead=ahead, 
      n_estimators=rf_n_estimators, max_depth=rf_max_depth, max_features=rf_max_features)
  
  # MLP
  elif 'mlp' in modelname:
    mlp_n_layers, mlp_hidden_dim, mlp_reg = get_default_downstream_model_params(modelname, datatype) \
      if '-' not in modelname else get_investigate_downstream_model_params(modelname)
    batch_size, lr = get_downstream_nn_training_params(modelname, datatype)
    return separate_train_mlp_predictor(
      dataset=dataset, datatype=datatype, output_dim=ahead,
      n_layers=mlp_n_layers, hidden_dim=mlp_hidden_dim, regularization=mlp_reg,
      batch_size=batch_size, lr=lr)
  
  # iTransformer
  elif 'itransformer' in modelname:
    batch_size, lr = get_downstream_nn_training_params(modelname, datatype)
    if 'pretrain' in modelname:
      lora_rank = 0 if len(modelname.split('-')) < 3 else int(modelname.split('-')[-1])
      return separate_train_itransformer_predictor(
        dataset=dataset, datatype=datatype, output_dim=ahead, 
        batch_size=batch_size, lr=lr, adjust_lr=adjust_lr, pretrain=True, lora_rank=lora_rank)
    elif 'selftrain' in modelname:
      lora_rank = 0 if len(modelname.split('-')) < 4 else int(modelname.split('-')[-1])
      return separate_train_itransformer_predictor(
        dataset=dataset, datatype=datatype, output_dim=ahead, 
        batch_size=batch_size, lr=lr, adjust_lr=adjust_lr, pretrain_self=modelname, lora_rank=lora_rank)
    else:
      itf_n_layers, itf_d_model, itf_d_ff, itf_n_heads = get_default_downstream_model_params(modelname, datatype) \
        if '-' not in modelname else get_investigate_downstream_model_params(modelname)
      return separate_train_itransformer_predictor(
        dataset=dataset, datatype=datatype, output_dim=ahead,
        n_layers=itf_n_layers, d_model=itf_d_model, d_ff=itf_d_ff, n_heads=itf_n_heads,
        batch_size=batch_size, lr=lr, adjust_lr=adjust_lr)
  
  else:
    raise RuntimeError(f"Prediction model {modelname} is not specified for training.")


def prediction_downstream_model(model, dataset, ahead, modelname=None, scaler=None):
  """ Make predictions with the prediction model on the given dataset """
  if 'gbdt' in modelname:
    return prediction_gbdt(model, dataset, ahead, scaler=scaler)
  elif 'rf' in modelname:
    return prediction_rf(model, dataset, ahead, scaler=scaler)
  elif 'mlp' in modelname:
    return prediction_mlp(model, dataset, scaler=scaler)
  elif 'itransformer' in modelname:
    return prediction_itransformer(model, dataset, scaler=scaler)
  else:
    raise RuntimeError(f"Prediction model {modelname} is not specified for test.")


def count_params_downstream(model, downstream_model):
  """ Count the number of parameters in the prediction model """ 
  if 'gbdt' in downstream_model:
    return count_params_gbdt(model)
  elif 'rf' in downstream_model:
    return count_params_rf(model)
  elif 'mlp' in downstream_model:
    return count_params_mlp(model)
  elif 'itransformer' in downstream_model:
    return count_params_itransformer(model)
  else:
    raise TypeError("Given prediction model is not valid for counting the number of paramaters.")









