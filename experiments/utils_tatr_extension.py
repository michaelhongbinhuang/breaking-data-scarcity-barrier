#############################
# Extended Settings of TATR #
#############################

def get_default_downstream_model_params(model_name, datatype):
  """ Get the default hyper-params. of downstream models """
  # GBDT
  if model_name == 'gbdt':
    gbdt_params = {'n_estimators': 200, 'max_depth': 2, 'max_features': None} if datatype == 'returns' \
      else {'n_estimators': 300, 'max_depth': 2, 'max_features': None}
    return gbdt_params['n_estimators'], gbdt_params['max_depth'], gbdt_params['max_features']
  
  # RF
  elif model_name == 'rf':
    rf_params = {'n_estimators':300, 'max_depth': 5, 'max_features': None} if datatype == 'returns' \
      else {'n_estimators': 300, 'max_depth': 4, 'max_features': None}
    return rf_params['n_estimators'], rf_params['max_depth'], rf_params['max_features']
  
  # MLP
  elif model_name == 'mlp':
    mlp_params = {'n_layers': 2, 'hidden_dim': 64, 'reg': 1} if datatype == 'returns' \
      else {'n_layers': 2, 'hidden_dim': 64, 'reg': 0}
    return mlp_params['n_layers'], mlp_params['hidden_dim'], mlp_params['reg']
  
  # iTransformer
  elif model_name == 'itransformer':
    itf_params = {'n_layers': 2, 'd_model': 64, 'd_ff': 512, 'n_heads': 4} if datatype == 'returns' \
      else {'n_layers': 8, 'd_model': 32, 'd_ff': 512, 'n_heads': 8}
    return itf_params['n_layers'], itf_params['d_model'], itf_params['d_ff'], itf_params['n_heads']
  
  else:
    raise ValueError(f"Invalid downstream model {model_name}")


def get_investigate_downstream_model_params(model_name):
  """ Get the hyper-params. of downstream models for investigation """
  # GBDT or RF
  if 'gbdt' in model_name or 'rf' in model_name:
    str_n_estimator = model_name.split('-')[1]
    n_estimators = 1000 * int(str_n_estimator[:-1]) if str_n_estimator[-1] == 'k' else int(str_n_estimator)
    max_depth = int(model_name.split('-')[2])
    max_features = int(model_name.split('-')[3]) if len(model_name.split('-')) > 3 else None
    return n_estimators, max_depth, max_features
  
  # MLP
  elif 'mlp' in model_name:
    n_layers = int(model_name.split('-')[1])
    hidden_dim = int(model_name.split('-')[2])
    str_reg = model_name.split('-')[3] if len(model_name.split('-')) > 3 else 1
    reg = int(str_reg)
    return n_layers, hidden_dim, reg
  
  # iTransformer
  elif 'itransformer' in model_name:
    n_layers = int(model_name.split('-')[1])
    d_model = int(model_name.split('-')[2])
    d_ff = int(model_name.split('-')[3])
    n_heads = int(model_name.split('-')[4])
    return n_layers, d_model, d_ff, n_heads
  
  else:
    raise ValueError(f"Invalid downstream model {model_name}")
  

def get_downstream_nn_training_params(modelname, datatype):
  """ Get the training hyper-params for downstream NN models """
  if 'mlp' in modelname:
    params = {'batch_size': 2048, 'lr': 1e-5} if datatype == 'returns' \
      else {'batch_size': 1024, 'lr': 1e-3}
    return params['batch_size'], params['lr']
  
  elif 'itransformer' in modelname:
    params = {'batch_size': 2048, 'lr': 7e-5} if datatype == 'returns' \
      else {'batch_size': 512, 'lr': 3e-5}
    return params['batch_size'], params['lr']
  
  else:
    raise ValueError(f"Invalid downstream model {modelname}")


def get_pretrain_downstream_model_params(dataname, modelname):
  """ Get hyperparams. of the pre-trained model """
  if modelname == 'itransformer':
    params = {
      
      'exchange': {
        'n_channels': 8, 'seq_len': 96, 'pred_len': 96, 
        # 'n_layers': 2, 'n_heads': 8, 'd_model': 128, 'd_ff': 128,   # Official
        'n_layers': 2, 'n_heads': 8, 'd_model': 128, 'd_ff': 256, 
        }, 
      }
  else:
    raise ValueError(f"Invalid dataset {dataname} or pre-trained model {modelname}")
  return params[dataname]


def get_model_colors(models):
  """ Get the colors of generative models for boxplot """
  colors = {
    'init': '#fe9f69', 
    'brownianmotion': '#ace0cf', 
    'armagarch': '#8ebcdb', 
    'bootstrap': '#a79fce', 
    'timegan': '#f5c1c8', 
    'ftsdiffusion-crf': '#4c9be6', 
    }
  return [colors.get(model, None) for model in models]


def get_aug_model_labels(list_aug_models):
  """ Get the labels of generative models for legend in figures """
  dict_aug_models = {
    'init': 'Historical', 
    'brownianmotion': 'BrownianMotion', 
    'armagarch': 'ARMA-GARCH', 
    'bootstrap': 'Bootstrap', 
    'timegan': 'TimeGAN', 
    'ftsdiffusion-crf': 'FTS-Diffusion', 
  }
  return [dict_aug_models[model] for model in list_aug_models]






  


