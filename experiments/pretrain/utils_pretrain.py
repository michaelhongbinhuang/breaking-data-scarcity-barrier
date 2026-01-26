from experiments.predictors.predictor_itransformer import iTransformerPredictor_pretrain



###############################################
# Settings of Self-trained Pre-trained Models #
###############################################

def init_pretrain_model(modelname, model_params):
  """ Initialize the pre-trained model with given params. """
  if modelname == 'itransformer':
    model = iTransformerPredictor_pretrain(
      input_dim=model_params['n_channels'], 
      output_dim=model_params['n_channels'], 
      seq_len=model_params['seq_len'], 
      pred_len=model_params['pred_len'], 
      d_model=model_params['d_model'], 
      n_heads=model_params['n_heads'], 
      e_layers=model_params['n_layers'], 
      d_ff=model_params['d_ff'], 
    )
  else:
    raise ValueError(f"Invalid pre-trained model: {modelname}")
  return model









