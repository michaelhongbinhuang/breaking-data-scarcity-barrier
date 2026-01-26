import matplotlib.pyplot as plt
import random
from experiments.tatr_init_only import *
from experiments.tatr_final_aug import *
from experiments.predictors.utils_predictors import adjust_learning_rate



########################################
# Demo Prediction by Downstream Models #
########################################

def demo_train_prediction(
    datatype,
    down_model, 
    dataname='sp500', 
    aug_model=None,
    aug_size=100,
    aug_run=0,
    window_size=60,
    predstep=1,
    ahead=1,
    batch_size=None, lr=None,
    adjust_lr=False,
    print_info=True,
    plot_pred=False,
    plot_scale=[700,1250],
    set_gpu=7,
    ):
  """ Demo of training and prediction """
  if 'mlp' in down_model or 'transformer' in down_model or 'tst' in down_model:
    batch_size, lr = get_downstream_nn_training_params(down_model) if batch_size is None else (batch_size, lr)
    print(f"-- {dataname.upper()} {datatype.upper()} - t{predstep} | {down_model} adjust_lr: {adjust_lr} batch_size: {batch_size} lr: {lr:.0e} --")
  else:
    print(f"-- {dataname.upper()} {datatype.upper()} - t{predstep} | {down_model} --")
  # Load data
  init_timeseries = load_tatr_init(dataname=dataname)
  init_timeseries = convert_datatype(init_timeseries, datatype=datatype)
  init_dataset = init_tatr_set(init_timeseries, window_size=window_size, predstep=predstep, stride=1)
  test_timeseries = load_tatr_test(dataname=dataname)
  test_timeseries = convert_datatype(test_timeseries, datatype=datatype)
  test_dataset = init_tatr_set(test_timeseries, window_size=window_size, predstep=predstep, stride=1)
  y_benchmark_avg = prediction_benchmark(test_dataset, ahead, benchmark='avg')
  
  # Set device
  if torch.cuda.is_available():
    torch.cuda.set_device(set_gpu)

  # Train downstream models on init. data
  print("- Training on Init. Data -")
  model_init = separate_train_downstream_model(
    init_dataset, ahead, datatype=datatype, modelname=down_model, 
    adjust_lr=adjust_lr
    )
  n_params_init = count_params_downstream(model_init, down_model)
  print(f"# of parameters in {down_model.upper()}: {n_params_init}")
  y_true, y_pred_init = prediction_downstream_model(model_init, test_dataset, ahead, modelname=down_model)
  if print_info:
    print_demo_pred_info(y_true, y_pred_init, y_benchmark_avg, down_model)

  # Train downstream models on aug. data
  if aug_model:
    print("- Training on Aug. Data -")
    curr_dataset = init_dataset
    augmentations = load_tatr_augmentation(aug_run, aug_model=aug_model, n_aug_year=100)
    TRADING_DAYS_PER_YEAR = 252
    n_aug_times = int(np.ceil(aug_size * TRADING_DAYS_PER_YEAR / (window_size + 1)))
    returns_models = {'armagarch', 'bootstrap', 'rcgan', 'timegan'}
    for i in range(n_aug_times):
      aug_timeseries = augmentations[i]
      if aug_model not in returns_models:
        aug_timeseries = convert_datatype(aug_timeseries, datatype=datatype)
      else:
        if datatype == 'returns':
          aug_timeseries = aug_timeseries
      aug_dataset = Timeseries2Dataset_Downstream(aug_timeseries, window_size=window_size, predstep=predstep, stride=1)
      curr_dataset = concat_datasets_downstream(curr_dataset, aug_dataset)
    model_aug = separate_train_downstream_model(
      curr_dataset, ahead, datatype=datatype, modelname=down_model, 
      adjust_lr=adjust_lr,
      )
    n_params_aug = count_params_downstream(model_aug, down_model)
    print(f"# of parameters in {down_model.upper()}: {n_params_aug}")
    _, y_pred_aug = prediction_downstream_model(model_aug, test_dataset, ahead, modelname=down_model)
    if print_info:
      print_demo_pred_info(y_true, y_pred_aug, y_benchmark_avg, down_model, aug_model)
  
  # Plot results
  if plot_pred:
    plt.figure(figsize=(10, 2))
    plt.plot(y_true[plot_scale[0]:plot_scale[1]], alpha=0.2, color='red', label='true')
    plt.plot(y_pred_init[plot_scale[0]:plot_scale[1]], color='green', label=f"{down_model}-init")
    if aug_model:
      plt.plot(y_pred_aug[plot_scale[0]:plot_scale[1]], color='blue', label=f"{down_model}-aug")
    plt.plot(y_benchmark_avg[plot_scale[0]:plot_scale[1]], color='orange', label='benchmark')
    plt.legend(loc='upper left')
    plt.show()


def print_demo_pred_info(y_true, y_pred, y_benchmark, downstream_model, aug_model=None):
  """ Print the info of demo prediction """
  r2_init = downstream_test_criterion(y_true, y_pred, y_benchmark=y_benchmark, criterion='r2')
  if aug_model:
    print(f"R^2 of y_pred by {downstream_model.upper()} with {aug_model.upper()} compared to y_true: {r2_init:.5f}")
  else:
    print(f"R^2 of y_pred by {downstream_model.upper()} compared to y_true: {r2_init:.5f}")



#########################################################
# Investigation of Overfitting in Downstream Prediction #
#########################################################

def investigate_downstream_nn_loss(
    datatype,
    down_model,
    aug_model=None,
    batch_size=None,
    lr=None,
    loss_fn=None,
    adjust_lr=False, 
    aug_size=100,
    aug_run=0,
    window_size=60,
    predstep=1,
    ahead=1,
    set_gpu=7,
    print_r2=False,
    ):
  """ Print the loss of the trained downstream model """
  batch_size, lr = get_downstream_nn_training_params(down_model) if batch_size is None else (batch_size, lr)
  print(f"-- {datatype.upper()} - t{predstep} | {down_model} | batch_size: {batch_size} lr: {lr:.0e} loss_fn: {loss_fn} adjust_lr: {adjust_lr} --")
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)

  # Load data
  init_timeseries = load_tatr_init()
  init_timeseries = convert_datatype(init_timeseries, datatype=datatype)
  init_dataset = init_tatr_set(init_timeseries, window_size=window_size, predstep=predstep, stride=1)
  test_timeseries = load_tatr_test()
  test_timeseries = convert_datatype(test_timeseries, datatype=datatype)
  test_dataset = init_tatr_set(test_timeseries, window_size=window_size, predstep=predstep, stride=1)
  
  # Set device
  if torch.cuda.is_available():
    torch.cuda.set_device(set_gpu)
  device = get_device(set_gpu=set_gpu)
  context_window = init_dataset.shape[1]
  
  # Train downstream models on init. data
  print("- Training on Init. Data -")
  model_init = get_downstream_nn_for_loss_investigation(
    down_model, datatype, loss_fn, 
    device=device, context_window=context_window, 
    )
  model_init, train_loss_init, test_loss_init = train_downstream_nn_for_loss_investigation(
    model_init, n_epochs=100, batch_size=batch_size, lr=lr, 
    train_dataset=init_dataset, test_dataset=test_dataset, 
    ahead=ahead, device=device, adjust_lr=adjust_lr)
  print(f"Final train loss of {down_model} - init: {train_loss_init[-1]:.7f}")
  print(f"Final test loss of {down_model} - init: {test_loss_init[-1]:.7f}")
  
  # Train downstream models on aug. data
  print("- Training on Aug. Data -")
  curr_dataset = init_dataset
  augmentations = load_tatr_augmentation(aug_run, aug_model=aug_model, n_aug_year=100)
  TRADING_DAYS_PER_YEAR = 252
  n_aug_times = int(np.ceil(aug_size * TRADING_DAYS_PER_YEAR / (window_size + 1)))
  returns_models = {'armagarch', 'bootstrap', 'rcgan', 'timegan'}
  for i in range(n_aug_times):
    aug_timeseries = augmentations[i]
    if aug_model not in returns_models:
      aug_timeseries = convert_datatype(aug_timeseries, datatype=datatype)
    else:
      if datatype == 'returns':
        aug_timeseries = aug_timeseries
    aug_dataset = Timeseries2Dataset_Downstream(aug_timeseries, window_size=window_size, predstep=predstep, stride=1)
    curr_dataset = concat_datasets_downstream(curr_dataset, aug_dataset)
  model_aug = get_downstream_nn_for_loss_investigation(
    down_model, datatype, loss_fn, 
    device=device, context_window=context_window, 
    )
  model_aug, train_loss_aug, test_loss_aug = train_downstream_nn_for_loss_investigation(
    model_aug, n_epochs=100, batch_size=batch_size, lr=lr,
    train_dataset=curr_dataset, test_dataset=test_dataset, 
    ahead=ahead, device=device, adjust_lr=adjust_lr)
  print(f"Training hyper-params. - batch_size: {batch_size}, lr: {lr:.0e}")
  print(f"Final train loss of {down_model} - aug: {train_loss_aug[-1]:.7f}")
  print(f"Final test loss of {down_model} - aug: {test_loss_aug[-1]:.7f}")
  
  # Print R-squared
  if print_r2:
    y_benchmark_avg = prediction_benchmark(test_dataset, ahead, benchmark='avg')
    y_true, y_pred_init = prediction_downstream_model(model_init, test_dataset, ahead, modelname=down_model)
    print_demo_pred_info(y_true, y_pred_init, y_benchmark_avg, down_model)
    _, y_pred_aug = prediction_downstream_model(model_aug, test_dataset, ahead, modelname=down_model)
    print_demo_pred_info(y_true, y_pred_aug, y_benchmark_avg, down_model, aug_model=aug_model)

  # Plot training and test losses of init. and aug.
  fig, axs = plt.subplots(1, 2, figsize=(10, 2))
  axs[0].plot(train_loss_init, label='train')
  axs[0].plot(test_loss_init, label='test')
  axs[0].set_title(f"{down_model} - init {model_init.loss_fn}")
  axs[0].legend()
  axs[1].plot(train_loss_aug, label='train')
  axs[1].plot(test_loss_aug, label='test')
  axs[1].set_title(f"{down_model} - aug {model_aug.loss_fn}")
  axs[1].legend()
  plt.show()


def get_downstream_nn_for_loss_investigation(modelname, datatype, loss_fn=None, ahead=1, device=None, **kwargs):
  """ Get the downstream NN model for investigation of overfitting """
  device = device if device is not None else get_device()
  if 'mlp' in modelname:
    mlp_n_layers, mlp_hidden_dim, mlp_reg = get_default_downstream_model_params(modelname) \
      if '-' not in modelname else get_investigate_downstream_model_params(modelname)
    if loss_fn is None:
      loss_fn = 'mae' if datatype == 'returns' else 'mse'
    model = MLPPredictor(
      input_dim=1, hidden_dim=mlp_hidden_dim, output_dim=ahead, 
      n_layers=mlp_n_layers, loss_fn=loss_fn, regularization=mlp_reg).to(device)
  
  elif 'itransformer' in modelname:
    itf_n_layers, itf_d_model, itf_d_ff, itf_n_heads = get_default_downstream_model_params(modelname) \
      if '-' not in modelname else get_investigate_downstream_model_params(modelname)
    if loss_fn is None:
      loss_fn = 'mae' if datatype == 'returns' else 'mse'
    model = iTransformerPredictor(
      e_layers=itf_n_layers, d_model=itf_d_model, d_ff=itf_d_ff, n_heads=itf_n_heads, 
      loss_fn=loss_fn).to(device)
  
  return model


def train_downstream_nn_for_loss_investigation(model, n_epochs, batch_size, lr, train_dataset, test_dataset, ahead=1, device=None, adjust_lr=False):
  """ Train the downstream NN model for investigation of overfitting """
  loss_fn = set_loss_fn_itransformer(model.loss_fn)
  if isinstance(model, MLPPredictor):
    regularization = model.regularization
    l1_weight = 1e-5 if regularization == 1 else 0
    l2_weight = 1e-5 if regularization == 2 else 0
  optimizer = optim.Adam(model.parameters(), lr=lr)
  dataset_size = len(train_dataset)
  num_batches = (dataset_size + batch_size - 1) // batch_size  # Ensure we cover the entire dataset
  train_loss = []
  test_loss = []
  train_steps = 1
  if adjust_lr:
    adjust_learning_rate(optimizer, train_steps)
  for epoch in range(n_epochs):
    # Training
    model.train()
    for i in range(num_batches):
      start_idx = i * batch_size
      end_idx = min(start_idx + batch_size, dataset_size)
      X = train_dataset[start_idx:end_idx, :-ahead].unsqueeze(-1).to(device)
      y = train_dataset[start_idx:end_idx, -ahead:].to(device)
      y_ = model(X).squeeze(-1)
      loss = loss_fn(y_, y)
      if isinstance(model, MLPPredictor):
        # L1 regularization
        l1_penalty = sum(p.abs().sum() for p in model.parameters()) if l1_weight !=0 else 0
        # L2 regularization
        l2_penalty = sum(p.pow(2.0).sum() for p in model.parameters()) if l2_weight !=0 else 0
        loss += l1_weight * l1_penalty + l2_weight * l2_penalty
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_steps += 1
    if adjust_lr:
      adjust_learning_rate(optimizer, train_steps)
    # Store avg. training loss and test loss at each epoch
    model.eval()
    with torch.no_grad():
      # Training loss
      # train_loss.append(loss.item())
      X_train = train_dataset[2*len(test_dataset):3*len(test_dataset), :-ahead].unsqueeze(-1).to(device)
      y_train = train_dataset[2*len(test_dataset):3*len(test_dataset), -ahead:].to(device)
      y_train_ = model(X_train).squeeze(-1)
      train_loss.append(loss_fn(y_train_, y_train).item())
      # Test loss
      X_test = test_dataset[:, :-ahead].unsqueeze(-1).to(device)
      y_test = test_dataset[:, -ahead:].to(device)
      y_test_ = model(X_test).squeeze(-1)
      test_loss.append(loss_fn(y_test_, y_test).item())
  return model, train_loss, test_loss









