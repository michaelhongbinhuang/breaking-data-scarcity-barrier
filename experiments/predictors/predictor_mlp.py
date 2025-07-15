import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.sys_config import get_device



##############################
# Downstream MLP-based model #
##############################

class MLPPredictor(nn.Module):
  """ MLP-based predictor with output shaped correctly for one-dimensional prediction """
  def __init__(self, input_dim, hidden_dim, output_dim, n_layers, loss_fn, regularization=0):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.dropout = 0.1
    self.n_layers = n_layers
    self.loss_fn = loss_fn
    self.regularization = regularization
    # Build the network layers
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU(inplace=False))
    layers.append(nn.Dropout(self.dropout))
    for i in range(1, n_layers):
      if (i + 1) % 2 == 0 and i + 1 < n_layers:
        layers.append(SkipConnection(hidden_dim, hidden_dim))
      else:
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.Dropout(self.dropout))
    self.network = nn.Sequential(*layers)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    _, seq_len, _ = x.shape
    x = x.reshape(-1, self.input_dim)            # [batch_size * seq_len, input_dim]
    x = self.network(x)
    x = x.reshape(-1, seq_len, self.hidden_dim)  # [batch_size, seq_len, hidden_dim]
    x = x.mean(dim=1)
    y = self.fc(x)
    return y


class SkipConnection(nn.Module):
  """ Skip Connection Block for intermediate layers in the MLP """
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dropout_rate = 0.1
    self.fc1 = nn.Linear(input_dim, output_dim)
    self.relu1 = nn.ReLU(inplace=False)
    self.dropout1 = nn.Dropout(self.dropout_rate)
    self.fc2 = nn.Linear(output_dim, output_dim)
    self.relu2 = nn.ReLU(inplace=False)
    self.dropout2 = nn.Dropout(self.dropout_rate)

  def forward(self, x):
    identity = x
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.dropout1(out)
    out = self.fc2(out)
    out = self.relu2(out)
    out = self.dropout2(out)
    out = out + identity
    return out



#############################################
# Separately train the downstream predictor #
#############################################

def separate_train_mlp_predictor(
    dataset, 
    datatype, 
    output_dim=1, 
    input_dim=1, 
    n_layers=2, 
    hidden_dim=32, 
    loss_fn=None, 
    regularization=0, 
    n_epochs=100, 
    batch_size=1024, 
    lr=1e-3,
    ):
  """ Separately train the MLP-based predictor on the given dataset """
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  device = get_device(train_nn=False)
  if loss_fn is None:
    loss_fn = 'mae' if datatype == 'returns' else 'mse'
  model = MLPPredictor(input_dim, hidden_dim, output_dim, n_layers, loss_fn, regularization).to(device)
  loss_fn = set_loss_fn_mlp(loss_fn)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  l1_weight = 1e-5 if regularization == 1 else 0
  l2_weight = 1e-5 if regularization == 2 else 0
  dataset_size = len(dataset)
  num_batches = (dataset_size + batch_size - 1) // batch_size     # Ensure we cover the entire dataset
  for epoch in range(n_epochs):
    model.train()
    for i in range(num_batches):
      start_idx = i * batch_size
      end_idx = min(start_idx + batch_size, dataset_size)
      X = dataset[start_idx:end_idx, :-output_dim].unsqueeze(-1).to(device)
      y = dataset[start_idx:end_idx, -output_dim:].to(device)
      y_ = model(X)
      # print(y.shape, y_.shape)
      loss = loss_fn(y_, y)
      # L1 regularization
      l1_penalty = sum(p.abs().sum() for p in model.parameters()) if l1_weight !=0 else 0
      # L2 regularization
      l2_penalty = sum(p.pow(2.0).sum() for p in model.parameters()) if l2_weight !=0 else 0
      loss += l1_weight * l1_penalty + l2_weight * l2_penalty
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  return model


def set_loss_fn_mlp(criterion='mae'):
  """ Set the loss function for the downstream MLP-based model """
  criterion = criterion.lower()
  if criterion == 'mse':
    return nn.MSELoss()
  elif criterion == 'mae':
    return nn.L1Loss()
  else:
    raise NotImplementedError


def count_params_mlp(model):
  """ Count the number of parameters in the downstream MLP-based model """
  return sum(p.numel() for p in model.parameters())



##############################################
# Test the downstream predictor on real data #
##############################################

def prediction_mlp(model, dataset, scaler=None, plot_fig=False):
  """ Test the trained downstream model on real test dataset """
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  device = get_device(train_nn=False)
  model.to(device)
  model.eval()
  ahead = model.output_dim
  with torch.no_grad():
    X, y = dataset[:, :-ahead].unsqueeze(-1).to(device), dataset[:, -ahead:].to(device)
    y_ = model(X)
  y_test = y.detach().cpu().numpy()
  y_pred = y_.detach().cpu().numpy()
  if scaler is not None:
    y_test = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred)
  if plot_fig:
    plt.figure(figsize=(7, 2))
    plt.plot(y_test, color='red')
    plt.plot(y_pred, color='blue')
  return y_test.ravel(), y_pred.ravel()









