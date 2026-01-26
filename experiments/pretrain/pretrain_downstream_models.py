import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch

from experiments.pretrain.utils_pretrain import *
from experiments.utils_tatr_extension import get_pretrain_downstream_model_params



############################
# Train Pre-trained Models #
############################

class PretrainLearner:
  def __init__(
      self, model, dataset, 
      modelname=None, dataname=None, 
      batch_size=32, lr=0.0001, epochs=100, 
      target_window=96, 
      adjust_lr=True, adjust_factor=0.001, 
      device=torch.device("cpu"), 
      store_model=False, 
      ):
    self.device = device
    self.model = model.to(device)
    self.batch_size = batch_size
    train_dataset, valid_dataset, test_dataset = dataset
    self.train_datalen = len(train_dataset)
    self.valid_datalen = len(valid_dataset)
    self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    self.valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    self.lr=lr
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.loss = torch.nn.MSELoss()
    self.epochs = epochs
    self.target_window=target_window
    self.best_weight = self.model.state_dict()
    self.adjust_lr = adjust_lr
    self.adjust_factor = adjust_factor
    self.store_model = store_model
    if store_model:
      self.pretrain_model_path = "./experiments/pretrain/trained_models/"
      self.stored_model_name = f"{modelname}_{dataname}"
  
  def adjust_learning_rate(self, steps, warmup_step=300, printout=False):
    if steps**(-0.5) < steps * (warmup_step**-1.5):
      lr_adjust = (16**-0.5) * (steps**-0.5) * self.adjust_factor
    else:
      lr_adjust = (16**-0.5) * (steps * (warmup_step**-1.5)) * self.adjust_factor

    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr_adjust
    if printout: 
      print(f"Updating learning rate to {lr_adjust}")
    return 

  def train(self):
    print("-- Training Phase --")
    best_valid_loss = np.inf
    train_history = []
    valid_history = []
    train_steps = 1
    if self.adjust_lr:
      self.adjust_learning_rate(train_steps)
    for epoch in range(self.epochs):
      # Train
      self.model.train()
      iter_count = 0
      total_loss = 0
      for train_x, train_y in self.train_dataloader:
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        pred_y = self.model(train_x)
        # print(pred_y.shape, train_y.shape)
        loss = self.loss(pred_y, train_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
        iter_count += 1
        train_steps += 1
      if self.adjust_lr:
        self.adjust_learning_rate(train_steps)

      # Valid
      self.model.eval()
      valid_iter_count = 0
      valid_total_loss = 0
      with torch.no_grad():
        for valid_x, valid_y in self.valid_dataloader:
          valid_x = valid_x.to(self.device)
          valid_y = valid_y.to(self.device)
          pred_y = self.model(valid_x)
          loss = self.loss(pred_y, valid_y)
          valid_total_loss += loss.item()
          valid_iter_count += 1
      
      total_loss /= iter_count
      valid_total_loss /= valid_iter_count
      log_string = f"Epoch: {epoch:3d} | MSE loss: {total_loss:.4f}, MSE valid loss: {valid_total_loss:.4f}"
      if best_valid_loss >= valid_total_loss:
        self.best_weight = self.model.state_dict()
        best_valid_loss = valid_total_loss
        log_string += " -> Best score! Weights of the model are updated!"
        if self.store_model:
          torch.save(self.model.state_dict(), self.pretrain_model_path + self.stored_model_name +'.pth')
          torch.save(self.model, self.pretrain_model_path + self.stored_model_name +'.pt')
          log_string += " (stored)"
      train_history.append(total_loss)
      valid_history.append(valid_total_loss)
      print(log_string)
    return train_history, valid_history

  def test(self):
    print("-- Testing Phase --")
    self.model.load_state_dict(self.best_weight)
    self.model.eval()
    iter_count = 0
    total_loss = 0
    with torch.no_grad():
      for test_x, test_y in self.test_dataloader:
        test_x = test_x.to(self.device)
        test_y = test_y.to(self.device)
        pred_y = self.model(test_x)
        loss = self.loss(pred_y, test_y)
        total_loss += loss.item()
        iter_count += 1
    total_loss /= iter_count
    print(f"MSE test loss: {total_loss:.4f}")


def train_pretrain_models(
  dataname, 
  modelname, 
  model_params=None, 
  lr=1e-4, 
  epochs=100, 
  set_gpu=0, 
  store_model=False, 
  ):
  # Init.
  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  device = torch.device(f"cuda:{set_gpu}") if torch.cuda.is_available() else torch.device("cpu")
  model_params = get_pretrain_downstream_model_params(dataname, modelname) if model_params is None else model_params
  data_seq_len = model_params['seq_len'] if model_params is not None else 96
  data_pred_len = model_params['pred_len'] if model_params is not None else 96
  pretrain_train_dataset = PretrainDataset(dataname=dataname, mode='train', seq_len=data_seq_len, pred_len=data_pred_len)
  pretrain_valid_dataset = PretrainDataset(dataname=dataname, mode='val', seq_len=data_seq_len, pred_len=data_pred_len)
  pretrain_test_dataset = PretrainDataset(dataname=dataname, mode='test', seq_len=data_seq_len, pred_len=data_pred_len)
  pretrain_dataset = (pretrain_train_dataset, pretrain_valid_dataset, pretrain_test_dataset)
  pretrain_model = init_pretrain_model(modelname, model_params)

  # Train
  pretrain_learner = PretrainLearner(
    model=pretrain_model, dataset=pretrain_dataset, 
    modelname=modelname, dataname=dataname, 
    lr=lr, epochs=epochs, 
    adjust_lr=True, adjust_factor=0.001, 
    device=device, 
    store_model=store_model, 
    )
  train_history, valid_history = pretrain_learner.train()

  # Plot training progress
  fig, ax = plt.subplots(1, 1, figsize=(5, 2))
  ax.plot(np.arange(1, pretrain_learner.epochs+1), train_history, label="Pretrain Train")
  ax.plot(np.arange(1, pretrain_learner.epochs+1), valid_history, label="Pretrain Valid")
  ax.set_xlabel("Epochs")
  ax.set_ylabel("MSE Error")
  ax.legend(loc="upper right")
  plt.show()

  # Test
  pretrain_learner.test()
  return pretrain_learner


def load_pretrain_model(modelname, dataname, set_gpu):
  """ Load self-trained pretrain models """
  model_params = get_pretrain_downstream_model_params(dataname, modelname)
  model = init_pretrain_model(modelname, model_params)
  pretrain_model_path = "./experiments/pretrain/trained_models/"
  pretrain_model_name = f"{modelname}_{dataname}.pth"
  device = torch.device(f"cuda:{set_gpu}") if torch.cuda.is_available() else torch.device("cpu")
  model.load_state_dict(torch.load(pretrain_model_path + pretrain_model_name, map_location=device))
  return model.eval()



#############################
# Load Pre-training Dataset #
#############################

class PretrainDataset(torch.utils.data.Dataset):
  """ Build the pre-trained dataset in torch tensors """
  def __init__(self, dataname, mode="train", seq_len=96, pred_len=96, scale=True):
    super().__init__()
    self.dataname = dataname
    self.mode = mode
    assert mode in ['train', 'test', 'val']
    type_map = {'train': 0, 'val': 1, 'test': 2}
    self.set_type = type_map[mode]
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.scale = scale
    self.__read_data__()

  def __read_data__(self):
    self.path = self.get_path()
    df = pd.read_csv(self.path)
    x_y = df.iloc[:,1:]
    time_stamp = df.iloc[:,0]
    if "ETT" in self.path:
      border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
      border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    else:
      n_total = len(df)
      n_train = int(n_total * 0.7)
      n_test = int(n_total * 0.2)
      n_valid = n_total - n_train - n_test
      border1s = [0, n_train - self.seq_len, n_total - n_test - self.seq_len]
      border2s = [n_train, n_train + n_valid, n_total]
    border1 = border1s[self.set_type]
    border2 = border2s[self.set_type]
    if self.scale:
      train_x_y = x_y.iloc[border1s[0]: border2s[0]]
      self.scaler = StandardScaler()
      self.scaler.fit(train_x_y.to_numpy(dtype=np.float32))
      x_y = self.scaler.transform(x_y.to_numpy(dtype=np.float32))
    else:
      x_y = x_y.to_numpy(dtype=np.float32)
    time_stamp = time_stamp.to_numpy()   
    self.data_x = x_y[border1: border2, :]
    self.data_y = x_y[border1: border2, :]
    self.data_stamp = time_stamp[border1: border2]
    
  def __getitem__(self, index):
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end
    r_end = r_begin + self.pred_len
    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]
    return seq_x, seq_y

  def __len__(self):
    return len(self.data_x) - self.seq_len - self.pred_len + 1
  
  def get_path(self):
    filename = f"{self.dataname}.csv" if self.dataname != 'ETT' else f"{self.dataname}h1.csv"
    path = f"./experiments/pretrain/data/{self.dataname}/{filename}"
    return path

  def get_shape(self):
    print(f"Dataset {self.dataname} shape | X: {self.data_x.shape}, y: {self.data_y.shape}")

  def show_samples(self):
    print(f"X: {self.data_x[0]}")
    print(f"y: {self.data_y[0]}")

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)









