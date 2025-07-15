import torch
import torch.optim as optim
torch.set_default_dtype(torch.float)

from models.pattern_recognition_module import train_recognition_module
from models.pattern_generation_module import get_dataloader_generation, train_generation_module, build_pgm
from models.pattern_evolution_module_crf import train_evolution_module_crf
from models.model_params import *
from utils.load_data import load_historical_fts



def train_framework(dataname='sp500', store_model=True):
  """ Train the generative framework """
  fts = load_historical_fts(dataname=dataname).values
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  train_recognition(fts, store_model)
  dataloader_pgm = get_dataloader_generation(dataname=dataname)
  train_generation(dataloader_pgm, dataname=dataname, device=device, store_model=store_model)
  train_evolution(dataname=dataname, store_model=store_model)


def train_recognition(fts, dataname='sp500', store_model=True):
  """ Train the pattern recognition module """
  _ = train_recognition_module(
    fts, 
    dataname=dataname, 
    n_clusters=prm_params['k'], 
    l_min=prm_params['l_min'], l_max=prm_params['l_max'], 
    max_iters=prm_params['max_iters'], 
    init_strategy=prm_params['init_strategy'], 
    barycenter=prm_params['barycenter'], 
    plot_progress=False, 
    plot_loss=True, 
    store_res=store_model, 
    )


def train_generation(dataloader, dataname='sp500', device=None, store_model=True):
  """ Train the pattern generation module """
  pgm = build_pgm(device).to(device)
  n_epochs = pgm_params['n_epochs']
  lr = pgm_params['lr']
  loss_weights = pgm_params['loss_weights']
  pad_weight = pgm_params['pad_weight']
  scale_weight = pgm_params['scale_weight']
  optimizer = optim.Adam(pgm.parameters(), lr)
  train_generation_module(
    pgm, 
    dataloader, 
    n_epochs, 
    optimizer, 
    loss_weights, 
    pad_weight=pad_weight, 
    scale_weight=scale_weight, 
    condition=True, 
    device=device, 
    display=True, 
    store_model=store_model, 
    dataname=dataname, 
    )


def train_evolution(dataname='sp500', store_model=True):
  """ Train the pattern evolution module """
  pem = train_evolution_module_crf(
  algo='lbfgs', 
  c1=0.1, c2=0.001, 
  n_iter=1000, 
  store_model=store_model, 
  dataname=dataname, 
  )









