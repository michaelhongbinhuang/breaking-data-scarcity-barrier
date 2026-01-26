import torch
import torch.optim as optim
torch.set_default_dtype(torch.float)

from models.pattern_recognition_module import train_recognition_module
from models.pattern_generation_module import get_dataloader_generation, train_generation_module, build_pgm
from models.pattern_evolution_module_crf import train_evolution_module_crf
from models.model_params import prm_params, pgm_params
from utils.load_data import load_hist_fts



def train_framework(dataname='sp500', store_model=True, set_gpu=0):
  """ Train the generative framework """
  hist_fts = load_hist_fts(dataname=dataname)
  device = torch.device(f"cuda:{set_gpu}" if torch.cuda.is_available() else "cpu")
  train_recognition(fts=hist_fts, dataname=dataname, store_model=store_model)
  dataloader_pgm = get_dataloader_generation(dataname=dataname)
  train_generation(dataloader=dataloader_pgm, dataname=dataname, device=device, store_model=store_model)
  train_evolution(dataname=dataname, store_model=store_model)


def train_recognition(fts, dataname='sp500', store_model=True, plot_progress=False, plot_loss=False):
  """ Train the pattern recognition module """
  train_recognition_module(
    fts, 
    dataname=dataname, 
    n_clusters=prm_params['k'], 
    l_min=prm_params['l_min'], l_max=prm_params['l_max'], 
    max_iters=prm_params['max_iters'], 
    init_strategy=prm_params['init_strategy'], 
    barycenter=prm_params['barycenter'], 
    center_length=prm_params['center_length'], 
    plot_progress=plot_progress, 
    plot_loss=plot_loss, 
    store_res=store_model, 
    )


def train_generation(dataloader, dataname='sp500', device=None, store_model=True, plot_progress=False):
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
    display=plot_progress, 
    store_model=store_model, 
    dataname=dataname, 
    )


def train_evolution(dataname='sp500', store_model=True):
  """ Train the pattern evolution module """
  train_evolution_module_crf(
    algo='lbfgs', 
    c1=0.1, c2=0.001, 
    n_iter=1000, 
    store_model=store_model, 
    dataname=dataname, 
    )


def train_generative_model_demo():
  train_framework(dataname='sp500', store_model=True)
  print("-- Training of the generative model completed")









