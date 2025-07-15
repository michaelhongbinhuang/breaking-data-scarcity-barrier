import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
torch.set_default_dtype(torch.float)
from tslearn.metrics import SoftDTWLossPyTorch as SoftDTWLoss
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from models.scaling_autoencoder import *
from models.pattern_conditioned_diffusion import *
from models.utils_generation import *
from models.model_params import pgm_params



#############################
# Pattern generation module #
#############################

class PatternGenerationModule(nn.Module):
  """
    Pattern generation module incorporating scaling AE and pattern-conditioned diffusion network (random noise schedule for diffusion network)
    Args
      sae: torch network, scaling AE network
      pcdm: torch network, pattern-conditioned diffusion network
      condition: bool, True for learning conditioned on patterns in pcdm
  """
  def __init__(self, sae, pcdm, condition=True, device=None):
    super().__init__()
    self.sae = sae.to(device)
    self.pcdm = pcdm.to(device)
    self.n_steps = pcdm.n_steps
    self.condition = condition
    self.device = device

  def forward(self, x, p, lengths):
    x, p = x.to(self.device), p.to(self.device)
    batch_size = x.shape[0]
    # SAE encoder
    z, (z_hidden, z_cell) = self.sae.encoder(x, lengths)
    # SAE decoder
    packed_z = pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
    x_ = self.sae.decoder(packed_z)
    x_ = x_.squeeze(-1)
    z_out = z.reshape(x_.size())

    # PCDM diffusion
    z = z.reshape(batch_size, 1, -1)
    p = p.unsqueeze(1)
    if self.condition:
      z = z - p
    epsilon = torch.randn_like(z).to(self.device)
    t = torch.randint(0, self.n_steps, (batch_size,)).to(self.device)
    z_noisy = self.pcdm.forward(z, t, epsilon, p).to(self.device)
    # PCDM denoising
    epsilon_theta = self.pcdm.backward(z_noisy, t, p).to(self.device)

    return x_, epsilon, epsilon_theta, z_out

  def generate(self, p, lengths):
    self.sae.eval()
    self.pcdm.eval()
    p = p.to(self.device)
    with torch.no_grad():
      p = p.unsqueeze(1)
      # Sample noise
      batch_size, n_channels, series_len = p.shape
      z_noisy = torch.randn_like(p).to(self.device)
      # PCDM denoising
      z_ = self.denoising_process(z_noisy, p,
                                  batch_size, n_channels, series_len).to(self.device)
      # SAE decoder
      if self.condition:
        z_ = z_ + p
      z_ = z_.reshape(batch_size, -1, 1)
      lengths = lengths.to('cpu', dtype=torch.int64)
      packed_z = pack_padded_sequence(z_, lengths, batch_first=True, enforce_sorted=False)
      x_ = self.sae.decoder(packed_z).to(self.device)
      x_ = x_.squeeze(-1)
    return x_, z_.reshape(x_.size())

  def denoising_process(self, z_noisy, p,
                        batch_size, n_channels, series_len):
    z_ = z_noisy
    for _, t in enumerate(list(range(self.n_steps))[::-1]):
      timestep = torch.full((batch_size,), t, dtype=torch.float32, device=self.device)
      e_theta = self.pcdm.backward(z_, timestep, p)
      alpha_t = self.pcdm.alphas[t]
      alpha_t_bar = self.pcdm.alpha_bars[t]
      z_ = (1 / alpha_t.sqrt()) * (z_ - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * e_theta)
      if t > 0:
        eta = torch.randn(batch_size, n_channels, series_len).to(self.device)
        beta_t = self.pcdm.betas[t]
        prev_alpha_t_bar = self.pcdm.alpha_bars[t-1] if t > 0 else self.pcdm.alphas[0]
        beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
        sigma_t = beta_tilda_t.sqrt()
        z_ += sigma_t * eta
    return z_



#######################################
# Train the pattern generation module #
#######################################

def train_generation_module(
    pgm, 
    dataloader, 
    n_epochs, 
    optimizer, 
    loss_weights, 
    pad_weight=1, 
    scale_weight=0, 
    condition=True, 
    device=None, 
    display=False, 
    store_model=False, 
    dataname='sp500', 
    ):
  model_path = f"trained_models/{dataname}/"
  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  pgm.to(device)
  pgm.train()
  mse = nn.MSELoss(reduction='sum')
  softdtw = SoftDTWLoss(gamma=.1)
  hist_loss = []
  hist_loss_sae = []
  hist_loss_pcdm = []
  best_loss = float("inf")
  for epoch in tqdm(range(n_epochs)):
    pgm.train()
    total_loss = 0.0
    total_loss_sae = 0.0
    total_loss_pcdm = 0.0
    n_batches = 0
    for x, p, lengths in dataloader:
      x, p = x.to(device), p.to(device)
      optimizer.zero_grad()
      batch_size = x.shape[0]

      x_0_, epsilon, epsilon_theta, z = pgm.forward(x, p, lengths)

      # Compute the loss and update the params.
      mask_data = (torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)).int().to(device)
      mask_pad = (torch.ones_like(mask_data) - mask_data).to(device)
      loss_sae = loss_weights[0] * (mse(x * mask_data, x_0_ * mask_data) + pad_weight * mse(x * mask_pad, x_0_ * mask_pad))
      if scale_weight != 0: # Optional: soft DTW between original data and latent representation, better interpretability
        loss_scale = scale_weight * softdtw(x.unsqueeze(1), z.unsqueeze(1)).sum()
        loss_sae += loss_scale
      loss_pcdm = loss_weights[-1] * mse(epsilon, epsilon_theta)
      loss = loss_sae + loss_pcdm
      total_loss += loss.item()
      total_loss_sae += loss_sae.item()
      total_loss_pcdm += loss_pcdm.item()
      loss.backward()
      optimizer.step()
      n_batches += 1

    epoch_loss = total_loss / n_batches
    hist_loss.append(epoch_loss)
    hist_loss_sae.append(total_loss_sae / n_batches)
    hist_loss_pcdm.append(total_loss_pcdm / n_batches)
    log_string = f"Epoch {epoch + 1:3d}/{n_epochs:3d} - loss: {epoch_loss:.5f}"
    log_string += f" | loss_sae: {total_loss_sae / n_batches:.5f} loss_pcdm: {total_loss_pcdm / n_batches:.5f}"

    # Store the optimal model
    if best_loss > epoch_loss:
      log_string += " --> Best model ever"
      best_loss = epoch_loss
      if store_model:
        modelname = f"pgm_{dataname}"
        torch.save(pgm.state_dict(), model_path + modelname + '.pth')
        torch.save(pgm, model_path + modelname + '.pt')
        log_string += " (stored)"
    print(log_string)
    if (epoch == 0 or (epoch+1) % 10 == 0) and display:
        display_progress(pgm, dataloader, condition, device=device)

  # Plot the loss
  plt.figure(figsize=(4, 1.5))
  plt.plot(hist_loss, color='red', label='l_total')
  plt.plot(hist_loss_sae, color='orange', label='l_sae')
  plt.plot(hist_loss_pcdm, color='blue', label='l_pcdm')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(loc='upper right')


def display_progress(pgm, dataloader, condition=True, device=None):
  """ Display the training progess of pattern generation module """
  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  pgm.eval()
  with torch.no_grad():
    for x, p, lengths in dataloader:
      x, p = x.to(device), p.to(device)
      batch_size, series_len = p.shape[0], p.shape[-1]
      # Reconstruct original segments
      x_decode, _, _, z_decode = pgm.forward(x, p, lengths)

      # Create new segments from noise
      x_, z_ = pgm.generate(p, lengths)

      # Plot decoded and generated samples
      if torch.cuda.is_available():
        x, x_decode, z_decode, x_, z_, p = cuda2cpu([x, x_decode, z_decode, x_, z_, p])
      n_rows = 9
      n_cols = batch_size
      fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.25, n_rows * 0.7))
      for i in range(batch_size):
        axs[0, i].plot(x[i][:lengths[i]], color='red')
        axs[1, i].plot(x[i] - p[i], linestyle=':', color='black')
        axs[2, i].plot(x_decode[i][:lengths[i]], linestyle='-.', color='red')
        axs[3, i].plot(x_decode[i] - p[i], linestyle=':', color='black')
        axs[4, i].plot(z_decode[i], linestyle='--', color='purple')
        axs[5, i].plot(x_[i][:lengths[i]], color='blue')
        axs[6, i].plot(x_[i] - p[i], linestyle=':', color='black')
        axs[7, i].plot(z_[i], linestyle='--', color='purple')
        axs[8, i].plot(p[i], color='gray')
        for j in range(n_rows):
          axs[j, i].tick_params(axis='x', labelsize=7)
          axs[j, i].tick_params(axis='y', labelsize=7)
      ylabels = ['x', 'x-p', 'x\'', 'x\'-p', 'z', 'x_', 'x_-p', 'z_', 'p']
      for j in range(n_rows):
        axs[j, 0].set_ylabel(ylabels[j])
      plt.subplots_adjust(wspace=0.5, hspace=0.5)
      plt.show()
      break



##############################################
# Load the trained pattern generation module #
##############################################

def build_sae(device):
  """ Instantiate the scaling AE """
  input_dim = pgm_params['sae_input_dim']
  hidden_dim = pgm_params['sae_hidden_dim']
  output_dim = pgm_params['sae_output_dim']
  custom_pad_length = pgm_params['sae_custom_pad_length']
  sae = ScalingAE(input_dim, hidden_dim, output_dim, custom_pad_length, device)
  return sae

def build_pcdm(device):
  """ Instantiate the pattern-conditioned diffusion network """
  n_steps = pgm_params['pcdm_n_steps']
  series_length = pgm_params['pcdm_series_length']
  channels = pgm_params['pcdm_channels']
  latent_dim = pgm_params['pcdm_latent_dim']
  time_embed_dim= pgm_params['pcdm_time_embed_dim']
  time_hidden_dim = pgm_params['pcdm_time_hidden_dim']
  min_beta, max_beta = pgm_params['pcdm_min_beta'], pgm_params['pcdm_max_beta']
  tcn = TCN2(
    n_steps, series_length, channels, 
    input_dim=latent_dim, 
    time_embed_dim=time_embed_dim, time_hidden_dim=time_hidden_dim, 
    device=device, 
    )
  pcdm = PCDM(
    tcn, 
    n_steps=n_steps, 
    min_beta=min_beta, max_beta=max_beta, 
    device=device,
    )
  return pcdm

def build_pgm(device):
  """ Instantiate the pattern generation module """
  sae = build_sae(device).to(device)
  pcdm = build_pcdm(device).to(device)
  pgm = PatternGenerationModule(sae, pcdm, condition=True, device=device).to(device)
  return pgm

def load_pattern_generation_module(dataname='sp500', state_dict=True, device=None):
  """ Load the trained pattern generation module with default (or input) hyper-parameters """
  model_path =f"trained_models/{dataname}/"
  modelname = f"pgm_{dataname}.pth" if state_dict else f"pgm_{dataname}.pt"
  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  pgm = build_pgm(device)
  pgm.load_state_dict(torch.load(model_path + modelname, map_location=device, weights_only=True))
  pgm.eval()
  return pgm









