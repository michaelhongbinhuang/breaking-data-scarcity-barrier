import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from experiments.predictors.layer_transformer_Embed import DataEmbedding_inverted
from experiments.predictors.layer_transformer_EncDec import Encoder, EncoderLayer
from experiments.predictors.layer_transformer_SelfAttention import FullAttention, AttentionLayer
from experiments.predictors.utils_predictors import adjust_learning_rate
from experiments.utils_tatr_extension import get_pretrain_downstream_model_params
from experiments.pretrain.lora import apply_lora_to_pretrain
from utils.sys_config import get_device



#################################
# Downstream iTransformer model #
#################################

class iTransformerPredictor(nn.Module):
  """ Basic iTransformer. Code from https://github.com/thuml/iTransformer, with minor modifications """
  def __init__(
      self,
      output_dim=1,
      seq_len=60,
      d_model=256, n_heads=8,
      e_layers=2, factor=3, d_ff=256,
      use_norm=True, loss_fn='mae',
      ):
    super().__init__()
    self.output_dim = output_dim
    self.seq_len = seq_len
    self.e_layers = e_layers
    self.d_model = d_model
    self.d_ff = d_ff
    self.n_heads = n_heads
    self.use_norm = use_norm
    self.loss_fn = loss_fn.lower()
    # Embedding
    self.enc_embedding = DataEmbedding_inverted(seq_len, d_model)
    # Encoder-only architecture
    self.encoder = Encoder(
      [
        EncoderLayer(
          AttentionLayer(
            FullAttention(False, factor), 
            d_model, n_heads),
          d_model,
          d_ff,
        ) for l in range(e_layers)
      ],
      norm_layer=torch.nn.LayerNorm(d_model)
    )
    self.projector = nn.Linear(d_model, output_dim, bias=True)

  def forecast(self, x):
    if self.use_norm:
      # Normalization from Non-stationary Transformer
      means = x.mean(1, keepdim=True).detach()
      x = x - means
      stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
      x /= stdev

    _, _, N = x.shape # B L N
    # B: batch_size;  E: d_model; 
    # L: seq_len;     S: pred_len;
    # N: number of variate (tokens), can also includes covariates

    # Embedding
    # B L N -> B N E        (B L N -> B L E in the vanilla Transformer)
    enc_out = self.enc_embedding(x) # covariates (e.g timestamp) can be also embedded as tokens
    
    # B N E -> B N E        (B L E -> B L E in the vanilla Transformer)
    # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
    enc_out, attns = self.encoder(enc_out, attn_mask=None)

    # B N E -> B N S -> B S N 
    dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

    if self.use_norm:
      # De-Normalization from Non-stationary Transformer
      dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_dim, 1))
      dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.output_dim, 1))
    return dec_out

  def forward(self, x):
    dec_out = self.forecast(x)
    return dec_out[:, -self.output_dim:, :]  # [B, L, D]


class iTransformerPredictor_with_pretrain_self(nn.Module):
  """ iTransformer based on the self-trained pre-trained model """
  def __init__(
      self,
      input_dim=1, output_dim=1, 
      seq_len=60, pred_len=1, 
      pretrain_model_name='itransformer-selftrain-ETT', 
      lora_rank=0, 
      loss_fn='mae', 
      ):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.channels = input_dim
    self.input_length = seq_len
    self.output_length = pred_len
    self.pretrain_model_name = '-'.join(pretrain_model_name.split('-')[:3])
    self.loss_fn = loss_fn.lower()
    self.lora_rank = lora_rank

    # Pretrain
    self.pretrain = self.load_pretrain_model()
    self.pretrain_channels = self.pretrain.input_dim
    self.pretrain_input_length = self.pretrain.seq_len
    self.pretrain_output_length = self.pretrain.pred_len
    for param in self.pretrain.parameters():
      param.requires_grad = False

    # LoRA
    if self.lora_rank:
      self.pretrain = apply_lora_to_pretrain(self.pretrain, rank=self.lora_rank, self_train=True)
      for layer in self.pretrain.encoder.attn_layers:
        layer.attention.query_projection.lora_A.requires_grad = True  # Train LoRA A
        layer.attention.query_projection.lora_B.requires_grad = True  # Train LoRA B
        layer.attention.value_projection.lora_A.requires_grad = True
        layer.attention.value_projection.lora_B.requires_grad = True
    
    # Projection heads
    self.input_proj = nn.Linear(self.channels, self.pretrain_channels)
    self.temporal_proj = nn.Linear(self.input_length, self.pretrain_input_length)
    self.channel_reduce = nn.Linear(self.pretrain_channels, self.channels)
    self.temporal_reduce = nn.Linear(self.pretrain_output_length, self.output_length)
    self.backbone = nn.Sequential(
      self.input_proj, self.temporal_proj, 
      self.pretrain, 
      self.channel_reduce, self.temporal_reduce,
      )

  def load_pretrain_model(self):
    pretrain_modelname, pretrain_dataname = self.pretrain_model_name.split('-')[0], self.pretrain_model_name.split('-')[-1]
    pretrain_model_params = get_pretrain_downstream_model_params(pretrain_dataname, pretrain_modelname)
    pretrain_model = iTransformerPredictor_pretrain(
      input_dim=pretrain_model_params['n_channels'], 
      output_dim=pretrain_model_params['n_channels'], 
      seq_len=pretrain_model_params['seq_len'], 
      pred_len=pretrain_model_params['pred_len'], 
      d_model=pretrain_model_params['d_model'], 
      n_heads=pretrain_model_params['n_heads'], 
      e_layers=pretrain_model_params['n_layers'], 
      d_ff=pretrain_model_params['d_ff'], 
    )
    pretrain_model_path = "./experiments/pretrain/trained_models/"
    pretrain_model_name = f"{pretrain_modelname}_{pretrain_dataname}.pth"
    curr_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    pretrain_model.load_state_dict(torch.load(pretrain_model_path + pretrain_model_name, map_location=curr_device))
    return pretrain_model#.eval()

  def forward(self, x):
    x = self.input_proj(x)                  # [B, input_length, pretrain_channels] for linear input_proj
    x = x.permute(0, 2, 1)                  # [B, pretrain_channels, input_length] if using linear input_proj
    x = self.temporal_proj(x)               # [B, pretrain_channels, pretrain_input_length]
    x = x.permute(0, 2, 1)                  # [B, pretrain_input_length, pretrain_channels] (matches pre-trained input shape)
    with torch.no_grad():
      x = self.pretrain(x)                  # [B, pretrain_input_length, pretrain_channels] (pre-trained output shape)
    x = self.channel_reduce(x)              # [B, pretrain_output_length, channels]
    x = x.permute(0, 2, 1)                  # [B, channels, pretrain_output_length]
    x = self.temporal_reduce(x)             # [B, channels, output_length]
    return x


class iTransformerPredictor_pretrain(nn.Module):
  """ Self-trained pre-trained iTransformer """
  def __init__(
      self,
      input_dim=1, output_dim=1,
      seq_len=96, pred_len=96, 
      d_model=256, n_heads=8, 
      e_layers=2, factor=3, d_ff=256, 
      use_norm=True, loss_fn='mse',
      ):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.e_layers = e_layers
    self.d_model = d_model
    self.d_ff = d_ff
    self.n_heads = n_heads
    self.use_norm = use_norm
    self.loss_fn = loss_fn.lower()
    # Embedding
    self.enc_embedding = DataEmbedding_inverted(seq_len, d_model)
    # Encoder-only architecture
    self.encoder = Encoder(
      [
        EncoderLayer(
          AttentionLayer(
            FullAttention(False, factor), 
            d_model, n_heads),
          d_model,
          d_ff,
        ) for l in range(e_layers)
      ],
      norm_layer=torch.nn.LayerNorm(d_model)
    )
    self.projector = nn.Linear(d_model, pred_len, bias=True)

  def forecast(self, x):
    if self.use_norm:
      means = x.mean(1, keepdim=True).detach()
      x = x - means
      stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
      x /= stdev
    _, _, N = x.shape
    enc_out = self.enc_embedding(x)
    enc_out, attns = self.encoder(enc_out, attn_mask=None)
    dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

    if self.use_norm:
      dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
      dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    return dec_out

  def forward(self, x):
    dec_out = self.forecast(x)
    return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class iTransformerPredictor_pretrain_multitask(nn.Module):
  """ Self-trained pre-trained iTransformer """
  def __init__(
      self,
      input_dim=1, output_dim=1,
      seq_len=96, pred_len=96, 
      d_model=256, n_heads=8, 
      e_layers=2, factor=3, d_ff=256, 
      pred_len_multitask=1, 
      use_norm=True, loss_fn='mse',
      ):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.pred_len_multitask = pred_len_multitask
    self.e_layers = e_layers
    self.d_model = d_model
    self.d_ff = d_ff
    self.n_heads = n_heads
    self.use_norm = use_norm
    self.loss_fn = loss_fn.lower()
    # Embedding
    self.enc_embedding = DataEmbedding_inverted(seq_len, d_model)
    # Encoder-only architecture
    self.encoder = Encoder(
      [
        EncoderLayer(
          AttentionLayer(
            FullAttention(False, factor), 
            d_model, n_heads),
          d_model,
          d_ff,
        ) for l in range(e_layers)
      ],
      norm_layer=torch.nn.LayerNorm(d_model)
    )
    self.projector = nn.Linear(d_model, pred_len, bias=True)
    self.projector = nn.Linear(d_model, pred_len_multitask, bias=True)

  def forecast(self, x):
    if self.use_norm:
      means = x.mean(1, keepdim=True).detach()
      x = x - means
      stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
      x /= stdev
    _, _, N = x.shape
    enc_out = self.enc_embedding(x)
    enc_out, attns = self.encoder(enc_out, attn_mask=None)
    dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
    dec_out_multitask = self.projector_multitask(enc_out).permute(0, 2, 1)[:, :, :N]

    if self.use_norm:
      dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
      dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    return dec_out, dec_out_multitask

  def forward(self, x):
    dec_out, dec_out_multitask = self.forecast(x)
    return dec_out[:, -self.pred_len:, :], dec_out_multitask[:, -self.pred_len_multitask:, :]



#############################################
# Separately train the downstream predictor #
#############################################

def separate_train_itransformer_predictor(
    dataset,
    datatype='returns',
    output_dim=1,
    n_layers=2,
    d_model=64,
    d_ff=512,
    n_heads=4,
    loss_fn=None,
    n_epochs=100,
    batch_size=512,
    lr=3e-5, 
    adjust_lr=False, 
    pretrain=False, 
    pretrain_self=None, 
    lora_rank=0, 
    ):
  """ Separately train the iTransformer predictor on the given dataset """
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  torch.cuda.empty_cache()
  device = get_device(train_nn=True)
  if loss_fn is None:
    loss_fn = 'mae' if datatype == 'returns' else 'mse'
  
  # Tune a self-trained pre-trained model
  if pretrain_self is not None:
    model = iTransformerPredictor_with_pretrain_self(
      pretrain_model_name=pretrain_self, lora_rank=lora_rank, 
      loss_fn=loss_fn).to(device)
    # lr = 1e-6 if datatype == 'returns' else 5e-5
  
  # Tune an existing pre-trained model (Not Found)
  elif pretrain:
    pass

  # Train from scratch
  else:
    model = iTransformerPredictor(
      e_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, 
      loss_fn=loss_fn).to(device)
  # print(f"Model params. {count_params_itransformer(model)}")
  loss_fn = set_loss_fn_itransformer(loss_fn)
  optimizer = optim.Adam(model.parameters(), lr=lr)
  dataset_size = len(dataset)
  num_batches = (dataset_size + batch_size - 1) // batch_size  # Ensure we cover the entire dataset
  train_steps = 1
  if adjust_lr:
    adjust_learning_rate(optimizer, train_steps)
  for epoch in range(n_epochs):
    model.train()
    for i in range(num_batches):
      start_idx = i * batch_size
      end_idx = min(start_idx + batch_size, dataset_size)
      X = dataset[start_idx:end_idx, :-output_dim].unsqueeze(-1).to(device)
      y = dataset[start_idx:end_idx, -output_dim:].to(device)
      y_ = model(X).squeeze(-1)
      # print(y_.shape, y.shape)
      loss = loss_fn(y_, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_steps += 1
    if adjust_lr:
      adjust_learning_rate(optimizer, train_steps)
  return model


def set_loss_fn_itransformer(criterion='mae'):
  """ Set the loss function for the downstream iTransformer model """
  criterion = criterion.lower()
  if criterion == 'mse':
    return nn.MSELoss()
  elif criterion == 'mae':
    return nn.L1Loss()


def count_params_itransformer(model):
  """ Count the number of parameters in the downstream iTransformer model """
  return sum(p.numel() for p in model.parameters())



##############################################
# Test the downstream predictor on real data #
##############################################

def prediction_itransformer(model, dataset, scaler=None, plot_fig=False):
  """ Test the trained downstream model on real test dataset """
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)
  device = get_device(train_nn=True)
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









