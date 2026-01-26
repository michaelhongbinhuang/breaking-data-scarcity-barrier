import numpy as np
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
  def __init__(self, original_layer, rank=8):
    super().__init__()
    self.original_layer = original_layer  # Frozen pre-trained layer
    self.rank = rank

    # Freeze original weights
    for param in self.original_layer.parameters():
      param.requires_grad = False
    
    # Add low-rank adapters
    d_in, d_out = original_layer.weight.shape
    self.lora_A = nn.Parameter(torch.randn(rank, d_in))   # LoRA matrix A: from d_in to rank (reducing dimension)
    self.lora_B = nn.Parameter(torch.randn(d_out, rank))  # LoRA matrix B: from rank to d_out (expanding dimension)
    nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
    nn.init.kaiming_uniform_(self.lora_B, a=np.sqrt(5))

  def forward(self, x):
    original_output = self.original_layer(x)    # Original output: Wx
    lora_output = x @ self.lora_A.T             # [batch_size * sequence_length, rank]
    lora_output = lora_output @ self.lora_B.T   # [batch_size * sequence_length, d_out] LoRA output: BAx
    return original_output + lora_output


def apply_lora_to_pretrain(pretrain_model, rank=8, self_train=False, is_patchtst=False):
  """ Apply LoRA to the given pre-trained model """
  if self_train:
   # Iterate through transformer encoder layers and Replace query/value projections with LoRA
   if is_patchtst:
     for layer in pretrain_model.backbone.encoder.layers:
      layer.self_attn.W_Q = LoRALayer(layer.self_attn.W_Q, rank)
      layer.self_attn.W_V = LoRALayer(layer.self_attn.W_V, rank)
   else:
    for layer in pretrain_model.encoder.attn_layers:
      layer.attention.query_projection = LoRALayer(layer.attention.query_projection, rank)
      layer.attention.value_projection = LoRALayer(layer.attention.value_projection, rank)
  else:
    # Iterate through transformer encoder layers and Replace query/value projections with LoRA
    for layer in pretrain_model.model.encoder.layers:
      layer.self_attn.q_proj = LoRALayer(layer.self_attn.q_proj, rank)
      layer.self_attn.v_proj = LoRALayer(layer.self_attn.v_proj, rank)
  return pretrain_model









