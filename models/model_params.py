prm_params = {
  'k': 14,
  'l_min': 10,
  'l_max': 21,
  'barycenter': 'dba',
  'init_strategy': 'kmeans++'
}


pgm_params = {
  'sae_input_dim': 1,
  'sae_hidden_dim': 1,
  'sae_output_dim': 1,
  'sae_custom_pad_length': prm_params['l_max'],
  'pcdm_n_steps': 100,
  'pcdm_series_length': prm_params['l_max'],
  'pcdm_latent_dim': 1,
  'pcdm_time_embed_dim': 32,
  'pcdm_time_hidden_dim': 32,
  'pcdm_channels': [48, 64, 80, 80, 64, 48],
  'pcdm_min_beta': 1e-4,
  'pcdm_max_beta': 0.02,
  'n_patterns': prm_params['k'],
  'n_steps': 100,
  'batch_size': 16,
  'n_epochs': 400,
  'lr': 4e-4,
  'loss_weights': [0.98, 0.01],
  'pad_weight': 1,
  'scale_weight': 0.01
}









