from experiments.utils_tatr import *
from experiments.utils_tatr_extension import *
from experiments.tatr_init_only import *
from experiments.tatr_final_aug import *

fig_path = 'figs/'



#############################
# Comparisons across Models #
#############################

def compare_tatr_augmodels_entire(
    pred_models, 
    aug_models, 
    dataname='sp500', 
    datatypes=None, 
    n_augmentations=100, 
    show_boxplot=False, 
    show_barplot=True, 
    percentage=False, 
    agg_scheme='median', 
    convert_benchmark=False, 
    bar_spacing=None, 
    show_params=False, 
    legend_loc='upper left', 
    y_range=None, 
    store_fig=False, 
    ):
  """ Compare the TATR results augmented on the entire training set by different generative models """
  datatypes = ['returns', 'vol_change'] if datatypes is None else datatypes
  fig_width = 3 * len(pred_models) if len(pred_models) > 2 else 4 * len(pred_models)
  fig_height = 5 * len(datatypes)
  fig, axs = plt.subplots(len(datatypes), 1, figsize=(fig_width, fig_height))
  if len(datatypes) == 1:
    axs = [axs]
  for ax, datatype in zip(axs, datatypes):
    # Load the results
    tatr_results = load_tatr_results_r2(datatype, pred_models, aug_models, dataname, n_augmentations, convert_benchmark)
    pretrain = ('itransformer' in pred_models and 'itransformer-selftrain-exchange' in pred_models) \
      or ('transformer' in pred_models and 'transformer-selftrain-exchange' in pred_models)
    
    # Plot the boxplot results
    if show_boxplot:
      plot_tatr_results_boxplot_violin(
        ax, datatype, pred_models, aug_models, tatr_results, 
        show_params=show_params, 
        pretrain=pretrain, legend_loc=legend_loc, y_range=y_range, 
        )
    
    # Plot the barplot results
    if show_barplot:
      plot_tatr_results_barplot(
        ax, datatype, pred_models, aug_models, tatr_results, dataname=dataname, 
        percentage=percentage, show_params=show_params, agg=agg_scheme,
        spacing=bar_spacing, pretrain=pretrain, legend_loc=legend_loc, y_range=y_range, 
        )
  fig.tight_layout()
  
  # Store the figure
  if store_fig:
    res_name = 'srmodel' if show_barplot else 'r2'
    datatype_name = datatypes[0] if len(datatypes) == 1 else 'both'
    if pretrain: 
      file_name = f"res_{res_name}_{dataname}_{datatype_name}_pretrain.pdf"
    elif convert_benchmark:
      file_name = f"res_{res_name}_{dataname}_{datatype_name}_benchmark0.pdf"
    else: 
      file_name = f"res_{res_name}_{dataname}_{datatype_name}.pdf"
    fig.savefig(fig_path + file_name, dpi=300, bbox_inches='tight')
    print(f"Figure stored: {file_name}")


def load_tatr_results_r2(
    datatype, pred_models, aug_models, 
    dataname='sp500', n_augmentations=100, convert_benchmark=False, 
    ):
  """ Load the R-squared values of prediction experiments. Return shape: (#aug_models, n_runs, #pred_models) """
  n_aug = f"-an{n_augmentations}"
  tatr_results_all = []
  for pred_model in pred_models:
    # Load the TATR init. results
    pred_model_init = pred_model
    df_init_entire = load_res_tatr_init_only(pred_model_init, datatype, dataname)
    r2_init = df_init_entire.values
    if convert_benchmark:
      r2_init = convert_r_squared_benchmark(r2_init, dataname=dataname, datatype=datatype)
    tatr_init = r2_init.reshape(1, -1, 1)
    # Load the TATR aug. results
    tatr_augs = []
    for aug_model in aug_models:
      res_tatr_final_aug_r2_path = f"res/{dataname}/r2/aug_{aug_model}/"
      filename = f"tatr_{dataname}-{datatype}_finalaug_{aug_model}{n_aug}_{pred_model}_r2avg.csv"
      df = pd.read_csv(res_tatr_final_aug_r2_path + filename, index_col=False)
      r2_aug =  df['r2'].values
      if convert_benchmark:
        r2_aug = convert_r_squared_benchmark(r2_aug, dataname=dataname, datatype=datatype)
      tatr_aug = r2_aug.reshape(1, -1, 1)
      tatr_augs.append(tatr_aug)
    tatr_augs = np.concatenate(tatr_augs, axis=0)
    tatr_results = np.concatenate([tatr_init, tatr_augs], axis=0)
    tatr_results_all.append(tatr_results)
  tatr_results_all = np.concatenate(tatr_results_all, axis=2)
  return tatr_results_all   # Shape: (#aug_models, n_runs, #pred_models)


def stats_agg(res, scheme='median'):
  """ Compute the statistics of TATR results """
  if scheme == 'median':
    return np.median(res)
  elif scheme == 'avg' or scheme == 'mean':
    return res.mean()
  elif scheme == 'max':
    return np.max(res, axis=0)
  elif scheme == '95p' or scheme == '95%' or scheme == '95':
    return np.percentile(res, 95)
  elif scheme == 'min':
    return np.min(res, axis=0)
  else:
    raise ValueError(f"Invalid statstics aggregation scheme: {scheme}")


def compute_tatr_results_sr(tatr_results_all, dataname='sp500', agg='median', percentage=False):
  """ Compute the sr_model of prediction models """
  SR_BH = compute_sr_bh(dataname)
  tatr_sr = np.zeros((tatr_results_all.shape[2], tatr_results_all.shape[0] - 1))
  tatr_sr_init = np.zeros((tatr_results_all.shape[2], 1))
  delta_tatr_sr = np.zeros((tatr_results_all.shape[2], tatr_results_all.shape[0] - 1))
  for i in range(tatr_results_all.shape[2]):
    r2_init = tatr_results_all[0, :, i]
    sr_init = compute_model_sharpe_ratio(stats_agg(r2_init, agg), dataname=dataname)
    tatr_sr_init[i, 0] = sr_init
    for j in range(1, tatr_results_all.shape[0]):
      # Sharpe ratio based improvement
      r2_model = tatr_results_all[j, :, i]
      sr_model = compute_model_sharpe_ratio(stats_agg(r2_model, agg), dataname=dataname)
      tatr_sr[i, j-1] = sr_model if not percentage else (sr_model / sr_init - 1) * 100
      delta_sr = (sr_model / SR_BH) - 1
      delta_tatr_sr[i, j-1] = delta_sr * 100
  return tatr_sr, delta_tatr_sr, tatr_sr_init


def plot_tatr_results_boxplot_violin(
    ax, datatype, pred_models, aug_models, tatr_results, 
    show_params=False, 
    pretrain=True, legend_loc='upper left', y_range=None, 
    ):
  """ Plot the boxplots (violin shape) of TATR results """
  list_aug_models = ['init'] + aug_models
  label_aug_models = get_aug_model_labels(list_aug_models)
  colors = get_model_colors(list_aug_models)
  data_for_boxplot = []
  init_data_for_boxplot = []
  positions = []
  init_position = []
  positions_offset = np.linspace(-0.35, 0.35, len(list_aug_models))
  for i, _ in enumerate(pred_models):
    for j, aug_model in enumerate(list_aug_models):
      pos = i + positions_offset[j]
      if aug_model == 'init':
        init_data_for_boxplot.append(tatr_results[j, :, i])
        init_position.append(pos)
      else:
        data_for_boxplot.append(tatr_results[j, :, i])
        positions.append(pos)
  vplot = ax.violinplot(data_for_boxplot, positions=positions, widths=0.17, showmedians=False)
  bplot = ax.boxplot(init_data_for_boxplot, positions=init_position, widths=0.11, patch_artist=True)

  # Set the x- and y-axis range of plots
  if y_range is None:
    y_ranges_returns = [-0.1, 0.05]
    y_ranges_volchange = [-0.1, 0.2]
    y_range = y_ranges_returns if datatype == 'returns' else y_ranges_volchange
  ax.set_ylim(y_range[0], y_range[1])

  # Set the x- and y-axis labels
  ax.set_xlabel('Prediction Models')
  ax.set_ylabel('OOS R-squared')
  pred_model_labels = [modify_pred_models(pred_model, pretrain) for pred_model in pred_models]
  pred_model_labels = [pred_model if '-' not in pred_model else pred_model.split('-')[0] for pred_model in pred_model_labels] \
    if not show_params else pred_model_labels
  ax.set_xticks(ticks=range(len(pred_models)), labels=pred_model_labels)
  
  # Style violins
  for body, color in zip(vplot['bodies'], np.tile(colors[1:], len(pred_models))):
    body.set_facecolor(color)
    # body.set_edgecolor('gray')
    body.set_alpha(1)
  vplot['cbars'].set_linewidth(1)
  vplot['cmaxes'].set_linewidth(1)
  vplot['cmins'].set_linewidth(1)

  # Style init box
  for patch, color in zip(bplot['boxes'], np.tile(colors[0], len(pred_models))):
    patch.set_facecolor(color)
  for cap, color in zip(bplot['caps'], np.tile(np.repeat(colors[0], 2), len(pred_models) * 2)):
    cap.set(color=color)
  for whisker, color in zip(bplot['whiskers'], np.tile(np.repeat(colors[0], 2), len(pred_models) * 2)):
    whisker.set(color=color)

  # Style legends
  legend_patches = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
  set_legend_loc(ax, legend_patches, label_aug_models, loc=legend_loc)

  # Plot grid
  ax.yaxis.grid(True, alpha=0.5, linestyle='--', linewidth=0.5)

  # Style space and font sizes
  plt.subplots_adjust(hspace=0.4)
  plt.rcParams['axes.labelsize'] = 15  # Axis labels font size
  plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels font size
  plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels font size
  plt.rcParams['legend.fontsize'] = 11  # Legend font size


def plot_tatr_results_barplot(
    ax, datatype, pred_models, aug_models, tatr_results, dataname='sp500', 
    percentage=False, show_params=False, agg='median',
    spacing=None, pretrain=True, legend_loc='upper left', y_range=None, 
    ):
  """ Plot the barplots of TATR results """
  list_aug_models = ['init'] + aug_models
  label_aug_models = get_aug_model_labels(list_aug_models)
  if datatype == 'returns':
    colors = get_model_colors(list_aug_models)
    tatr_sr, _, tatr_sr_init = compute_tatr_results_sr(tatr_results, dataname=dataname, agg=agg, percentage=percentage)
    if not percentage:
      SR_BH = compute_sr_bh(dataname)
      COLOR_BH = ['#c4d8e9']
      sr_bh = np.array([SR_BH] * len(pred_models)).reshape(-1, 1)
      sr_model_values = np.concatenate([tatr_sr_init, tatr_sr], axis=1)
      bar_values = np.concatenate([sr_bh, sr_model_values], axis=1)   # shape: [#pred_models, #sr_hb + #sr_init + #sr_augs]
      label_aug_models = ['BH'] + label_aug_models
      colors = COLOR_BH + colors
    else:
      bar_values = tatr_sr    # shape: [#pred_models, #sr_hb + #sr_init + #sr_augs]
      colors = colors[1:]
      label_aug_models = label_aug_models[1:]
  else:
    raise NotImplementedError
  
  # Set the space of plots
  if y_range is None:
    y_range_min = max(0, np.min(bar_values) - 0.1 * np.min(bar_values))
    y_range_max_space = 0.27 if 'upper' in legend_loc else 0.15
    y_range_max = np.max(bar_values[np.isfinite(bar_values)]) + y_range_max_space * np.max(bar_values[np.isfinite(bar_values)])
  else:
    y_range_min, y_range_max = y_range
  GROUP_SPACING = 2 if spacing is None else spacing[0]
  BAR_WIDTH = 0.2 if spacing is None else spacing[1]
  for i in range(len(pred_models)):
    plot_bars(ax, i, bar_values, colors, label_aug_models, bar_width=BAR_WIDTH, group_spacing=GROUP_SPACING)
    ax.set_ylim(y_range_min, y_range_max)
  
  # Set the labels and titles of plots
  ax.set_xlabel('Prediction Models')
  ax.set_ylabel('Sharpe Ratio' if datatype == 'returns' else 'Delta R-squared')
  pred_model_labels = [modify_pred_models(pred_model, pretrain) for pred_model in pred_models]
  pred_model_labels = [pred_model if '-' not in pred_model else pred_model.split('-')[0] for pred_model in pred_model_labels] \
    if not show_params else pred_model_labels
  ax.set_xticks(ticks=np.arange(len(pred_models)) * GROUP_SPACING, labels=pred_model_labels)

  # Style legends
  legend_patches = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
  set_legend_loc(ax, legend_patches, label_aug_models, loc=legend_loc)

  # Style grid
  ax.yaxis.grid(True, alpha=0.5, linestyle='--', linewidth=0.5)

  # Style space and fonts
  plt.subplots_adjust(hspace=0.4)
  plt.rcParams['axes.labelsize'] = 15  # Axis labels font size
  plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels font size
  plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels font size
  plt.rcParams['legend.fontsize'] = 11  # Legend font size

def plot_bars(ax, i, values, colors, label_aug_models, bar_width, group_spacing):
  """ Plot the barplots of given values """
  total_width = bar_width * (values.shape[1] - 0.5)
  offsets = np.linspace(-total_width/2, total_width/2, values.shape[1])
  base_x = i * group_spacing
  for j in range(values.shape[1]):
    rect = ax.bar(base_x + offsets[j], values[i, j], bar_width, color=colors[j], alpha=0.9, label=label_aug_models[j])
    autolabel(ax, rect, n=values.shape[1])

def autolabel(ax, rects, n):
  """ Attach a text label above each bar in rects, displaying its height """
  for rect in rects:
    height = rect.get_height()
    # Determine whether height is nan
    neg = np.isnan(height) or height < 0
    height_disp = height - 0.02 if not neg else 0
    height_label = f"{round(height, 2):.2f}" if not neg else 'NA'
    color = '#161616' if not neg else 'gray'
    ax.annotate(
      f"{height_label}",
      xy=(rect.get_x() + rect.get_width() / 2, height_disp),
      xytext=(0, n),  # n points vertical offset
      fontsize=11,
      textcoords="offset points",
      ha='center', va='bottom',
      color=color, rotation=90,
      )


def modify_pred_models(modelname, pretrain=False):
  """ Modify the names of down. models """
  if not pretrain:
    if 'gbdt' in modelname:
      return 'GBRT'
    elif 'rf' in modelname:
      return 'RF'
    elif 'mlp' in modelname:
      return 'MLP'
    elif 'itransformer' in modelname:
      return 'iTransformer'
    else:
      raise ValueError(f"Invalid prediction model {modelname} for y-labels")
  else:
    if 'itransformer' in modelname and len(modelname.split('-')) == 1:
      return 'iTransformer-NPT'
    elif 'itransformer' in modelname and len(modelname.split('-')) > 1:
      return 'iTransformer-PT'
    else:
      raise ValueError(f"Invalid pre-trained prediction model {modelname} for y-labels")


def set_legend_loc(ax, legend_patches, label_aug_models, loc='upper left', narrow=False):
  """ Set the legend location of plots """
  if loc == 'lower right':
    ax.legend(legend_patches, label_aug_models, loc='lower left', bbox_to_anchor=(1, 0))  # Lower right outside
  elif 'upper left' in loc:
    col_space = 1 if 'wide' in loc else (4 if 'narrow' in loc else 2)
    n_cols_legend = np.ceil(len(label_aug_models) / col_space)
    ax.legend(
      legend_patches, label_aug_models, 
      loc='upper left', ncol=n_cols_legend, 
      bbox_to_anchor=(0.0, 1.0), columnspacing=0.8, 
      frameon=True)  # Top center outside
  else:
    raise ValueError(f"Invalid legend location {loc}")



#############################################
# Comparisons among Increment Augmentations #
#############################################

def compare_tatr_augmodels_incremental(
    pred_models, 
    n_augmentations, 
    dataname='sp500', 
    datatypes=None, 
    aug_models='', 
    show_boxplot=False, 
    show_barplot=True, 
    percentage=False, 
    agg_scheme='median', 
    bar_spacing=None,  
    show_params=False, 
    legend_loc='upper left', 
    y_range=None, 
    store_fig=False, 
    ):
  """ Compare the TATR results augmented on the entire training set by different generative models """
  datatypes = ['returns', 'vol_change'] if datatypes is None else datatypes
  fig_width = 3 * len(pred_models) if len(pred_models) > 2 else 4 * len(pred_models)
  fig_height = 5 * len(datatypes)
  fig, axs = plt.subplots(len(datatypes), 1, figsize=(fig_width, fig_height))
  if len(datatypes) == 1:
    axs = [axs]
  for ax, datatype in zip(axs, datatypes):
    # Load the results
    tatr_results = load_tatr_results_r2_incremental(datatype, pred_models, n_augmentations, dataname, aug_models)

    # Plot the boxplot results
    if show_boxplot:
      plot_tatr_results_boxplot_violin(
        ax, datatype, pred_models, n_augmentations, tatr_results, 
        show_params=show_params, 
        legend_loc=legend_loc, y_range=y_range, 
        )

    # Plot the barplot results
    if show_barplot:
      plot_tatr_results_barplot(
        ax, datatype, pred_models, n_augmentations, tatr_results, 
        percentage=percentage, show_params=show_params, agg=agg_scheme,
        spacing=bar_spacing, legend_loc=legend_loc, y_range=y_range, 
        )
  fig.tight_layout()
  
  # Store the figure
  if store_fig:
    res_name = 'srmodel' if show_barplot else 'r2'
    datatype_name = datatypes[0] if len(datatypes) == 1 else 'both'
    fig.savefig(fig_path + f"res_{res_name}_{dataname}_{datatype_name}_incremental.pdf", dpi=300, bbox_inches='tight')
    print(f"Figure stored: res_{res_name}_{dataname}_{datatype_name}_incremental.pdf")

def load_tatr_results_r2_incremental(datatype, pred_models, n_augmentations, dataname='sp500', aug_model=''):
  """ Load the R-squared values of prediction experiments. Return shape: (#aug_models, n_runs, #pred_models) """
  tatr_results_all = []
  for pred_model in pred_models:
    # Load the TATR init. results
    pred_model_init = pred_model
    df_init_entire = load_res_tatr_init_only(pred_model_init, datatype, dataname)
    tatr_init = df_init_entire.values.reshape(1, -1, 1)
    # Load the TATR aug. results
    tatr_augs = []
    for n_augmentation in n_augmentations:
      n_aug = f"-an{n_augmentation}"
      res_tatr_final_aug_r2_path = f"res/{dataname}/r2/aug_{aug_model}/"
      filename = f"tatr_{dataname}-{datatype}_finalaug_{aug_model}{n_aug}_{pred_model}_r2avg.csv"
      df = pd.read_csv(res_tatr_final_aug_r2_path + filename, index_col=False)
      tatr_aug = df['r2'].values.reshape(1, -1, 1)
      tatr_augs.append(tatr_aug)
    tatr_augs = np.concatenate(tatr_augs, axis=0)
    tatr_results = np.concatenate([tatr_init, tatr_augs], axis=0)
    tatr_results_all.append(tatr_results)
  tatr_results_all = np.concatenate(tatr_results_all, axis=2)
  return tatr_results_all   # Shape: (#n_augs, n_runs, #pred_models)









