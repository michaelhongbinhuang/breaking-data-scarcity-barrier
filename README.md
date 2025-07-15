# Breaking the Data Scarcity Barrier: Scale-Invariant Data Augmentation Significantly Improves Return Prediction across Models

This repository provides the code for the PNAS manuscript: Breaking the Data Scarcity Barrier: Scale-Invariant Data Augmentation Significantly Improves Return Prediction across Models.


## Data

Market data for the experiments is sourced from Yahoo! Finance using the Python package [yfinance](https://pypi.org/project/yfinance/).

Execute the following command to download and set up the market data of asset returns in our experiments:

```
`bash ./run_data_download.sh`
```

Options:

--dataname: Asset


## Training Generative Models

To train the generative framework, run the following command:

```
`bash ./run_generative_training.sh`
```

Options:

--dataname: Asset


*Note: The training process for the generative framework involves inherent randomness.*

Once training is complete, generate synthetic data for subsequent prediction experiments by running:

```
`bash ./run_data_augmentation.sh`
```

Options:

--dataname: Asset

--aug_models: Generative models in our manuscript

--n_runs: Number of trials


## Training Prediction Models

To train the prediction models on historical data and augmented data, and to store the resulting predictions, execute the following command:

```
`bash ./run_prediction_training.sh`
```

Options:

--dataname: Asset

--aug_models: Generative models in our manuscript

--pred_models: Prediction models in our manuscript

--n_runs: Number of trials

--adjust_lr: Whether adjust the learning rate during training (for neural networks)

--set_gpu: Assign the GPU
