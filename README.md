# Codebase for "Breaking the Data Scarcity Barrier: Significant Return Prediction Improvement across Models through Scale-Invariant Data Augmentation"

This repository provides the code for our manuscript - Breaking the Data Scarcity Barrier: Significant Return Prediction Improvement across Models through Scale-Invariant Data Augmentation.



## Installation

Please follow these steps to install and set up the project.


**Step 1. Download the code**

Clone or download this repository to access the code.


**Step 2. Install Python**

This project is written in Python. If Python is not installed on your device, you can download and install it from the [official Python download page](https://www.python.org/downloads/). Python 3.11 is recommended.


**Step 3. Intall the required packages**

Install the necessary Python packages listed in `requirements.txt` using the following command:

```
pip install -r requirements.txt --constraint constraints.txt
```
*Note: Certain packages require specific versions of other dependencies, which have been specified in constraints.txt.*

This process might take 5–10 minutes, depending on your hardware and network speed.


**Tested Systems**

This code has been tested on the following systems:

* Linux with an AMD EPYC 7763 64-Core CPU, 3.9 TiB of RAM, and 8 NVIDIA A800 GPUs
* Linux with an AMD EPYC 9754 128-Core CPU and 1.5 TiB of RAM
* macOS with an Apple M2 Pro chip and 16 GB of RAM



## Instructions

**Run the demo**

Execute the following commands to run the demo without any manual inputs:

```
bash ./run_demo.sh
```

*Notes: Data downloads may be interrupted due to unstable network connections (e.g., in regions with restricted access) or temporary outages of the Yahoo Finance API, often caused by exceeding the request limit. Please verify the network connection and check the status of the [Yahoo Finance API](https://ranaroussi.github.io/yfinance/index.html) if issues occur.*



**Expected output and runtime**

To keep the terminal window easy to follow, only progress information are displayed for each step.

The final results of the implied Sharpe ratios and out-of-sample (OOS) R-squared values are stored in the path: `figs/`.


**Expected runtime**

The runtime for completing each step in the above demo on the Linux with an AMD EPYC 7763 64-Core CPU, 3.9 TiB of RAM, and 8 NVIDIA A800 GPUs is as follows:

* Data collection: ~ 1 minutes
* Generative models training: ~ 2 hours
* Data augmentation: ~ 2 hours
* Prediction models training: ~ 4 hours


## Contact

If you have any questions, please feel free to contact: 

Hongbin Huang (hongbin.huang@my.cityu.edu.hk)
