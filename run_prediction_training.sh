#!/bin/bash

dataname="sp500"
aug_models=""
down_models=""
n_runs=100
n_augmentations="100"
adjust_lr=True
set_gpu=0

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataname) dataname="$2"; shift ;;
    --aug_models) aug_model="$2"; shift ;;
    --down_models) down_model="$2"; shift ;;
    --n_runs) n_runs="$2"; shift ;;
    --n_augmentations) n_augmentations="$2"; shift ;;
    --adjust_lr) adjust_lr="$2"; shift ;;
    --set_gpu) set_gpu="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

python3 script_prediction_training.py \
  --datanames ${datanames} \
  --aug_models ${aug_models} \
  --down_models ${down_models} \
  --n_runs ${n_runs} \
  --n_augmentations ${n_augmentations} \
  --adjust_lr ${adjust_lr} \
  --set_gpu ${set_gpu} \
  --store_res True
