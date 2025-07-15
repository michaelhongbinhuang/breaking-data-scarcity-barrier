#!/bin/bash

dataname="sp500"
aug_model=""
n_runs=100

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataname) dataname="$2"; shift ;;
    --aug_model) aug_model="$2"; shift ;;
    --n_runs) n_runs="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

python3 script_data_augmentation.py \
  --dataname ${dataname} \
  --aug_model ${aug_model} \
  --n_runs ${n_runs} \
  --idle_cpu 0
