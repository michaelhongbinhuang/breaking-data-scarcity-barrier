#!/bin/bash

dataname="sp500"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --dataname) dataname="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

python3 script_data_download.py --dataname ${dataname}
