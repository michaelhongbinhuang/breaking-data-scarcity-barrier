import argparse

from models.train_framework import train_framework



###########################
# Traing Generative Model #
###########################

if __name__ == "__main__":
  # Set up inputs
  parser = argparse.ArgumentParser(description="Download Market Data")
  parser.add_argument("--dataname", type=str, default='sp500', help="Data name")
  args = parser.parse_args()

  # Settings
  dataname = args.dataname
  
  # Training
  train_framework(dataname=dataname)









