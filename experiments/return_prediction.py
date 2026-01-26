from experiments.tatr_init_only import return_prediction_init
from experiments.tatr_final_aug import return_prediction_aug
from experiments.tatr_show_results import return_prediction_res



def return_prediction_demo():
  return_prediction_init()
  return_prediction_aug()
  return_prediction_res()
  print("-- Return prediction completed")









