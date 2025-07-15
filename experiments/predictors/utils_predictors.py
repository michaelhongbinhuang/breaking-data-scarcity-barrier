#######################################
# Adjust Learning Rates for Large NNs #
#######################################

def adjust_learning_rate(optimizer, steps, adjust_factor=0.001, warmup_step=300, printout=False):
  if steps**(-0.5) < steps * (warmup_step**-1.5):
    lr_adjust = (16**-0.5) * (steps**-0.5) * adjust_factor
  else:
    lr_adjust = (16**-0.5) * (steps * (warmup_step**-1.5)) * adjust_factor

  for param_group in optimizer.param_groups:
    param_group['lr'] = lr_adjust
  if printout: 
    print(f"Updating learning rate to {lr_adjust}")
  return 









