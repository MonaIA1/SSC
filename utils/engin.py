import os
import torch
from torch.utils.tensorboard import SummaryWriter 
from datetime import datetime 
from pathlib import Path 
import numpy as np
import wandb
from torch.optim.lr_scheduler import _LRScheduler
class EarlyStopping:
    
    def __init__(self, patience=10, verbose=False, delta=0, trace_func=print): # https://github.com/Bjarten/early-stopping-pytorch
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
                            
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    
    def __call__(self, model, model_name, expr_name, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, model_name, expr_name, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, model_name, expr_name, val_loss)
            self.counter = 0
    
    def save_checkpoint(self, model, model_name, expr_name, val_loss):
        # Saves model when validation loss decrease.
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model, model_name, expr_name)
        self.val_loss_min = val_loss

################################# Model saving #######################################
def save_model(model, model_name, expr_name):
  
  model_path = Path("saved_models")
  model_path.mkdir(parents=True, 
                   exist_ok=True 
  )

  
  timestamp = datetime.now().strftime("%Y-%m-%d") 
  full_model_name = model_name +"_"+ expr_name + "_" + timestamp +".pth"
  model_save_path = model_path / full_model_name

  
  print(f"Saving model to: {model_save_path}")
  
  # save model in saved_models dir
  torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
             f=model_save_path)
  # save model in wandb dir
  torch.save(model.state_dict(), os.path.join(wandb.run.dir, full_model_name))
  
  
  # Get the model size in bytes then convert to megabytes
  loaded_model_size = Path(model_save_path).stat().st_size // (1024*1024)
  print(f"{model_name}_{expr_name} model size: {loaded_model_size} MB")
  
      
                    
##################################### train time cal ##############################
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time    

############################## trainig summary #####################################

def create_writer(experiment_name: str, 
                  model_name: str,
                  fold_num = None) -> torch.utils.tensorboard.writer.SummaryWriter():
   
    timestamp = datetime.now().strftime("%Y-%m-%d") 

    if fold_num:
        fold_num = "fold" + str(fold_num)
        log_dir = os.path.join("runs", timestamp, model_name, experiment_name, fold_num)
    else:
        log_dir = os.path.join("runs", timestamp, model_name, experiment_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
####################################################################################

