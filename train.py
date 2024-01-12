
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
import argparse
from torch.optim.lr_scheduler import OneCycleLR
from timeit import default_timer as timer 
# Import tqdm for progress bar
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.data_setup import SSCData
from utils.file_utils import get_all_preprocessed_prefixes 
from utils.engin import EarlyStopping, create_writer,save_model, print_train_time 
from model.network import get_res_unet
from model.losses import WeightedCrossEntropyLoss, WCE_k3Clusters
from model.metrics import comp_IoU, m_IoU
from torchsummary import summary

###### set seeds ###################################################
SEED_VAL = 1234
np.random.seed(SEED_VAL)
random.seed(SEED_VAL)
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(SEED_VAL)
torch.manual_seed(SEED_VAL)
torch.cuda.manual_seed(SEED_VAL)
torch.cuda.manual_seed_all(SEED_VAL)
# when running on the CuDNN backend
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False
####################################################################
 
MODEL_NAME ='ResUNet'
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 2
BASE_LR= 0.01
DECAY = 0.0005
EPOCHS = 100
#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 6 # 2 to eliminate DataLoader running slow or even freeze
FOLD_NUM = 1

#import wandb
#wandb.login(key='add your key')
######################################################################

def parse_arguments():
    global DATASET, EXPR_NAME, MODEL_NAME, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, BASE_LR, DECAY, EPOCHS, PREPROC_PATH
    print("\n Training Script\n")

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", help="Target dataset", type=str, choices=['NYU', 'NYUCAD'])
    parser.add_argument("--model_name", help="model name", type=str)
    parser.add_argument("--expr_name", help="experiment name", type=str)
    parser.add_argument("--train_batch_size", help="Trainig batch size. Default: "+str(TRAIN_BATCH_SIZE),
                        type=int, default=TRAIN_BATCH_SIZE, required=False)
    parser.add_argument("--val_batch_size",  help="Val batch size. Default: "+str(VAL_BATCH_SIZE),
                        type=int, default=VAL_BATCH_SIZE, required=False)
    parser.add_argument("--base_lr", help="Base LR. Default " + str(BASE_LR),
                        type=float, default=BASE_LR, required=False)
    parser.add_argument("--decay", help="Weight decay. Default: " + str(DECAY),
                        type=float, default=DECAY, required=False)
    parser.add_argument("--epochs", help="How many epochs? Default: " + str(EPOCHS),
                        type=int, default=EPOCHS, required=False)
   
    args = parser.parse_args()


    ####################################################
    DATASET = args.dataset
    EXPR_NAME = args.expr_name
    MODEL_NAME = args.model_name
    EPOCHS= args.epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    BASE_LR= args.base_lr
    DECAY = args.decay
    
    if(DATASET == 'NYU')or (DATASET == 'nyu'): 
      PREPROC_PATH = './data/NYU_train_preproc/'
    elif (DATASET == 'NYUCAD') or (DATASET == 'nyucad'):
      PREPROC_PATH = './data/NYUCAD_train_preproc/'
    ###################################################
   

def train():
  global FOLD_NUM  
    

  ######################################################
  print("Model: "+ MODEL_NAME+"_"+EXPR_NAME)
  print("Dataset path: "+ PREPROC_PATH)
  
  # get file prefixes
  file_prefixes = get_all_preprocessed_prefixes(PREPROC_PATH, criteria='*.npz')
  df = pd.DataFrame(data={"filename": file_prefixes})
  
  
  ####################################################################################################################
  kf = KFold(n_splits=3, random_state=42, shuffle=True)
  
  for fold, (train_ids, val_ids) in enumerate(kf.split(df)):
      
      train_df= df.iloc[train_ids]
      val_df= df.iloc[val_ids]
      train_ds = SSCData(train_df['filename'].tolist())
      val_ds = SSCData(val_df['filename'].tolist())
      
      
      print(f'Items in train_dataset ',len(train_ds), ' |  Items in val_dataset ',len(val_ds))
  
      train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS, pin_memory=True) 
      val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True)
  
      # check
      print(f"Length of train dataloader: {len(train_loader)} batches ({len(train_ds)}/{TRAIN_BATCH_SIZE})")
      print(f"Length of val dataloader: {len(val_loader)} batches ({len(val_ds)}/{VAL_BATCH_SIZE})")
  
      ############################################################################################
      # model initialization for each fold
      if  MODEL_NAME == 'ResUNet':
        model = get_res_unet()
        
      # If there are multiple GPUs, wrap the model with nn.DataParallel
      if torch.cuda.device_count() > 1:
          print("Using", torch.cuda.device_count(), "GPUs!")
          model = nn.DataParallel(model)
      
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.to(device)
      ############################################################################################
      ############################################################################################
      # define loss
      #loss_function = WeightedCrossEntropyLoss() # custom weighted loss (CE + re-sampling)
      loss_function = WCE_k3Clusters(device)
      print("Loss.function: "+ str(loss_function))
      print("-------------------------------------------------------------")
      ############################################################################################
      ############################################################################################
      
      # define the optimizer
      optimizer = optim.SGD(model.parameters(), lr=BASE_LR, weight_decay=DECAY, momentum=0.9)
      
      # Set up the scheduler
      scheduler = OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=EPOCHS)

      
      # initialize the timer
      train_time_start = timer()
      
      # initialize wandb for each fold
      #run = wandb.init(project=f'{MODEL_NAME}_{EXPR_NAME}', group='k-fold', name=f'fold-{FOLD_NUM}', reinit=True)
      
      # initialize the early_stopping object
      early_stopping = EarlyStopping(patience=15, verbose=True)
      
      print(f"Processing FOLD NUM: {FOLD_NUM}")
      print(f'Number of EPOCHS in this fold: {EPOCHS}')  
      
      for epoch in tqdm(range(EPOCHS)):
          ### Training
          model.train()
          train_loss, train_IoU, train_prec, train_recall, train_mIoU = 0,0,0,0,0
          
          ####################
         
          # Add a loop to loop through training batches
          for batch_num, sample in enumerate(train_loader):
              loss, t_IoU, t_prec, t_recall, t_mIoU = 0,0,0,0,0
              seg_list=[]
              inputs, targets, weights, masks = sample['vox_tsdf'].to(device), sample['vox_lbl'].to(device), sample['vox_weight'].to(device),sample['vox_mask'].to(device), # inputs (tsdf), targets (semantic labels), weights, masks (for evaluation)
        
              # 1. Forward pass
              ##########################
              outputs = model(inputs)
              ########################### 
              
              # 2. Calculate loss (per batch)
              loss = loss_function(outputs, targets, weights)
              
              train_loss += loss # accumulatively add up the loss  
              
              # Calculate IoU and mIoU
              t_IoU, t_prec, t_recall = comp_IoU(outputs, targets, masks)
              t_mIoU,seg_list = m_IoU(outputs, targets)
              train_IoU += t_IoU
              train_prec += t_prec
              train_recall += t_recall
              train_mIoU += t_mIoU 
              
              #######################
              
              # 3. Optimizer zero grad
              optimizer.zero_grad()
  
              # 4. Loss backward
              loss.backward()
  
              # 5. Optimizer step
              optimizer.step()
          
              # Step the scheduler
              scheduler.step()
          
          # Divide total train loss, train_IoU, and  train mIoU by length of train dataloader (average loss per batch per epoch)
          train_loss /= len(train_loader)
          train_IoU /= len(train_loader) 
          train_prec /= len(train_loader) 
          train_recall /= len(train_loader) 
          train_mIoU /= len(train_loader) 
        ############################## validation #######################################    
          
          model.eval()
          with torch.inference_mode():
              val_loss, val_IoU, val_prec, val_recall, val_mIoU = 0,0,0,0,0
              ####################
              for batch_num, sample in enumerate(val_loader):
                  v_IoU, v_prec, v_recall, v_mIoU = 0,0,0,0
                  seg_lis =[]     
                  inputs, targets, weights, masks = sample['vox_tsdf'].to(device), sample['vox_lbl'].to(device), sample['vox_weight'].to(device),sample['vox_mask'].to(device)
                  outputs = model(inputs)
                 
              ########################### 
                  val_loss += loss_function(outputs, targets, weights)
               
                  v_IoU, v_prec, v_recall = comp_IoU(outputs, targets, masks)
                  v_mIoU, seg_lis = m_IoU(outputs, targets)
                   
                  # accumulate the values over patches
                  val_IoU += v_IoU
                  val_prec += v_prec
                  val_recall += v_recall
                  val_mIoU += v_mIoU
                  ######################
              
              # Divide total val loss, val_IoU, val_mIoU by length of val dataloader (per batch per epoch)
              val_loss /= len(val_loader)
              val_IoU /= len(val_loader)
              val_prec /= len(val_loader)
              val_recall /= len(val_loader)
              val_mIoU /= len(val_loader)
              #############################
              expr_name_fold = EXPR_NAME+"_fold"+ str(FOLD_NUM)
              
              # Early stopping
              early_stopping(model, MODEL_NAME, expr_name_fold, val_loss)
              
              if early_stopping.early_stop:
                print("Early stopping")
                expr_name_fold = " "
                break
              
          print(f"Epoch: {epoch}")
          print(f"train Loss: {train_loss:.5f} | train IoU: {train_IoU:.2f} |train precision: {train_prec:.2f} | train recall: {train_recall:.2f} | train mIoU: {train_mIoU:.2f}")
          print(f"val Loss: {val_loss:.5f} | val IoU: {val_IoU:.2f} |val precision: {val_prec:.2f} | val recall: {val_recall:.2f} | val mIoU: {val_mIoU:.2f}")
          print(f"................................................................................................................................................................")
          
          ###########################################################################################################
          # log metrics to wandb
          #wandb.log({"train Loss": train_loss, "train IoU":train_IoU, "train precision": train_prec, "train recall":train_recall, "train mIoU": train_mIoU, "val Loss": val_loss, "val IoU":val_IoU, "val precision": val_prec, "val recall":val_recall, "val mIoU": val_mIoU },step=epoch )
        ##############################################################################################################
        
      # Calculate training time for each fold      
      train_time_end = timer()
      total_train_time_model_0 = print_train_time(start=train_time_start, 
                                                end=train_time_end,
                                                device=str(next(model.parameters()).device))
      
      ##########################################
      # Close this run for wandb
      run.finish()
      FOLD_NUM+=1
      print(f"----------------------------------------------------------------------------------------------------------------------------------------------------------------")

# Main Function
def main():
    parse_arguments()
    train()
    
if __name__ == '__main__':
  main()
