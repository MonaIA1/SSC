import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
import argparse
from tqdm.auto import tqdm

from utils.data_setup import SSCData
from utils.file_utils import get_all_preprocessed_prefixes 
from utils.engin import EarlyStopping, create_writer,save_model, print_train_time 
from model.network import get_res_unet
from model.losses import WeightedCrossEntropyLoss, WCE_BalancedClusters, WCE_k3Clusters
from model.metrics import comp_IoU, m_IoU
from torch.utils.tensorboard import SummaryWriter



######################################################################
MODEL_NAME ='ResUNet'
BATCH_SIZE = 1
#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 2 # to eliminate DataLoader running slow or even freeze
FOLD_NUM = 1
SAVED_WEIGHTES = ''

def parse_arguments():
    global DATASET, MODEL_NAME,EXPR_NAME, PREPROC_PATH, SAVED_WEIGHTES
    print("\n Evaluation Script\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Target dataset", type=str, choices=['NYU', 'NYUCAD'])
    parser.add_argument("--model_name", help="model name", type=str)
    parser.add_argument("--expr_name", help="experiment name", type=str)
    parser.add_argument('--weights', metavar='File', type=str, nargs='+',
                    help='a model weight file to process')
    args = parser.parse_args()
    ####################################################
    DATASET = args.dataset
    MODEL_NAME = args.model_name
    SAVED_WEIGHTES = args.weights
    EXPR_NAME = args.expr_name
    
    if(DATASET == 'NYU')or (DATASET == 'nyu'):  
      PREPROC_PATH = './data/NYU_test_preproc/'
    elif (DATASET == 'NYUCAD') or (DATASET == 'nyucad'):
      PREPROC_PATH = './data/NYUCAD_test_preproc/'
    ###################################################
    
    
def model_evalauteion ():
  ################################### testing data loader ####################################################################
  file_prefixes_test = get_all_preprocessed_prefixes(PREPROC_PATH, criteria='*.npz')
  df_test = pd.DataFrame(data={"filename": file_prefixes_test})
  test_ds = SSCData(df_test['filename']) 
  test_loader = DataLoader(test_ds, batch_size= BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS)
  fold_num = 1
  #########################################
  print("Dataset path:" , PREPROC_PATH)
  print("Model weights:", SAVED_WEIGHTES)
  
  # Make device agnostic code
  dev = 'cuda:0'
  device = torch.device(dev) if torch.cuda.is_available() else torch.device("cpu")
  
  # define loss
  loss_function = WCE_BalancedClusters(device)
  
  for w in range(len(SAVED_WEIGHTES)):
  
    if  MODEL_NAME == 'ResUNet':
      model_eval = get_res_unet()
    elif MODEL_NAME == 'mmnet':
      model_eval = get_mmnet()
    elif MODEL_NAME == 'mmnet_early': 
      model_eval = get_mmnet_depth_rgb_early() 
    elif MODEL_NAME == 'mmnet_mid': 
      model_eval = get_mmnet_depth_rgb_mid() 
     
    # initialize wandb for each weighted fold
    run = wandb.init(project=f'{MODEL_NAME}_{EXPR_NAME}_testing', group='k-fold', name=f'fold-{fold_num}', reinit=True)
    
    # Because of parallization on multiple GPUs during the training we need to
    # 1- Load the saved state dict
    state_dict = torch.load(f=SAVED_WEIGHTES[w])

    # 2- Create a new state dict in which 'module.' prefix is removed
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 3- Load the new state dict to your model
    model_eval.load_state_dict(new_state_dict)
    
    # Send model to GPU
    model_eval = model_eval.to(device)
  
    model_eval.eval()
    with torch.inference_mode():
          total_loss, total_IoU, total_prec, total_recall, total_mIoU = 0,0,0,0,0
          #####################
          ceil_ls = []
          floor_ls = []
          wall_ls = []
          wind_ls = []
          chair_ls = []
          bed_ls = []
          sofa_ls = []
          table_ls = []
          tvs_ls = []
          furn_ls = []
          objs_ls = []
          ####################
          for batch_num, sample in enumerate(test_loader):
              test_loss = 0
              test_IoU = 0
              test_mIoU = 0
              test_prec = 0
              test_recall = 0
              seg_lis =[]
              inputs, targets, weights, masks = sample['vox_tsdf'].to(device), sample['vox_lbl'].to(device), sample['vox_weight'].to(device),sample['vox_mask'].to(device)
              
              #outputs =   model_eval(inputs)
              if  (MODEL_NAME == 'ResUNet'):
                outputs = model_eval(inputs)
              
              
              test_loss = loss_function(outputs, targets, weights)
              
              test_IoU, test_prec, test_recall = comp_IoU(outputs, targets, masks)
              test_mIoU, seg_lis = m_IoU(outputs, targets)
              print(f"IoU: {test_IoU} | precision: {test_prec} | recall: {test_recall} | mIoU: {test_mIoU}")
              print(f"*************************************************")
              total_loss += test_loss
              total_IoU += test_IoU
              total_prec += test_prec
              total_recall += test_recall
              
              ######################
              ceil_ls.append(seg_lis[0])
              floor_ls.append( seg_lis[1])
              wall_ls.append(seg_lis[2])
              wind_ls.append(seg_lis[3])
              chair_ls.append(seg_lis[4])
              bed_ls.append(seg_lis[5])
              sofa_ls.append(seg_lis[6])
              table_ls.append(seg_lis[7])
              tvs_ls.append(seg_lis[8])
              furn_ls.append(seg_lis[9])
              objs_ls.append(seg_lis[10])
              #######################
          total_loss /= len(test_loader)
          total_IoU /= len(test_loader)
          total_prec /= len(test_loader)
          total_recall /= len(test_loader)
          
          #############################
          ceil_ls = sum(ceil_ls)/73
          floor_ls = sum(floor_ls)/639
          wall_ls = sum(wall_ls)/606
          wind_ls = sum(wind_ls)/146
          chair_ls = sum(chair_ls)/267
          bed_ls = sum(bed_ls)/152
          sofa_ls = sum(sofa_ls)/138
          table_ls = sum (table_ls)/296
          tvs_ls = sum(tvs_ls)/37
          furn_ls = sum(furn_ls)/529
          objs_ls = sum(objs_ls)/630
          total_mIoU = (ceil_ls+floor_ls+wall_ls+wind_ls+chair_ls+bed_ls+sofa_ls+table_ls+tvs_ls+furn_ls+objs_ls ) /11
          ###########################
          print(f"Average scores for fold-{fold_num}:")
          print(f"AVG loss: {total_loss} | AVG IoU: {total_IoU} | AVG precision: {total_prec} | AVG recall: {total_recall} | AVG mIoU: {total_mIoU}")
          # log metrics to wandb
          #wandb.log({"AVG test Loss": total_loss, "AVG test IoU":total_IoU, "AVG test precision": total_prec, "AVG test recall":total_recall, "AVG test mIoU": total_mIoU})
          #wandb.log({"AVG ceil": ceil_ls, "AVG floor":floor_ls, "AVG wall": wall_ls, "AVG wind":wind_ls, "AVG chair": chair_ls, "AVG bed": bed_ls, "AVG sofa" : sofa_ls, "AVG table": table_ls, "AVG tvs":tvs_ls, "AVG furn": furn_ls ,"AVG objs": objs_ls})
    
    
    # Close this run for wandb
    run.finish()
    fold_num+=1
    print(f"----------------------------------------------------------------------------------------------------------------------------------------------------------------")

# Main Function
def main():

    parse_arguments()
    model_evalauteion()
    
if __name__ == '__main__':
  main()        

