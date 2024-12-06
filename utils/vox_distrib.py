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

PREPROC_PATH = './data/NYU_train_preproc/'
def vox_dist():
  
  file_prefixes = get_all_preprocessed_prefixes(PREPROC_PATH, criteria='*.npz')
  df_t = pd.DataFrame(data={"filename": file_prefixes})
  train_ds = SSCData(df_t['filename']) 

  print(f'Items in train_dataset ',len(train_ds))

  train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,num_workers=1)
  occl_emp_count=0
  
  #######################################################
  # Make device agnostic code
  dev = 'cuda:0'
  device = torch.device(dev) if torch.cuda.is_available() else torch.device("cpu")
  ######################################################
  
  for batch_num, sample in enumerate(train_loader):
      x, y, w, m = sample['vox_tsdf'].to(device), sample['vox_lbl'].to(device), sample['vox_weight'].to(device),sample['vox_mask'].to(device) # x (input)-> tsdf, y(target)-> gt_labels, w-> weights of occluded and occupied regions =1 while other voxels =0. m-> masks of the occluded and occupied regions
# Main Function
      # Count the number of elements
      occl_emp_count += (m == 0.25).sum() # occluded empty
   
  print (f' occl_emp_count: {occl_emp_count}')

  
def main():
    vox_dist()
    
if __name__ == '__main__':
  main()        
