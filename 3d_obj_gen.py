import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from utils.data_setup import SSCData
from utils.file_utils import get_all_preprocessed_prefixes
from model.SSC_networks import get_SSCNet
from model.network import get_res_unet
from utils.visual_utils import voxel_export, obj_export
import argparse


# data paths
GT_PATH = './NYU_gt_pred/'
OUTPUT_PATH = './obj/'
BATCH_SIZE = 1
EXPR_NAME =''
#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 2 # to eliminate DataLoader running slow or even freeze
SAVED_WEIGHTES = ''
######################################################################
def parse_arguments():
    global EXPR_NAME, GT_PATH, OUTPUT_PATH, SAVED_WEIGHTES, MODEL_NAME
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="model name", type=str)
    parser.add_argument("--expr_name", help="experiment name", type=str)
    parser.add_argument("--gt_path", help="gt path of the original data. Default: "+GT_PATH, type=str, default=GT_PATH, required=False)
    parser.add_argument("--output_path", help="output obj path. Default: "+OUTPUT_PATH, type=str, default=OUTPUT_PATH, required=False)
    parser.add_argument('--weights', help='a model weight file to process', metavar='File', type=str)
    args = parser.parse_args()

    #####################################################
    MODEL_NAME = args.model_name
    EXPR_NAME = args.expr_name
    GT_PATH = args.gt_path
    OUTPUT_PATH = args.output_path
    SAVED_WEIGHTES = args.weights
##################################################################

def obj_generation():
  print('Obj path', GT_PATH) 
  v_unit = 0.02
  xs, ys, zs = 60, 36, 60
  file_prefixes = get_all_preprocessed_prefixes(GT_PATH, criteria="*.npz")
  
  df = pd.DataFrame(data={"filename": file_prefixes})
  gt_data = SSCData(df['filename'].tolist())
  
  gt_loader = DataLoader(gt_data, batch_size= 1, shuffle=False,num_workers=NUM_WORKERS)
  
  if  MODEL_NAME == 'ResUNet':
      model_ = get_res_unet()
  
  elif MODEL_NAME == 'SSCNet':
      model_ = get_SSCNet()
  
  #######################################################
  # Make device agnostic code
  dev = 'cuda:0'
  device = torch.device(dev) if torch.cuda.is_available() else torch.device("cpu")
  ######################################################
  
  # Because of parallization on multiple GPUs during the training we need to
  # 1- Load the saved state dict
  state_dict = torch.load(f=SAVED_WEIGHTES)

  # 2- Create a new state dict in which 'module.' prefix is removed
  new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

  # 3- Load the new state dict to your model
  model_.load_state_dict(new_state_dict)
  
  # Send model to GPU
  model_ = model_.to(device)
  
  for batch_num, sample in enumerate(gt_loader):
      x, y, w, m = sample['vox_tsdf'].to(device), sample['vox_lbl'].to(device), sample['vox_weight'].to(device),sample['vox_mask'].to(device), sample['rgb'].to(device)  
            
      out_file = os.path.basename(file_prefixes[batch_num])
      print(f'file: {out_file}')
      
      y = y.argmax(dim=1)
      label= y[0]*w[0]
      print(f"label {label.shape}")
      
      print(torch.unique(label))
      voxel_export(OUTPUT_PATH + MODEL_NAME + '_'+ EXPR_NAME+'_'+ out_file+'.bin', label) 
      obj_export(OUTPUT_PATH + MODEL_NAME + '_'+ EXPR_NAME+'_'+ str(out_file)+'_obj', label, (xs,ys,zs), 0, 0, 0, v_unit, include_top=True,triangular=False,inner_faces=True)
      
      # get the predictions
      model_.eval()
      with torch.inference_mode():   
        pred =  model_(x)
        print(f'pred shape:{pred.shape}')
        pred = pred.argmax(dim=1)
        print(f"predection {pred.shape}")
        label_pred= pred[0]*w[0]
        
        print(f"label_pred shape: {label_pred.shape}")
        print(torch.unique(label_pred))
        
        voxel_export(OUTPUT_PATH + MODEL_NAME + '_'+ EXPR_NAME+'_'+"pred_"+out_file+'.bin', label_pred) 
        obj_export(OUTPUT_PATH + MODEL_NAME + '_'+ EXPR_NAME+'_'+"pred_"+str(out_file)+'_obj', label_pred, (xs,ys,zs), 0, 0, 0, v_unit, include_top=True,triangular=False,inner_faces=True)
      print(f'-------------------------------------------------------------------------------------------------------') 
# Main Function
def main():

    parse_arguments()
    obj_generation()
    
if __name__ == '__main__':
  main()        

