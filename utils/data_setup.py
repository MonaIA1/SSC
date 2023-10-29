
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torchvision import transforms
import glob
import cv2
from PIL import Image

class SSCData(Dataset):
    def __init__(self, file_prefixes):
      
        self.file_prefixes = file_prefixes
        
    def __len__(self):
        return len(self.file_prefixes)

    def __getitem__(self, index):
        file_prefix = self.file_prefixes[index]
        npz_file = file_prefix + '.npz'
        loaded = np.load(npz_file)
        
        # turn the 3D data to pytorch tensor format
        x_batch = torch.from_numpy(loaded['tsdf']).reshape( 1, 240, 144, 240)
        y_batch = torch.from_numpy(loaded['lbl']).long()
        y_batch = one_hot(y_batch, num_classes=12).float().permute(3, 0, 1, 2).float()
        w_batch = torch.from_numpy(loaded['weights']).float()
        m_batch = torch.from_numpy(loaded['masks']).float()
        #d_batch = torch.from_numpy(loaded['mapping'])
        sample = {
            'vox_tsdf': x_batch,
            'vox_lbl': y_batch,
            'vox_weight': w_batch, # resampling
            'vox_mask':  m_batch,
            #'vox_depth': d_batch,
        }

        return sample

