
import os
from utils.file_utils import *
import numpy as np # linear algebra
import multiprocessing
import argparse
import time
import random
from utils.py_cuda import *

##################################
# Parameters
##################################

########################################################################################
####################################### NYUCAD Dataset #################################
#BASE_PATH = './data/NYUCADtrain'
#DEST_PATH = './data/NYUCAD_train_preproc'
#####################################
BASE_PATH = './data/NYUCADtest'
DEST_PATH = './data/NYUCAD_test_preproc'
#####################################

#######################################################################################
####################################### NYU Dataset ###################################
#BASE_PATH = './data/NYUtrain'
#DEST_PATH = './data/NYU_train_preproc'
#####################################
#BASE_PATH = './data/NYUtest'
#DEST_PATH = './data/NYU_test_preproc'
####################################################################################

DEVICE = 0
THREADS = 4

# Globals
proc_prefixes = multiprocessing.Value('i',0)
total_prefixes = 0
processed = 0


##################################

def parse_arguments():
    global BASE_PATH, DATASET, DEST_PATH, DEVICE, THREADS

    print("\nSSC PREPROCESSOR\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", help="Base path of the original data. Default: "+BASE_PATH, type=str, default=BASE_PATH, required=False)
    parser.add_argument("--dest_path", help="Destination path. Default: "+DEST_PATH, type=str, default=DEST_PATH, required=False)
    parser.add_argument("--device", help="CUDA device. Default " + str(DEVICE), type=int, default=DEVICE, required=False)
    parser.add_argument("--threads", help="Concurrent threads. Default " + str(THREADS), type=int, default=THREADS, required=False)
    args = parser.parse_args()

    DATASET = args.dataset
    BASE_PATH = args.base_path
    DEST_PATH = args.dest_path
    DEVICE = args.device
    THREADS = args.threads
    
    
def process_multi(depth_prefixes):
    global BASE_PATH, DATASET, DEST_PATH, DEVICE, THREADS, to_process, proc_prefixes, total_time, processed
    
    file_prefix = depth_prefixes
    
    lib_sscnet_setup(device=DEVICE, num_threads=128, K=None, frame_shape=(640, 480), v_unit=0.02, v_margin=0.24)
    
    dest_prefix = DEST_PATH + file_prefix[len(BASE_PATH):]
    preproc_file = dest_prefix + '.npz'

    directory = os.path.split(dest_prefix)[0]

    with proc_prefixes.get_lock():
        if not os.path.exists(directory):
            os.makedirs(directory)

    shape = (240, 144, 240)
    
    ################################################################################################
    vox_tsdf, segmentation_label, vox_weights, vox_masks, depth_mapping = process(file_prefix, voxel_shape=shape, down_scale=4)
    ################################################################################################
    
    down_shape=(shape[0]//4, shape[1]//4, shape[2]//4 )
    
    if np.sum(segmentation_label) == 0:
      print(file_prefix)
      exit(-1)
    else:
     ##########################################################################################################
      mapping = depth_mapping.astype(np.int64)
      
      np.savez_compressed(preproc_file,
                           tsdf=vox_tsdf.reshape((shape[0], shape[1], shape[2], 1)),
                           lbl=segmentation_label.reshape((down_shape[0], down_shape[1], down_shape[2])),
                           weights= vox_weights.reshape((down_shape[0], down_shape[1], down_shape[2])),
                           masks= vox_masks.reshape((down_shape[0], down_shape[1], down_shape[2] )),
                           mapping = mapping.reshape((shape[0]*shape[1]* shape[2], 1))) 
      
      
   ##################################################################################################################
    with proc_prefixes.get_lock():
        proc_prefixes.value += 1
        
        counter = proc_prefixes.value
        mean_time = (time.time() - total_time)/counter
        eta_time = mean_time * (to_process - counter)
        eta_h = eta_time // (60*60)
        eta_m = (eta_time - (eta_h*60*60))//60
        eta_s = eta_time - (eta_h*60*60) - eta_m * 60
        perc = 100 * counter/to_process

        print("  %3.2f%%  Mean Processing Time: %.2f seconds ETA: %02d:%02d:%02d     " % (perc, mean_time, eta_h, eta_m, eta_s), end="\r")

# Main Function
def Run():
    global BASE_PATH, DATASET, DEST_PATH, DEVICE, THREADS, to_process, proc_prefixes, total_time
    
    print(" .")
    
    data_path = os.path.join(BASE_PATH)
    
    print("Data path %s" % data_path)
    depth_prefixes  = get_file_prefixes_from_path(data_path)
    
    print(" ..")

    total_time = time.time()
 
    if to_process>0:
        pool = multiprocessing.Pool(processes=THREADS) 
        pool.map(process_multi, depth_prefixes)
        pool.close()
        pool.join()

    print("Total time: %s seconds                                                       " % (time.time() - total_time))
       
if __name__ == '__main__':
  Run()



