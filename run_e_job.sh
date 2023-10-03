#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00          # walltime

python evaluation.py --dataset NYUCAD --model_name ResUNet --expr_name 3NYUCAD_WCE_BalancedClusters_60 --weights "./saved_models/ResUNet_3NYUCAD_WCE_BalancedClusters_60_fold1_2023-08-21.pth" "./saved_models/ResUNet_3NYUCAD_WCE_BalancedClusters_60_fold2_2023-08-21.pth" "./saved_models/ResUNet_3NYUCAD_WCE_BalancedClusters_60_fold3_2023-08-21.pth" 