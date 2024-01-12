#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00          # walltime

python evaluation.py --dataset NYUCAD --model_name ResUNet --expr_name k3_depthOnly_p15 --weights "./saved_models/ResUNet_k3_depthOnly_p15_fold1_2023-12-01.pth" "./saved_models/ResUNet_k3_depthOnly_p15_fold2_2023-12-02.pth" "./saved_models/ResUNet_k3_depthOnly_p15_fold3_2023-12-02.pth" 
