#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00          # walltime
python 3d_obj_gen.py --model_name SSCNet --expr_name 3_60_fold2 --gt_path './NYU_gt_pred/' --output_path './obj/' --weights "./saved_models/SSCNet_3_60_fold2_2023-08-28.pth"