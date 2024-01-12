#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00          # walltime
python 3d_obj_gen.py --model_name ResUNet --expr_name NYUCAD_k3_depth_f3 --gt_path './NYUCAD_gt_pred/' --output_path './obj/' --weights "./saved_models/ResUNet_k3_depthOnly_p15_fold3_2023-12-02.pth"
