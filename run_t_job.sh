#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --mem=100GB
#SBATCH --time=60:00:00          # walltime
export CUDA_VISIBLE_DEVICES=0,1,2

python train.py --dataset NYU --model_name ResUNet --expr_name WBCE --train_batch_size 4 --val_batch_size 2 --base_lr 0.01 --decay 0.0005 --epochs 60
