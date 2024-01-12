#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00          # walltime
python preproc_tsdf.py --base_path './data/NYUCADtrain' --dest_path './data/NYUCAD_train_preproc'
