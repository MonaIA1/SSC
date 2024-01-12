# 3D Semantic Scene Completion from a Depth Map with Unsupervised Learning for Semantics Prioritisation
![SSC results from depth maps only by NYUv2 dataset. The 3D models displayed from different viewpoints to illustrate the scene completion in the occluded regions and scene semantics. Each object represented by a unique color and circles show the main differences between GT and the predictions by SSCNet model and ours.]
([/path/to/your/image](https://github.com/MonaIA1/SSC/blob/master/images/OursVSSSCNet.png))

The following demo shows the predicted 3D models from the NYUCAD dataset, comparing scenarios where our proposed weighting method was applied versus when they were not.
[Demo on NYUCAD datset](https://drive.google.com/file/d/17LIwaj8-4rG7cBkZ-pSWsmvu5Q68rvQc/view?usp=sharing)

This is the setup environment, dataset preperation and the source code for reproducing the results of the paper, available in:
Please cite the paper if part of the code is used for anypurpose.

## Setup
### Hardware Requirement
At least one NVIDIA GPU 
### Environmet Setup
- conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
- conda install -c conda-forge tqdm
- conda install -c conda-forge tensorboard
- conda install -c conda-forge scikit-image
- conda install -c conda-forge scikit-learn
- conda install -c conda-forge pandas
- conda install -c conda-forge numpy
- conda install -c conda-forge opencv

## Datasets
To download the NYU dataset depth maps and the 3D ground truth semantics, we use those used by SSCNet paper available in: https://github.com/shurans/sscnet/blob/master/download_data.sh
               
         NYUtrain 
                    |-- xxxxx_0000.png // depth map
                    |-- xxxxx_0000.bin //3D Ground Truth
         NYUtest
                    |-- xxxxx_0000.png 
                    |-- xxxxx_0000.bin
        
    
         NYUCADtrain 
                    |-- xxxxx_0000.png
                    |-- xxxxx_0000.bin 
         NYUCADtest
                    |-- xxxxx_0000.png
                    |-- xxxxx_0000.bin
### Data Preperation
To generate 3D volumes for train and evaluate the SSC model:

1. compile the CUDA code: 
   `nvcc -std=c++11 --ptxas-options=-v --compiler-options '-fPIC' -o lib_preproc.so --shared lib_preproc.cu`
   
2. To process the 3D volumes use `run_job.sh`, or run:
   `python preproc_tsdf.py --base_path './directory_path_to_be_processed' --dest_path './directory_path_of_processed_data'`

   Ex: `python preproc_tsdf.py --base_path './data/NYUtest' --dest_path './data/NYU_test_preproc'`

### SSC Training
For training the 3D model, use `run_t_job.sh`, or run:  

`python train.py --dataset NYU --model_name ResUNet --expr_name xxxx --train_batch_size 4 --val_batch_size 2 --base_lr 0.01 --decay 0.0005 --epochs 100`  

(check help --help for additional information)

### SSC Evaluaion
For the 3D model evaluation, use `run_e_job.sh`, or run:  

`python evaluation.py --dataset NYUCAD --model_name ResUNet --expr_name xxxx --weights "./saved_models/path1.pth" "./saved_models/path2.pth" "./saved_models/path3.pth" `  


### 3D SSC Object Generation
To generate the 3D scenes for GT and predictions. Use `run_obj_job.sh`, or run:  

`python 3d_obj_gen.py --model_name ResUNet --expr_name xxxx --gt_path './NYU_gt_pred/' --output_path './obj/' --weights "./saved_models/path1.pth"`
