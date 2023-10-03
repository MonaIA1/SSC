
import ctypes
import numpy as np
import cv2
import os

####################################################################
### image resolution
frame_width = 640 # in pixels
frame_height = 480
####################################################################
def get_segmentation_class_map():
    return np.array([0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11,
                                   11, 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11], dtype=np.int32)
def get_class_names():
    return ["ceil.", "floor", "wall ", "wind.", "chair", "bed  ", "sofa ", "table", "tvs  ", "furn.", "objs."]


#nvcc --ptxas-options=-v --compiler-options '-fPIC' -o lib_preproc.so --shared lib_preproc.cu (old)
#nvcc -std=c++11 --ptxas-options=-v --compiler-options '-fPIC' -o lib_preproc.so --shared lib_preproc.cu


_lib = ctypes.CDLL('src/lib_preproc.so')


_lib.Process.argtypes = (ctypes.c_char_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_int,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p,
                              ctypes.c_void_p, # for depth_mapping_3d
                              ctypes.c_void_p, 
                              #ctypes.c_void_p
                              )


_lib.setup.argtypes = (ctypes.c_int,
              ctypes.c_int,
              ctypes.c_void_p,
              ctypes.c_int,
              ctypes.c_int,
              ctypes.c_float,
              ctypes.c_float)



def lib_sscnet_setup(device=0, num_threads=1024, K=None, frame_shape=(640, 480), v_unit=0.02, v_margin=0.24, debug=0):

    global _lib
    frame_width = frame_shape[0]
    frame_height = frame_shape[1]

    if K is None:
        K = np.array([518.8579, 0.0, frame_width / 2.0, 0.0, 518.8579, frame_height / 2.0, 0.0, 0.0, 1.0],dtype=np.float32) # camera intrinsic parameters

    _lib.setup(ctypes.c_int(device),
                  ctypes.c_int(num_threads),
                  K.ctypes.data_as(ctypes.c_void_p),
                  ctypes.c_int(frame_width),
                  ctypes.c_int(frame_height),
                  ctypes.c_float(v_unit),
                  ctypes.c_float(v_margin),
                  ctypes.c_int(debug)
               )

def process(file_prefix, voxel_shape, down_scale = 4):
    global _lib
    

    vox_origin = np.ones(3,dtype=np.float32)
    cam_pose = np.ones(16,dtype=np.float32)
    num_voxels = voxel_shape[0] * voxel_shape[1] * voxel_shape[2]
    vox_size = np.array([voxel_shape[0], voxel_shape[1], voxel_shape[2]], dtype=np.int32)
    segmentation_class_map = get_segmentation_class_map()
    segmentation_label = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.int32)

    vox_weights = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)
    vox_masks = np.zeros(num_voxels//(down_scale*down_scale*down_scale), dtype=np.float32)
    ##################
    depth_mapping = np.ones(num_voxels, dtype=np.float32) * (-1)
    ##################
    depth_image = cv2.imread(file_prefix+'.png', cv2.IMREAD_ANYDEPTH)
    vox_tsdf = np.zeros(num_voxels, dtype=np.float32)
   
    


    _lib.Process(ctypes.c_char_p(bytes(file_prefix+'.bin','utf-8')),
                      cam_pose.ctypes.data_as(ctypes.c_void_p),
                      vox_size.ctypes.data_as(ctypes.c_void_p),
                      vox_origin.ctypes.data_as(ctypes.c_void_p),
                      ctypes.c_int(down_scale),
                      segmentation_class_map.ctypes.data_as(ctypes.c_void_p),
                      depth_image.ctypes.data_as(ctypes.c_void_p),
                      vox_tsdf.ctypes.data_as(ctypes.c_void_p),
                      vox_weights.ctypes.data_as(ctypes.c_void_p),
                      vox_masks.ctypes.data_as(ctypes.c_void_p),
                      #################################################
                      depth_mapping.ctypes.data_as(ctypes.c_void_p),
                      #################################################
                      segmentation_label.ctypes.data_as(ctypes.c_void_p)
                     
                      
                      ) 
    ##########################################################################                                 
    return vox_tsdf, segmentation_label, vox_weights, vox_masks, depth_mapping
    
    #########################################################################

