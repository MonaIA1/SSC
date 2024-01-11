
/*  
    Data preperation
    Author: Mona Alawadh (Sept, 2023)
    *Original Caffe Code: Shuran Song (https://github.com/shurans/sscnet)
    ** https://gitlab.com/UnBVision/spawn/-/blob/main/src/lib_preproc.cu?ref_type=heads
    to compile use:
    nvcc -std=c++11 --ptxas-options=-v --compiler-options '-fPIC' -o lib_preproc.so --shared lib_preproc.cu  
*/

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

typedef high_resolution_clock::time_point clock_tick;

// Camera information
int frame_width = 640; // in pixels
int frame_height = 480;
float vox_unit = 0.02;
float vox_margin = 0.24;
int NUM_THREADS = 128;
int DEVICE = 0;
float* parameters_GPU;
int debug = 0;

#define NUM_CLASSES (256)
#define MAX_DOWN_SIZE (1000)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


float* cam_K; //cam_K = np.array([518.8579, 0.0, frame_width / 2.0, 0.0, 518.8579, frame_height / 2.0, 0.0, 0.0, 1.0],dtype=np.float32) # camera intrinsic parameters
float cam_info[27];

float* create_parameters_GPU() {

    float parameters[13];
    for (int i = 0; i < 9; i++)
        parameters[i] = cam_K[i];
    parameters[9] = frame_width;
    parameters[10] = frame_height;
    parameters[11] = vox_unit;
    parameters[12] = vox_margin;

    float* parameters_GPU;

    cudaMalloc(&parameters_GPU, 13 * sizeof(float));
    cudaMemcpy(parameters_GPU, parameters, 13 * sizeof(float), cudaMemcpyHostToDevice);

    return (parameters_GPU);

}

clock_tick start_timer() {
    return (high_resolution_clock::now());
}

void end_timer(clock_tick t1, const char msg[]) {
    if (debug == 1) {
        clock_tick t2 = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(t2 - t1).count();
        printf("%s: %ld(ms)\n", msg, duration);
    }
}


void setup_CPP(int device, int num_threads, float* K, int fw, int fh, float v_unit, float v_margin, int debug_flag) {
    cam_K = K;
    DEVICE = device;
    NUM_THREADS = num_threads;
    frame_width = fw; // in pixels
    frame_height = fh;
    vox_unit = v_unit;
    vox_margin = v_margin;

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, DEVICE);
    cudaSetDevice(DEVICE);

    parameters_GPU = create_parameters_GPU();

    if (debug_flag == 1) {

        printf("\nUsing GPU: %s - (device %d)\n", deviceProperties.name, DEVICE);
        printf("Threads per block: %d\n", NUM_THREADS);
    }

    debug = debug_flag;

}


__device__
void get_parameters_GPU(float* parameters_GPU,
    float** cam_K_GPU, int* frame_width_GPU, int* frame_height_GPU,
    float* vox_unit_GPU, float* vox_margin_GPU) {
    *cam_K_GPU = parameters_GPU;
    *frame_width_GPU = int(parameters_GPU[9]);
    *frame_height_GPU = int(parameters_GPU[10]);
    *vox_unit_GPU = parameters_GPU[11];
    *vox_margin_GPU = parameters_GPU[12];
}


void destroy_parameters_GPU(float* parameters_GPU) {

    cudaFree(parameters_GPU);

}

__device__
float modeLargerZero(const int* values, int size) {
    int count_vector[NUM_CLASSES] = { 0 };

    for (int i = 0; i < size; ++i)
        if (values[i] > 0)
            count_vector[values[i]]++;

    int md = 0;
    int freq = 0;

    for (int i = 0; i < NUM_CLASSES; i++)
        if (count_vector[i] > freq) {
            freq = count_vector[i];
            md = i;
        }
    return md;
}

// find mode 
__device__
float mode(const int* values, int size) {
    int count_vector[NUM_CLASSES] = { 0 };

    for (int i = 0; i < size; ++i)
        count_vector[values[i]]++;

    int md = 0;
    int freq = 0;

    for (int i = 0; i < NUM_CLASSES; i++)
        if (count_vector[i] > freq) {
            freq = count_vector[i];
            md = i;
        }
    return md;
}

__global__
void Downsample_Kernel(int* in_vox_size, int* out_vox_size,
    int* in_labels, float* in_tsdf, unsigned char* in_grid_GPU,
    int* out_labels, float* out_tsdf,
    int label_downscale, unsigned char* out_grid_GPU) {

    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (vox_idx >= out_vox_size[0] * out_vox_size[1] * out_vox_size[2]) {
        return;
    }

    int down_size = label_downscale * label_downscale * label_downscale;

    int emptyT = int((0.95 * down_size)); //Empty Threshold

    int z = vox_idx / (out_vox_size[0] * out_vox_size[1]);
    int y = (vox_idx - (z * out_vox_size[0] * out_vox_size[1])) / out_vox_size[0];
    int x = vox_idx - (z * out_vox_size[0] * out_vox_size[1]) - (y * out_vox_size[0]);

    int label_vals[MAX_DOWN_SIZE] = { 0 };
    int count_vals = 0;
    float tsdf_val = 0.0f;

    int zero_count = 0;
    int zero_surface_count = 0;

    for (int tmp_x = x * label_downscale; tmp_x < (x + 1) * label_downscale; ++tmp_x) {
        for (int tmp_y = y * label_downscale; tmp_y < (y + 1) * label_downscale; ++tmp_y) {
            for (int tmp_z = z * label_downscale; tmp_z < (z + 1) * label_downscale; ++tmp_z) {
                int tmp_vox_idx = tmp_z * in_vox_size[0] * in_vox_size[1] + tmp_y * in_vox_size[0] + tmp_x;
                label_vals[count_vals] = int(in_labels[tmp_vox_idx]);
                count_vals += 1;

                if (in_labels[tmp_vox_idx] == 0 || in_labels[tmp_vox_idx] == 255) {

                    zero_count++;
                }

                if (in_grid_GPU[tmp_vox_idx] == 0 || in_labels[tmp_vox_idx] == 255) {
                    zero_surface_count++;
                }

                tsdf_val += in_tsdf[tmp_vox_idx];

            }
        }
    }


    if (zero_count > emptyT) {
        out_labels[vox_idx] = float(mode(label_vals, down_size));
    }
    else {
        out_labels[vox_idx] = float(modeLargerZero(label_vals, down_size)); // object label mode without zeros
    }

    if (zero_surface_count > emptyT) {
        out_grid_GPU[vox_idx] = 0;
    }
    else {
        out_grid_GPU[vox_idx] = 1;
    }

    out_tsdf[vox_idx] = tsdf_val / down_size;

}

void ReadVoxLabel_CPP(const std::string& filename,
    float* vox_origin,
    float* cam_pose,
    int* vox_size,
    int* segmentation_class_map,
    int* segmentation_label_fullscale)
{

    //downsample lable
    clock_tick t1 = start_timer();

    // Open file
    std::ifstream fid(filename, std::ios::binary);

    end_timer(t1, "open");


    // Read voxel origin in world coordinates
    for (int i = 0; i < 3; ++i) {
        fid.read((char*)&vox_origin[i], sizeof(float));
    }
    end_timer(t1, "origin");

    // Read camera pose
    for (int i = 0; i < 16; ++i) {
        fid.read((char*)&cam_pose[i], sizeof(float)); //cam_pose is 4x4 transformation matrix contains rotation and translation: [[[ 0.99320972  0.02637034 - 0.11330903  0. ][0.11620127 - 0.27193201  0.95527476  0.][-0.00562143 - 0.96195489 - 0.27314997  1.23215246][0.          0.          0.          1.]] ]
    }
    end_timer(t1, "pose");

    // Read voxel label data from file (RLE compression)
    std::vector<unsigned int> scene_vox_RLE;
    while (!fid.eof()) {
        int tmp;
        fid.read((char*)&tmp, sizeof(int));
        if (!fid.eof())
            scene_vox_RLE.push_back(tmp);
    }
    end_timer(t1, "read");


    // Reconstruct voxel label volume from RLE
    int vox_idx = 0;
    //int object_count=0;
    for (size_t i = 0; i < scene_vox_RLE.size() / 2; ++i) {
        unsigned int vox_val = scene_vox_RLE[i * 2];
        unsigned int vox_iter = scene_vox_RLE[i * 2 + 1];

        for (size_t j = 0; j < vox_iter; ++j) {
            if (vox_val == 255) {                        //255: Out of view frustum
                segmentation_label_fullscale[vox_idx] = 255; //12 classes 0 - 11 + 12=Outside room
            }
            else {
                segmentation_label_fullscale[vox_idx] = segmentation_class_map[vox_val];
            }
            vox_idx++;
        }
    }

    end_timer(t1, "voxel");
}


void DownsampleLabel_CPP(int* vox_size,
    int out_scale,
    int* segmentation_label_fullscale,
    float* vox_tsdf_fullscale,
    int* segmentation_label_downscale,
    float* vox_weights,
    float* vox_masks,
    /////////////////////////////////////////
    //float *vox_grid,
    unsigned char* vox_grid,
    /////////////////////////////////////////
    int* segmentation_class_map) {

    //downsample lable
    clock_tick t1 = start_timer();

    int num_voxels_in = vox_size[0] * vox_size[1] * vox_size[2];
    int label_downscale = 4; //out_scale
    int num_voxels_down = num_voxels_in / (label_downscale * label_downscale * label_downscale);
    int out_vox_size[3];
    //////////////////////////////////////////////////////////
    float* vox_tsdf_down = new float[num_voxels_down];
    //float *vox_grid_downscale = new float[num_voxels_down];
    unsigned char* vox_grid_downscale = new unsigned char[num_voxels_down];
    //////////////////////////////////////////////////////
    out_vox_size[0] = vox_size[0] / label_downscale;
    out_vox_size[1] = vox_size[1] / label_downscale;
    out_vox_size[2] = vox_size[2] / label_downscale;

    int* in_vox_size_GPU;
    int* out_vox_size_GPU;
    int* in_labels_GPU;
    int* out_labels_GPU;
    float* in_tsdf_GPU;
    float* out_tsdf_GPU;
    unsigned char* in_grid_GPU;
    unsigned char* out_grid_GPU;

    cudaMalloc(&in_vox_size_GPU, 3 * sizeof(int));
    cudaMalloc(&out_vox_size_GPU, 3 * sizeof(int));
    cudaMalloc(&in_labels_GPU, num_voxels_in * sizeof(int));
    cudaMalloc(&in_tsdf_GPU, num_voxels_in * sizeof(float));
    cudaMalloc(&in_grid_GPU, num_voxels_in * sizeof(unsigned char));
    cudaMalloc(&out_labels_GPU, num_voxels_down * sizeof(int));
    cudaMalloc(&out_tsdf_GPU, num_voxels_down * sizeof(float));
    cudaMalloc(&out_grid_GPU, num_voxels_down * sizeof(unsigned char));

    cudaMemcpy(in_vox_size_GPU, vox_size, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_vox_size_GPU, out_vox_size, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_labels_GPU, segmentation_label_fullscale, num_voxels_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(in_tsdf_GPU, vox_tsdf_fullscale, num_voxels_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(in_grid_GPU, vox_grid, num_voxels_in * sizeof(unsigned char), cudaMemcpyHostToDevice);


    int BLOCK_NUM = int((num_voxels_down + size_t(NUM_THREADS) - 1) / NUM_THREADS);

    Downsample_Kernel << < BLOCK_NUM, NUM_THREADS >> > (in_vox_size_GPU, out_vox_size_GPU,
        in_labels_GPU, in_tsdf_GPU, in_grid_GPU,
        out_labels_GPU, out_tsdf_GPU,
        label_downscale, out_grid_GPU);

    cudaDeviceSynchronize();

    end_timer(t1, "Downsample duration");

    cudaMemcpy(segmentation_label_downscale, out_labels_GPU, num_voxels_down * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vox_tsdf_down, out_tsdf_GPU, num_voxels_down * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vox_grid_downscale, out_grid_GPU, num_voxels_down * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    //cudaMemcpy(segmentation_label_fullscale, in_labels_GPU, num_voxels_in * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(in_vox_size_GPU);
    cudaFree(out_vox_size_GPU);
    cudaFree(in_labels_GPU);
    cudaFree(out_labels_GPU);
    cudaFree(in_tsdf_GPU);
    cudaFree(out_tsdf_GPU);
    cudaFree(in_grid_GPU);
    cudaFree(out_grid_GPU);


    // Find number of occluded and occupied voxels
    // Save voxel indices of background

    std::vector<int> bg_voxel_idx;
    std::vector<int> occl_voxel_idx; // occluded_occupied var


    memset(vox_weights, 0, num_voxels_down * sizeof(float));
    memset(vox_masks, 0, num_voxels_down * sizeof(float));


    //calculate the label weights
    for (int i = 0; i < num_voxels_down; ++i) {

        if ((segmentation_label_downscale[i]) > 0 && (segmentation_label_downscale[i] < 255)) { //Occupied voxels in the room
            vox_weights[i] = 1; //occupied voxels

            if (vox_tsdf_down[i] < -0.5) {
                vox_masks[i] = 1; // occupied under surface
                occl_voxel_idx.push_back(i);
            }
            else {
                vox_masks[i] = 0.5; //occupied on surface
            }
        }

        else if ((vox_tsdf_down[i] < -0.5) && (segmentation_label_downscale[i] < 255)) {
            bg_voxel_idx.push_back(i); // background voxels in unobserved region in the room (empty occluded voxels)
        }

        else if (segmentation_label_downscale[i] == 255) {  //outside room
            segmentation_label_downscale[i] = 0;
        }

    }
    // Raise the weight for a few indices of background voxels
    std::random_device tmp_rand_rd; //create a random_device object to generate random seeds
    std::mt19937 tmp_rand_mt(tmp_rand_rd()); //create an mt19937 random number generator, seeded with a value from the random_device
    int segcout = 0;
    int sample_obj_ratio = 2;
    int segtotal = floor(sample_obj_ratio * occl_voxel_idx.size());
    if (bg_voxel_idx.size() > 0) {
        std::uniform_real_distribution<double> tmp_rand_dist(0, (float)(bg_voxel_idx.size()) - 0.0001);
        for (int i = 0; i < occl_voxel_idx.size(); ++i) {
            int rand_idx = (int)(std::floor(tmp_rand_dist(tmp_rand_mt)));

            if (segcout < segtotal && segmentation_label_downscale[bg_voxel_idx[rand_idx]] < 255) {
                // background voxels within room 
                vox_weights[bg_voxel_idx[rand_idx]] = 1;
                vox_masks[bg_voxel_idx[rand_idx]] = 0.25;
                segcout++;
            }
        }
    }

    end_timer(t1, "Downsample duration + copy");

    delete[] vox_tsdf_down;
    delete[] vox_grid_downscale;


}


__global__
void depth2Grid(float* cam_pose, int* vox_size, float* vox_origin, float* depth_data, unsigned char* vox_grid,
    //float *vox_grid, 
    ////////////////////////////////////////////////
    float* depth_mapping_3d_GPU,
    ////////////////////////////////////////////////     
    float* parameters_GPU) {


    float* cam_K_GPU;
    int frame_width_GPU, frame_height_GPU;
    float vox_unit_GPU, vox_margin_GPU;

    get_parameters_GPU(parameters_GPU, &cam_K_GPU, &frame_width_GPU, &frame_height_GPU,
        &vox_unit_GPU, &vox_margin_GPU);


    // Get point in world coordinate
    // conver from image corrdinate (point_depth) --> camera coordinate (point_cam) --> world coordinate (point_base)
    int pixel_x = blockIdx.x;
    int pixel_y = threadIdx.x;

    if (pixel_x >= frame_width_GPU || pixel_y >= frame_height_GPU)
        return;



    float point_depth = depth_data[pixel_y * frame_width_GPU + pixel_x];

    float point_cam[3] = { 0 };
    point_cam[0] = (pixel_x - cam_K_GPU[2]) * point_depth / cam_K_GPU[0]; //cam_K = np.array([518.8579, 0.0, frame_width / 2.0, 0.0, 518.8579, frame_height / 2.0, 0.0, 0.0, 1.0], dtype = np.float32) # camera intrinsic parameters
    point_cam[1] = (pixel_y - cam_K_GPU[5]) * point_depth / cam_K_GPU[4];
    point_cam[2] = point_depth;

    float point_base[3] = { 0 };

    point_base[0] = cam_pose[0 * 4 + 0] * point_cam[0] + cam_pose[0 * 4 + 1] * point_cam[1] + cam_pose[0 * 4 + 2] * point_cam[2];
    point_base[1] = cam_pose[1 * 4 + 0] * point_cam[0] + cam_pose[1 * 4 + 1] * point_cam[1] + cam_pose[1 * 4 + 2] * point_cam[2];
    point_base[2] = cam_pose[2 * 4 + 0] * point_cam[0] + cam_pose[2 * 4 + 1] * point_cam[1] + cam_pose[2 * 4 + 2] * point_cam[2];

    point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
    point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
    point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];


    // World coordinate into grid 
    int z = (int)floor((point_base[0] - vox_origin[0]) / vox_unit_GPU);
    int x = (int)floor((point_base[1] - vox_origin[1]) / vox_unit_GPU);
    int y = (int)floor((point_base[2] - vox_origin[2]) / vox_unit_GPU);

    // mark vox_out with 1.0
    if (x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]) {
        int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
        vox_grid[vox_idx] = 1;
        //vox_grid[vox_idx] = float(1.0);
        depth_mapping_3d_GPU[pixel_y * frame_width_GPU + pixel_x] = vox_idx;

    }
}


__global__
void SquaredDistanceTransform(float* cam_pose, int* vox_size, float* vox_origin, float* depth_data, unsigned char* vox_grid,
    //float *vox_grid, 
    float* vox_tsdf, float* parameters_GPU) {

    float* cam_K_GPU = parameters_GPU;
    int frame_width_GPU = int(parameters_GPU[9]), frame_height_GPU = int(parameters_GPU[10]);
    float vox_unit_GPU = parameters_GPU[11], vox_margin_GPU = parameters_GPU[12];


    int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (vox_idx >= vox_size[0] * vox_size[1] * vox_size[2]) {
        return;
    }


    int z = float((vox_idx / (vox_size[0] * vox_size[1])) % vox_size[2]);
    int y = float((vox_idx / vox_size[0]) % vox_size[1]);
    int x = float(vox_idx % vox_size[0]);
    int search_region = (int)(vox_margin_GPU / vox_unit_GPU);

    if (vox_grid[vox_idx] > 0) {
        vox_tsdf[vox_idx] = 0; //0
        return;
    }

    // Compute the 3D point in world coordinates and transform it into the camera's coordinate system.

    // Get point in world coordinates
    float point_base[3] = { 0 };
    point_base[0] = float(z) * vox_unit_GPU + vox_origin[0];
    point_base[1] = float(x) * vox_unit_GPU + vox_origin[1];
    point_base[2] = float(y) * vox_unit_GPU + vox_origin[2];

    // Get point in current camera coordinates
    // Check if the point's depth is within a valid range.
    float point_cam[3] = { 0 };
    point_base[0] = point_base[0] - cam_pose[0 * 4 + 3];
    point_base[1] = point_base[1] - cam_pose[1 * 4 + 3];
    point_base[2] = point_base[2] - cam_pose[2 * 4 + 3];
    point_cam[0] = cam_pose[0 * 4 + 0] * point_base[0] + cam_pose[1 * 4 + 0] * point_base[1] + cam_pose[2 * 4 + 0] * point_base[2];
    point_cam[1] = cam_pose[0 * 4 + 1] * point_base[0] + cam_pose[1 * 4 + 1] * point_base[1] + cam_pose[2 * 4 + 1] * point_base[2];
    point_cam[2] = cam_pose[0 * 4 + 2] * point_base[0] + cam_pose[1 * 4 + 2] * point_base[1] + cam_pose[2 * 4 + 2] * point_base[2];
    if (point_cam[2] <= 0) {
        return;
    }
    // Project point to 2D
    // project the 3D point to 2D image coordinates and check if it is within the frame boundaries.
    int pixel_x = roundf(cam_K_GPU[0] * (point_cam[0] / point_cam[2]) + cam_K_GPU[2]);
    int pixel_y = roundf(cam_K_GPU[4] * (point_cam[1] / point_cam[2]) + cam_K_GPU[5]);
    if (pixel_x < 0 || pixel_x >= frame_width_GPU || pixel_y < 0 || pixel_y >= frame_height_GPU) { // outside FOV
        return;
    }

    // Get depth
    //retrieve the depth value for the projected 2D point and check if it is within a valid range; if not, set the TSDF value to -1.
    float point_depth = depth_data[pixel_y * frame_width_GPU + pixel_x];
    if (point_depth < float(0.5f) || point_depth > float(8.0f))
    {
        return;
    }
    if (roundf(point_depth) == 0) { // mising depth
        vox_tsdf[vox_idx] = float(-1.0);
        return;
    }

    // Get depth difference between the measured depth value from the depth image (point_depth) and the depth value of the transformed 3D point in the camera coordinate system (point_cam[2])
    // sign value will be used when updating the TSDF values based on the search region. It provides information about whether the current voxel is in front of or behind the object's surface 
    float sign;
    if (abs(point_depth - point_cam[2]) < 0.0001) {
        sign = 1; // avoid NaN
    }
    else {
        sign = (point_depth - point_cam[2]) / abs(point_depth - point_cam[2]);
    }

    vox_tsdf[vox_idx] = sign;

    //compute the minimum TSDF value for the current voxel within the search region, considering the occupied voxels in the grid
    for (int iix = max(0, x - search_region); iix < min((int)vox_size[0], x + search_region + 1); iix++) {
        for (int iiy = max(0, y - search_region); iiy < min((int)vox_size[1], y + search_region + 1); iiy++) {
            for (int iiz = max(0, z - search_region); iiz < min((int)vox_size[2], z + search_region + 1); iiz++) {

                int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
                if (vox_grid[iidx] > 0) {

                    float xd = abs(x - iix);
                    float yd = abs(y - iiy);
                    float zd = abs(z - iiz);
                    float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd) / (float)search_region;
                    if (tsdf_value < abs(vox_tsdf[vox_idx])) {
                        vox_tsdf[vox_idx] = float(tsdf_value * sign);
                    }
                }
            }
        }
    }
}

void getDepthData_cpp(unsigned char* depth_image, float* depth_data) {
    unsigned short depth_raw;
    for (int i = 0; i < frame_height * frame_width; ++i) {
        depth_raw = ((((unsigned short)depth_image[i * 2 + 1]) << 8) + ((unsigned short)depth_image[i * 2 + 0]));
        depth_raw = (depth_raw << 13 | depth_raw >> 3); // 3 and 13 are bit shift values used to convert the input depth map depth from a 16 - bit unsigned integer format to a 32 - bit floating - point format
        depth_data[i] = float((float)depth_raw / 1000.0f);

    }

}


void ComputeTSDF_CPP(float* cam_pose, int* vox_size, float* vox_origin, unsigned char* depth_image, unsigned char* vox_grid,
    //float *vox_grid,
    ////////////
    float* depth_mapping,
    /// /////////   
    float* vox_tsdf) {

    //cout << "\nComputeTSDF_CPP\n";
    clock_tick t1 = start_timer();

    float* depth_data = new float[frame_height * frame_width];
    getDepthData_cpp(depth_image, depth_data);

    int num_voxels = vox_size[0] * vox_size[1] * vox_size[2];


    float* cam_pose_GPU;
    float* vox_origin_GPU;
    float* depth_data_GPU;
    float* vox_tsdf_GPU;
    unsigned char* vox_grid_GPU;
    /////////////////////////////////////////
    float* depth_mapping_3d_GPU;
    ////////////////////////////////////////
    int* vox_size_GPU;

    cudaMalloc(&cam_pose_GPU, 16 * sizeof(float));
    cudaMalloc(&vox_size_GPU, 3 * sizeof(int));
    cudaMalloc(&vox_origin_GPU, 3 * sizeof(float));

    cudaMalloc(&depth_data_GPU, frame_height * frame_width * sizeof(float));
    cudaMalloc(&vox_grid_GPU, num_voxels * sizeof(unsigned char));
    cudaMalloc(&vox_tsdf_GPU, num_voxels * sizeof(float));
    ////////////////////////////////////////
    cudaMalloc(&depth_mapping_3d_GPU, frame_height * frame_width * sizeof(float));
    ///////////////////////////////////////
    cudaMemset(vox_tsdf_GPU, 0, num_voxels * sizeof(float));
    cudaMemset(vox_grid_GPU, 0, num_voxels * sizeof(unsigned char));
    ///////////////////////////////////////
    //cudaMemset(&depth_mapping_3d_GPU,0, frame_height * frame_width * sizeof(float));

    cudaMemcpy(cam_pose_GPU, cam_pose, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vox_size_GPU, vox_size, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vox_origin_GPU, vox_origin, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(depth_data_GPU, depth_data, frame_height * frame_width * sizeof(float), cudaMemcpyHostToDevice);
    //////////////////////////////////////
    cudaMemcpy(depth_mapping_3d_GPU, depth_mapping, frame_height * frame_width * sizeof(float), cudaMemcpyHostToDevice);

    end_timer(t1, "Prepare duration");

    t1 = start_timer();
    // from depth map to binaray voxel representation
    depth2Grid << <frame_width, frame_height >> > (cam_pose_GPU, vox_size_GPU, vox_origin_GPU, depth_data_GPU, vox_grid_GPU, depth_mapping_3d_GPU, parameters_GPU);
    /// //////////////////////////////////
    cudaMemcpy(depth_mapping, depth_mapping_3d_GPU, frame_height * frame_width * sizeof(float), cudaMemcpyDeviceToHost);
    /// //////////////////////////////////////////
    cudaDeviceSynchronize();

    end_timer(t1, "depth2Grid duration");


    // distance transform
    int BLOCK_NUM = int((num_voxels + size_t(NUM_THREADS) - 1) / NUM_THREADS);

    t1 = start_timer();

    SquaredDistanceTransform << < BLOCK_NUM, NUM_THREADS >> > (cam_pose_GPU, vox_size_GPU, vox_origin_GPU, depth_data_GPU, vox_grid_GPU, vox_tsdf_GPU, parameters_GPU);

    end_timer(t1, "SquaredDistanceTransform duration");

    t1 = start_timer();

    cudaMemcpy(vox_grid, vox_grid_GPU, num_voxels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vox_tsdf, vox_tsdf_GPU, num_voxels * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] depth_data;


    cudaFree(cam_pose_GPU);
    cudaFree(vox_size_GPU);
    cudaFree(vox_origin_GPU);
    cudaFree(depth_data_GPU);
    cudaFree(vox_grid_GPU);
    //////////////////////////////////////
    cudaFree(depth_mapping_3d_GPU);
    cudaFree(vox_tsdf_GPU);

    end_timer(t1, "closeup duration");

}

void FlipTSDF_CPP(int* vox_size, float* vox_tsdf) {

    for (int vox_idx = 0; vox_idx < vox_size[0] * vox_size[1] * vox_size[2]; vox_idx++) {
        float value = float(vox_tsdf[vox_idx]);
        if (value > 1)
            value = 1;
        else if (value < -1)
            value = -1;

        float sign = 1;
        if (abs(value) > 0.001)
            sign = value / abs(value);

        vox_tsdf[vox_idx] = sign * (max(0.001, (1.0 - abs(value))));
    }
}

void Process_CPP(const char* filename,
    float* cam_pose,
    int* vox_size,
    float* vox_origin,
    int out_scale,
    int* segmentation_class_map,
    unsigned char* depth_data,
    float* vox_tsdf,
    float* vox_weights,
    float* vox_masks,
    //////////////////////////////
    //float* vox_grid, 
    float* depth_mapping,
    //////////////////////////////
    int* segmentation_label_downscale
    //int* segmentation_label_fullscale
)
{

    clock_tick t1 = start_timer();

    int num_voxels = vox_size[0] * vox_size[1] * vox_size[2];

    
    int* segmentation_label_fullscale;
    segmentation_label_fullscale = (int*)malloc((vox_size[0] * vox_size[1] * vox_size[2]) * sizeof(int));
    

    //int object_count;
      
    ReadVoxLabel_CPP(filename, vox_origin, cam_pose, vox_size, segmentation_class_map, segmentation_label_fullscale);

    end_timer(t1, "ReadVoxLabel_CPP");



    unsigned char* vox_grid = new unsigned char[num_voxels];
    memset(vox_grid, 0, num_voxels * sizeof(unsigned char));

    //ComputeTSDF_CPP(cam_pose, vox_size, vox_origin, depth_data, vox_grid, vox_tsdf);
    ////////////////////////////////////////////////////////////////////////////////////////////////
    ComputeTSDF_CPP(cam_pose, vox_size, vox_origin, depth_data, vox_grid, depth_mapping, vox_tsdf);


    end_timer(t1, "ComputeTSDF_edges_CPP");

    DownsampleLabel_CPP(vox_size,
        out_scale,
        segmentation_label_fullscale,
        vox_tsdf,
        segmentation_label_downscale,
        vox_weights,
        vox_masks,
        vox_grid,
        segmentation_class_map);

    end_timer(t1, "DownsampleLabel_CPP");

    FlipTSDF_CPP(vox_size, vox_tsdf);

    end_timer(t1, "FlipTSDF_CPP");

    t1 = start_timer();
    free(segmentation_label_fullscale); 
    end_timer(t1, "free");


}


extern "C" {
    void Process(const char* filename,
        float* cam_pose,
        int* vox_size,
        float* vox_origin,
        int out_scale,
        int* segmentation_class_map,
        unsigned char* depth_data,
        float* vox_tsdf,
        float* vox_weights,
        float* vox_masks,
        float* depth_mapping,
        ////////////////////////////////
        int* segmentation_label_downscale
        
    ) {
        Process_CPP(filename,
            cam_pose,
            vox_size,
            vox_origin,
            out_scale,
            segmentation_class_map,
            depth_data,
            vox_tsdf,
            vox_weights,
            vox_masks,
            /////////////////////////////////////////
            depth_mapping,
            /////////////////////////////////////////
            segmentation_label_downscale
            
        );
    }

    void setup(int device, int num_threads, float* K, int fw, int fh, float v_unit, float v_margin, int debug_flag) {
        setup_CPP(device, num_threads, K, fw, fh, v_unit, v_margin, debug_flag);
    }
}
