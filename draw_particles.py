import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import numpy as np
from gpu_utils import get_max_threads_per_block

kernel_draw_circles = """
__global__ void draw_circles( float *testIm, float *targetIm, int nbr_circles, int check, float *x_coordinates, float *y_coordinates, float *radius, int im_size){
    for(int c = 0; c < nbr_circles; c++){
        float color[3]{0,0,0};
        int count = 0;
        int id = (y_coordinates[c])*im_size + x_coordinates[c];
        if (id < check && id >= 0) {
            color[0] = color[0] + targetIm[3*id];
            color[1] = color[1] + targetIm[3*id+1];
            color[2] = color[2] + targetIm[3*id+2];
            count++;
        }
        int r = 10;
        if (radius[c] < r){r = radius[c];}
        for(int i = 0; i<r; i++){
            for(int j = 0; j<r; j++){
                int id = (y_coordinates[c]+j)*im_size + x_coordinates[c]+i;
                if (i*i+j*j<r*r && id < check && id >= 0){
                    color[0] = color[0] + targetIm[3*id];
                    color[1] = color[1] + targetIm[3*id+1];
                    color[2] = color[2] + targetIm[3*id+2];
                    count++;
                }
                id = (y_coordinates[c]-j)*im_size + x_coordinates[c]+i;
                if (i*i+j*j<r*r && id < check && id >= 0){
                    color[0] = color[0] + targetIm[3*id];
                    color[1] = color[1] + targetIm[3*id+1];
                    color[2] = color[2] + targetIm[3*id+2];
                    count++;
                }
                id = (y_coordinates[c]-j)*im_size + x_coordinates[c]-i;
                if (i*i+j*j<r*r  && id < check && id >= 0){
                    color[0] = color[0] + targetIm[3*id];
                    color[1] = color[1] + targetIm[3*id+1];
                    color[2] = color[2] + targetIm[3*id+2];
                    count++;
                }
                id = (y_coordinates[c]+j)*im_size + x_coordinates[c]-i;
                if (i*i+j*j<r*r && id < check && id >= 0){
                    color[0] = color[0] + targetIm[3*id];
                    color[1] = color[1] + targetIm[3*id+1];
                    color[2] = color[2] + targetIm[3*id+2];
                    count++;
                }
            }
        }
        color[0] = color[0]/count;
        color[1] = color[1] / count;
        color[2] = color[2] / count;

        int idx = (threadIdx.x ) + blockDim.x * blockIdx.x ;
        int pixel_y = idx / im_size ;
        int pixel_x = idx % im_size ;
        float dist_from_center = (pixel_x-x_coordinates[c])*(pixel_x-x_coordinates[c]) + (pixel_y-y_coordinates[c])*(pixel_y-y_coordinates[c]) ;
        if(idx *3 < check*3 && dist_from_center < radius[c]*radius[c])
        {
            testIm[idx*3]= color[0];
            testIm[idx*3+1]= color[1];
            testIm[idx*3+2]= color[2];
        }
    }
}
"""

def setup_cuda_memory(targetIm, testIm, center_pos_x, center_pos_y, radius):
    data = {
        "px_target": np.array(targetIm).astype(np.float32),
        "px_test": np.array(testIm).astype(np.float32),
        "x_coordinates": np.array(center_pos_x).astype(np.float32),
        "y_coordinates": np.array(center_pos_y).astype(np.float32),
        "radius": np.array(radius).astype(np.float32)
    }

    # for k, v in data.items():
    #     print(f"{k}: shape={v.shape}, dtype={v.dtype}")

    d_memory = {k: cuda.mem_alloc(v.nbytes) for k, v in data.items()}
    stream = cuda.Stream()
    for k, v in data.items():
        cuda.memcpy_htod_async(d_memory[k], v, stream)
    
    return d_memory, stream

def Draw_particules(targetIm, testIm, center_pos_x, center_pos_y, radius):
    # Compile and get kernel function
    mod = SourceModule(kernel_draw_circles)
    circle_func = mod.get_function("draw_circles")

    d_memory, stream = setup_cuda_memory(targetIm, testIm, center_pos_x, center_pos_y, radius)
    d_px_test = d_memory["px_test"]
    d_px_target = d_memory["px_target"]
    x_coord = d_memory["x_coordinates"]
    y_coord = d_memory["y_coordinates"]
    radius_cuda = d_memory["radius"]

    BLOCK_SIZE = get_max_threads_per_block()
    block = (BLOCK_SIZE, 1, 1)
    totalPixels = int(targetIm.shape[0] * targetIm.shape[1])
    gridRounded = (totalPixels + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (int(gridRounded), 1, 1)

    # Print types of parameters
    # print(f"Types of parameters passed to kernel function:")
    # print(f"d_px_test: {type(d_px_test)}, d_px_target: {type(d_px_target)}")
    # print(f"len(radius): {type(int(len(radius)))}, totalPixels: {type(int(totalPixels))}")
    # print(f"x_coord: {type(x_coord)}, y_coord: {type(y_coord)}, radius_cuda: {type(radius_cuda)}")
    # print(f"targetIm.shape[1]: {type(int(targetIm.shape[1]))}")
    # print(f"block: {block}, grid: {grid}")

    try:
        circle_func(d_px_test, d_px_target, np.int32(len(radius)), np.int32(totalPixels), x_coord, y_coord, radius_cuda, np.int32(targetIm.shape[1]), block=block, grid=grid, stream=stream)
        stream.synchronize()  # Synchronize to catch errors early
        # print("Kernel execution successful")
    except pycuda._driver.LogicError as e:
        print(f"totalPixels: {totalPixels}, grid: {grid}, block: {block}")
        print(f"CUDA Error: {e}")
    except pycuda._driver.LaunchError as e:
        print(f"CUDA Launch Error: {e}")
    except TypeError as e:
        print(f"Type Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

    res = np.empty_like(targetIm, dtype=np.float32)
    try:
        cuda.memcpy_dtoh_async(res, d_px_test, stream)
        stream.synchronize()  # Ensure all operations are completed
        # print("memcpy_dtoh_async successful")
    except pycuda._driver.LogicError as e:
        print(f"CUDA Error during memcpy_dtoh: {e}")
    except TypeError as e:
        print(f"Type Error during memcpy_dtoh: {e}")
    except Exception as e:
        print(f"Unexpected Error during memcpy_dtoh: {e}")

    if res is None:
        print("Error: res is None after memcpy_dtoh_async")
    else:
        # print("res is valid")
        res = np.uint8(res)
        # print("Conversion to uint8 successful")

    return res
