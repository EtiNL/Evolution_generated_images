import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from gpu_utils import get_max_threads_per_block
import asyncio

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

async def optimize_kernel_config(totalPixels):
    BLOCK_SIZE = 1024  # A100 optimal block size
    grid_size = (totalPixels + BLOCK_SIZE - 1) // BLOCK_SIZE
    block = (BLOCK_SIZE, 1, 1)
    grid = (grid_size, 1, 1)
    return block, grid

async def setup_cuda_memory(targetIm, testIm, center_pos_x, center_pos_y, radius):
    data = {
        "px_target": np.array(targetIm).astype(np.float32),
        "px_test": np.array(testIm).astype(np.float32),
        "x_coordinates": np.array(center_pos_x).astype(np.float32),
        "y_coordinates": np.array(center_pos_y).astype(np.float32),
        "radius": np.array(radius).astype(np.float32)
    }

    d_memory = {k: cuda.mem_alloc(v.nbytes) for k, v in data.items()}
    stream = cuda.Stream()
    for k, v in data.items():
        cuda.memcpy_htod_async(d_memory[k], v, stream)
    
    return d_memory, stream

async def Draw_particules(targetIm, testIm, center_pos_x, center_pos_y, radius, semaphore=None):
    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_context = cuda_device.make_context()

    try:
        await semaphore.acquire()
        try:
            mod = SourceModule(kernel_draw_circles)
            circle_func = mod.get_function("draw_circles")

            d_memory, stream = await setup_cuda_memory(targetIm, testIm, center_pos_x, center_pos_y, radius)
            block, grid = await optimize_kernel_config(int(targetIm.shape[0] * targetIm.shape[1]))

            circle_func(
                d_memory["px_test"], d_memory["px_target"], 
                np.int32(len(radius)), np.int32(int(targetIm.shape[0] * targetIm.shape[1])), 
                d_memory["x_coordinates"], d_memory["y_coordinates"], d_memory["radius"], 
                np.int32(targetIm.shape[1]), 
                block=block, grid=grid, stream=stream
            )
            stream.synchronize()

            res = np.empty_like(targetIm, dtype=np.float32)
            cuda.memcpy_dtoh_async(res, d_memory["px_test"], stream)
            stream.synchronize()
        except Exception as e:
            print(f"Exception in draw_particles: {e}")
        finally:
            cuda_context.pop()
            semaphore.release()
    except Exception as e:
        print(f"Exception in draw_particles: {e}")
    return np.uint8(res)
