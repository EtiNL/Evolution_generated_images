import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from gpu_utils import get_max_threads_per_block
import asyncio

kernel_score = """
__global__ void score(float *testIm, float *targetIm, float *score, int nbr_circles, int check, float *x_coordinates, float *y_coordinates, float *radius, int im_size) {
    int idx = (threadIdx.x) + blockDim.x * blockIdx.x;
    if (idx < check) {
        int pixel_y = idx / im_size;
        int pixel_x = idx % im_size;
        for (int c = 0; c < nbr_circles; c++) {
            float color[3] = {0, 0, 0};
            int count = 0;
            int id = (int(y_coordinates[c]) * im_size + int(x_coordinates[c])) * 3;
            if (id >= 0 && id < check * 3) {
                color[0] = targetIm[id];
                color[1] = targetIm[id + 1];
                color[2] = targetIm[id + 2];
                count++;
                int r = int(radius[c]);
                if (r < 10) r = 10;
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < r; j++) {
                        id = ((int(y_coordinates[c]) + j) * im_size + int(x_coordinates[c]) + i) * 3;
                        if (i * i + j * j < r * r && id >= 0 && id < check * 3) {
                            color[0] += targetIm[id];
                            color[1] += targetIm[id + 1];
                            color[2] += targetIm[id + 2];
                            count++;
                        }
                    }
                }
                color[0] /= count;
                color[1] /= count;
                color[2] /= count;
                float dist_from_center = (pixel_x - x_coordinates[c]) * (pixel_x - x_coordinates[c]) + (pixel_y - y_coordinates[c]) * (pixel_y - y_coordinates[c]);
                if (dist_from_center < radius[c] * radius[c]) {
                    float color_swap[3] = {testIm[idx * 3], testIm[idx * 3 + 1], testIm[idx * 3 + 2]};
                    testIm[idx * 3] = color[0];
                    testIm[idx * 3 + 1] = color[1];
                    testIm[idx * 3 + 2] = color[2];
                    score[c * check + idx] = (abs(testIm[idx * 3] - targetIm[idx * 3]) + abs(testIm[idx * 3 + 1] - targetIm[idx * 3 + 1]) + abs(testIm[idx * 3 + 2] - targetIm[idx * 3 + 2])) / 3;
                    testIm[idx * 3] = color_swap[0];
                    testIm[idx * 3 + 1] = color_swap[1];
                    testIm[idx * 3 + 2] = color_swap[2];
                } else {
                    score[c * check + idx] = (abs(testIm[idx * 3] - targetIm[idx * 3]) + abs(testIm[idx * 3 + 1] - targetIm[idx * 3 + 1]) + abs(testIm[idx * 3 + 2] - targetIm[idx * 3 + 2])) / 3;
                }
            }
        }
    }
}
"""

kernel_loss = """
__global__ void loss(float *testIm, float *targetIm, float *score, int check, int im_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < check) {
        score[idx] = (abs(testIm[idx * 3] - targetIm[idx * 3]) +
                      abs(testIm[idx * 3 + 1] - targetIm[idx * 3 + 1]) +
                      abs(testIm[idx * 3 + 2] - targetIm[idx * 3 + 2])) / 3.0;
    }
}
"""

async def setup_cuda_memory(targetIm, testIm):
    try:
        data = {
            "px_target": np.array(targetIm).astype(np.float32),
            "px_test": np.array(testIm).astype(np.float32),
        }

        d_memory = {k: cuda.mem_alloc(v.nbytes) for k, v in data.items()}
        stream = cuda.Stream()
        for k, v in data.items():
            cuda.memcpy_htod_async(d_memory[k], v, stream)
        
        stream.synchronize()
        return d_memory, stream
    except Exception as e:
        print(f"Exception in setup_cuda_memory: {e}")
        return None, None

async def score_generation(targetIm, testIm, center_pos_x, center_pos_y, radius, semaphore=None):
    cuda.init()
    cuda_device = cuda.Device(0)
    cuda_context = cuda_device.make_context()

    try:
        await semaphore.acquire()
        try:
            mod = SourceModule(kernel_score)
            circle_func = mod.get_function("score")

            d_memory, stream = await setup_cuda_memory(targetIm, testIm, center_pos_x, center_pos_y, radius)

            totalPixels = int(targetIm.shape[0] * targetIm.shape[1])
            BLOCK_SIZE = get_max_threads_per_block()
            grid = ((totalPixels + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
            block = (BLOCK_SIZE, 1, 1)

            score = np.zeros((len(radius), totalPixels)).astype(np.float32)
            score_gpu = cuda.mem_alloc(score.nbytes)
            cuda.memcpy_htod_async(score_gpu, score, stream)

            nbr_circles = np.int32(len(radius))
            totalPixels_np = np.int32(totalPixels)
            im_size = np.int32(targetIm.shape[1])

            circle_func(d_memory["px_test"], d_memory["px_target"], score_gpu, 
                        nbr_circles, totalPixels_np, 
                        d_memory["x_coordinates"], d_memory["y_coordinates"], d_memory["radius"], 
                        im_size, block=block, grid=grid, stream=stream)
            stream.synchronize()

            cuda.memcpy_dtoh(score, score_gpu)
        finally:
            semaphore.release()
    finally:
        cuda_context.pop()
    return np.sum(score, axis=1) / totalPixels

async def loss(targetIm, testIm, semaphore=None):
    # print("Starting loss calculation...")
    try:
        # print("Acquire semaphore...")
        await semaphore.acquire()
        cuda.init()
        cuda_device = cuda.Device(0)
        cuda_context = cuda_device.make_context()
        # print("Acquired semaphore")
        try:
            mod = SourceModule(kernel_loss)
            loss_func = mod.get_function("loss")

            d_memory, stream = await setup_cuda_memory(targetIm, testIm)
            if d_memory is None:
                raise RuntimeError("Failed to setup CUDA memory.")

            totalPixels = int(targetIm.shape[0] * targetIm.shape[1])
            BLOCK_SIZE = get_max_threads_per_block()
            grid = ((totalPixels + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
            block = (BLOCK_SIZE, 1, 1)

            score = np.zeros(totalPixels).astype(np.float32)
            score_gpu = cuda.mem_alloc(score.nbytes)
            cuda.memcpy_htod_async(score_gpu, score, stream)

            # print("Launching CUDA kernel...")
            loss_func(d_memory["px_test"], d_memory["px_target"], score_gpu, np.int32(totalPixels), np.int32(targetIm.shape[1]), block=block, grid=grid, stream=stream)
            stream.synchronize()
            # print("CUDA kernel completed.")

            cuda.memcpy_dtoh_async(score, score_gpu, stream)
            stream.synchronize()

            total_loss = np.sum(score) / totalPixels
            # print(f"Loss calculation completed. Total loss: {total_loss}")
            return total_loss
        except Exception as e:
            print(f"Exception during loss calculation: {e}")
        
    except Exception as e:
        print(f"Exception in loss function: {e}")
    finally:
        cuda_context.pop()
        # print("CUDA context popped after loss calculation.")
        semaphore.release()
        # print("Semaphore released after loss calculation.")