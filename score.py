import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import compiler
import pycuda.driver as cuda
import pycuda.autoinit             # PyCuda autoinit
import numpy
import numpy as np
from gpu_utils import *

kernel_score = """
__global__ void score( float *testIm, float *targetIm, float *score, int nbr_circles, int check, float *x_coordinates, float *y_coordinates, float *radius, int im_size){
    int idx = (threadIdx.x ) + blockDim.x * blockIdx.x;
    if (idx < check) { // Ensure idx is within bounds
        int pixel_y = idx / im_size;
        int pixel_x = idx % im_size;
        for(int c = 0; c < nbr_circles; c++){
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
                for(int i = 0; i < r; i++) {
                    for(int j = 0; j < r; j++) {
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




def score_generation(targetIm, testIm, center_pos_x, center_pos_y, radius):
    # Compile and get kernel function
    mod = SourceModule(kernel_score)
    circle_func = mod.get_function("score")

    px_target = numpy.array(targetIm).astype(numpy.float32)
    d_px_target = cuda.mem_alloc(px_target.nbytes)
    cuda.memcpy_htod(d_px_target, px_target)

    px_test = numpy.array(testIm).astype(numpy.float32)
    d_px_test = cuda.mem_alloc(px_test.nbytes)
    cuda.memcpy_htod(d_px_test, px_test)

    x_coordinates = numpy.array(center_pos_x).astype(numpy.float32)
    x_coord = cuda.mem_alloc(x_coordinates.nbytes)
    cuda.memcpy_htod(x_coord, x_coordinates)

    y_coordinates = numpy.array(center_pos_y).astype(numpy.float32)
    y_coord = cuda.mem_alloc(y_coordinates.nbytes)
    cuda.memcpy_htod(y_coord, y_coordinates)

    radius = numpy.array(radius).astype(numpy.float32)
    radius_cuda = cuda.mem_alloc(radius.nbytes)
    cuda.memcpy_htod(radius_cuda, radius)

    score = numpy.zeros((radius.shape[0], targetIm.shape[0] * targetIm.shape[1])).astype(numpy.float32)
    score_gpu = cuda.mem_alloc(score.nbytes)
    cuda.memcpy_htod(score_gpu, score)

    BLOCK_SIZE = get_max_threads_per_block()
    block = (BLOCK_SIZE, 1, 1)
    totalPixels = targetIm.shape[0] * targetIm.shape[1]
    gridRounded = (totalPixels + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (gridRounded, 1, 1)

    # Ensure types before launching the kernel
    assert isinstance(d_px_test, pycuda.driver.DeviceAllocation)
    assert isinstance(d_px_target, pycuda.driver.DeviceAllocation)
    assert isinstance(score_gpu, pycuda.driver.DeviceAllocation)
    assert isinstance(x_coord, pycuda.driver.DeviceAllocation)
    assert isinstance(y_coord, pycuda.driver.DeviceAllocation)
    assert isinstance(radius_cuda, pycuda.driver.DeviceAllocation)

    # Launch the kernel with proper type conversions
    circle_func(d_px_test, d_px_target, score_gpu, 
                numpy.int32(len(radius)), 
                numpy.int32(totalPixels), 
                x_coord, y_coord, radius_cuda, 
                numpy.int32(targetIm.shape[1]), 
                block=block, grid=grid)

    # Force synchronization and error checking
    cuda.Context.synchronize()

    cuda.memcpy_dtoh(score, score_gpu)

    return np.sum(score, axis=1) / totalPixels








kernel_loss = """
__global__ void loss(float *testIm, float *targetIm, float *score, int check, int im_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < check) { // Ensure idx is within bounds
        score[idx] = (abs(testIm[idx * 3] - targetIm[idx * 3]) +
                      abs(testIm[idx * 3 + 1] - targetIm[idx * 3 + 1]) +
                      abs(testIm[idx * 3 + 2] - targetIm[idx * 3 + 2])) / 3.0;
    }
}
"""
def loss(targetIm, testIm):
    # Compile and get kernel function
    mod = SourceModule(kernel_loss)
    loss_func = mod.get_function("loss")

    px_target = np.array(targetIm).astype(np.float32)
    px_test = np.array(testIm).astype(np.float32)
    totalPixels = px_target.shape[0] * px_target.shape[1]

    # Ensure the sizes are correct
    assert px_target.size == px_test.size, "Target and Test images must have the same size"

    d_px_target = cuda.mem_alloc(px_target.nbytes)
    d_px_test = cuda.mem_alloc(px_test.nbytes)
    score = np.zeros(px_target.shape[0] * px_target.shape[1]).astype(np.float32)
    score_gpu = cuda.mem_alloc(score.nbytes)

    cuda.memcpy_htod(d_px_target, px_target)
    cuda.memcpy_htod(d_px_test, px_test)
    cuda.memcpy_htod(score_gpu, score)

    BLOCK_SIZE = get_max_threads_per_block()
    gridRounded = (totalPixels + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (gridRounded, 1, 1)
    block = (BLOCK_SIZE, 1, 1)

    # Print types and shapes for debugging
    # print(f"d_px_test type: {type(d_px_test)}, size: {px_test.nbytes}")
    # print(f"d_px_target type: {type(d_px_target)}, size: {px_target.nbytes}")
    # print(f"score_gpu type: {type(score_gpu)}, size: {score.nbytes}")
    # print(f"totalPixels: {totalPixels}, targetIm.shape[1]: {targetIm.shape[1]}")
    # print(f"grid: {grid}, block: {block}")

    loss_func(d_px_test, d_px_target, score_gpu, np.int32(totalPixels), np.int32(targetIm.shape[1]), block=block, grid=grid)

    cuda.memcpy_dtoh(score, score_gpu)

    return np.sum(score) / totalPixels

if __name__=='__main__':
    from PIL import Image
    import cv2 as cv
    # im = np.array(Image.open('raw_data/vangogh2.jpg'))
    # nbr_particules = 100
    # center_pos_x = np.random.randint(im.shape[0], size=nbr_particules)
    # center_pos_y = np.random.randint(im.shape[1], size=nbr_particules)
    # radius = np.random.randint(2, high=50, size=nbr_particules)
    # score = score_generation(im,im,center_pos_x,center_pos_y,radius)
    # print(score)
    # cv2.imshow("Reslt-img55", cv2.cvtColor(pil_im, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)


    from draw_particles import Draw_particules

    im = np.zeros((100,100,3))

    center_pos_x = np.array([50])
    center_pos_y = np.array([50])
    radius = np.array([50])

    score = score_generation(im,im,center_pos_x,center_pos_y,radius)
    print(score)
    draw = Draw_particules(im, np.copy(im), center_pos_x, center_pos_y, radius)
    scoreloss = loss(im, draw)
    print(scoreloss)

    # cv.imshow('particle', draw)
    # cv.imshow('test', im)
    # cv.waitKey(0)
