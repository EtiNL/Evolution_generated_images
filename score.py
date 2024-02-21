import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import compiler
import pycuda.driver as cuda
import pycuda.autoinit             # PyCuda autoinit
import pycuda.gpuarray as gpuarray
import numpy
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2 as cv

kernel_score = """

__global__ void score(float *testIm, float *targetIm, float *score, int nbr_circles, int check, float *x_coordinates, float *y_coordinates, float *radius, int im_size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int pixel_y = idx / im_size;
    int pixel_x = idx % im_size;
    int circle_idx = blockIdx.y;

    if (circle_idx < nbr_circles) {
        float color[3] = {0, 0, 0};
        // Getting color
        int count = 0;
        int id = (y_coordinates[circle_idx]) * im_size + x_coordinates[circle_idx];
        color[0] = color[0] + targetIm[3 * id];
        color[1] = color[1] + targetIm[3 * id + 1];
        color[2] = color[2] + targetIm[3 * id + 2];
        count++;
        int r = 5;
        if (radius[circle_idx] < r){
                if (radius[circle_idx] > 1){r = radius[circle_idx];}
                else{r = 1;}
                }
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < r; j++) {
                int id = (y_coordinates[circle_idx] + j) * im_size + x_coordinates[circle_idx] + i;
                if (i * i + j * j < r * r && id < check && id > 0) {
                    color[0] = color[0] + targetIm[3 * id];
                    color[1] = color[1] + targetIm[3 * id + 1];
                    color[2] = color[2] + targetIm[3 * id + 2];
                    count++;
                }
                id = (y_coordinates[circle_idx] - j) * im_size + x_coordinates[circle_idx] + i;
                if (i * i + j * j < r * r && id < check && id > 0) {
                    color[0] = color[0] + targetIm[3 * id];
                    color[1] = color[1] + targetIm[3 * id + 1];
                    color[2] = color[2] + targetIm[3 * id + 2];
                    count++;
                }
                id = (y_coordinates[circle_idx] - j) * im_size + x_coordinates[circle_idx] - i;
                if (i * i + j * j < r * r && id < check && id > 0) {
                    color[0] = color[0] + targetIm[3 * id];
                    color[1] = color[1] + targetIm[3 * id + 1];
                    color[2] = color[2] + targetIm[3 * id + 2];
                    count++;
                }
                id = (y_coordinates[circle_idx] + j) * im_size + x_coordinates[circle_idx] - i;
                if (i * i + j * j < r * r && id < check && id > 0) {
                    color[0] = color[0] + targetIm[3 * id];
                    color[1] = color[1] + targetIm[3 * id + 1];
                    color[2] = color[2] + targetIm[3 * id + 2];
                    count++;
                }
            }
        }
        color[0] = color[0] / count;
        color[1] = color[1] / count;
        color[2] = color[2] / count;
        float dist_from_center = (pixel_x - x_coordinates[circle_idx]) * (pixel_x - x_coordinates[circle_idx]) + (pixel_y - y_coordinates[circle_idx]) * (pixel_y - y_coordinates[circle_idx]);
        if (idx * 3 < check * 3 && dist_from_center < radius[circle_idx] * radius[circle_idx]) {
            // Assign color
            float color_swap[3] = {testIm[idx * 3], testIm[idx * 3 + 1], testIm[idx * 3 + 2]};
            testIm[idx * 3] = color[0];
            testIm[idx * 3 + 1] = color[1];
            testIm[idx * 3 + 2] = color[2];
            score[circle_idx * check + idx] = (abs(testIm[idx * 3] - targetIm[idx * 3]) + abs(testIm[idx * 3 + 1] - targetIm[idx * 3 + 1]) + abs(testIm[idx * 3 + 2] - targetIm[idx * 3 + 2])) / 3;
            testIm[idx * 3] = color_swap[0];
            testIm[idx * 3 + 1] = color_swap[1];
            testIm[idx * 3 + 2] = color_swap[2];
        } else {
            score[circle_idx * check + idx] = (abs(testIm[idx * 3] - targetIm[idx * 3]) + abs(testIm[idx * 3 + 1] - targetIm[idx * 3 + 1]) + abs(testIm[idx * 3 + 2] - targetIm[idx * 3 + 2])) / 3;
        }
    }
}

    """




def score_generation(targetIm, testIm, center_pos_x, center_pos_y, radius):
    try:
        # Compile and get kernel function
        mod = SourceModule(kernel_score)
        circle_func = mod.get_function("score")

        d_px_target = gpuarray.to_gpu(targetIm.astype(np.float32))
        d_px_test = gpuarray.to_gpu(testIm.astype(np.float32))
        x_coordinates = gpuarray.to_gpu(center_pos_x.astype(np.int32))
        y_coordinates = gpuarray.to_gpu(center_pos_y.astype(np.int32))
        radius_cuda = gpuarray.to_gpu(radius.astype(np.int32))

        # Create a buffer to hold the result of the kernel computation
        score_result_gpu = gpuarray.zeros((radius.shape[0], targetIm.shape[0] * targetIm.shape[1]), dtype=np.float32)

        # Define score_gpu buffer
        score_gpu = gpuarray.zeros((radius.shape[0], targetIm.shape[0] * targetIm.shape[1]), dtype=np.float32)

        BLOCK_SIZE = 1024
        total_pixels = targetIm.shape[0] * targetIm.shape[1]
        num_blocks_x = (total_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks_y = radius.shape[0]
        grid_size = (num_blocks_x, num_blocks_y, 1)
        block_size = (BLOCK_SIZE, 1, 1)

        # Call the kernel function with the result buffer
        circle_func(d_px_test, d_px_target, score_result_gpu, numpy.int32(len(radius)), numpy.int32(total_pixels), x_coordinates, y_coordinates, radius_cuda, numpy.int32(targetIm.shape[1]), block=block_size, grid=grid_size)

        # Copy the result back to the original score_gpu buffer if needed
        score_gpu.set(score_result_gpu)

        # Process the result further if necessary
        score = (score_gpu.get()).reshape((radius.shape[0], targetIm.shape[0], targetIm.shape[1]))
        score = np.array([np.sum(entropy(score[i,:,:].astype(numpy.uint8), disk(10))) / total_pixels for i in range(radius.shape[0])])

        return score * 10

    except cuda.Error as e:
        print("CUDA error:", e)
        return 0

kernel_loss = """

    __global__ void loss( float *testIm, float *targetIm, float *score, int check, int im_size){


        int idx = (threadIdx.x ) + blockDim.x * blockIdx.x ;
        score[idx] = (abs(testIm[idx*3]-targetIm[idx*3])+abs(testIm[idx*3+1]-targetIm[idx*3+1])+abs(testIm[idx*3+2]-targetIm[idx*3+2]))/3;

        }

    """
def loss(targetIm, testIm):

    #Compile and get kernel function
    mod = SourceModule(kernel_loss)
    loss_func = mod.get_function("loss")


    d_px_target = gpuarray.to_gpu(targetIm.astype(np.float32))
    d_px_test = gpuarray.to_gpu(testIm.astype(np.float32))
    # Create a buffer to hold the result of the kernel computation
    score_result_gpu = gpuarray.zeros(targetIm.shape[:2], dtype=np.float32)

    # Define score_gpu buffer
    score_gpu = gpuarray.zeros(targetIm.shape[:2], dtype=np.float32)

    BLOCK_SIZE = 1024
    block = (BLOCK_SIZE,1,1)
    totalPixels = numpy.int32(targetIm.shape[0]*targetIm.shape[1])
    gridRounded=int(targetIm.shape[0]*targetIm.shape[1]/BLOCK_SIZE)+1
    grid = (gridRounded,1,1)




    loss_func(d_px_test, d_px_target, score_result_gpu, totalPixels, numpy.int32(targetIm.shape[1]) ,block=block,grid = grid)

    # Copy the result back to the original score_gpu buffer if needed
    score_gpu.set(score_result_gpu)

    # Process the result further if necessary
    score = (score_gpu.get()).reshape(targetIm.shape[:2])



    return np.sum(entropy(score.astype(np.uint8), disk(10)))/totalPixels*10

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

    targetIm = np.array(Image.open('raw_data/point.jpg'))
    genIm = np.zeros_like(targetIm)

    center_pos_x = np.array([50,150])
    center_pos_y = np.array([50,150])
    radius = np.array([50,100])

    score = score_generation(targetIm,genIm,center_pos_x,center_pos_y,radius)
    print(score)
    draw = Draw_particules(targetIm, np.copy(genIm), center_pos_x, center_pos_y, radius)
    scoreloss = loss(targetIm, draw)
    print(scoreloss)

    # cv.imshow('particle', draw)
    # cv.imshow('test', im)
    cv.waitKey(0)
