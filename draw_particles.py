import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import cv2 as cv

kernel_draw_circles = """
#include <math.h>

__global__ void draw_circles(float *testIm, float *targetIm, int im_width, int im_height, float *x_coordinates, float *y_coordinates, float *radius, int num_circles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_linear = idy * im_width + idx;

    if (idx < im_width && idy < im_height) {
        for (int c = 0; c < num_circles; ++c) {
            float dx = idx - x_coordinates[c];
            float dy = idy - y_coordinates[c];
            float dist = sqrt(dx * dx + dy * dy);

            if (dist < radius[c]) {
                int id = (int)(y_coordinates[c]) * im_width + (int)(x_coordinates[c]);
                float color[3] = {0, 0, 0};
                int count = 0;

                for (int i = 0; i < radius[c]; ++i) {
                    for (int j = 0; j < radius[c]; ++j) {
                        int id_i = (int)(y_coordinates[c] + j) * im_width + (int)(x_coordinates[c] + i);
                        int id_j = (int)(y_coordinates[c] - j) * im_width + (int)(x_coordinates[c] + i);
                        int id_k = (int)(y_coordinates[c] + j) * im_width + (int)(x_coordinates[c] - i);
                        int id_l = (int)(y_coordinates[c] - j) * im_width + (int)(x_coordinates[c] - i);

                        if (id_i >= 0 && id_i < im_width * im_height) {
                            color[0] += targetIm[3 * id_i];
                            color[1] += targetIm[3 * id_i + 1];
                            color[2] += targetIm[3 * id_i + 2];
                            count++;
                        }
                        if (id_j >= 0 && id_j < im_width * im_height) {
                            color[0] += targetIm[3 * id_j];
                            color[1] += targetIm[3 * id_j + 1];
                            color[2] += targetIm[3 * id_j + 2];
                            count++;
                        }
                        if (id_k >= 0 && id_k < im_width * im_height) {
                            color[0] += targetIm[3 * id_k];
                            color[1] += targetIm[3 * id_k + 1];
                            color[2] += targetIm[3 * id_k + 2];
                            count++;
                        }
                        if (id_l >= 0 && id_l < im_width * im_height) {
                            color[0] += targetIm[3 * id_l];
                            color[1] += targetIm[3 * id_l + 1];
                            color[2] += targetIm[3 * id_l + 2];
                            count++;
                        }
                    }
                }

                color[0] /= count;
                color[1] /= count;
                color[2] /= count;

                testIm[3 * idx_linear] = color[0];
                testIm[3 * idx_linear + 1] = color[1];
                testIm[3 * idx_linear + 2] = color[2];
            }
        }
    }
}
"""

def Draw_circles(targetIm, testIm, center_pos_x, center_pos_y, radius):
    mod = SourceModule(kernel_draw_circles)
    draw_func = mod.get_function("draw_circles")

    try:
        num_circles = len(center_pos_x)
        im_height, im_width, _ = targetIm.shape

        d_px_target = gpuarray.to_gpu(targetIm.astype(np.float32))
        d_px_test = gpuarray.to_gpu(testIm.astype(np.float32))
        x_coordinates = gpuarray.to_gpu(center_pos_x.astype(np.float32))
        y_coordinates = gpuarray.to_gpu(center_pos_y.astype(np.float32))
        radius_cuda = gpuarray.to_gpu(radius.astype(np.float32))

        block_size = (16, 16, 1)
        grid_size = ((im_width + block_size[0] - 1) // block_size[0], (im_height + block_size[1] - 1) // block_size[1])

        draw_func(d_px_test, d_px_target, np.int32(im_width), np.int32(im_height), x_coordinates, y_coordinates, radius_cuda, np.int32(num_circles), block=block_size, grid=grid_size)

        return d_px_test.get().astype(np.uint8)

    except cuda.CudaError as e:
        print("CUDA error:", e)
        return np.zeros_like(targetIm)

if __name__ == '__main__':
    from PIL import Image
    targetIm = np.array(Image.open('raw_data/point.jpg'))
    genIm = np.zeros_like(targetIm)

    center_pos_x = np.array([100, 200])  # Example x coordinates of circle centers
    center_pos_y = np.array([100, 200])  # Example y coordinates of circle centers
    radius = np.array([50, 80])          # Example radii of circles

    draw = Draw_circles(targetIm, genIm, center_pos_x, center_pos_y, radius)
    cv.imshow('Circles', cv.cvtColor(draw, cv.COLOR_RGB2BGR))
    cv.imshow('Original', cv.cvtColor(targetIm, cv.COLOR_RGB2BGR))
    cv.waitKey(0)
