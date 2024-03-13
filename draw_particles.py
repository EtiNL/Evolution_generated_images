import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import compiler
import pycuda.driver as cuda
import pycuda.autoinit             # PyCuda autoinit
import numpy
import numpy as np
import cv2 as cv

kernel_draw_circles = """

    __global__ void draw_circles( float *testIm, float *targetIm, int nbr_circles, int check, float *x_coordinates, float *y_coordinates, float *radius, int im_size){

        for(int c = 0; c < nbr_circles; c++){

            float color[3]{0,0,0};
            // getting color
            int count = 0;
            int id = (y_coordinates[c])*im_size + x_coordinates[c];
            color[0] = color[0] + targetIm[3*id];
            color[1] = color[1] + targetIm[3*id+1];
            color[2] = color[2] + targetIm[3*id+2];
            count++;
            int r = 10;
            if (radius[c] < r){r = radius[c];}
            for(int i = 0; i<r; i++){
                for(int j = 0; j<r; j++){

                    int id = (y_coordinates[c]+j)*im_size + x_coordinates[c]+i;
                    if (i*i+j*j<r*r && id < check && id > 0){

                      color[0] = color[0] + targetIm[3*id];
                      color[1] = color[1] + targetIm[3*id+1];
                      color[2] = color[2] + targetIm[3*id+2];
                      count++;
                      }
                    id = (y_coordinates[c]-j)*im_size + x_coordinates[c]+i;
                    if (i*i+j*j<r*r && id < check && id > 0){

                      color[0] = color[0] + targetIm[3*id];
                      color[1] = color[1] + targetIm[3*id+1];
                      color[2] = color[2] + targetIm[3*id+2];
                      count++;
                      }
                    id = (y_coordinates[c]-j)*im_size + x_coordinates[c]-i;
                    if (i*i+j*j<r*r  && id < check && id > 0){

                      color[0] = color[0] + targetIm[3*id];
                      color[1] = color[1] + targetIm[3*id+1];
                      color[2] = color[2] + targetIm[3*id+2];
                      count++;
                      }
                    id = (y_coordinates[c]+j)*im_size + x_coordinates[c]-i;
                    if (i*i+j*j<r*r && id < check && id > 0){

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

            //assign color

            testIm[idx*3]= color[0];
            testIm[idx*3+1]= color[1];
            testIm[idx*3+2]= color[2];
            }
        }
        }

    """




def Draw_particules(targetIm, testIm, center_pos_x, center_pos_y, radius):

    #Compile and get kernel function
    mod = SourceModule(kernel_draw_circles)
    circle_func = mod.get_function("draw_circles")

    px_target = numpy.array(targetIm).astype(numpy.float32)
    d_px_target = cuda.mem_alloc(px_target.nbytes)
    cuda.memcpy_htod(d_px_target, px_target)

    px_test = numpy.array(testIm).astype(numpy.float32)
    d_px_test = cuda.mem_alloc(px_test.nbytes)
    cuda.memcpy_htod(d_px_test, px_test)

    x_coordinates = numpy.array(center_pos_x).astype(numpy.float32)
    x_coord = cuda.mem_alloc(x_coordinates.nbytes)
    cuda.memcpy_htod(x_coord,x_coordinates)

    y_coordinates = numpy.array(center_pos_y).astype(numpy.float32)
    y_coord = cuda.mem_alloc(y_coordinates.nbytes)
    cuda.memcpy_htod(y_coord,y_coordinates)

    radius = numpy.array(radius).astype(numpy.float32)
    radius_cuda = cuda.mem_alloc(radius.nbytes)
    cuda.memcpy_htod(radius_cuda,radius)

    # print(x_coordinates)
    # print(y_coordinates)

    BLOCK_SIZE = 1024
    block = (BLOCK_SIZE,1,1)
    totalPixels = numpy.int32(targetIm.shape[0]*targetIm.shape[1])
    gridRounded=int(targetIm.shape[0]*targetIm.shape[1]/BLOCK_SIZE)+1
    grid = (gridRounded,1,1)

    circle_func(d_px_test, d_px_target, numpy.int32(len(radius)), totalPixels, x_coord, y_coord, radius_cuda, numpy.int32(targetIm.shape[1]) ,block=block,grid = grid)

    res = numpy.empty_like(px_target)
    cuda.memcpy_dtoh(res, d_px_test)

    res = (numpy.uint8(res))
    # pil_im = Image.fromarray(bwPx,mode ="RGB")
    return res
