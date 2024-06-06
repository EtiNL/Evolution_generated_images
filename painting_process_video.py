from particles_class import Particles
import argparse
import numpy as np
import cv2
import time
from PIL import Image
from score import loss
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tqdm import tqdm

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
class Video_particle_manager:

    def __init__(self,filename,fps,targetImg_path,video_duration,particles):
        """Handle the particles that have been selected and draw them doubling the number of drawn particles each second
        and draw them at a random time within the second_frame"""
        targetImg = np.array(Image.open(targetImg_path))
        self.filename = filename
        self.fps = fps
        self.targetIm = targetImg
        self.frame = np.zeros_like(self.targetIm)
        self.particles_queue = []
        self.video_duration = video_duration
        # print(particles)
        self.particles = [particles[i,:] for i in range(particles.shape[0])]
        self.spawn_times = self.generate_spawn_time_for_particules()
        self.number_end = 0

    def Nbr_particles_per_second(self, t):
        if t < self.fps:
            return 0
        elif len(self.particles) > (int(len(self.particles)/(2**self.video_duration+1))+1)*(2**(t // self.fps)-1):
            return (int(len(self.particles)/(2**self.video_duration+1))+1)*2**(t // self.fps -1)
        elif len(self.particles) > (int(len(self.particles)/(2**self.video_duration+1))+1)*(2**(t // self.fps-1)-1):
            return len(self.particles) - (int(len(self.particles)/(2**self.video_duration+1))+1)*(2**(t // self.fps-1)-1)
        else:
            return 0

    def generate_spawn_time_for_particules(self):
        spawn_times = []
        for t in range(1,self.video_duration):
            Nbr_particle = self.Nbr_particles_per_second(t*self.fps)
            # print(t*self.fps,Nbr_particle)
            spawn = list(np.sort(np.random.randint((t-1)*self.fps, high=t*self.fps, size=Nbr_particle, dtype=int)))
            for spawn_time in spawn:
                spawn_times.append(spawn_time)
        # print(spawn_times)
        return spawn_times

    def update_particles_queue_time(self):
        for particle in self.particles_queue:
            #particle = [x,y,radius,relative_time]
            if particle[3] != -1 and particle[3]*6/self.fps < 6:
                #growing phase
                # print('growing: ', particle,self.Nbr_particles_per_second(particle[4]))
                # print(particle[3],particle[3]*6/self.fps * self.Nbr_particles_per_second(particle[4]))
                particle[3]+=1
                # particle[2] = particle[2]*(1-np.exp(-particle[3]*6/self.fps * self.Nbr_particles_per_second))
            else:
                # the particle reached its max radius and need to be removed from the queue after been drawn
                #print('end: ', particle)
                #print(particle[3],particle[3]*6/self.fps * self.Nbr_particles_per_second(particle[4]))
                self.number_end+=1
                particle[3] = -1

    def timed_radius(self,rad,t,t_spawn):
        if t >= 0:
            # print('Nbr_particle test: ',self.Nbr_particles_per_second((t_spawn//self.fps +1)*self.fps), 't_spawn :',t_spawn)
            #print([self.Nbr_particles_per_second(t) for t in self.spawn_times])
            return rad*(1-np.exp(-t*6/self.fps))  #* self.Nbr_particles_per_second((t_spawn//self.fps +1)*self.fps))
        else:
            return rad

    def video_generation(self):

        video=cv2.VideoWriter(f'{self.filename}.avi',cv2.VideoWriter_fourcc(*'DIVX'),fps = self.fps , frameSize=(self.targetIm.shape[1],self.targetIm.shape[0]))

        #print([self.Nbr_particles_per_second(t) for t in self.spawn_times])

        particle_counter=0
        for t in tqdm(range(self.fps*self.video_duration)):
            #print(f'spawn time {particle_counter}', self.spawn_times[particle_counter])
            if particle_counter<len(self.spawn_times):

                while self.spawn_times[particle_counter] == t:
                    x, y, radius = self.particles.pop(0)
                    self.particles_queue.append([x, y, radius, 0, t])#zero represents the time since its spawn and t the spawn time
                    if particle_counter<len(self.spawn_times)-1:
                        particle_counter+=1
                    else:
                        break
                # print(self.spawn_times)
                # self.spawn_times = self.spawn_times[i:]
                # print(self.spawn_times)
                # print('   ')
            x_coord = []
            y_coord = []
            radius = []

            ind_filter = []
            for i in range(len(self.particles_queue)):
                x, y, rad, t_since_spawn, spawn_time = self.particles_queue[i]
                x_coord.append(x)
                y_coord.append(y)
                radius.append(self.timed_radius(rad,t_since_spawn,spawn_time))
                if t_since_spawn != -1:
                    ind_filter.append(i)
            #print(ind_filter)
            self.particles_queue = [self.particles_queue[i] for i in ind_filter]

            self.update_particles_queue_time()

            if len(x_coord)>0:
                # print(t//self.fps)
                # print(self.particles_queue)
                # print(np.array(radius))
                # print(' ')
                self.frame = Draw_particules(self.targetIm, self.frame, np.array(x_coord), np.array(y_coord), np.array(radius))

            video.write(self.frame.astype(np.uint8))

        video.release()
        print('Number of particles that ended growing: ',self.number_end)

def generate_particles(targetImg_path, number_gen, scoring = 'entropy'):
    targetImg = np.array(Image.open(targetImg_path))
    genImg = np.zeros_like(targetImg)
    ds_coef = 16

    particles = []
    loss_val = loss(targetImg, genImg)

    for i in tqdm(range(number_gen)):
        radius_mean = []
        particle_found = False
        
        while particle_found == False:
            gen = Particles(100, 1, targetImg, genImg, ds_coef)
            # cv2.imshow(f"Target", cv2.cvtColor(targetImg, cv2.COLOR_RGB2BGR))
            # if intermediary_show: cv2.imshow(f"gen{i}_init", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
            gen.keep_n_best(n=10)
            # if intermediary_show: cv2.imshow(f"gen{i}_keep", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
            gen.generate_noise(0.1)
            # if intermediary_show: cv2.imshow(f"gen{i}_noise", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
            gen.keep_n_best()
            radius_mean, loss_val, genImg, particle_found = gen.draw_particules(radius_mean, loss_val)
            # print(loss_gen, loss_val)
            # print(' ')
            if particle_found:
                particles.append([gen.scale(gen.center_pos_x[0]),gen.scale(gen.center_pos_y[0]), gen.scale(gen.radius[0])])
                radius_mean.append(gen.radius[0])
            if len(radius_mean)>=10:
                radius_mean.pop(0)
            # cv2.imshow(f"gen{i}", cv2.cvtColor(genImg, cv2.COLOR_RGB2BGR))


        if np.mean(np.array(radius_mean)) < 5:
            ds_coef = int(ds_coef//2)
            print(f'Gen {i}, ds_coef = {ds_coef}')
            if ds_coef < 2 and np.mean(np.array(radius_mean)) < 2:
                break

    # cv2.waitKey(0)

    with open(f'{targetImg_path}_{len(particles)}_particles.npy', 'wb') as f:
        np.save(f, np.array(particles))

    print(f'Sucessfully created and saved {len(particles)} particles')
    
    return f'{targetImg_path}_{len(particles)}_particles.npy'

def main(target_Img_path, nbr_gen):
    start = time.time()
    path_generated_particles = generate_particles(target_Img_path,nbr_gen,scoring ='intensity')
    # plt.plot(mean_rad)
    with open(path_generated_particles, 'rb') as f:
        particles = np.load(f)
    execution_time = time.time() - start
    print(execution_time//(60*60),'h ', execution_time//60-(execution_time//(60*60))*60,'min ', round(execution_time%60))
    start = time.time()
    Video_particle_manager(path_generated_particles,24,target_Img_path,20,particles).video_generation()
    execution_time = time.time() - start
    print(execution_time//(60*60),'h ', execution_time//60-(execution_time//(60*60))*60,'min ', round(execution_time%60))

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog='Generate video',
                    description='',
                    epilog='')
    
    parser.add_argument('filename') 
    parser.add_argument('nbr_gen')
    args = parser.parse_args()
    main(args.filename, int(args.nbr_gen))

