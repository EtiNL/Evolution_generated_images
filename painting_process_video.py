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
from gpu_utils import get_max_threads_per_block
from draw_particles import Draw_particules


class Video_particle_manager:

    def __init__(self, filename, fps, targetImg_path, video_duration, particles: np.array):
        """
        Initialize the Video_particle_manager with the given parameters.
        
        Parameters:
        filename (str): The name of the output video file.
        fps (int): Frames per second for the video.
        targetImg_path (str): Path to the target image.
        video_duration (int): Duration of the video in seconds.
        particles (ndarray): Array containing particle data [x, y, radius].
        """
        self.filename = filename
        self.fps = fps
        self.targetIm = np.array(Image.open(targetImg_path))
        self.frame = np.zeros_like(self.targetIm)
        self.particles = [particles[i, :] for i in range(particles.shape[0])]
        self.video_duration = video_duration
        self.particles_queue = []
        self.spawn_times = self.generate_spawn_time_for_particles()
        self.number_end = 0

    def Nbr_particles_per_second(self, t):
        """
        Calculate the number of particles to spawn each second.
        
        Parameters:
        t (int): Current time in frames.
        
        Returns:
        int: Number of particles to spawn.
        """
        if t < self.fps:
            return 0
        elif len(self.particles) > (int(len(self.particles) / (2 ** self.video_duration + 1)) + 1) * (2 ** (t // self.fps) - 1):
            return (int(len(self.particles) / (2 ** self.video_duration + 1)) + 1) * 2 ** (t // self.fps - 1)
        elif len(self.particles) > (int(len(self.particles) / (2 ** self.video_duration + 1)) + 1) * (2 ** (t // self.fps - 1) - 1):
            return len(self.particles) - (int(len(self.particles) / (2 ** self.video_duration + 1)) + 1) * (2 ** (t // self.fps - 1) - 1)
        else:
            return 0

    def generate_spawn_time_for_particles(self):
        """
        Generate spawn times for particles.
        
        Returns:
        list: List of spawn times in frames.
        """
        spawn_times = []
        for t in range(1, self.video_duration):
            Nbr_particle = self.Nbr_particles_per_second(t * self.fps)
            spawn = list(np.sort(np.random.randint((t - 1) * self.fps, high=t * self.fps, size=Nbr_particle, dtype=int)))
            for spawn_time in spawn:
                spawn_times.append(spawn_time)
        return spawn_times

    def update_particles_queue_time(self):
        """
        Update the time for particles in the queue.
        """
        for particle in self.particles_queue:
            if particle[3] != -1 and particle[3] * 6 / self.fps < 6:
                particle[3] += 1
            else:
                self.number_end += 1
                particle[3] = -1

    def timed_radius(self, rad, t, t_spawn):
        """
        Calculate the radius of a particle based on the time since its spawn.
        
        Parameters:
        rad (float): Initial radius of the particle.
        t (int): Time since the particle spawned in frames.
        t_spawn (int): Spawn time of the particle in frames.
        
        Returns:
        float: Adjusted radius of the particle.
        """
        if t >= 0:
            return rad * (1 - np.exp(-t * 6 / self.fps))
        else:
            return rad

    def video_generation(self):
        """
        Generate the video with particles.
        """
        video = cv2.VideoWriter(f'{self.filename}.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps=self.fps, frameSize=(self.targetIm.shape[1], self.targetIm.shape[0]))

        particle_counter = 0
        t_morph = None
        countdown_t_morph = 2*self.fps
        morph_duration = 2*self.fps
        
        for t in tqdm(range(self.fps * self.video_duration)):
            if particle_counter < len(self.spawn_times):
                while self.spawn_times[particle_counter] == t:
                    x, y, radius = self.particles.pop(0)
                    self.particles_queue.append([x, y, radius, 0, t])
                    if particle_counter < len(self.spawn_times) - 1:
                        particle_counter += 1
                    else:
                        break

            x_coord = []
            y_coord = []
            radius = []
            ind_filter = []
            for i in range(len(self.particles_queue)):
                x, y, rad, t_since_spawn, spawn_time = self.particles_queue[i]
                x_coord.append(x)
                y_coord.append(y)
                radius.append(self.timed_radius(rad, t_since_spawn, spawn_time))
                if t_since_spawn != -1:
                    ind_filter.append(i)

            self.particles_queue = [self.particles_queue[i] for i in ind_filter]
            self.update_particles_queue_time()

            if len(x_coord) > 0:
                self.frame = Draw_particules(self.targetIm, self.frame, np.array(x_coord), np.array(y_coord), np.array(radius))
                video.write(cv2.cvtColor(self.frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
            # Adding morphing effect
            if len(self.particles) == 0:
                if t_morph == None and countdown_t_morph>0:
                    countdown_t_morph -= 1
                    
                else: 
                    if t_morph == None:
                        t_morph = t
                
                    if t - t_morph <= morph_duration:
                        alpha = (t - t_morph) / morph_duration
                        morphed_frame = cv2.addWeighted(cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR), 1 - alpha, cv2.cvtColor(self.targetIm, cv2.COLOR_RGB2BGR), alpha, 0)
                        video.write(morphed_frame)
                    
                    else: video.write(cv2.cvtColor(self.targetIm, cv2.COLOR_RGB2BGR))

        video.release()
        print('Number of particles that ended growing: ', self.number_end)

def generate_particles(targetImg_path, number_gen, particles_per_gen):
    targetImg = np.array(Image.open(targetImg_path))
    genImg = np.zeros_like(targetImg)
    ds_coef = 16

    particles = []
    loss_val = loss(targetImg, genImg)
    time_between_particles = time.time()

    for i in tqdm(range(number_gen)):
        radius_mean = []
        particle_found = False
        
        while particle_found == False:
            gen = Particles(particles_per_gen, 1, targetImg, genImg, ds_coef)
            # cv2.imshow(f"Target", cv2.cvtColor(targetImg, cv2.COLOR_RGB2BGR))
            # if intermediary_show: cv2.imshow(f"gen{i}_init", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
            gen.keep_n_best(n=2)
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

        time_between_particles = time.time()
        if np.mean(np.array(radius_mean)) < 2:
            ds_coef = int(ds_coef//2)
            print(f'Gen {i}, ds_coef = {ds_coef}')
            if ds_coef < 2 and np.mean(np.array(radius_mean)) < 2:
                break

    # cv2.waitKey(0)

    with open(f'{targetImg_path}_{len(particles)}_particles.npy', 'wb') as f:
        np.save(f, np.array(particles))

    print(f'Sucessfully created and saved {len(particles)} particles')
    
    return f'{targetImg_path}_{len(particles)}_particles.npy'

def main(target_Img_path, nbr_gen, particles_per_gen):
    start = time.time() 
    path_generated_particles = generate_particles(target_Img_path,nbr_gen,particles_per_gen)
    # plt.plot(mean_rad)
    with open(path_generated_particles, 'rb') as f:
        particles = np.load(f)
    execution_time = time.time() - start
    print(execution_time//(60*60),'h ', execution_time//60-(execution_time//(60*60))*60,'min ', round(execution_time%60))
    start = time.time()
    Video_particle_manager(path_generated_particles,24,target_Img_path,20,particles).video_generation()
    execution_time = time.time() - start
    print(execution_time//(60*60),'h ', execution_time//60-(execution_time//(60*60))*60,'min ', round(execution_time%60))
    
def main_intermediary(target_Img_path, saved_particles_path, nbr_gen, particles_per_gen):
    with open(saved_particles_path, 'rb') as f:
        particles = list(np.load(f))
    particles_copy =  np.array(particles.copy())
    targetImg = np.array(Image.open(target_Img_path))
    genImg = np.zeros_like(targetImg)
    print(nbr_gen > len(particles))
    if nbr_gen > len(particles):
        genImg = Draw_particules(targetImg, genImg, particles_copy[:,0], particles_copy[:,1], particles_copy[:,2])

        ds_coef = 16

        loss_val = loss(targetImg, genImg)
        radius_mean = []
        for i in tqdm(range(nbr_gen - len(particles))):
            particle_found = False

            while particle_found == False:
                gen = Particles(particles_per_gen, 1, targetImg, genImg, ds_coef)
                # cv2.imshow(f"Target", cv2.cvtColor(targetImg, cv2.COLOR_RGB2BGR))
                # if intermediary_show: cv2.imshow(f"gen{i}_init", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
                gen.keep_n_best(n=2)
                # if intermediary_show: cv2.imshow(f"gen{i}_keep", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
                gen.generate_noise(0.1)
                # if intermediary_show: cv2.imshow(f"gen{i}_noise", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
                gen.keep_n_best()
                radius_mean, loss_val, genImg, particle_found = gen.draw_particules(radius_mean, loss_val)
                # print(loss_gen, loss_val)
                # print(' ')
                if particle_found:
                    particles.append([gen.scale(gen.center_pos_x[0]), gen.scale(gen.center_pos_y[0]), gen.scale(gen.radius[0])])
                    radius_mean.append(gen.radius[0])
                if len(radius_mean) >= 10:
                    radius_mean.pop(0)
                # cv2.imshow(f"gen{i}", cv2.cvtColor(genImg, cv2.COLOR_RGB2BGR))

            if np.mean(np.array(radius_mean)) < 1.2:
                ds_coef = int(ds_coef // 2)
                print(f'Gen {i}, ds_coef = {ds_coef}')
                if ds_coef < 2 and np.mean(np.array(radius_mean)) < 2:
                    break

        # cv2.waitKey(0)

        with open(f'{target_Img_path}_{len(particles)}_particles.npy', 'wb') as f:
            np.save(f, np.array(particles))

        print(f'Successfully created and saved {len(particles)} particles')

    Video_particle_manager(f'{target_Img_path.split(".")[0]}_{len(particles)}_particles', 24, target_Img_path, 20, np.array(particles)).video_generation()


if __name__=='__main__':
    # parser = argparse.ArgumentParser(
    #                 prog='Generate video',
    #                 description='',
    #                 epilog='')
    
    # parser.add_argument('filename') 
    # parser.add_argument('nbr_gen')
    # parser.add_argument('particles_per_gen')
    # args = parser.parse_args()
    # print(args.filename, int(args.nbr_gen), int(args.particles_per_gen))
    # main(args.filename, int(args.nbr_gen), int(args.particles_per_gen))
    main_intermediary("La_force_des_vagues.JPG", "La_force_des_vagues.JPG_5000_particles.npy", 5000, 20)

