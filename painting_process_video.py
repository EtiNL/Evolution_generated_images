from particles_class import Particles
import numpy as np
import cv2
import time
from PIL import Image
from score import loss
from skimage.filters.rank import entropy
from skimage.morphology import disk
from draw_particles import Draw_circles
from tqdm import tqdm

class Video_particle_manager:

    def __init__(self,filename,fps,targetImg,video_duration,particles):
        """Handle the particles that have been selected and draw them doubling the number of drawn particles each second
        and draw them at a random time within the second_frame"""
        self.filename = filename
        self.fps = fps
        self.targetIm = targetImg
        self.frame = np.zeros_like(self.targetIm)
        self.particles_queue = []
        self.video_duration = video_duration
        print(particles)
        self.particles = [particles[i,:] for i in range(particles.shape[0])]
        self.spawn_times = self.generate_spawn_time_for_particules()

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
            print(t*self.fps,Nbr_particle)
            spawn = list(np.sort(np.random.randint(t*self.fps, high=(t+1)*self.fps, size=Nbr_particle, dtype=int)))
            for spawn_time in spawn:
                spawn_times.append(spawn_time)
        print(spawn_times)
        return spawn_times

    def update_particles_queue_time(self):
        for particle in self.particles_queue:
            #particle = [x,y,radius,relative_time]
            if particle[3] != -1 and particle[3]*6/self.fps * self.Nbr_particles_per_second(particle[4]) < 6:
                #growing phase
                print('growing: ', particle,self.Nbr_particles_per_second(particle[4]))
                print(particle[3],particle[3]*6/self.fps * self.Nbr_particles_per_second(particle[4]))
                particle[3]+=1
                # particle[2] = particle[2]*(1-np.exp(-particle[3]*6/self.fps * self.Nbr_particles_per_second))
            else:
                # the particle reached its max radius and need to be removed from the queue after been drawn
                print('end: ', particle)
                print(particle[3],particle[3]*6/self.fps * self.Nbr_particles_per_second(particle[4]))
                particle[3] = -1

    def timed_radius(self,rad,t,t_spawn):
        if t >= 0:
            print('Nbr_particle test: ',self.Nbr_particles_per_second(t_spawn), 't_spawn :',t_spawn)
            print([self.Nbr_particles_per_second(t) for t in self.spawn_times])
            return rad*(1-np.exp(-t*6/self.fps * self.Nbr_particles_per_second(t_spawn)))
        else:
            return rad

    def video_generation(self):

        video=cv2.VideoWriter(f'{self.filename}.avi',cv2.VideoWriter_fourcc(*'DIVX'),fps = self.fps , frameSize=(self.targetIm.shape[1],self.targetIm.shape[0]))

        print([self.Nbr_particles_per_second(t) for t in self.spawn_times])

        particle_counter=0
        for t in tqdm(range(self.fps*self.video_duration)):
            print(f'spawn time {particle_counter}', self.spawn_times[particle_counter])
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
                if t_since_spawn == -1:
                    ind_filter.append(i)
            print(ind_filter)
            self.particles_queue = [self.particles_queue[i] for i in range(len(self.particles_queue)) if i not in ind_filter]

            self.update_particles_queue_time()

            if len(x_coord)>0:
                # print(t//self.fps)
                # print(self.particles_queue)
                # print(np.array(radius))
                # print(' ')
                self.frame = Draw_circles(self.targetIm, self.frame, np.array(x_coord), np.array(y_coord), np.array(radius))

            video.write(self.frame.astype(np.uint8))

        video.release()

def generate_particles(targetImg,number_gen,filename):
    genImg = np.zeros_like(targetImg)
    loss_val = loss(targetImg, genImg)
    ds_coef = 16

    particles = []

    for i in tqdm(range(number_gen)):
        radius_mean = []
        particle_found = False
        while not particle_found:
            gen = Particles(10, 1, targetImg, genImg, ds_coef)
            # cv2.imshow(f"Target", cv2.cvtColor(targetImg, cv2.COLOR_RGB2BGR))
            # if intermediary_show: cv2.imshow(f"gen{i}_init", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
            gen.keep_n_best(n=3)
            # if intermediary_show: cv2.imshow(f"gen{i}_keep", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
            gen.generate_noise(0.1)
            # if intermediary_show: cv2.imshow(f"gen{i}_noise", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
            gen.keep_n_best()
            loss_gen, genImg = gen.draw_particules(loss_val)
            if loss_gen < loss_val:
                particle_found = True
                particles.append([gen.scale(gen.center_pos_x[0]),gen.scale(gen.center_pos_y[0]), gen.scale(gen.radius[0])])
                radius_mean.append(gen.radius[0])
                loss_val = loss_gen
                # cv2.imshow(f"gen{i}", cv2.cvtColor(genImg, cv2.COLOR_RGB2BGR))


        if i%10==0:
            if np.mean(np.array(radius_mean)) < 2 and ds_coef>4:
                ds_coef = int(ds_coef//2)
                radius_mean = []

    # cv2.waitKey(0)

    with open(f'{filename}.npy', 'wb') as f:
        np.save(f, np.array(particles))

    print(f'Sucessfully created and saved particles at {filename}.npy')



if __name__=='__main__':
    start = time.time()
    targetImg = np.array(Image.open('raw_data/point.jpg'))
    generate_particles(targetImg,500,'raw_data/points_20')
    with open('raw_data/points_500.npy', 'rb') as f:
        particles = np.load(f)
    execution_time = time.time()
    print(execution_time//(60*60),'h ', execution_time//60-(execution_time//(60*60))*60,'min ', round(execution_time%60))
    # start = time.time()
    # Video_particle_manager('raw_data/points_500 ',24,targetImg,20,particles).video_generation()
    # print(execution_time//(60*60),'h ', execution_time//60-(execution_time//(60*60))*60,'min ', round(execution_time%60))
