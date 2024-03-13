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
                self.frame = Draw_circles(self.targetIm, self.frame, np.array(x_coord), np.array(y_coord), np.array(radius))

            video.write(self.frame.astype(np.uint8))

        video.release()
        print('Number of particles that ended growing: ',self.number_end)

def generate_particles(targetImg,number_gen,filename, scoring = 'entropy'):
    genImg = np.ones_like(targetImg)*255
    ds_coef = 1

    particles = []
    Mean_rad = []

    for i in tqdm(range(number_gen)):
        radius_mean = []
        particle_found = False
        gen = Particles(50  , 1, targetImg, genImg, ds_coef, scoring)
        # cv2.imshow(f"Target", cv2.cvtColor(targetImg, cv2.COLOR_RGB2BGR))
        # if intermediary_show: cv2.imshow(f"gen{i}_init", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
        gen.keep_n_best(n=10)
        # if intermediary_show: cv2.imshow(f"gen{i}_keep", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
        gen.generate_noise(0.1)
        # if intermediary_show: cv2.imshow(f"gen{i}_noise", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
        gen.keep_n_best()
        genImg = gen.draw_particules()
        # print(loss_gen, loss_val)
        # print(' ')
        particles.append([gen.scale(gen.center_pos_x[0]),gen.scale(gen.center_pos_y[0]), gen.scale(gen.radius[0])])
        radius_mean.append(gen.radius[0])
        # cv2.imshow(f"gen{i}", cv2.cvtColor(genImg, cv2.COLOR_RGB2BGR))


        if i%10==0:
            # print(np.mean(np.array(radius_mean)))
            # if np.mean(np.array(radius_mean)) < 5: #and ds_coef>4:
                # ds_coef = int(ds_coef//2)
            Mean_rad.append(np.mean(gen.scale(np.array(radius_mean))))
            radius_mean = []

    # cv2.waitKey(0)

    with open(f'{filename}.npy', 'wb') as f:
        np.save(f, np.array(particles))

    print(f'Sucessfully created and saved particles at {filename}.npy')
    return Mean_rad



if __name__=='__main__':
    import matplotlib.pyplot as plt
    start = time.time()
    targetImg = np.array(Image.open('raw_data/point.jpg'))
    mean_rad = generate_particles(targetImg,100,'raw_data/points_100',scoring ='intensity')
    # plt.plot(mean_rad)
    with open('raw_data/points_100.npy', 'rb') as f:
        particles = np.load(f)
    execution_time = time.time() - start
    print(execution_time//(60*60),'h ', execution_time//60-(execution_time//(60*60))*60,'min ', round(execution_time%60))
    start = time.time()
    Video_particle_manager('raw_data/points_10',24,targetImg,20,particles).video_generation()
    execution_time = time.time() - start
    print(execution_time//(60*60),'h ', execution_time//60-(execution_time//(60*60))*60,'min ', round(execution_time%60))
    plt.show()
