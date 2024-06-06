import numpy as np
from score import score_generation, loss
from draw_particles import Draw_particules
import time
import cv2 as cv

class Particles:
    def __init__(self, nbr_particules, n_keep, target_img, particle_img, ds_coef):
        self.image_shape = target_img.shape
        self.ds_coef = ds_coef
        self.nbr_particules = nbr_particules
        self.n_keep = n_keep
        self.target_img = target_img
        self.ds_target_img = cv.resize(target_img, (self.image_shape[1]//ds_coef,self.image_shape[0]//ds_coef))
        self.particle_img = particle_img
        self.ds_particle_img = cv.resize(particle_img, (self.image_shape[1]//ds_coef,self.image_shape[0]//ds_coef))

        # self.type = np.random.randint(nbr_types, size=nbr_particules)
        # center_pos = np.random.randint((0,0), high=(image_shape[1],image_shape[0]), size=(nbr_particules, 2))
        self.center_pos_x, self.center_pos_y = self.depth_map_distribuated_center_pos()
        # self.orientation = np.random.randint(2*np.pi, size=nbr_particules)
        # self.max_radius = max(min(self.image_shape[0:2])//20*radius_decay,10)
        # self.radius = np.random.randint(2, high=self.max_radius, size=nbr_particules)
        self.max_radius = max(min(self.image_shape[0:2])//(ds_coef*4),4.5)
        self.radius, self.score = self.binary_search_radius()


    def binary_search_radius(self):
        radius = []
        score = []
        for i in range(self.center_pos_x.shape[0]):
            epsilon = 2
            rad_plus = self.max_radius
            rad_minus = 2
            rad_mid = (rad_plus-rad_minus)/2
            loss_plus = loss(self.ds_target_img, Draw_particules(self.ds_target_img, np.copy(self.ds_particle_img), np.array([self.center_pos_x[i]]), np.array([self.center_pos_y[i]]), np.array([rad_plus])))
            loss_minus = loss(self.ds_target_img, Draw_particules(self.ds_target_img, np.copy(self.ds_particle_img), np.array([self.center_pos_x[i]]), np.array([self.center_pos_y[i]]), np.array([rad_minus])))
            loss_mid = loss(self.ds_target_img, Draw_particules(self.ds_target_img, np.copy(self.ds_particle_img), np.array([self.center_pos_x[i]]), np.array([self.center_pos_y[i]]), np.array([rad_mid])))

            while np.abs(rad_plus-rad_minus)>epsilon:

                if np.min([loss_plus, loss_mid, loss_minus]) == loss_plus:
                    swap = rad_mid
                    rad_mid += (rad_plus-rad_mid)/2
                    rad_minus = swap
                    loss_minus = loss(self.ds_target_img, Draw_particules(self.ds_target_img, np.copy(self.ds_particle_img), np.array([self.center_pos_x[i]]), np.array([self.center_pos_y[i]]), np.array([rad_minus])))
                    loss_mid = loss(self.ds_target_img, Draw_particules(self.ds_target_img, np.copy(self.ds_particle_img), np.array([self.center_pos_x[i]]), np.array([self.center_pos_y[i]]), np.array([rad_mid])))

                elif np.min([loss_plus, loss_mid, loss_minus]) == loss_minus:
                    swap = rad_mid
                    rad_mid -= (rad_mid-rad_minus)/2
                    rad_plus = swap
                    loss_plus = loss(self.ds_target_img, Draw_particules(self.ds_target_img, np.copy(self.ds_particle_img), np.array([self.center_pos_x[i]]), np.array([self.center_pos_y[i]]), np.array([rad_plus])))
                    loss_mid = loss(self.ds_target_img, Draw_particules(self.ds_target_img, np.copy(self.ds_particle_img), np.array([self.center_pos_x[i]]), np.array([self.center_pos_y[i]]), np.array([rad_mid])))
                else:
                    rad_minus += (rad_mid-rad_minus)/2
                    rad_plus -= (rad_plus-rad_mid)/2
                    loss_plus = loss(self.ds_target_img, Draw_particules(self.ds_target_img, np.copy(self.ds_particle_img), np.array([self.center_pos_x[i]]), np.array([self.center_pos_y[i]]), np.array([rad_plus])))
                    loss_mid = loss(self.ds_target_img, Draw_particules(self.ds_target_img, np.copy(self.ds_particle_img), np.array([self.center_pos_x[i]]), np.array([self.center_pos_y[i]]), np.array([rad_minus])))


            if np.min([loss_plus, loss_mid, loss_minus]) == loss_plus:
                radius.append(rad_plus)
                score.append(loss_plus)
            elif np.min([loss_plus, loss_mid, loss_minus]) == loss_mid:
                radius.append(rad_mid)
                score.append(loss_mid)
            else:
                radius.append(rad_minus)
                score.append(loss_minus)
        return np.array(radius), np.array(score)

    def depth_map_distribuated_center_pos(self):
        depth_map = np.sum(np.abs(self.ds_target_img - self.ds_particle_img), axis = 2)/3
        max_depth, min_depth = np.max(depth_map), np.min(depth_map)
        depth_map = (depth_map - min_depth)/(max_depth - min_depth)
        distrib_depth = depth_map/np.sum(depth_map)
        # Create a flat copy of the array
        flat_distrib_depth = distrib_depth.flatten()

        # Then, sample an index from the 1D array with the
        # probability distribution from the original array
        sample_index = np.random.choice(flat_distrib_depth.size, size = self.nbr_particules, p=flat_distrib_depth)

        # Take this index and adjust it so it matches the original array
        center_pos_y, center_pos_x = np.unravel_index(sample_index, distrib_depth.shape)
        return center_pos_x, center_pos_y


    def get_score(self):
        return score_generation(self.ds_particle_img, self.ds_target_img, self.center_pos_x, self.center_pos_y, self.radius)
    def keep_n_best(self, n = 0):
        if n==0: n = self.n_keep
        # print(score)
        # print(self.center_pos_x)
        # print(self.center_pos_y)
        kept_particle_index = np.argsort(self.score)[:n]
        # print(kept_particle_index)
        self.center_pos_x = self.center_pos_x[kept_particle_index.astype(int)]
        self.center_pos_y = self.center_pos_y[kept_particle_index.astype(int)]
        self.radius = self.radius[kept_particle_index.astype(int)]

        #reordering by radius size
        # ind_radius = np.argsort(-self.radius)
        # self.center_pos_x = self.center_pos_x[ind_radius.astype(int)]
        # self.center_pos_y = self.center_pos_y[ind_radius.astype(int)]
        # self.radius = self.radius[ind_radius.astype(int)]
        # print(self.center_pos_x)
        # print(self.center_pos_y)

    def generate_noise(self, level, nbr_derivates = 3):
        pos = zip(list(self.center_pos_x),list(self.center_pos_y),list(self.radius))
        pos_x_noise = []
        pos_y_noise = []
        radius_noise = []
        for x,y,r in pos:
            pos_x_noise.append(x)
            pos_y_noise.append(y)
            radius_noise.append(r)


            for i in range(nbr_derivates):
                x_noise = float((np.random.rand(1)-0.5)*2)*level*self.image_shape[0]
                if x_noise+x<0: x_noise = 0
                if x_noise+x>self.image_shape[0]: x_noise = 0
                pos_x_noise.append(x+x_noise)

                y_noise = float((np.random.rand(1)-0.5)*2)*level*self.image_shape[1]
                if y_noise+y<0: y_noise = 0
                if y_noise+y>self.image_shape[0]: y_noise = 0
                pos_y_noise.append(y+y_noise)

                r_noise = float((np.random.rand(1)-0.5)*2)*level*self.max_radius
                if r_noise+r<2: r_noise = 0
                radius_noise.append(r+r_noise)
        self.center_pos_x = np.array(pos_x_noise)
        self.center_pos_y = np.array(pos_y_noise)
        self.radius = np.array(radius_noise)
        self.score = self.get_score()

    def scale(self,arr):
        return self.ds_coef * arr

    def draw_particules(self, rad, previous_loss):

        new_img = Draw_particules(self.target_img, np.copy(self.particle_img), self.scale(self.center_pos_x), self.scale(self.center_pos_y), self.scale(self.radius))

        loss_val = loss(self.target_img, new_img)
        if previous_loss < loss_val:
            return rad, previous_loss, self.particle_img, False

        else:
            rad.append(self.radius[0])
            return rad, loss_val, new_img, True
    
