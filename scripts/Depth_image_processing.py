#!/usr/bin/env python3.6

import numpy as np
import time

import pyransac3d as pyrsc
import math
import random
from copy import deepcopy
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped


class SLZ_detection:
    def __init__(self):
        # path = '123.npy'
        # self.image = np.load(path)
        print('module started!')
        self.image = None
        # rospy.Subscriber("/camera/depth/points", Image, self.save_depth_image)
        # self.pub_point = rospy.Publisher('slz_point', PoseStamped, queue_size=2)
        self.i = 0
        self.best_SLZ = None

        self.edge_region = [0., 0., 0., 0.]

    def save_depth_image(self, data):
        self.image = data
        if self.i % 10:
            self.det_SLZ(self.image)
        self.i += 1

    def crop_edge(self, image, x_crop, y_crop):
        x, y,_ = np.shape(image)
        crop_image = image[x_crop: x-x_crop, y_crop: y-y_crop]
            
        return crop_image

    def create_image_grid(self, image, dim_grid):
        x, y, _ = np.shape(image)
        list_grid_image = []
        ith_grid_image = []
        shape_grid = np.array([x//dim_grid[0], y//dim_grid[1]])
        for i in range(shape_grid[0]):
            ith_grid_image = []
            for j in range(shape_grid[1]):
                grid_image = image[i*dim_grid[0]:(i+1)*dim_grid[0], j*dim_grid[1]:(j+1)*dim_grid[1]]
                ith_grid_image.append(grid_image)
            list_grid_image.append(ith_grid_image)
        return list_grid_image

    def calc_plane_equation(self, grid_image):
        plane = pyrsc.Plane()
        x, y, z = np.shape(grid_image)
        image_array = np.reshape(grid_image, (x*y, 3))

        sample_image = image_array[range(0, x*y, 7)]
        try:
            # start = time.time()
            best_eq, best_inliers = plane.fit(sample_image, 0.1, maxIteration = 50)
            # print("time :", time.time() - start)

            return best_eq, best_inliers
        except:
            return 0, 0

    def calc_slopeness(self, plane_eq):
        n_z = plane_eq[2] / np.sqrt(plane_eq[0]**2 + plane_eq[1]**2 + plane_eq[2]**2)
        alpha = np.arccos(abs(n_z))
        alpha_th = math.radians(15)
        if abs(alpha) < alpha_th:
            bool_slopeness = 1
        else:
            bool_slopeness = 0
        return bool_slopeness, alpha
            
    def calc_roughness(self, plane_eq, points):
        nx, ny, _ = np.shape(points)
        points_vector = np.reshape(points,(nx*ny, 3))
        points = points_vector
        z_plane_vector = -1*(plane_eq[0] * points[:,0] + plane_eq[1] * points[:,1] + plane_eq[3])/plane_eq[2]
        
        error_z = points[:,2] - z_plane_vector
        sum_error = sum(map(abs, error_z)) / (nx * ny)
        
        if sum_error > 0.05:
            return 0, sum_error
        else:
            return 1, sum_error

    def convolution(self, image, s=1): # s is stride
        filter = np.array([[1,1,1],[1,1,1],[1,1,1]])
        mx, my = np.shape(image)
        fx, fy = np.shape(filter)
        if (mx-fx)%s==0 and (my-fy)%s==0:
            # print('convolution possible')
            pass 
        else: 
            print('convolution impossible') 
            print('(mx-fx)%s=',(mx-fx)%s) 
            print('(my-fy)%s=',(my-fy)%s) 
            return
        
        o = []
        
        for i in range(0, mx-fx+1, s):
            for j in range(0, my-fy+1, s):
                o.append((image[i:i+fx, j:j+fy]*filter).sum())
        ow = int((mx-fx)/s)+1
        oh = int((my-fy)/s)+1
        # print('Size of output = (%s, %s)' %(ow,oh))
        o = np.array(o).reshape(ow, oh)
        return o 

    def estimate_SLZ(self, conv_image, level, image, crop_num, dim_grid, param_state):
        
        nx, ny = np.shape(conv_image)
        image_line = conv_image.reshape(nx * ny)

        index = np.where(image_line == 9**level)
        ii = np.array(index[0] // ny)
        jj = np.array(index[0] % ny)

        # ii = (ii +1/2 + level + crop_num[0]) * dim_grid[0]
        # jj = (jj +1/2 + level + crop_num[1]) * dim_grid[1]
        
        idxs = np.array([ii,jj]).T

        param = np.array([param_state[int(i + level), int(j + level)] for i, j in idxs])

        idxs_2 = (idxs + 1/2 + level + crop_num) * dim_grid
        
        radius = []
        for i, j in idxs_2:
            s = int(40*level)
            i, j = int(i), int(j)
            u = min(image[i + s, j - s : j + s, 1])
            d = max(image[i - s, j - s : j + s, 1])
            l = max(image[i - s : i + s , j - s, 0]) # recently changed, check when the error occured
            r = min(image[i - s : i + s , j + s, 0])
            ver_d = u - d
            hor_d = r - l
            rad = min(ver_d, hor_d)/2
            if rad < 0:
                rad = 0
                print('radius is minus')
            radius.append(rad)

        pre_state = np.array([image[int(i), int(j)] for i, j in idxs_2])  

        state = np.c_[pre_state, radius, param]
        
        return state # n x 5 (x_t, y_t, z_t, r, alpha, ri)

    def calc_state_vector(self, center_index_1, center_index_2):
        for point_2 in center_index_2:
            n = len(center_index_1)
            for idx in range(len(center_index_1)):
                point_1 = center_index_1[n - idx -1]
                diff = point_2[0:2] - point_1[0:2]
                dist = np.linalg.norm(diff)
                if dist < point_2[3]:
                    center_index_1 = np.delete(center_index_1, n - idx -1, axis = 0)
        center_index_1 = center_index_1.tolist()
        center_index_2 = center_index_2.tolist()

        if len(center_index_1):
            state_vector = np.vstack([center_index_1, center_index_2])
        else:
            state_vector = deepcopy(center_index_2)
        return state_vector


    def det_sub_SLZ(self, grid_image, dim_grid, plane_time):    
        start = time.time()

        best_eq, best_inliers = self.calc_plane_equation(grid_image)
        
        plane_time += time.time() - start 

        if best_eq == 0 or best_eq == []:
            bool_sub_SLZ = 0
            return bool_sub_SLZ, plane_time, 0, 0
        
        else:
            bool_slopeness, alpha = self.calc_slopeness(best_eq)
            bool_roughness, sum_error = self.calc_roughness(best_eq, grid_image)    
            if bool_slopeness and bool_roughness:
                bool_sub_SLZ = 1
            else:
                bool_sub_SLZ = 0
                
            return bool_sub_SLZ, plane_time, alpha, sum_error

    def det_SLZ(self, image):
        self.image_region = self.find_region(image)
        # output: coordinates of SLZ and shape, extension
        list_SLZ = []
        
        dim_grid = np.array([40, 40])
        crop_num = [2, 2]
        crop_image = self.crop_edge(image, dim_grid[0]*crop_num[0], dim_grid[1]*crop_num[1])

        list_grid_image = self.create_image_grid(crop_image, dim_grid)
        gx, gy, _, _, _ = np.shape(list_grid_image)
        
        plane_time = 0
        param_state = []
        for i in range(gx):
            for j in range(gy):
                bool_sub_SLZ, plane_time, alpha, sum_error = self.det_sub_SLZ(list_grid_image[i][j], dim_grid, plane_time)

                if bool_sub_SLZ:
                    list_SLZ.append(1)
                else:
                    list_SLZ.append(0)
                param_state.append([alpha, sum_error])

        list_SLZ = np.array(list_SLZ).reshape(gx, gy)
        param_state = np.array(param_state).reshape(gx, gy, 2)
        
        conv_image_1 = self.convolution(list_SLZ)
        center_1 = self.estimate_SLZ(conv_image_1, 1, image, crop_num, dim_grid, param_state)
        
        conv_image_2 = self.convolution(conv_image_1)
        center_2 = self.estimate_SLZ(conv_image_2, 2, image, crop_num, dim_grid, param_state)
        
        state_vector = deepcopy(center_1)
        iter = 3
        for _ in range(2):
        # while True:
            if len(center_2):
                state_vector = self.calc_state_vector(center_1, center_2)
                
                conv_image_3 = self.convolution(conv_image_2)
                center_1 = deepcopy(center_2)
                center_2 = self.estimate_SLZ(conv_image_3, iter, image, crop_num, dim_grid, param_state)
                conv_image_1 = deepcopy(conv_image_2)
                conv_image_2 = deepcopy(conv_image_3)
                iter += 1
            else:
                break

        print('plane_time : ', plane_time)

        self.best_SLZ = state_vector


    def find_region(self, us_image):
        x_min = min(min(us_image[:, 0, 0]), min(us_image[0, :, 0]), min(us_image[:, -1, 0]), min(us_image[-1, :, 0]))
        y_min = min(min(us_image[:, 0, 1]), min(us_image[0, :, 1]), min(us_image[:, -1, 1]), min(us_image[-1, :, 1]))
        x_max = max(max(us_image[:, 0, 0]), max(us_image[0, :, 0]), max(us_image[:, -1, 0]), max(us_image[-1, :, 0]))
        y_max = max(max(us_image[:, 0, 1]), max(us_image[0, :, 1]), max(us_image[:, -1, 1]), max(us_image[-1, :, 1]))

        return [x_min, x_max, y_min, y_max]