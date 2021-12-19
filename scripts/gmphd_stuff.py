#!/usr/bin/env python

import numpy as np
import math
from Gmphd import Gmphd, GmphdComponent
from filterpy.common.discretization import  Q_discrete_white_noise
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge, CvBridgeError


def pcd_coord_transform(pos, state_vector):
    # (P_t - P_o) + P_o --> P_t
    x, y, z =  state_vector[0] + pos[0], state_vector[2] + pos[1], state_vector[4] + pos[2]
    # doesnot have to consider rotation?
    state_vector[0], state_vector[2], state_vector[4] = x, y, z
    return state_vector

def calc_score(weight, loc):
    # calculate the score of each SLZ candidates
    # coeff_weight = 20. # weight, which should be normalized
    # coeff_dist = - 0.07 * 0.2 # distance from ownship to SLZ
    coeff_slope = -1. # slope of SLZ
    coeff_ri = -4. # roughness index of SLZ

    # dist = np.linalg.norm([loc[0], loc[2], loc[4]])
    # dist = np.linalg.norm([loc[0], loc[2]]) # horizontal distance?
    # print('weight: ', weight, 'dist: ', dist, 'slope: ', loc[7], 'ri: ', loc[8])
    if len(loc):
        # score = (c1 * weight) + (c2 * distance btw uav and target) + (c3 * slope) + (c4 * roughness index)
        # score = coeff_weight * weight + coeff_dist * dist + coeff_slope * loc[7] + coeff_ri * loc[8]
        score = coeff_slope * loc[7] + coeff_ri * loc[8]
    else:
        score = -10000.
        print('loc is empty')
    return score

def updateandprune(g, obsset, dt, vel):
    f = create_F(dt)
    print("-------------------------------------------------------------------")
    # update GM-PHD
    g.update(obsset, f, vel) 
    # merge similar components and prune the components with low weight
    g.prune(maxcomponents=50, mergethresh=0.5) 

def birth_gmm(pos, vel, weight):

    P_0 = np.array([[1, 0.1, 0, 0, 0, 0, 0, 0, 0],
                    [0.1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0.1, 0, 0, 0, 0, 0],
                    [0, 0, 0.1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0.1, 0, 0, 0],
                    [0, 0, 0, 0, 0.1, 1, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]],  dtype=float) * 1.  # initial uncertainty

    # create birth component
    # GmphdComponent(weight, state, covariance)
    x_sample = np.random.uniform(-20, 20, 4)
    y_sample = np.random.uniform(-20, 20, 4)
    birth_set = [GmphdComponent(weight, [x, - vel[0], y, -vel[1], - pos[2], - vel[2], \
        2., 0.2, 0.05], P_0) for x in x_sample for y in y_sample]
    return birth_set

def create_F(dt):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0
    F = np.array([[1, dt, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, dt, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, dt, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)
    return F

def create_H():
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]

    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]],  dtype=float)
    return H

def create_H_star():
    H_star = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0,  0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]],  dtype=float)
    return H_star

def init_PHD(pos, vel):
    # parameter variables
    # freq_est = 5.

    # measurement variables
    dim_vis = 3
    dim_s = 1
    dim_alpha = 1
    dim_ri  = 1

    z_std_vis = 1.
    z_std_s = 1.
    z_std_alpha = 1.
    z_std_ri = 1.

    # filter variable
    survivalprob = 1 # hypothetical probabilities
    detectprob = 1
    clutterintensity = 0.001

    dt = 2 # 1./freq_est

    Q_r_o = Q_discrete_white_noise(dim=2, dt=dt, var=1. ** 2, block_size=3)
    Q_s = np.eye(1)*1e-1
    Q_alpha = np.eye(1)*1e-1
    Q_ri = np.eye(1)*1e-1
    Q = block_diag(Q_r_o, Q_s, Q_alpha, Q_ri)

    R_vis =  np.eye(dim_vis)*(z_std_vis**2)
    R_s =  np.eye(dim_s)*(z_std_s**2)
    R_alpha = np.eye(dim_alpha)*(z_std_alpha**2)
    R_ri = np.eye(dim_ri)*(z_std_ri**2)
    R = block_diag(R_vis, R_s, R_alpha, R_ri)

    transnmatrix = create_F(dt)
    obsnmatrix = create_H()

    birthgmm = birth_gmm(pos, vel, 1.0)

    h_star = create_H_star()

    g = Gmphd(survivalprob, detectprob, transnmatrix, Q, obsnmatrix, R, clutterintensity, birthgmm, h_star)

    return g
##

'''
def slz_plot(list_state):
    fig = plt.figure()
    axes = plt.axes(xlim=(-20, 20), ylim=(-20, 20))
    # print('shape of list_state : ', np.shape(list_state))
    if np.array(list_state).ndim == 1:
        pass
    else:
        for state in list_state:
            # x.append(state[0])
            # y.append(state[1])
            # r.append(state[3])

            circle = plt.Circle((state[0], state[1]), state[3], fill=False)
            axes.add_patch(circle)
    
        # plt.plot(x, y, 'o')
        fig.show()
'''

def slz_drawing(list_idx, image_msg, i):
    cv_bridge = CvBridge()
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(e)

    if np.array(list_idx).ndim == 1:
        list_idx = [list_idx]
    for element_idx in list_idx:
        if element_idx[2] > 0:
            cv2.circle(cv_image, (int(element_idx[1]), int(element_idx[0])), int(element_idx[2]), (255, 255, 255), 2)
    # cv2.imshow('circled SLZ', cv_image)
    dr = '/home/lics-hm/Documents/data/experiment_figure/circled_image/1112-5/sample_image_%d.jpg' %i
    # cv2.imwrite(dr, cv_image)
    cv2.imshow('circled image', cv_image)
    cv2.waitKey(0)


def slz_show(list_idx, image_msg, i, best_slz, ctrl):
    cv_bridge = CvBridge()
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
    except CvBridgeError as e:
        print(e)

    if np.array(list_idx).ndim == 1:
        list_idx = [list_idx]
    for element_idx in list_idx:
        if element_idx[2] > 0:
            cv2.circle(cv_image, (int(element_idx[1]), int(element_idx[0])), int(element_idx[2]), (255, 255, 255), 2)
    # cv2.imshow('circled SLZ', cv_image)
    dr = '/home/lics-hm/Documents/data/experiment_figure/circled_image/1112-5/sample_image_%d.jpg' %i
    cv2.imwrite(dr, cv_image)
    # cv2.waitKey(3)