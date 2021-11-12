#!/usr/bin/env python
import rospy
import numpy as np
import time
from tf.transformations import euler_from_quaternion

from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import PoseStamped, TwistStamped
# from main import q, s, m, p, f
# from Algorithms import *
from Gmphd import * 
from gmphd_stuff import *


class Main_GMPHD:
    def __init__(self):
        self.slz_state = [] # state vector in the image frame considering yaw of the camera
        self.state_ct = None # state vector in the world frame
        self.flag_main_init = False # flag for initializing PHD filter
        self.flag_PHD_init = False # flag indicating that PHD has initialized
        self.flag_phd_done = False # flag indicating that PHD has updated at least once
        self.flag_score = False # flag for initializing max_score when the height level is changed
        self.flag_slz = False # flag indicating that slz state is recently updated(subscribed)
        self.flag_slz_updated = False 
        self.g = None
        self.weight = 0.
        # self.est_state = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.pos = [0., 0., 0.] # position of ownship (NEU?)
        self.vel = [0., 0., 0.] # velocity of ownship (NEU?)
        self.att = [0., 0., 0.] # attitude of ownship (rpy)
        self.edge = [0., 0., 0., 0.] # [x_min, x_max, y_min, y_max]

        rospy.Subscriber("custom/slz_point/states", Float32MultiArray, self.save_slz)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.save_pose)
        rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped, self.save_vel)
        rospy.Subscriber("/custom/flag_main", Bool, self.save_flag_main)
        rospy.Subscriber("/custom/flag_score", Bool, self.save_flag_score)
        rospy.Subscriber("/custom/slz_point/edge", Float32MultiArray, self.save_edge)

        self.pub_gmphd = rospy.Publisher('/custom/gmphd/result', Float32MultiArray, queue_size=2)
        self.pub_gmphd_flag = rospy.Publisher('/custom/flag_phd', Bool, queue_size=2)

    ######################################################################################################
    ##################################### Callback for subscribe #########################################
    ######################################################################################################
    def save_edge(self, msg):
        if self.flag_PHD_init:
            # self.g.edge_prev = self.g.edge
            self.g.edge = msg.data # [x_min, x_max, y_min, y_max]
            self.g.flag_meas_update = True

    def save_slz(self, msg):
        msg_data = np.reshape(msg.data, (np.shape(msg.data)[0]/6, 6))
        # if not self.flag_slz_updated:  # to verify that GM-PHD filter works well, let the measurement be fixed
        self.slz_state = [[-data[1], data[0], - data[2], data[3], data[4], data[5]] for data in msg_data]
        # self.flag_slz_updated = True
        self.flag_slz = True
        # slz_plot(self.slz_state)

    def save_pose(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        qw = msg.pose.orientation.w
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z

        # attitude transformation from Quaternion to Euler
        quaternion_list = [qx, qy, qz, qw]
        self.pos = [x, y, z]
        (self.att[0], self.att[1], self.att[2]) = euler_from_quaternion(quaternion_list)


    def save_vel(self, msg):
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vz = msg.twist.linear.z
        self.vel = [vx, vy, vz]

    def save_flag_main(self, msg):
        self.flag_main_init = msg.data

    def save_flag_score(self, msg):
        self.flag_score = msg.data

    ######################################################################################################
    ###################################### Assign for publish ############################################
    ######################################################################################################


    def assign_gmphd_result(self, est_state, weight):
        msg = Float32MultiArray()
        state = np.append(est_state, weight)
        msg.data = state
        return msg

    def assign_flag_gmphd(self, flag):
        msg = Bool()
        msg.data = flag
        return msg


    ######################################################################################################
    ############################################# Main ###################################################
    ######################################################################################################

    def main(self, max_score, dt):
        # if the PHD filter is permitted to init from main script and it is not initialized yet
        if self.flag_main_init and not self.flag_PHD_init:
            self.g = init_PHD(self.pos, self.vel)
            self.flag_PHD_init = True

        # if the PHD filter is permitted to update from main script and it is initialized
        elif self.flag_main_init and self.flag_PHD_init:
            # make birth gm-phd component
            # birthgmm = birth_gmm(self.pos, self.vel, 1.0)

            # if slz state is not recently updated, then does not consider the dynamics
            if not self.flag_slz:
                dt = 0
            else:
                self.flag_slz = False
            # predict -> update -> merge -> prune
            if self.g.flag_meas_update:
                self.g.flag_meas_update = False
                updateandprune(self.g, self.slz_state, dt)
                # extract a state which has max. weight from weight distribution of gm-phd components
                
                est_state, weight = self.g.extractstatesmax()
                print('est_state: ', est_state)

                # if the extracted state has the highest score, update it as the target point
                if len(est_state):
                    score = calc_score(weight, est_state)

                    if score > max_score:
                        max_score = score

                        self.weight = weight
                        self.state_ct = pcd_coord_transform(self.pos, est_state)

                # publish best SLZ to main module
                msg_state = self.assign_gmphd_result(self.state_ct, self.weight)
                self.pub_gmphd.publish(msg_state)

                if not self.flag_phd_done:
                    self.flag_phd_done = True

                # publish flag indicating phd filter has updated to main module 
                msg_flag = self.assign_flag_gmphd(self.flag_phd_done)
                self.pub_gmphd_flag.publish(msg_flag)

                # print('maximum score is ', max_score)

        return max_score


if __name__ == '__main__':
    rospy.init_node('gmphd')

    # freq = 5. # p['freq_est']
    # rate = rospy.Rate(freq)

    main_gmphd = Main_GMPHD()
    max_score = -10000.
    time_1 = time.time()
    time_2 = time.time()

    while not rospy.is_shutdown():
        # initialize max score when the flag_score is changed since the scenario level is change in the main script
        if main_gmphd.flag_score: # cannot be applied properly
            max_score = -10000.

        # calculate the time difference between time steps
        dt = time_2 - time_1
        time_1 = time.time()
        # execute main function of gmphd once
        score = main_gmphd.main(max_score, dt)
        time_2 = time.time()

        # update the max score
        if score > max_score:
            max_score = score
        # rate.sleep()
