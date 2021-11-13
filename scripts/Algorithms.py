#!/usr/bin/env python
import numpy as np
# from Gmphd import *


def ctrl(c, q, phase, pos_default):
    # distance values to the target point


    if phase == 0:  # waypoint
        e_x = pos_default[0] - q['x_o']
        e_y = pos_default[1] - q['y_o']
        e_z = pos_default[2] - q['z_o']
        e_tot = np.linalg.norm([e_x, e_y, e_z])
        # if total distance to the target exceeds 3m, move at the constant velocity
        # if e_tot > 1:
        v_tot = 1
        v_x_cmd = e_x / e_tot * v_tot
        v_y_cmd = e_y / e_tot * v_tot
        v_z_cmd = e_z / e_tot * v_tot
        # if total distance to the target is less than 3m, move gradually to the target
        # else:
        #     v_x_cmd = e_x
        #     v_y_cmd = e_y
        #     v_z_cmd = e_z

    elif phase == 1:  # searching
        e_x_1 = q['x_t'] - q['x_o']
        e_y_1 = q['y_t'] - q['y_o']
        # e_x_1 = pos_default[0] - q['x_o']
        # e_y_1 = pos_default[1] - q['y_o']
        e_z_1 = pos_default[2] - q['z_o']
        e_hor_1 = np.linalg.norm([e_x_1, e_y_1])

        if e_hor_1 > 1:
            v_tot = 1
            v_x_cmd = e_x_1 / e_hor_1 * v_tot
            v_y_cmd = e_y_1 / e_hor_1 * v_tot
            v_z_cmd = e_z_1
        # if total distance to the target is less than 3m, move gradually to the target
        else:
            v_x_cmd = 0. # e_x_1
            v_y_cmd = 0. # e_y_1
            v_z_cmd = 0. # e_z_1


    elif phase == 2:  # landing
        e_z_2 = pos_default[2] - q['z_o']
        
        v_x_cmd = 0.
        v_y_cmd = 0.
        
        if e_z_2 > 5:
            v_z_cmd = e_z_2
        # if altitude is less than 2m, descent gradually without horizontal movement
        elif e_z_2 < 5:
            v_z_cmd = e_z_2

    c['vx'] = v_x_cmd
    c['vy'] = v_y_cmd
    c['vz'] = v_z_cmd

    return c


def assign_m_o_2_q_o(q, m):
    q['x_o'] = m['x_o']
    q['y_o'] = m['y_o']
    q['z_o'] = m['z_o']
    q['roll_o'] = m['roll_o']
    q['pitch_o'] = m['pitch_o']
    q['yaw_o'] = m['yaw_o']
    return q