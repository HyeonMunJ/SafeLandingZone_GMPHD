#!/usr/bin/env python
# from syntheticexamplestuff import updateandprune
from Variables import filter_state, meas_state, alg_state, ctrl_state, status, param, phd_state
from Subscribers import *
from Algorithms import *
from Publishers import Publishers
from Services import Services
from Scenario import *
from Utils import *
from Gmphd import Gmphd
# from gmphd_main import *
# from Depth_image_processing import SLZ_detection
# from syntheticexamplestuff import *
import time

# if __name__ == '__main__':

rospy.init_node('auto_land')

# flight mode object
modes = Services()

# obj_AL : Autonomous_landing
q = filter_state()
m = meas_state()
a = alg_state()
c = ctrl_state()
f = phd_state()
p = param()
s = status()

sub = Subscribers(q=q, m=m, p=p, s=s)
pub = Publishers()
scenario = Scenario()
# transform_point_cloud = TransformPointCloud()
# slz_detection = SLZ_detection()
# gmphd = Gmphd()

# Make sure the drone is armed
while not m['armed']:
    modes.setArm()
    rospy.sleep(0.1)

# activate OFFBOARD mode
modes.setOffboardMode()

# start point
msg_cmd_sp = pub.assign_cmd_sp(scenario.sp_pos)

# move to the start altitude
k = 0
while k < 200:
    pub.pub_cmd_sp.publish(msg_cmd_sp)
    rospy.sleep(0.05)
    k += 1

# set frequency
freq_est = p['freq_est']
freq_ctrl = p['freq_ctrl']
freq_rviz = p['freq_rviz']
freq = lcm_pr(freq_est, freq_ctrl)

rate = rospy.Rate(freq)

count = 1
rate.sleep() # T_0 is sometimes measured as 0 so sleep first

T_0 = rospy.get_rostime().to_time()
slz_record = []
while not rospy.is_shutdown():
    T_now = rospy.get_rostime().to_time()

    # estimation
    if count % (freq / freq_est) == 0:
        T_vis = m['T_vis']
        q = assign_m_o_2_q_o(q, m)
        # detect the SLZ from pointcloud array and transform into world frame
        # state_vector = sub.slz_state
        # state_ct = pcd_coord_transform(q, state_vector)

        if s['phase'] != -1:
            # initialize the PHD filter
            if s['flag_PHD_init'] == False:
                print('start landing procedure!!!')
                s['flag_PHD_init'] = True

            # update the PHD filter and max score zone
            elif s['flag_PHD_init'] and s['flag_PHD_update']:

                q['x_t'], q['y_t'], q['z_t'] = m['est_state'][0], m['est_state'][2], m['est_state'][4]

                print('best zone : ', float(q['x_t']), float(q['y_t']), float(q['z_t']))
                print('phase : ', s['phase'], 'ownship height : ', q['z_o'])
            
    msg_flag_main = pub.assign_flag_main(s['flag_PHD_init'])
    pub.pub_flag_main.publish(msg_flag_main)

    msg_flag_score = pub.assign_flag_score(s['flag_score_init'])
    pub.pub_flag_score.publish(msg_flag_score)
    # if s['flag_score_init']:
    #     s['flag_score_init'] = False

    # control
    msg_cmd_mount = pub.assign_cmd_mount()
    pub.pub_cmd_mount.publish(msg_cmd_mount)

    if count % (freq / freq_ctrl) == 0:
        pos_default = scenario.target_pos_2(T_now - T_0, s, q)
        c = ctrl(c=c, q=q, phase=s['phase'], pos_default=pos_default)
        msg_flag_reinit = pub.assign_flag_reinit(s['flag_reinit'])
        msg_phase = pub.assign_phase(s)
        msg_cmd_vel = pub.assign_cmd_vel(c)
        pub.pub_flag_reinit.publish(msg_flag_reinit)
        pub.pub_phase.publish(msg_phase)
        pub.pub_cmd_vel.publish(msg_cmd_vel)

        if s['flag_reinit']:
            s['flag_reinit'] = False

    if count % (freq / freq_rviz) == 0:
        pass

    if count >= freq:
        count = 1
    else:
        count = count + 1
    # count += 1


    rate.sleep()


