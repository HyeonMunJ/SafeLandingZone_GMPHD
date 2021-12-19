#!/usr/bin/env python
import numpy as np

class Scenario():
    def __init__(self):
        # ======== ROS communicators declaration ========
        self.sp_pos = [0.0, 0.0, 5.0]

        # self.timeline = np.array([0,20,30,40,50,60,70,80])
        # self.timeline = np.array([0,30,60,90,120])
        # self.timeline = np.array([0,20,40,60, 100])
        # self.timeline = np.array([0,20,40,60, 100])
        # self.timeline = np.array([0,30,60,90,110])
        # self.timeline = np.array([0,20,50,120,140,150])
        self.timeline = np.array([0,20,35,70,90,110])


        self.N_wp = np.shape(self.timeline)[0]
        self.goal = np.zeros((self.N_wp,3))
        self.goal[0] = [0,0,30]
        # self.goal[1] = [0,0,30]
        self.goal[1] = [25,-25,30]
        self.goal[2] = [0,0,30]
        self.goal[3] = [0,0,30]
        self.goal[4] = [0,0,20]
        self.goal[5] = [0,0,5]

        '''
        self.goal[0] = [0,0,30]
        self.goal[1] = [0,0,30]
        self.goal[2] = [0,0,20]
        self.goal[3] = [0,0,0]
        '''
        '''
        self.goal[0] = [0,0,30]
        self.goal[1] = [0,0,30]
        self.goal[2] = [0,0,30]
        self.goal[3] = [0,0,30]
        self.goal[4] = [0,0,30]
        '''

        self.record_idx = 3
        self.time_step = 0

        self.flag_record = False


    def target_pos(self, T, s, q):
        idx = np.argmax(np.where(self.timeline<=T))
        # when the height level changed, initialized the score
        if idx != 0 and self.record_idx != idx:
            s['flag_score_init'] = True
            # s['phase'] = 2
            self.record_idx = idx
        # when the UAV reaches the target attitude, it starts searching
        if abs(self.goal[idx][2] - q['z_o']) < 1.:
            # s['phase'] = 1
            pass
        return self.goal[idx]

    def target_pos_2(self, T, s, q, m):
        if T > (self.time_step +1) * 300:
            self.time_step += 1
            s['flag_reinit'] = True
        loop_time = T - self.time_step * 300
        
        idx = np.argmax(np.where(self.timeline<=loop_time))
        if idx < 2:
            # change phase
            s['phase'] = -1
            s['flag_score_init'] = False
            # s['flag_score_init'] = True
        elif idx == 2:
            s['flag_score_init'] = False
            s['phase'] = 0
        elif self.record_idx != idx:
            s['flag_score_init'] = True
            s['phase'] = 2
            self.record_idx = idx
        elif abs(self.goal[idx][2] - q['z_o']) < 1.: # case check
            s['flag_score_init'] = False
            s['phase'] = 1

        if 0:  
        # if not self.flag_record and s['flag_PHD_init'] and idx >= 3:
            with open("/home/lics-hm/Documents/data/experiment_data/target_position_1123/slz_position.txt", "a+") as file:
                file.write("x = %f, v_x = %f, y = %f, v_y = %f, z = %f, v_z = %f, r = %f, alpha = %f, ri = %f, score = %f\n" \
                    %(float(m['est_state'][0]), float(m['est_state'][1]), float(m['est_state'][2]), float(m['est_state'][3]), float(m['est_state'][4]),\
                        float(m['est_state'][5]), float(m['est_state'][6]), float(m['est_state'][7]), float(m['est_state'][8]), float(m['score'])))
            self.flag_record = True
        return self.goal[idx]
        