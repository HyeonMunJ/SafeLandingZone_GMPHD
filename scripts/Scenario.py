#!/usr/bin/env python
import numpy as np

class Scenario():
    def __init__(self):
        # ======== ROS communicators declaration ========
        self.sp_pos = [0.0, 0.0, 5.0]

        # self.timeline = np.array([0,20,30,40,50,60,70,80])
        self.timeline = np.array([0,30,60,90,120])
        self.N_wp = np.shape(self.timeline)[0]
        self.goal = np.zeros((self.N_wp,3))
        '''
        self.goal[0] = [0,0,30]
        self.goal[1] = [0,0,30]
        self.goal[2] = [0,0,20]
        self.goal[3] = [0,0,10]
        self.goal[4] = [0,0,0]
        '''
        self.goal[0] = [0,0,30]
        self.goal[1] = [0,0,30]
        self.goal[2] = [0,0,30]
        self.goal[3] = [0,0,30]
        self.goal[4] = [0,0,30]
        self.record_idx = 1


    def target_pos(self, T, s, q):
        idx = np.argmax(np.where(self.timeline<=T))
        # when the height level changed, initialized the score
        if idx != 0 and self.record_idx != idx:
            s['flag_score_init'] = True
            # s['phase'] = 2
            self.record_idx = idx
        # when the UAV reaches the target attitude, it starts searching
        if abs(self.goal[idx][2] - q['z_o']) < 1.:
            s['phase'] = 1
            # pass
        return self.goal[idx]

