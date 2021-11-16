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
        self.timeline = np.array([0,20,50,120,130,140])

        self.N_wp = np.shape(self.timeline)[0]
        self.goal = np.zeros((self.N_wp,3))
        self.goal[0] = [0,0,30]
        # self.goal[1] = [0,0,30]
        self.goal[1] = [50,-50,30]
        self.goal[2] = [0,0,30]
        self.goal[3] = [0,0,20]
        self.goal[4] = [0,0,10]
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

        self.record_idx = 1
        self.time_step = 0


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

    def target_pos_2(self, T, s, q):
        if T > (self.time_step +1) * 150:
            self.time_step += 1
            s['flag_reinit'] = True
        loop_time = T - self.time_step * 150
        
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
        elif abs(self.goal[idx][2] - q['z_o']) < 1.: # case 잘 나뉘는지 확인
            s['flag_score_init'] = False
            s['phase'] = 1
        return self.goal[idx]