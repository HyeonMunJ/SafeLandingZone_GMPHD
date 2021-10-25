import numpy as np

class filter_state:
    def __init__(self):
        self.dim = 7

        '''
        # ownship (enu)
        self.r_x = 0
        self.rv_x = 0
        self.r_y = 0
        self.rv_y = 0
        self.r_z = 0
        self.rv_z = 0
        '''
        self.x_o = 0.
        self.vx_o = 0.
        self.y_o = 0.
        self.vy_o = 0.
        self.z_o = 0.
        self.vz_o = 0.
        self.roll_o = 0.
        self.d_roll_o = 0.
        self.pitch_o = 0.
        self.d_pitch_o = 0.
        self.yaw_o = 0.
        self.d_yaw_o = 0.

        self.x_t = 0.
        self.y_t = 0.
        self.z_t = 0.

        # target  (enu)
        self.s = 0
        self.alpha = 0
        self.ri = 0

        # ownship

        self.P = np.ones((self.dim ,self.dim ))

    def __getitem__(self,key):
        return getattr(self, key)

    def __setitem__(self,key,value):
        if hasattr(self, key):
            return setattr(self, key, value)
        else:
            print('not defined variable')

class meas_state:
    def __init__(self):
        self.dim = 4
        self.dim_vis = 3
        self.dim_s = 1
        self.dim_alpha = 1
        self.dim_ri  = 1

        # ownship
        self.x_o = 0
        self.vx_o = 0
        self.y_o = 0
        self.vy_o = 0
        self.z_o = 0
        self.vz_o = 0
        self.roll_o = 0
        self.d_roll_o = 0
        self.pitch_o = 0
        self.d_pitch_o = 0
        self.yaw_o = 0
        self.d_yaw_o = 0

        # target
        # (ned)
        self.est_state = None
        self.weight = None

        # std of meas
        self.z_std_vis = 50.
        self.z_std_s = 0.01
        self.z_std_alpha = 0.01
        self.z_std_ri = 0.01
        self.z_std_bad = 100.

        # timestamp
        self.T_o = 0
        self.T_vis = 0 # vision
        self.T_r = 0 # LRF
        self.T_s = 0 # ownship state

        self.imu_roll = 0
        self.imu_pitch = 0
        self.imu_yaw = 0

        # vehicle armed
        self.armed = False

    def __getitem__(self,key):
        return getattr(self, key)

    def __setitem__(self,key,value):
        if hasattr(self, key):
            return setattr(self, key, value)
        else:
            print('not defined variable')

class param:
    def __init__(self):
        self.freq_est = 5.
        self.freq_ctrl = 20.
        self.freq_rviz = 10.

        self.N_H = 2 # number of hypothesis (0:landing condition, 1:not landing condition)
        self.th_L = 0.8 # threshold for landing possible

        self.cm = np.array([[0.6, 0.4], [0.4, 0.5]]) # confusion matrix
        self.P_H_min = 0.01

        self.cam_bound = [1080, 1920] # camera resolution
        self.th_viscen = 500 # threshold for target centered

    def __getitem__(self,key):
        return getattr(self, key)

class alg_state:
    def __init__(self):
        self.P_H = np.array([0.1, 0.9]) # hypothetical probabilities

    def __getitem__(self,key):
        return getattr(self, key)

    def __setitem__(self,key,value):
        if hasattr(self, key):
            return setattr(self, key, value)
        else:
            print('not defined variable')

class phd_state:
    def __init__(self):
        self.birthprob = 0.1
        self.survivalprob = 0.975 # hypothetical probabilities
        self.detectprob = 0.95
        self.clutterintensitytot = 5
        self.clutterintensity = 0.002

    def __getitem__(self,key):
        return getattr(self, key)

    def __setitem__(self,key,value):
        if hasattr(self, key):
            return setattr(self, key, value)
        else:
            print('not defined variable')

class ctrl_state:
    def __init__(self):
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.d_roll = 0
        self.d_pitch = 0
        self.d_yaw = 0

    def __getitem__(self,key):
        return getattr(self, key)

    def __setitem__(self,key,value):
        if hasattr(self, key):
            return setattr(self, key, value)
        else:
            print('not defined variable')

class status:
    def __init__(self):
        self.phase = 0 # 0:scenario, 1:landing
        self.flag_PHD_init = False # checker for KF initiation
        self.flag_score_init = False
        self.flag_PHD_update = False

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            return setattr(self, key, value)
        else:
            print('not defined variable')