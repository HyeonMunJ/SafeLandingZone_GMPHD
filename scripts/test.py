from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common.discretization import  Q_discrete_white_noise
import numpy as np


def fx(x, dt):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]], dtype=float)
    return np.dot(F, x)

def hx(x, alpha1):
   # measurement function - convert state into a measurement
   # where measurements are [x_pos, y_pos]
   return np.array([x[0], x[2]])

dt = 0.1
points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)

ukf.x = np.array([-1., 1., -1., 1]) # initial state
ukf.P *= 0.2 # initial uncertainty

z_std = 0.1

ukf.R = np.diag([z_std**2, z_std**2]) # 1 standard
ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)

zs = [[i+np.random.randn()*z_std, i+np.random.randn()*z_std] for i in range(50)] # measurements

for z in zs:
    ukf.predict()
    ukf.update(z, alpha1=1)
    print(ukf.x, 'log-likelihood', ukf.log_likelihood)