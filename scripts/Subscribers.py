#!/usr/bin/env python
import rospy
# import tf2_ros
# import tf2_py as tf2
import numpy as np

from sensor_msgs.msg import Range, Imu, Image
from geometry_msgs.msg import PoseStamped, TwistStamped, PointStamped
from std_msgs.msg import Float32MultiArray, Bool
from mavros_msgs.msg import State

from tf.transformations import euler_from_quaternion
# from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
# from transform_point_cloud.cfg import LookupTransformConfig

class Subscribers():
    def __init__(self, q, m, p, s):
        # q : filter states
        # m : measured states
        self.q = q
        self.m = m
        self.p = p
        self.s = s
        self.depth_image = None
        self.slz_state = np.array([[0,0,0,0,0,0]])

        # ======== ROS communicators declaration ========z
        # Subscriber
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.save_pose)
        rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped, self.save_vel)
        rospy.Subscriber("/mavros/px4flow/ground_distance", Range, self.save_LRF)
        rospy.Subscriber("/target_pixel", PointStamped, self.save_target_pixel)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.save_depth_image)
        # rospy.Subscriber("custom/slz_point/states", Float32MultiArray, self.save_slz)
        rospy.Subscriber("/mavros/state", State, self.save_state)
        rospy.Subscriber('/custom/gmphd/result', Float32MultiArray, self.save_gmphd)
        rospy.Subscriber('/custom/flag_phd', Bool, self.save_flag_phd)

        # rospy.Subscriber('/mavros/imu/data', Imu, self.save_imu)

    def save_imu(self, msg):
        orientation_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        (self.m['imu_roll'], self.m['imu_pitch'], self.m['imu_yaw']) = euler_from_quaternion(orientation_list)

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
        self.m['x_o'], self.m['y_o'], self.m['z_o'] = x, y, z
        (self.m['roll_o'], self.m['pitch_o'], self.m['yaw_o']) = euler_from_quaternion(quaternion_list)
        self.m['T_o'] = msg.header.stamp.to_sec()

    def save_vel(self, msg):
        vx = msg.twist.linear.x
        vy = msg.twist.linear.y
        vz = msg.twist.linear.z

        d_roll = msg.twist.angular.x
        d_pitch = msg.twist.angular.y
        d_yaw = msg.twist.angular.z

        self.m['vx_o'], self.m['vy_o'], self.m['vz_o'] = vx, vy, vz
        self.m['d_roll_o'], self.m['d_pitch_o'], self.m['d_yaw_o'] = d_roll, d_pitch, d_yaw

    def save_LRF(self, msg):
        self.m['r'] = msg.range
        self.m['T_r'] = msg.header.stamp.to_sec()

    def save_target_pixel(self, msg):
        # self.m['px_t'] = msg.point.x
        # self.m['py_t'] = msg.point.y
        self.m['px_t'] = -(msg.point.y - self.p['cam_bound'][0] / 2)
        self.m['py_t'] = (msg.point.x - self.p['cam_bound'][1] / 2)
        self.m['T_vis'] = msg.header.stamp.to_sec() ##

    def save_depth_image(self, msg):
        self.depth_image = msg

    def save_slz(self, msg):
        shape_msg = np.shape(msg.data)
        self.slz_state = np.reshape(msg.data, (shape_msg[0]/6, 6))

    def save_state(self, msg):
        self.m['armed'] = msg.armed

    def save_gmphd(self, msg):
        self.m['est_state'] = msg.data[:6]
        self.m['weight'] = msg.data[6]

    def save_flag_phd(self, msg):
        self.s['flag_PHD_update'] = msg.data

'''
class TransformPointCloud:
    def __init__(self):
        self.config = None
        # self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(12))
        self.tf_buffer = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf_buffer)
        # self.pub = rospy.Publisher("point_cloud_transformed", PointCloud2, queue_size=2)
        # self.sub = rospy.Subscriber("/camera/depth/points", PointCloud2,
        #                             self.point_cloud_callback, queue_size=2)


    def point_cloud_callback(self, msg):
        trans = self.tf_buffer.lookup_transform("base_link", "map", rospy.Time())
        cloud_out = do_transform_cloud(msg, trans)
        self.pub.publish(cloud_out)
'''