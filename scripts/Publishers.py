#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist, Point
from mavros_msgs.msg import MountControl
from mavros_msgs.msg import MountControl, PositionTarget
from std_msgs.msg import Bool
# from visualization_msgs.msg import Marker
# import math

class Publishers():
    def __init__(self):
        # ======== ROS communicators declaration ========z
        self.pub_cmd_vel = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
        self.pub_cmd_mount = rospy.Publisher('mavros/mount_control/command', MountControl, queue_size=10)
        self.pub_cmd_sp = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=2)
        self.pub_flag_main = rospy.Publisher('/custom/flag_main', Bool, queue_size=2)
        self.pub_flag_score = rospy.Publisher('/custom/flag_score', Bool, queue_size=2)
        # self.pub_state_helipad = rospy.Publisher('helipad_marker', Marker, queue_size=2)
        # self.pub_state_ownship = rospy.Publisher('ownship_marker', Marker, queue_size=2)

    def assign_cmd_vel(self, c):
        msg = Twist()
        msg.linear.x = c['vx']
        msg.linear.y = c['vy']
        msg.linear.z = c['vz']
        msg.angular.x = c['d_roll']
        msg.angular.y = c['d_pitch']
        msg.angular.z = c['d_yaw']
        return msg

    def assign_cmd_mount(self):

        msg = MountControl()
        msg.header.frame_id = 'map'
        msg.mode = 2
        msg.roll = 0
        msg.pitch = 90.
        msg.yaw = 0

        return msg

    def assign_cmd_sp(self, pos):
        msg = PositionTarget()
        msg.type_mask = int('010111111000', 2)
        msg.coordinate_frame = 1
        msg.position = Point(pos[0], pos[1], pos[2])
        return msg

    def assign_flag_main(self, flag):
        msg = Bool()
        msg.data = flag

        return msg

    def assign_flag_score(self, flag):
        msg = Bool()
        msg.data = flag

        return msg