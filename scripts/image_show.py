#!/usr/bin/env python
import cv2
import rospy
import sys
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point

from main import state_vector

class image_converter:

  def __init__(self):

    self.bridge = CvBridge()
    rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
    self.target_pixel = PointStamped()
    self.target_pixel.point = Point(0.0, 0.0, 0.0)

  def callback(self, data):
    try:
      #self.target_pixel.header.stamp = rospy.Time.now()

      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    if state_vector is not None:
        for pixels in state_vector[:,0:2]:
            cv2.rectangle(cv_image, (pixels[0] - 60, pixels[1] - 60), (pixels[0] + 60, pixels[1] + 60))

    cv_image_resize = cv2.resize(cv_image, (960, 540))
    cv2.imshow('SLZ detection (not merged size)', cv_image_resize)
    cv2.waitKey(3)

def main(args):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()

  rate = rospy.Rate(20.0)
  while not rospy.is_shutdown():
      rate.sleep()


if __name__ == '__main__':
	try:    
		main(sys.argv)
	except rospy.ROSInterruptException:
		pass
