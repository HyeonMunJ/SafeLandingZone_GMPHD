#!/usr/bin/env python3.6

# from .registry import converts_from_numpy, converts_to_numpy

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np
import rospy
import time
import numpy.lib.recfunctions as rf

from Depth_image_processing import SLZ_detection

class Pcd_2_array:
    def __init__(self, DUMMY_FIELD_PREFIX, pftype_to_nptype, pftype_sizes):
        rospy.init_node('test_pcd')
        # rospy.Subscriber("point_cloud_transformed", PointCloud2, self.test)
        rospy.Subscriber("/camera/depth/points", PointCloud2, self.test)
        self.pub_slz = rospy.Publisher('/custom/slz_point/states', Float32MultiArray, queue_size=2)
        self.pub_edge = rospy.Publisher('/custom/slz_point/edge', Float32MultiArray, queue_size=2)

        self.i = 0
        self.DUMMY_FIELD_PREFIX = DUMMY_FIELD_PREFIX
        self.pftype_to_nptype = pftype_to_nptype
        self.pftype_sizes = pftype_sizes

        self.slz_detection = SLZ_detection()

    def test(self, pcd):
        if self.i % 10 == 0:
            print('******************************')
            converted_pcd = self.pointcloud2_to_array(pcd, squeeze=False)

            pcd_array = rf.structured_to_unstructured(converted_pcd)

            # dr = '/home/lics-hm/Documents/data/slz_pcd/pcd_array_%d' % self.i
            # np.save(dr, pcd_array)

            # pcd array has the information of RGBD
            self.slz_detection.det_SLZ(pcd_array[:,:,:3])
            state_slz = self.slz_detection.best_SLZ

            # dr_2 = '/home/lics-hm/Documents/data/slz_pcd/state_slz_%d' % self.i
            # np.save(dr_2, state_slz)

            print('num of slz : ', len(state_slz))
            if len(state_slz):
                msg_slz = self.assign_state_slz(state_slz)
                msg_edge = self.assign_state_edge(self.slz_detection.image_region)
                self.pub_slz.publish(msg_slz)
                self.pub_edge.publish(msg_edge)
        self.i += 1

    # converts_to_numpy(PointField, plural=True)
    def fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(('%s%d' % (self.DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1
    
            dtype = self.pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += self.pftype_sizes[f.datatype] * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (self.DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1
        
        return np_dtype_list

    # converts_to_numpy(PointCloud2)
    def pointcloud2_to_array(self, cloud_msg, squeeze=True):
        ''' Converts a rospy PointCloud2 message to a numpy recordarray 
        
        Reshapes the returned array to have shape (height, width), even if the height is 1.
    
        The reason for using np.fromstring rather than struct.unpack is speed... especially
        for large point clouds, this will be <much> faster.
        '''
        # construct a numpy record type equivalent to the point type of this cloud
        dtype_list = self.fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

        # parse the cloud into an array
        cloud_arr = np.fromstring(cloud_msg.data, dtype_list)
    
        # remove the dummy fields that were added
        cloud_arr = cloud_arr[
            [fname for fname, _type in dtype_list if not (fname[:len(self.DUMMY_FIELD_PREFIX)] == self.DUMMY_FIELD_PREFIX)]]
        
        if squeeze and cloud_msg.height == 1:
            return np.reshape(cloud_arr, (cloud_msg.width,))
        else:
            return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


    def assign_state_slz(self, states):
        msg = Float32MultiArray()
        shape_state = np.shape(states)
        msg.data = np.reshape(states, (shape_state[0]*shape_state[1]))

        '''
        msg.layout.data_offset = 0 
        msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        msg.layout.dim[0].label = "slz"
        msg.layout.dim[0].size = shape_state[0]
        msg.layout.dim[0].stride = shape_state[0] * shape_state[1]
        msg.layout.dim[1].label = "channel"
        msg.layout.dim[1].size = shape_state[1]
        msg.layout.dim[1].stride = shape_state[1]
        '''
        return msg

    def assign_state_edge(self, edge):
        msg = Float32MultiArray()
        msg.data = edge

        return msg

# __docformat__ = "restructurednpyext en"

# prefix to the names of dummy fields we add to get byte alignment correct. this needs to not
# clash with any actual field names
DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}



main = Pcd_2_array( DUMMY_FIELD_PREFIX, pftype_to_nptype, pftype_sizes )
while True:
    rospy.sleep(0.1)

