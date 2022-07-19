

import numpy as np
import matplotlib.pyplot as plt

import rospy, time
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Header, MultiArrayDimension, Float32MultiArray
from geometry_msgs.msg import Pose

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import ros_numpy

import ctypes
import struct

import tf

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from filter_rate.filling_mask import FillingMask
# from filter_rate.loss import *

class Filling:
    def __init__(self, map_dim=200, odomt = "/auto_odom", lidart="/velodyne_points"):
        self.map_dim = map_dim #int(cfg['map']['map_dim'])
        self.resolution = 0.04
        self.use_mf_points = False
        self.clip = 200

        precision = 'float'#str(cfg['map']['precision'])
        if precision == 'double':
            self.d_type = float
        elif precision == 'half':
            self.d_type = np.float16
        else:
            self.d_type = np.float32

        # Buffers
        if not self.use_mf_points:
            self.frame_buf_size = 3#int(cfg['map']['frame_buf_size'])
            self.frame_buf = []
            self.odom_buf_size = 3#int(cfg['map']['odom_buf_size'])
            self.odom_buf = []
        self.map_layer = np.zeros((1, self.map_dim, self.map_dim), dtype=self.d_type)

        # Objects
        self.elevation_map = FillingMask(self.map_dim)

        # ROS workers
        # -- Localization info
        odom_topic = odomt
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odomCallback)

        # -- Point cloud
        pointcloud_topic = lidart
        self.point_sub = rospy.Subscriber(pointcloud_topic, PointCloud2, self.pointcloudCallback)

        map_topic = "/filter_grid"
        self.map_pub = rospy.Publisher(map_topic, GridMap, latch=True, queue_size=1)
        
        self.map_frame_id = "map"
        
    def odomCallback(self, odom):
        # Comapre the time stamp of odom and points frames
        odom_time = odom.header.stamp.to_nsec()
        matched = False
        for frame in self.frame_buf:
            frame_time = frame.header.stamp.to_nsec()
            if frame_time == odom_time:
                matched = True
                self.process(frame, odom)
                break

        if not matched:
            #rospy.logwarn("WRN: Can't find a matched points frame with current odom.")
            self.odom_buf.append(odom)
            if len(self.odom_buf) > self.odom_buf_size:
                self.odom_buf.pop(0)

    def pointcloudCallback(self, points):
        """
        Receive the incoming points frame, save it into the points frame buffer
        """
        frame_time = points.header.stamp.to_nsec()
        matched = False
        for odom in self.odom_buf:
            odom_time = odom.header.stamp.to_nsec()
            if odom_time == frame_time:
                matched = True
                self.process(points, odom)
                break

        if not matched:
            self.frame_buf.append(points)
            if len(self.frame_buf) > self.frame_buf_size:
                self.frame_buf.pop(0)

    def process(self, frame, odom):
        time_0 = time.time()

        # Update body (map) center
        self.elevation_map.updateMapCenter([
            odom.pose.pose.position.x, 
            odom.pose.pose.position.y
        ])
        
        # Points frame to np array
        frame_np = ros_numpy.numpify(frame)
        points = np.hstack(( frame_np['x'].reshape(-1, 1), frame_np['y'].reshape(-1, 1), frame_np['z'].reshape(-1, 1) )).astype(self.d_type, copy=False)

        # Get point transformation from odom
        if self.use_mf_points:
            t = np.zeros(3, dtype=self.d_type)
            R = np.eye(3, dtype=self.d_type)
        else:
            t = np.asarray([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]).astype(self.d_type, copy=False)
            R = tf.transformations.quaternion_matrix([
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            ])[:3, :3].astype(self.d_type, copy=False)

        time_1 = time.time()
        print("msg process time:", time_1-time_0)
        
        # Update the points features and generate elevation map
        self.map_layer = self.elevation_map.step(points, R, t)

        time_2 = time.time()
        print("main generation time:", time_2-time_1)

        self.publishMap(self.map_layer, odom.header.stamp)

        time_3 = time.time()
        print("map publish time:", time_3-time_2)
        print(" ----- Total time:", time_3-time_0)

    def publishMap(self, map_layer, time_stamp):
        if map_layer.shape[2]>self.clip:
            ocol = map_layer.shape[2]
            delta = ocol-self.clip
            print(f"enable clip to small one, from {ocol}x{ocol} to {self.clip}x{self.clip}")
            map_layer = map_layer[:, delta//2:delta//2+self.clip, delta//2:delta//2+self.clip]
            map_dim = self.clip
        header = Header()
        header.seq = 0
        header.stamp = time_stamp
        header.frame_id = self.map_frame_id

        center = self.elevation_map.getMapCenter()

        pose = Pose()
        pose.position.x = center[0]
        pose.position.y = center[1]

        # Map preprocess
        dim_0 = MultiArrayDimension()
        dim_0.label = "column_index"
        dim_0.size = map_layer.shape[2]
        dim_0.stride = map_layer.shape[2]*map_layer.shape[1]
        dim_1 = MultiArrayDimension()
        dim_1.label = "row_index"
        dim_1.size = map_layer.shape[1]
        dim_1.stride = map_layer.shape[1]

        map_inpaint_data = Float32MultiArray()
        map_inpaint_data.layout.dim = [dim_0, dim_1]
        map_inpaint_data.layout.data_offset = 0
        map_inpaint_data.data = np.rot90(map_layer[0].T, 2).reshape(-1)

        grid_map = GridMap()
        grid_map.info.header = header
        grid_map.info.resolution = self.resolution
        grid_map.info.length_x = self.resolution * map_dim
        grid_map.info.length_y = self.resolution * map_dim
        grid_map.info.pose = pose
        grid_map.layers = ["omask"]
        grid_map.data = [map_inpaint_data]

        self.map_pub.publish(grid_map)
        rospy.loginfo("Elevation map published")



if __name__ == '__main__':
    rospy.init_node('pub_filter_rate')
    variable = 500
    odom_topic, lidar_topic = "/auto_odom", "/velodyne_points"
    if rospy.has_param("map_dim"):
        variable = rospy.get_param("map_dim")
        odom_topic = rospy.get_param("odom_topic")
        lidar_topic = rospy.get_param("lidar_topic")
        print(f"getting rosparam: {variable}")

    mapping = Filling(map_dim=variable,odomt = odom_topic, lidart=lidar_topic)
    rospy.spin()