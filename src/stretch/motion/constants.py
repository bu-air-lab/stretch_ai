# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import math
import numpy as np

# Control loop rate in Hz
CONTROL_RATE = 20.0  # 20 Hz control loop

# Segway robot configuration
SEGWAY_BASE_FRAME = "segway/base_link"
SEGWAY_ODOM_FRAME = "segway/odom"
SEGWAY_CAMERA_FRAME = "segway/camera_link"
SEGWAY_LIDAR_FRAME = "segway/lidar_link"

# Segway robot dimensions
SEGWAY_WHEEL_BASE = 0.5  # meters
SEGWAY_WHEEL_RADIUS = 0.195  # meters
SEGWAY_WIDTH = 0.58  # meters
SEGWAY_LENGTH = 0.75  # meters
SEGWAY_HEIGHT = 1.2  # meters

# Robot behavior limits
MAX_LINEAR_SPEED = 1.0  # m/s
MAX_ANGULAR_SPEED = 0.8  # rad/s
DEFAULT_LINEAR_SPEED = 0.5  # m/s
DEFAULT_ANGULAR_SPEED = 0.5  # rad/s
ACCELERATION_LIMIT = 0.5  # m/s²
ANGULAR_ACCELERATION_LIMIT = 0.8  # rad/s²

# Navigation settings
NAVIGATION_GOAL_TOLERANCE_POSITION = 0.1  # meters
NAVIGATION_GOAL_TOLERANCE_ORIENTATION = 0.1  # radians
OBSTACLE_INFLATION_RADIUS = 0.4  # meters

# ROS topic names
CMD_VEL_TOPIC = "/segway/cmd_vel"
ODOM_TOPIC = "/segway/odom"
SCAN_TOPIC = "/segway/scan"
CAMERA_RGB_TOPIC = "/segway/camera/rgb/image_raw"
CAMERA_DEPTH_TOPIC = "/segway/camera/depth/image_raw"
CAMERA_INFO_TOPIC = "/segway/camera/rgb/camera_info"

# Segway default states and positions
SEGWAY_HOME_POSE = np.array([0.0, 0.0, 0.0])  # x, y, theta