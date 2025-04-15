# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from abc import ABC
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation


class Pose(ABC):
    """Stores estimated pose from a SLAM backend"""

    def __init__(
        self,
        timestamp: float = 0.0,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __repr__(self) -> str:
        return f"timestamp: {self.timestamp}, x: {self.x}, y: {self.y}, z: \
            {self.z}, roll: {self.roll}, pitch: {self.pitch}, yaw: {self.yaw}"

    def set_timestamp(self, timestamp: float):
        """set pose timestamp (seconds)"""
        self.timestamp = timestamp

    def set_x(self, x: float):
        """set pose x-value (meters)"""
        self.x = x

    def set_y(self, y: float):
        """set pose y-value (meters)"""
        self.y = y

    def set_z(self, z: float):
        """set pose z-value (meters)"""
        self.z = z

    def set_roll(self, roll: float):
        """set pose roll (radians)"""
        self.roll = roll

    def set_pitch(self, pitch: float):
        """set pose pitch (radians)"""
        self.pitch = pitch

    def set_yaw(self, yaw: float):
        """set pose yaw (radians)"""
        self.yaw = yaw

    def get_timestamp(self) -> float:
        """get pose timestamp"""
        return self.timestamp

    def get_x(self) -> float:
        """get pose x-value"""
        return self.x

    def get_y(self) -> float:
        """get pose y-value"""
        return self.y

    def get_z(self) -> float:
        """get pose z-value"""
        return self.z

    def get_roll(self) -> float:
        """get pose roll"""
        return self.roll

    def get_pitch(self) -> float:
        """get pose pitch"""
        return self.pitch

    def get_yaw(self) -> float:
        """get pose yaw"""
        return self.yaw

    def get_rotation_matrix(self) -> np.ndarray:
        """get rotation matrix from euler angles"""
        return Rotation.from_euler("xyz", [self.roll, self.pitch, self.yaw]).as_matrix()


class Slam(ABC):
    """slam base class"""

    def __init__(self):
        raise NotImplementedError

    def initialize(self):
        """initialize slam backend"""
        raise NotImplementedError

    def get_pose(self) -> Pose:
        """returns camera pose"""
        raise NotImplementedError

    def get_trajectory_points(self) -> List[Pose]:
        """returns camera trajectory points"""
        raise NotImplementedError

    def start(self):
        """starts slam"""
        raise NotImplementedError

from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import time


class NavigationStatus(Enum):
    """Status of a navigation command."""
    IDLE = 0
    ACTIVE = 1
    SUCCEEDED = 2
    FAILED = 3
    CANCELED = 4


class NavigationBase:
    """
    Base class for navigation with default implementations.
    
    This class provides basic navigation functionality and can be extended
    by platform-specific implementations.
    """
    
    def __init__(self):
        """Initialize the navigation base."""
        self.status = NavigationStatus.IDLE
        self.goal = None
        self.current_path = None
        self.obstacle_threshold = 0.5  # Default obstacle threshold in meters
        self.current_pose = [0.0, 0.0, 0.0]  # Default starting pose
        self.callbacks = {}
        
    def set_goal(self, goal_pose):
        """
        Set a new navigation goal.
        
        Args:
            goal_pose: Target pose [x, y, theta] or [x, y, z, qx, qy, qz, qw]
            
        Returns:
            bool: True if the goal was set successfully
        """
        self.goal = goal_pose
        self.status = NavigationStatus.IDLE
        return True
    
    def cancel_goal(self):
        """
        Cancel the current navigation goal.
        
        Returns:
            bool: True if cancellation was successful
        """
        self.goal = None
        self.current_path = None
        self.status = NavigationStatus.CANCELED
        return True
    
    def get_path(self, start_pose, goal_pose):
        """
        Plan a path from start pose to goal pose.
        
        Args:
            start_pose: Starting pose [x, y, theta] or [x, y, z, qx, qy, qz, qw]
            goal_pose: Target pose [x, y, theta] or [x, y, z, qx, qy, qz, qw]
            
        Returns:
            List of poses representing the path
        """
        # Simple default implementation - straight line path
        path = [start_pose]
        # Add some intermediate points along a straight line
        for i in range(1, 10):
            t = i / 10.0
            x = start_pose[0] + t * (goal_pose[0] - start_pose[0])
            y = start_pose[1] + t * (goal_pose[1] - start_pose[1])
            
            # For theta, interpolate using shortest path
            theta_diff = goal_pose[2] - start_pose[2]
            # Normalize to [-pi, pi]
            if theta_diff > np.pi:
                theta_diff -= 2 * np.pi
            elif theta_diff < -np.pi:
                theta_diff += 2 * np.pi
                
            theta = start_pose[2] + t * theta_diff
            path.append([x, y, theta])
            
        path.append(goal_pose)
        return path
    
    def execute_path(self, path):
        """
        Execute a precomputed path.
        
        Args:
            path: List of poses to follow
            
        Returns:
            bool: True if execution started successfully
        """
        self.current_path = path
        self.status = NavigationStatus.ACTIVE
        return True
    
    def navigate_to(self, goal_pose, planner="default"):
        """
        Navigate to a goal pose. Plans and executes a path.
        
        Args:
            goal_pose: Target pose [x, y, theta] or [x, y, z, qx, qy, qz, qw]
            planner: Name of the planner to use
            
        Returns:
            bool: True if navigation started successfully
        """
        self.set_goal(goal_pose)
        path = self.get_path(self.get_current_pose(), goal_pose)
        return self.execute_path(path)
    
    def get_status(self):
        """
        Get the current navigation status.
        
        Returns:
            NavigationStatus: Current status of the navigation system
        """
        return self.status
    
    def is_goal_reached(self):
        """
        Check if the current goal has been reached.
        
        Returns:
            bool: True if the goal is reached
        """
        return self.status == NavigationStatus.SUCCEEDED
    
    def update(self):
        """
        Update the navigation system. Should be called periodically.
        
        Returns:
            NavigationStatus: Current status of the navigation system
        """
        # In a real implementation, this would update the navigation state
        # based on sensor data and current pose
        return self.status
    
    def get_current_pose(self):
        """
        Get the current pose of the robot.
        
        Returns:
            list: Current pose [x, y, theta]
        """
        return self.current_pose.copy()
    
    def set_velocity(self, linear_velocity, angular_velocity):
        """
        Set the velocity of the robot directly.
        
        Args:
            linear_velocity: Linear velocity in m/s
            angular_velocity: Angular velocity in rad/s
            
        Returns:
            bool: True if successful
        """
        # Base implementation just updates internal state
        # Platform-specific implementation would send commands to hardware
        if linear_velocity == 0.0 and angular_velocity == 0.0:
            return self.stop()
        return True
    
    def stop(self):
        """
        Stop the robot immediately.
        
        Returns:
            bool: True if successful
        """
        # Base implementation just updates internal state
        if self.status == NavigationStatus.ACTIVE:
            self.status = NavigationStatus.CANCELED
        return True
    
    def explore(self, duration=None, max_distance=None):
        """
        Autonomously explore the environment.
        
        Args:
            duration: Optional maximum duration in seconds
            max_distance: Optional maximum distance to explore in meters
            
        Returns:
            bool: True if exploration started successfully
        """
        # Base implementation just updates state
        self.status = NavigationStatus.ACTIVE
        return True
    
    def is_path_clear(self, path):
        """
        Check if a path is clear of obstacles.
        
        Args:
            path: List of poses to check
            
        Returns:
            bool: True if path is clear
        """
        # Base implementation assumes no obstacles
        return True
    
    def set_obstacle_threshold(self, threshold):
        """
        Set the obstacle threshold distance.
        
        Args:
            threshold: Distance in meters at which objects are considered obstacles
            
        Returns:
            bool: True if successful
        """
        self.obstacle_threshold = threshold
        return True
    
    def get_obstacle_threshold(self):
        """
        Get the current obstacle threshold distance.
        
        Returns:
            float: Obstacle threshold in meters
        """
        return self.obstacle_threshold
    
    def register_callback(self, event_type, callback):
        """
        Register a callback for navigation events.
        
        Args:
            event_type: Type of event (e.g., "goal_reached", "obstacle_detected")
            callback: Function to call when the event occurs
            
        Returns:
            bool: True if successful
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
        return True
    
    def unregister_callback(self, event_type, callback=None):
        """
        Unregister a callback for navigation events.
        
        Args:
            event_type: Type of event
            callback: Function to unregister, or None to unregister all callbacks for this event
            
        Returns:
            bool: True if successful
        """
        if event_type not in self.callbacks:
            return False
        
        if callback is None:
            self.callbacks[event_type] = []
        else:
            try:
                self.callbacks[event_type].remove(callback)
            except ValueError:
                return False
        
        return True