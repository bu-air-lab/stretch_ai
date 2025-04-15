"""
Segway Integration with Stretch AI

This module provides the integration layer between the Segway robot's
ROS interface and the Stretch AI framework.
"""

import time
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

from stretch.utils.logger import Logger

logger = Logger(__name__)


class SegwayStretchIntegration:
    """
    Integration class that connects Segway robot capabilities with
    the higher-level Stretch AI framework operations.
    """
    
    def __init__(self, client, config):
        """
        Initialize the Segway-Stretch integration.
        
        Args:
            client: SegwayROSClient instance
            config: Configuration dictionary
        """
        self.client = client
        self.config = config
        
        # Extract configuration
        self.safety_distance = self.config.get('safety_distance', 0.5)  # meters
        self.bump_detection_threshold = self.config.get('bump_detection_threshold', 0.5)  # m/s²
        self.min_obstacle_distance = self.config.get('min_obstacle_distance', 0.3)  # meters
        
        # Speed limits
        self.max_linear_speed = self.config.get('max_linear_speed', 0.5)  # m/s
        self.max_angular_speed = self.config.get('max_angular_speed', 1.0)  # rad/s
        
        # Navigation parameters
        self.goal_tolerance = self.config.get('position_tolerance', 0.1)  # meters
        self.angle_tolerance = self.config.get('heading_tolerance', 0.1)  # radians
        
        # Initialize state
        self.current_operation = None
        self.is_moving = False
        self.obstacles_detected = False
        self.emergency_stop = False
        
    def move_forward(self, distance: float, speed: Optional[float] = None) -> bool:
        """
        Move the robot forward by a specified distance.
        
        Args:
            distance: Distance to move in meters (can be negative for reverse)
            speed: Linear speed in m/s (will use default if None)
            
        Returns:
            bool: True if the movement was successful, False otherwise
        """
        if speed is None:
            speed = self.config.get('move_speed', 0.3)  # Default speed
            
        # Limit speed to maximum
        speed = min(abs(speed), self.max_linear_speed)
        
        # Apply sign based on distance direction
        if distance < 0:
            speed = -speed
            distance = -distance
            
        logger.info(f"Moving forward {distance} meters at {speed} m/s")
        
        # Calculate current pose
        current_pose = self.client.get_pose()
        
        # Calculate target pose
        target_x = current_pose[0] + distance * np.cos(current_pose[2])
        target_y = current_pose[1] + distance * np.sin(current_pose[2])
        target_pose = [target_x, target_y, current_pose[2]]
        
        # Set current operation and state
        self.current_operation = "move_forward"
        self.is_moving = True
        
        # Move to target pose
        success = self.client.move_to(target_pose, relative=False, blocking=True)
        
        # Update state
        self.is_moving = False
        self.current_operation = None
        
        return success
        
    def rotate_in_place(self, angle: float, speed: Optional[float] = None) -> bool:
        """
        Rotate the robot in place by a specified angle.
        
        Args:
            angle: Angle to rotate in radians
            speed: Angular speed in rad/s (will use default if None)
            
        Returns:
            bool: True if the rotation was successful, False otherwise
        """
        if speed is None:
            speed = self.config.get('rotation_speed', 0.5)  # Default speed
            
        # Limit speed to maximum
        speed = min(abs(speed), self.max_angular_speed)
        
        logger.info(f"Rotating {angle} radians at {speed} rad/s")
        
        # Calculate current pose
        current_pose = self.client.get_pose()
        
        # Calculate target pose (same position, new orientation)
        target_pose = [current_pose[0], current_pose[1], current_pose[2] + angle]
        
        # Set current operation and state
        self.current_operation = "rotate_in_place"
        self.is_moving = True
        
        # Move to target pose
        success = self.client.move_to(target_pose, relative=False, blocking=True)
        
        # Update state
        self.is_moving = False
        self.current_operation = None
        
        return success
        
    def navigate_to(self, x: float, y: float, theta: Optional[float] = None) -> bool:
        """
        Navigate to a specified position and orientation.
        
        Args:
            x: Target X coordinate in meters
            y: Target Y coordinate in meters
            theta: Target orientation in radians (if None, will face target direction)
            
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        # Get current pose
        current_pose = self.client.get_pose()
        
        # Calculate direction to target
        dx = x - current_pose[0]
        dy = y - current_pose[1]
        
        # If theta not specified, face the target
        if theta is None:
            theta = np.arctan2(dy, dx)
            
        target_pose = [x, y, theta]
        
        logger.info(f"Navigating to {target_pose}")
        
        # Set current operation and state
        self.current_operation = "navigate_to"
        self.is_moving = True
        
        # Move to target pose
        success = self.client.move_to(target_pose, relative=False, blocking=True)
        
        # Update state
        self.is_moving = False
        self.current_operation = None
        
        return success
        
    def navigate_to_pose_sequence(self, poses: List[List[float]]) -> bool:
        """
        Navigate through a sequence of poses.
        
        Args:
            poses: List of [x, y, theta] poses
            
        Returns:
            bool: True if the entire sequence was successfully navigated
        """
        if not poses:
            return True
            
        logger.info(f"Navigating sequence of {len(poses)} poses")
        
        # Set current operation and state
        self.current_operation = "navigate_sequence"
        self.is_moving = True
        
        success = True
        
        for i, pose in enumerate(poses):
            logger.info(f"Moving to waypoint {i+1}/{len(poses)}")
            
            # Navigate to each pose
            result = self.client.move_to(pose, relative=False, blocking=True)
            
            if not result:
                logger.warning(f"Failed to reach waypoint {i+1}")
                success = False
                break
                
        # Update state
        self.is_moving = False
        self.current_operation = None
        
        return success
        
    def stop(self) -> bool:
        """
        Stop the robot immediately.
        
        Returns:
            bool: True if the stop command was sent successfully
        """
        logger.info("Emergency stop")
        
        self.is_moving = False
        self.current_operation = None
        
        self.client.stop_robot()
        return True
        
    def check_obstacles(self, threshold_distance: Optional[float] = None) -> bool:
        """
        Check if there are obstacles within a certain distance.
        
        Args:
            threshold_distance: Distance threshold in meters (uses safety_distance if None)
            
        Returns:
            bool: True if obstacles detected, False otherwise
        """
        if threshold_distance is None:
            threshold_distance = self.safety_distance
            
        # Get the latest observation
        observation = self.client.get_observation()
        if observation is None:
            return False
            
        # Check scan data for obstacles
        scan_points = observation.point_cloud.points
        
        # Convert to robot frame if necessary
        # For simplicity, assuming points are already in robot frame
        
        # Calculate distances to each point (in 2D)
        distances = np.sqrt(scan_points[:, 0]**2 + scan_points[:, 1]**2)
        
        # Check if any point is closer than threshold
        obstacles_detected = np.any(distances < threshold_distance)
        
        self.obstacles_detected = obstacles_detected
        return obstacles_detected
        
    def get_closest_obstacle_distance(self) -> float:
        """
        Get the distance to the closest obstacle.
        
        Returns:
            float: Distance to closest obstacle in meters, or infinity if none detected
        """
        # Get the latest observation
        observation = self.client.get_observation()
        if observation is None:
            return float('inf')
            
        # Check scan data for obstacles
        scan_points = observation.point_cloud.points
        
        if len(scan_points) == 0:
            return float('inf')
            
        # Calculate distances to each point (in 2D)
        distances = np.sqrt(scan_points[:, 0]**2 + scan_points[:, 1]**2)
        
        # Return the minimum distance
        return float(np.min(distances))
        
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the robot.
        
        Returns:
            dict: Dictionary containing robot state information
        """
        pose = self.client.get_pose()
        
        return {
            'position': [pose[0], pose[1]],
            'orientation': pose[2],
            'is_moving': self.is_moving,
            'current_operation': self.current_operation,
            'obstacles_detected': self.obstacles_detected,
            'emergency_stop': self.emergency_stop
        }
        
    def at_goal(self, x: float, y: float, theta: Optional[float] = None, 
                position_tolerance: Optional[float] = None, 
                angle_tolerance: Optional[float] = None) -> bool:
        """
        Check if the robot is at the specified goal position and orientation.
        
        Args:
            x: Goal X coordinate
            y: Goal Y coordinate
            theta: Goal orientation (can be None if orientation doesn't matter)
            position_tolerance: Position tolerance in meters (uses default if None)
            angle_tolerance: Angle tolerance in radians (uses default if None)
            
        Returns:
            bool: True if at goal, False otherwise
        """
        if position_tolerance is None:
            position_tolerance = self.goal_tolerance
            
        if angle_tolerance is None:
            angle_tolerance = self.angle_tolerance
            
        current_pose = self.client.get_pose()
        
        # Check position
        dx = current_pose[0] - x
        dy = current_pose[1] - y
        distance = np.sqrt(dx**2 + dy**2)
        
        position_ok = distance <= position_tolerance
        
        # Check orientation if it matters
        if theta is not None:
            angle_diff = self._normalize_angle(current_pose[2] - theta)
            orientation_ok = abs(angle_diff) <= angle_tolerance
        else:
            orientation_ok = True
            
        return position_ok and orientation_ok
        
    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to be between -pi and pi.
        
        Args:
            angle: Angle in radians
            
        Returns:
            float: Normalized angle
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle