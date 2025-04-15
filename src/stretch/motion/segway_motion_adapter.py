"""
Segway Motion Adapter for Stretch AI

This module implements an adapter for integrating the Segway robot's
motion capabilities with Stretch AI's advanced motion control framework.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import threading

from stretch.core.interfaces import HybridAction
from stretch.motion.base.base import MotionBase
from stretch.motion.control.traj_following_controller import TrajectoryFollowingController
from stretch.motion.control.feedback.velocity_controllers import VelocityController
from stretch.motion.constants import CONTROL_RATE
from stretch.motion.conversions import euler_to_quaternion, quaternion_to_euler
from stretch.motion.utils.geometry import transform_pose, get_yaw, normalize_angle
from stretch.utils.logger import Logger

logger = Logger(__name__)


class SegwayMotionAdapter:
    """
    Adapter for integrating Segway robot's motion capabilities with
    Stretch AI's advanced motion control framework.
    """
    
    def __init__(self, parameters, ros_client):
        """
        Initialize the Segway Motion Adapter.
        
        Args:
            parameters: Configuration parameters
            ros_client: SegwayROSClient instance
        """
        self.parameters = parameters
        self.ros_client = ros_client
        
        # Extract configuration
        self.motion_config = parameters.get("motion", {})
        self.robot_config = parameters.get("robot", {})
        self.nav_config = parameters.get("navigation", {})
        
        # Core motion parameters
        self.controller_update_rate = self.motion_config.get(
            "controller_update_rate", 
            self.nav_config.get("controller_update_rate", 10.0)
        )
        self.max_linear_speed = self.robot_config.get("max_linear_speed", 0.5)
        self.max_angular_speed = self.robot_config.get("max_angular_speed", 0.8)
        self.min_linear_speed = self.motion_config.get("min_linear_speed", 0.05)
        self.min_angular_speed = self.motion_config.get("min_angular_speed", 0.1)
        self.acceleration_limit = self.motion_config.get("acceleration_limit", 0.5)
        self.angular_acceleration_limit = self.motion_config.get("angular_acceleration_limit", 1.0)
        
        # Control parameters
        self.position_tolerance = self.motion_config.get(
            "position_tolerance", 
            self.nav_config.get("goal_tolerance_position", 0.1)
        )
        self.orientation_tolerance = self.motion_config.get(
            "orientation_tolerance", 
            self.nav_config.get("goal_tolerance_orientation", 0.1)
        )
        self.linear_p_gain = self.motion_config.get("linear_p_gain", 0.5)
        self.angular_p_gain = self.motion_config.get("angular_p_gain", 1.0)
        
        # Motion state
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.target_linear_velocity = 0.0
        self.target_angular_velocity = 0.0
        self.last_command_time = time.time()
        
        # Trajectory following state
        self.trajectory = None
        self.trajectory_index = 0
        self.trajectory_active = False
        
        # Thread control
        self.motion_thread = None
        self.motion_lock = threading.Lock()
        self.running = False
        
        # Initialize motion components
        self._initialize_motion_components()
        
        logger.info("SegwayMotionAdapter initialized")
    
    def _initialize_motion_components(self):
        """Initialize motion control components."""
        try:
            logger.info("Initializing motion components")
            
            # Initialize base motion controller
            self.motion_base = MotionBase()
            
            # Initialize velocity controller
            self.velocity_controller = VelocityController(
                kp_linear=self.linear_p_gain,
                kp_angular=self.angular_p_gain,
                max_linear_speed=self.max_linear_speed,
                max_angular_speed=self.max_angular_speed
            )
            
            # Initialize trajectory following controller
            self.trajectory_controller = TrajectoryFollowingController(
                position_tolerance=self.position_tolerance,
                orientation_tolerance=self.orientation_tolerance,
                max_linear_speed=self.max_linear_speed,
                max_angular_speed=self.max_angular_speed
            )
            
            logger.info("Motion components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing motion components: {e}")
            self.motion_base = None
            self.velocity_controller = None
            self.trajectory_controller = None
    
    def start(self):
        """Start the motion adapter."""
        if self.running:
            logger.warning("SegwayMotionAdapter already running")
            return
            
        self.running = True
        self.motion_thread = threading.Thread(target=self._motion_thread_func)
        self.motion_thread.daemon = True
        self.motion_thread.start()
        
        logger.info("SegwayMotionAdapter started")
    
    def stop(self):
        """Stop the motion adapter."""
        self.running = False
        
        # Stop any active motion
        self.stop_motion()
        
        if self.motion_thread:
            self.motion_thread.join(timeout=2.0)
            self.motion_thread = None
            
        logger.info("SegwayMotionAdapter stopped")
    
    def _motion_thread_func(self):
        """Background thread for motion control."""
        logger.info("Motion thread started")
        
        # Calculate sleep time from controller update rate
        sleep_time = 1.0 / self.controller_update_rate
        
        while self.running:
            try:
                # Get the current time
                current_time = time.time()
                
                # If we're following a trajectory
                if self.trajectory_active:
                    self._execute_trajectory_step()
                else:
                    # Otherwise, apply velocity commands with acceleration limits
                    self._apply_velocity_commands(current_time)
                
                # Sleep to maintain control rate
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error in motion thread: {e}")
                time.sleep(sleep_time)
        
        logger.info("Motion thread stopped")
    
    def _apply_velocity_commands(self, current_time):
        """
        Apply velocity commands with acceleration limits.
        
        Args:
            current_time: Current time
        """
        try:
            # Calculate time delta
            dt = current_time - self.last_command_time
            self.last_command_time = current_time
            
            # Limit acceleration
            if dt > 0:
                # Linear velocity
                linear_diff = self.target_linear_velocity - self.current_linear_velocity
                max_linear_change = self.acceleration_limit * dt
                if abs(linear_diff) > max_linear_change:
                    linear_change = max_linear_change * np.sign(linear_diff)
                else:
                    linear_change = linear_diff
                
                # Angular velocity
                angular_diff = self.target_angular_velocity - self.current_angular_velocity
                max_angular_change = self.angular_acceleration_limit * dt
                if abs(angular_diff) > max_angular_change:
                    angular_change = max_angular_change * np.sign(angular_diff)
                else:
                    angular_change = angular_diff
                
                # Update current velocities
                self.current_linear_velocity += linear_change
                self.current_angular_velocity += angular_change
            
            # Apply minimum speed thresholds
            if abs(self.current_linear_velocity) < self.min_linear_speed and self.target_linear_velocity != 0:
                self.current_linear_velocity = self.min_linear_speed * np.sign(self.current_linear_velocity)
            
            if abs(self.current_angular_velocity) < self.min_angular_speed and self.target_angular_velocity != 0:
                self.current_angular_velocity = self.min_angular_speed * np.sign(self.current_angular_velocity)
            
            # Send velocity commands to the robot
            self.ros_client.send_velocity_command(
                self.current_linear_velocity,
                self.current_angular_velocity
            )
            
            logger.debug(
                f"Applied velocity: linear={self.current_linear_velocity:.2f}, "
                f"angular={self.current_angular_velocity:.2f}"
            )
        except Exception as e:
            logger.error(f"Error applying velocity commands: {e}")
    
    def _execute_trajectory_step(self):
        """Execute one step of trajectory following."""
        try:
            with self.motion_lock:
                if self.trajectory is None or self.trajectory_index >= len(self.trajectory):
                    self.trajectory_active = False
                    self.stop_motion()
                    return
                
                # Get current robot pose
                xyt = self.ros_client.get_pose()
                if xyt is None:
                    logger.warning("Failed to get current pose, pausing trajectory execution")
                    return
                
                # Get current and target waypoints
                current_waypoint = self.trajectory[self.trajectory_index]
                
                # Check if we've reached the current waypoint
                x, y, theta = xyt
                waypoint_x, waypoint_y = current_waypoint[:2]
                waypoint_theta = current_waypoint[2] if len(current_waypoint) > 2 else None
                
                # Calculate distance and angle to waypoint
                dx = waypoint_x - x
                dy = waypoint_y - y
                distance = np.sqrt(dx*dx + dy*dy)
                
                # If we've reached the waypoint
                if distance < self.position_tolerance:
                    # If orientation matters, check orientation
                    if waypoint_theta is not None:
                        angle_diff = abs(normalize_angle(theta - waypoint_theta))
                        if angle_diff > self.orientation_tolerance:
                            # Turn to face the correct orientation
                            self.target_linear_velocity = 0.0
                            self.target_angular_velocity = self.angular_p_gain * normalize_angle(waypoint_theta - theta)
                            return
                    
                    # Move to next waypoint
                    self.trajectory_index += 1
                    
                    # If we're at the end of the trajectory
                    if self.trajectory_index >= len(self.trajectory):
                        self.trajectory_active = False
                        self.stop_motion()
                        logger.info("Trajectory completed")
                        return
                    
                    logger.debug(f"Reached waypoint {self.trajectory_index-1}, moving to next waypoint")
                    return
                
                # Otherwise, move towards the waypoint
                angle_to_waypoint = np.arctan2(dy, dx)
                angle_diff = normalize_angle(angle_to_waypoint - theta)
                
                # If we're not facing the waypoint, turn to face it
                if abs(angle_diff) > self.orientation_tolerance:
                    self.target_linear_velocity = 0.0
                    self.target_angular_velocity = self.angular_p_gain * angle_diff
                else:
                    # Otherwise, move towards the waypoint
                    self.target_linear_velocity = self.linear_p_gain * distance
                    self.target_angular_velocity = self.angular_p_gain * angle_diff
                
                # Limit velocity based on maximum speeds
                self.target_linear_velocity = np.clip(
                    self.target_linear_velocity,
                    -self.max_linear_speed,
                    self.max_linear_speed
                )
                self.target_angular_velocity = np.clip(
                    self.target_angular_velocity,
                    -self.max_angular_speed,
                    self.max_angular_speed
                )
        except Exception as e:
            logger.error(f"Error executing trajectory step: {e}")
    
    def set_velocity(self, linear_velocity, angular_velocity):
        """
        Set target velocity.
        
        Args:
            linear_velocity: Target linear velocity in m/s
            angular_velocity: Target angular velocity in rad/s
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.debug(f"Setting velocity: linear={linear_velocity}, angular={angular_velocity}")
            
            # Limit velocity based on maximum speeds
            linear_velocity = np.clip(linear_velocity, -self.max_linear_speed, self.max_linear_speed)
            angular_velocity = np.clip(angular_velocity, -self.max_angular_speed, self.max_angular_speed)
            
            # Update target velocities
            with self.motion_lock:
                self.target_linear_velocity = linear_velocity
                self.target_angular_velocity = angular_velocity
                self.trajectory_active = False
            
            return True
        except Exception as e:
            logger.error(f"Error setting velocity: {e}")
            return False
    
    def stop_motion(self):
        """
        Stop all motion.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Stopping motion")
            
            # Stop the robot immediately
            self.ros_client.send_velocity_command(0.0, 0.0)
            
            # Reset motion state
            with self.motion_lock:
                self.current_linear_velocity = 0.0
                self.current_angular_velocity = 0.0
                self.target_linear_velocity = 0.0
                self.target_angular_velocity = 0.0
                self.trajectory_active = False
                self.trajectory = None
                self.trajectory_index = 0
            
            return True
        except Exception as e:
            logger.error(f"Error stopping motion: {e}")
            return False
    
    def follow_trajectory(self, trajectory):
        """
        Follow a trajectory.
        
        Args:
            trajectory: List of waypoints [[x, y, theta], ...]
            
        Returns:
            bool: True if trajectory following started successfully, False otherwise
        """
        try:
            if trajectory is None or len(trajectory) == 0:
                logger.warning("Cannot follow empty trajectory")
                return False
            
            logger.info(f"Following trajectory with {len(trajectory)} waypoints")
            
            # Stop any current motion
            self.stop_motion()
            
            # Set trajectory
            with self.motion_lock:
                self.trajectory = trajectory
                self.trajectory_index = 0
                self.trajectory_active = True
            
            return True
        except Exception as e:
            logger.error(f"Error starting trajectory following: {e}")
            return False
    
    def move_to_pose(self, pose):
        """
        Move to a specific pose.
        
        Args:
            pose: Target pose [x, y, theta]
            
        Returns:
            bool: True if motion started successfully, False otherwise
        """
        try:
            logger.info(f"Moving to pose: {pose}")
            
            # Create a single-waypoint trajectory
            trajectory = [pose]
            
            # Follow the trajectory
            return self.follow_trajectory(trajectory)
        except Exception as e:
            logger.error(f"Error moving to pose: {e}")
            return False
    
    def rotate_in_place(self, angle):
        """
        Rotate in place by the specified angle.
        
        Args:
            angle: Rotation angle in radians
            
        Returns:
            bool: True if rotation started successfully, False otherwise
        """
        try:
            logger.info(f"Rotating in place by {angle} radians")
            
            # Get current pose
            xyt = self.ros_client.get_pose()
            if xyt is None:
                logger.warning("Failed to get current pose, cannot rotate")
                return False
            
            # Calculate target pose
            x, y, theta = xyt
            target_theta = normalize_angle(theta + angle)
            
            # Create a pose at the current position with the target orientation
            pose = [x, y, target_theta]
            
            # Move to the pose (will only change orientation since position is the same)
            return self.move_to_pose(pose)
        except Exception as e:
            logger.error(f"Error rotating in place: {e}")
            return False
    
    def is_motion_complete(self):
        """
        Check if motion is complete.
        
        Returns:
            bool: True if motion is complete, False otherwise
        """
        with self.motion_lock:
            # If we're following a trajectory, check if it's complete
            if self.trajectory_active:
                return False
            
            # Otherwise, check if we're stopped
            return (abs(self.current_linear_velocity) < 0.01 and 
                    abs(self.current_angular_velocity) < 0.01 and
                    abs(self.target_linear_velocity) < 0.01 and
                    abs(self.target_angular_velocity) < 0.01)
    
    def get_remaining_trajectory(self):
        """
        Get the remaining trajectory.
        
        Returns:
            list: Remaining trajectory waypoints
        """
        with self.motion_lock:
            if self.trajectory is None or not self.trajectory_active:
                return []
            
            return self.trajectory[self.trajectory_index:]
    
    def convert_action(self, action):
        """
        Convert a Stretch AI action to Segway motion commands.
        
        Args:
            action: Stretch AI HybridAction or native action
            
        Returns:
            bool: True if action converted and applied successfully, False otherwise
        """
        try:
            logger.debug(f"Converting action: {action}")
            
            # If action is a HybridAction, extract the native action
            if isinstance(action, HybridAction):
                # Check action type
                if action.is_discrete():
                    # Handle discrete action
                    discrete_action = action.get()
                    return self._handle_discrete_action(discrete_action)
                elif action.is_navigation():
                    # Handle navigation action
                    xyt = action.get()
                    return self.move_to_pose(xyt)
                else:
                    logger.warning(f"Unsupported action type: {action.action_type}")
                    return False
            else:
                # Handle native action based on type
                if hasattr(action, 'xyt'):
                    # Looks like a navigation action
                    return self.move_to_pose(action.xyt)
                elif hasattr(action, 'linear') and hasattr(action, 'angular'):
                    # Looks like a velocity command
                    return self.set_velocity(action.linear, action.angular)
                else:
                    logger.warning(f"Unknown action type: {type(action)}")
                    return False
        except Exception as e:
            logger.error(f"Error converting action: {e}")
            return False
    
    def _handle_discrete_action(self, discrete_action):
        """
        Handle a discrete action.
        
        Args:
            discrete_action: Discrete action
            
        Returns:
            bool: True if action handled successfully, False otherwise
        """
        try:
            # Check the type of discrete action
            if hasattr(discrete_action, 'value'):
                action_value = discrete_action.value
            else:
                action_value = discrete_action
            
            # Handle different discrete actions based on value
            # These values should match the DiscreteNavigationAction enum in the interface
            if action_value == 0:  # STOP
                return self.stop_motion()
            elif action_value == 1:  # MOVE_FORWARD
                return self.set_velocity(self.max_linear_speed, 0.0)
            elif action_value == 2:  # TURN_LEFT
                return self.set_velocity(0.0, self.max_angular_speed)
            elif action_value == 3:  # TURN_RIGHT
                return self.set_velocity(0.0, -self.max_angular_speed)
            elif action_value == 4:  # PICK_OBJECT
                logger.warning("PICK_OBJECT action not supported by Segway robot")
                return False
            elif action_value == 5:  # PLACE_OBJECT
                logger.warning("PLACE_OBJECT action not supported by Segway robot")
                return False
            elif action_value == 6:  # NAVIGATION_MODE
                # Handle this in the higher-level robot agent
                return True
            elif action_value == 7:  # MANIPULATION_MODE
                # Handle this in the higher-level robot agent
                return True
            else:
                logger.warning(f"Unknown discrete action value: {action_value}")
                return False
        except Exception as e:
            logger.error(f"Error handling discrete action: {e}")
            return False
    
    def set_control_parameters(self, **kwargs):
        """
        Set control parameters.
        
        Args:
            **kwargs: Control parameters to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Setting control parameters: {kwargs}")
            
            # Update parameters
            for key, value in kwargs.items():
                if key == "max_linear_speed":
                    self.max_linear_speed = value
                elif key == "max_angular_speed":
                    self.max_angular_speed = value
                elif key == "min_linear_speed":
                    self.min_linear_speed = value
                elif key == "min_angular_speed":
                    self.min_angular_speed = value
                elif key == "acceleration_limit":
                    self.acceleration_limit = value
                elif key == "angular_acceleration_limit":
                    self.angular_acceleration_limit = value
                elif key == "position_tolerance":
                    self.position_tolerance = value
                elif key == "orientation_tolerance":
                    self.orientation_tolerance = value
                elif key == "linear_p_gain":
                    self.linear_p_gain = value
                elif key == "angular_p_gain":
                    self.angular_p_gain = value
                else:
                    logger.warning(f"Unknown parameter: {key}")
            
            # Update controllers
            if self.velocity_controller is not None:
                self.velocity_controller.max_linear_speed = self.max_linear_speed
                self.velocity_controller.max_angular_speed = self.max_angular_speed
                self.velocity_controller.kp_linear = self.linear_p_gain
                self.velocity_controller.kp_angular = self.angular_p_gain
            
            if self.trajectory_controller is not None:
                self.trajectory_controller.position_tolerance = self.position_tolerance
                self.trajectory_controller.orientation_tolerance = self.orientation_tolerance
                self.trajectory_controller.max_linear_speed = self.max_linear_speed
                self.trajectory_controller.max_angular_speed = self.max_angular_speed
            
            return True
        except Exception as e:
            logger.error(f"Error setting control parameters: {e}")
            return False
    
    def get_motion_status(self):
        """
        Get motion status.
        
        Returns:
            dict: Motion status
        """
        with self.motion_lock:
            return {
                "trajectory_active": self.trajectory_active,
                "trajectory_index": self.trajectory_index if self.trajectory else None,
                "trajectory_length": len(self.trajectory) if self.trajectory else 0,
                "current_linear_velocity": self.current_linear_velocity,
                "current_angular_velocity": self.current_angular_velocity,
                "target_linear_velocity": self.target_linear_velocity,
                "target_angular_velocity": self.target_angular_velocity,
                "position_tolerance": self.position_tolerance,
                "orientation_tolerance": self.orientation_tolerance,
                "max_linear_speed": self.max_linear_speed,
                "max_angular_speed": self.max_angular_speed
            }