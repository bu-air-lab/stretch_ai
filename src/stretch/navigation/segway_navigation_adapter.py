"""
Segway Navigation Adapter for Stretch AI

This module implements an adapter for integrating the Segway robot's
navigation capabilities with Stretch AI's advanced navigation framework.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import queue

from stretch.core.interfaces import Observations, PointCloud
from stretch.navigation.base.base import NavigationBase
from stretch.navigation.utils.geometry import get_pose_in_reference_frame as transform_pose
from stretch.navigation.utils.geometry import euler_angles_to_quaternion
from stretch.motion.algo.a_star import AStar
from stretch.motion.algo.rrt import RRT
from stretch.motion.algo.simplify import simplify_path
from stretch.motion.control.goto_controller import GotoController
from stretch.utils.logger import Logger
from stretch.mapping.voxel import SparseVoxelMapNavigationSpace
from stretch.motion import ConfigurationSpace

# Check if ORBSLAM is available and import if possible
try:
    from stretch.navigation.orbslam import OrbSlam
    ORBSLAM_AVAILABLE = True
except ImportError:
    ORBSLAM_AVAILABLE = False

logger = Logger(__name__)


class SegwayNavigationAdapter:
    """
    Adapter for integrating Segway robot's navigation capabilities with
    Stretch AI's advanced navigation framework.
    """
    
    def __init__(self, parameters, ros_client, mapping_adapter=None):
        """
        Initialize the Segway Navigation Adapter.
        
        Args:
            parameters: Configuration parameters
            ros_client: SegwayROSClient instance
            mapping_adapter: SegwayMappingAdapter instance (optional)
        """
        self.parameters = parameters
        self.ros_client = ros_client
        self.mapping_adapter = mapping_adapter
        
        # Extract navigation configuration
        self.nav_config = parameters.get("navigation", {})
        self.robot_config = parameters.get("robot", {})
        
        # Core navigation parameters
        self.planning_method = self.nav_config.get("planning_method", "astar")
        self.obstacle_inflation = self.nav_config.get("obstacle_inflation", 0.3)
        self.path_simplification = self.nav_config.get("path_simplification", True)
        self.controller_update_rate = self.nav_config.get("controller_update_rate", 10.0)
        self.goal_tolerance_position = self.nav_config.get("goal_tolerance_position", 0.1)
        self.goal_tolerance_orientation = self.nav_config.get("goal_tolerance_orientation", 0.1)
        
        # Safety settings
        self.min_obstacle_distance = self.nav_config.get("min_obstacle_distance", 0.3)
        self.emergency_stop_distance = self.nav_config.get("emergency_stop_distance", 0.2)
        
        # Robot physical parameters
        self.max_linear_speed = self.robot_config.get("max_linear_speed", 0.5)
        self.max_angular_speed = self.robot_config.get("max_angular_speed", 0.8)
        self.footprint_radius = self.robot_config.get("footprint_radius", 0.35)
        
        # Localization settings
        self.use_orbslam = self.nav_config.get("use_orbslam", False) and ORBSLAM_AVAILABLE
        self.localization_method = self.nav_config.get("localization_method", "odom")
        
        # Navigation state
        self.current_path = None
        self.current_goal = None
        self.navigation_active = False
        self.emergency_stop = False
        self.pose_confidence = 1.0
        
        # For thread safety
        self.navigation_lock = threading.Lock()
        self.navigation_thread = None
        self.running = False
        
        # Initialize navigation components
        self._initialize_navigation_components()
        
        logger.info("SegwayNavigationAdapter initialized")
    
    def _initialize_navigation_components(self):
        """Initialize navigation components."""
        try:
            logger.info("Initializing navigation components")
            
            # Initialize base navigation
            self.navigation_base = NavigationBase()
            
            # Get the navigation space first - needed for planners
            self.navigation_space = self._get_navigation_space()
            
            # Only initialize planners if we have a valid navigation space
            if self.navigation_space is not None:
                # Initialize AStar
                astar_config = self.parameters.get("navigation.astar", {})
                try:
                    logger.debug("Initializing AStar planner with navigation space")
                    self.astar_planner = AStar(
                        space=self.navigation_space,
                        **astar_config
                    )
                    logger.info("AStar planner initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing AStar: {e}")
                    self.astar_planner = None
                    
                # Initialize RRT
                rrt_config = self.parameters.get("navigation.rrt", {})
                try:
                    logger.debug("Initializing RRT planner with navigation space")
                    self.rrt_planner = RRT(
                        space=self.navigation_space,
                        validate_fn=self.navigation_space.is_valid,
                        **rrt_config
                    )
                    logger.info("RRT planner initialized successfully")
                except Exception as e:
                    logger.error(f"Error initializing RRT: {e}")
                    self.rrt_planner = None
            else:
                logger.warning("Navigation space not available, skipping planner initialization")
                self.astar_planner = None
                self.rrt_planner = None
            
            # Initialize controller
            self.goto_controller = GotoController(
                position_tolerance=self.goal_tolerance_position,
                orientation_tolerance=self.goal_tolerance_orientation,
                max_linear_speed=self.max_linear_speed,
                max_angular_speed=self.max_angular_speed
            )
            
            # Initialize ORBSLAM
            self.orbslam = None
            if self.use_orbslam and ORBSLAM_AVAILABLE:
                logger.info("Initializing ORBSLAM")
                self.orbslam = OrbSlam(
                    vocabulary_path=self.nav_config.get("orbslam_vocabulary_path", ""),
                    settings_path=self.nav_config.get("orbslam_settings_path", "")
                )
            
            logger.info("Navigation components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing navigation components: {e}")
            self.navigation_base = None
            self.astar_planner = None
            self.rrt_planner = None
            self.goto_controller = None
            self.orbslam = None
    
    def start(self):
        """Start the navigation adapter."""
        if self.running:
            logger.warning("SegwayNavigationAdapter already running")
            return
            
        self.running = True
        self.navigation_thread = threading.Thread(target=self._navigation_thread_func)
        self.navigation_thread.daemon = True
        self.navigation_thread.start()
        
        logger.info("SegwayNavigationAdapter started")
    
    def stop(self):
        """Stop the navigation adapter."""
        self.running = False
        
        # Stop navigation if active
        self.stop_navigation()
        
        if self.navigation_thread:
            self.navigation_thread.join(timeout=2.0)
            self.navigation_thread = None
            
        logger.info("SegwayNavigationAdapter stopped")
    
    def _navigation_thread_func(self):
        """Background thread for navigation control."""
        logger.info("Navigation thread started")
        
        # Calculate sleep time from controller update rate
        sleep_time = 1.0 / self.controller_update_rate
        
        while self.running:
            try:
                # Check if navigation is active
                if not self.navigation_active:
                    time.sleep(sleep_time)
                    continue
                    
                # Get current robot pose
                current_pose = self._get_current_pose()
                if current_pose is None:
                    logger.warning("Failed to get current pose, pausing navigation")
                    time.sleep(sleep_time)
                    continue
                
                # Check for emergency stop conditions
                if self._check_emergency_stop():
                    logger.warning("Emergency stop condition detected")
                    self.emergency_stop = True
                    self.stop_navigation()
                    continue
                
                # Execute next navigation step
                with self.navigation_lock:
                    if self.current_path and self.current_goal:
                        self._execute_navigation_step(current_pose)
                
                # Sleep to maintain control rate
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error in navigation thread: {e}")
                time.sleep(sleep_time)
        
        logger.info("Navigation thread stopped")
    
    def quaternion_from_euler(roll, pitch, yaw):
        w, x, y, z = euler_angles_to_quaternion(roll, pitch, yaw)
        return (x, y, z, w) 

    def _get_current_pose(self):
        """
        Get the current robot pose.
        
        Returns:
            tuple: (position, orientation) where position is [x, y, z] and 
                  orientation is [x, y, z, w] quaternion, or None if pose cannot be determined
        """
        try:
            # Use ORBSLAM if enabled
            if self.use_orbslam and self.orbslam is not None:
                pose = self.orbslam.get_current_pose()
                if pose is not None:
                    return pose
            
            # Fall back to mapping adapter if available
            if self.mapping_adapter is not None:
                return self.mapping_adapter.get_robot_pose()
            
            # Fall back to ROS client
            xyt = self.ros_client.get_pose()
            if xyt is not None:
                x, y, theta = xyt
                position = np.array([x, y, 0.0])
                orientation = quaternion_from_euler(0, 0, theta)
                return position, orientation
            
            return None
        except Exception as e:
            logger.error(f"Error getting current pose: {e}")
            return None
    
    def _check_emergency_stop(self):
        """
        Check if emergency stop is needed.
        
        Returns:
            bool: True if emergency stop is needed, False otherwise
        """
        try:
            # Check for obstacles within emergency stop distance
            if self.mapping_adapter is not None:
                nearest_obstacle_distance = self.mapping_adapter.get_nearest_obstacle_distance()
                if nearest_obstacle_distance < self.emergency_stop_distance:
                    return True
            
            # Additional emergency stop checks could be added here
            
            return False
        except Exception as e:
            logger.error(f"Error checking emergency stop: {e}")
            return False
    
    def _execute_navigation_step(self, current_pose):
        """
        Execute one step of navigation.
        
        Args:
            current_pose: Current robot pose (position, orientation)
        """
        try:
            # Unpack current pose
            position, orientation = current_pose
            
            # Check if we've reached the goal
            if self._check_goal_reached(position, orientation):
                logger.info("Goal reached")
                self.navigation_active = False
                return
            
            # Find the next waypoint to follow
            next_waypoint = self._get_next_waypoint(position)
            if next_waypoint is None:
                logger.warning("No valid waypoint found, stopping navigation")
                self.navigation_active = False
                return
            
            # Get control commands for the waypoint
            linear_vel, angular_vel = self.goto_controller.compute_velocity(
                current_position=position[:2],
                current_yaw=self._get_yaw_from_quaternion(orientation),
                target_position=next_waypoint[:2],
                target_yaw=next_waypoint[2] if len(next_waypoint) > 2 else None
            )
            
            # Send velocity commands to the robot
            self.ros_client.send_velocity_command(linear_vel, angular_vel)
            
            logger.debug(f"Navigation step: linear_vel={linear_vel}, angular_vel={angular_vel}")
        except Exception as e:
            logger.error(f"Error executing navigation step: {e}")
    
    def _check_goal_reached(self, position, orientation):
        """
        Check if the goal has been reached.
        
        Args:
            position: Current position [x, y, z]
            orientation: Current orientation [x, y, z, w]
            
        Returns:
            bool: True if goal reached, False otherwise
        """
        with self.navigation_lock:
            if self.current_goal is None:
                return True
            
            # Extract goal position and orientation
            goal_position = self.current_goal[:2] if len(self.current_goal) >= 2 else None
            goal_yaw = self.current_goal[2] if len(self.current_goal) >= 3 else None
            
            # Check position tolerance
            if goal_position is not None:
                position_error = np.linalg.norm(np.array(position[:2]) - np.array(goal_position))
                if position_error > self.goal_tolerance_position:
                    return False
            
            # Check orientation tolerance if specified
            if goal_yaw is not None:
                current_yaw = self._get_yaw_from_quaternion(orientation)
                yaw_error = abs(self._normalize_angle(current_yaw - goal_yaw))
                if yaw_error > self.goal_tolerance_orientation:
                    return False
            
            return True
    
    def _get_next_waypoint(self, position):
        """
        Get the next waypoint to follow.
        
        Args:
            position: Current position [x, y, z]
            
        Returns:
            np.ndarray: Next waypoint [x, y, theta] or None if no waypoint available
        """
        with self.navigation_lock:
            if self.current_path is None or len(self.current_path) == 0:
                return None
            
            # Find the closest waypoint ahead of the robot
            current_position_2d = np.array(position[:2])
            min_distance = float('inf')
            next_waypoint_index = None
            
            for i, waypoint in enumerate(self.current_path):
                waypoint_2d = np.array(waypoint[:2])
                distance = np.linalg.norm(waypoint_2d - current_position_2d)
                
                if distance < min_distance:
                    min_distance = distance
                    next_waypoint_index = i
            
            # If we're close enough to the current waypoint, move to the next one
            if min_distance < self.goal_tolerance_position and next_waypoint_index < len(self.current_path) - 1:
                next_waypoint_index += 1
            
            # If we're at the last waypoint, return the goal
            if next_waypoint_index == len(self.current_path) - 1:
                return self.current_goal
            
            return self.current_path[next_waypoint_index]
    
    def _get_yaw_from_quaternion(self, quaternion):
        """
        Extract yaw angle from quaternion.
        
        Args:
            quaternion: Orientation quaternion [x, y, z, w]
            
        Returns:
            float: Yaw angle in radians
        """
        x, y, z, w = quaternion
        
        # Convert quaternion to Euler angles
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return yaw
    
    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi].
        
        Args:
            angle: Angle in radians
            
        Returns:
            float: Normalized angle in radians
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def plan_path(self, goal, use_rrt=False):
        """
        Plan a path to a goal position.
        
        Args:
            goal: Goal position [x, y] or [x, y, theta]
            use_rrt: If True, use RRT planner, otherwise use A*
            
        Returns:
            list: List of waypoints [[x, y, theta], ...] or None if no path found
        """
        try:
            logger.info(f"Planning path to goal: {goal}")
            
            # Get current position
            current_pose = self._get_current_pose()
            if current_pose is None:
                logger.warning("Cannot plan path: current pose unknown")
                return None
            
            position, _ = current_pose
            
            # Convert goal to 2D if needed
            goal_2d = goal[:2] if len(goal) > 2 else goal
            start_2d = position[:2]
            
            # Get obstacles from mapping adapter if available
            obstacles = []
            if self.mapping_adapter is not None:
                if hasattr(self.mapping_adapter, 'get_occupancy_grid'):
                    grid = self.mapping_adapter.get_occupancy_grid()
                    obstacles = grid.get_obstacles() if grid is not None else []
                elif hasattr(self.mapping_adapter, '_get_2d_obstacles_from_voxel_map'):
                    obstacles = self.mapping_adapter._get_2d_obstacles_from_voxel_map()
            
            # Determine planning method
            if use_rrt or self.planning_method == "rrt":
                # Use RRT planner
                if self.rrt_planner is not None:
                    path = self.rrt_planner.plan(
                        start=start_2d,
                        goal=goal_2d,
                        obstacles=obstacles
                    )
                else:
                    logger.warning("RRT planner not initialized, falling back to simple planning")
                    path = self._simple_plan_path(start_2d, goal_2d)
            else:
                # Use A* planner
                if self.astar_planner is not None:
                    path = self.astar_planner.plan(
                        start=start_2d,
                        goal=goal_2d,
                        obstacles=obstacles
                    )
                else:
                    logger.warning("A* planner not initialized, falling back to simple planning")
                    path = self._simple_plan_path(start_2d, goal_2d)
                
            # Return None if no path found
            if path is None or len(path) == 0:
                logger.warning(f"No path found to goal: {goal}")
                return None
                
            # Simplify path if enabled and simplify function is available
            if self.path_simplification and simplify_path is not None:
                try:
                    path = simplify_path(
                        path,
                        obstacles=obstacles,
                        obstacle_inflation=self.obstacle_inflation
                    )
                except Exception as e:
                    logger.error(f"Error simplifying path: {e}")
                
            # Add orientation to waypoints
            path_with_orientation = []
            for i, waypoint in enumerate(path):
                if i < len(path) - 1:
                    # Compute orientation based on direction to next waypoint
                    next_waypoint = path[i + 1]
                    dx = next_waypoint[0] - waypoint[0]
                    dy = next_waypoint[1] - waypoint[1]
                    theta = np.arctan2(dy, dx)
                    path_with_orientation.append([waypoint[0], waypoint[1], theta])
                else:
                    # Use goal orientation for final waypoint if specified
                    if len(goal) > 2:
                        path_with_orientation.append([waypoint[0], waypoint[1], goal[2]])
                    else:
                        # Keep previous orientation
                        theta = path_with_orientation[-1][2] if path_with_orientation else 0.0
                        path_with_orientation.append([waypoint[0], waypoint[1], theta])
                
            logger.info(f"Path planned with {len(path_with_orientation)} waypoints")
            return path_with_orientation
        except Exception as e:
            logger.error(f"Error planning path: {e}")
            return None
    
    def _simple_plan_path(self, start, goal):
        """
        Simple straight-line path planning.
        
        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            
        Returns:
            list: List of waypoints [[x, y], ...] or None if no path found
        """
        # Create a simple path with intermediate waypoints
        direction = np.array(goal) - np.array(start)
        distance = np.linalg.norm(direction)
        
        # Determine number of waypoints based on distance
        num_waypoints = max(2, int(distance / 0.5))
        
        # Create waypoints
        waypoints = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            waypoint = start + t * direction
            waypoints.append(waypoint.tolist())
            
        return waypoints
    
    def navigate_to(self, goal, use_rrt=False):
        """
        Navigate to a goal position.
        
        Args:
            goal: Goal position [x, y] or [x, y, theta]
            use_rrt: If True, use RRT planner, otherwise use A*
            
        Returns:
            bool: True if navigation started successfully, False otherwise
        """
        try:
            logger.info(f"Navigating to goal: {goal}")
            
            # Stop any ongoing navigation
            self.stop_navigation()
            
            # Plan path to goal
            path = self.plan_path(goal, use_rrt)
            if path is None:
                logger.warning(f"Cannot navigate: no path found to {goal}")
                return False
            
            # Update navigation state
            with self.navigation_lock:
                self.current_path = path
                self.current_goal = goal if len(goal) > 2 else np.append(goal, 0.0)  # Add default orientation if needed
                self.navigation_active = True
                self.emergency_stop = False
            
            logger.info(f"Navigation started to goal: {goal}")
            return True
        except Exception as e:
            logger.error(f"Error starting navigation: {e}")
            return False
    
    def stop_navigation(self):
        """Stop current navigation."""
        try:
            logger.info("Stopping navigation")
            
            # Update navigation state
            with self.navigation_lock:
                self.navigation_active = False
                self.current_path = None
                self.current_goal = None
            
            # Send stop command to robot
            self.ros_client.send_velocity_command(0.0, 0.0)
            
            return True
        except Exception as e:
            logger.error(f"Error stopping navigation: {e}")
            return False
    
    def is_navigating(self):
        """
        Check if navigation is active.
        
        Returns:
            bool: True if navigation is active, False otherwise
        """
        with self.navigation_lock:
            return self.navigation_active
    
    def is_emergency_stopped(self):
        """
        Check if emergency stop is active.
        
        Returns:
            bool: True if emergency stop is active, False otherwise
        """
        with self.navigation_lock:
            return self.emergency_stop
    
    def reset_emergency_stop(self):
        """
        Reset emergency stop.
        
        Returns:
            bool: True if reset successful, False otherwise
        """
        try:
            logger.info("Resetting emergency stop")
            
            with self.navigation_lock:
                self.emergency_stop = False
            
            return True
        except Exception as e:
            logger.error(f"Error resetting emergency stop: {e}")
            return False
    
    def set_goal_tolerance(self, position_tolerance, orientation_tolerance=None):
        """
        Set goal tolerance.
        
        Args:
            position_tolerance: Position tolerance in meters
            orientation_tolerance: Orientation tolerance in radians (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Setting goal tolerance: position={position_tolerance}, orientation={orientation_tolerance}")
            
            self.goal_tolerance_position = position_tolerance
            
            if orientation_tolerance is not None:
                self.goal_tolerance_orientation = orientation_tolerance
            
            # Update controller if available
            if self.goto_controller is not None:
                self.goto_controller.position_tolerance = position_tolerance
                if orientation_tolerance is not None:
                    self.goto_controller.orientation_tolerance = orientation_tolerance
            
            return True
        except Exception as e:
            logger.error(f"Error setting goal tolerance: {e}")
            return False
    
    def set_max_speed(self, linear_speed, angular_speed=None):
        """
        Set maximum speed.
        
        Args:
            linear_speed: Maximum linear speed in m/s
            angular_speed: Maximum angular speed in rad/s (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Setting max speed: linear={linear_speed}, angular={angular_speed}")
            
            self.max_linear_speed = linear_speed
            
            if angular_speed is not None:
                self.max_angular_speed = angular_speed
            
            # Update controller if available
            if self.goto_controller is not None:
                self.goto_controller.max_linear_speed = linear_speed
                if angular_speed is not None:
                    self.goto_controller.max_angular_speed = angular_speed
            
            return True
        except Exception as e:
            logger.error(f"Error setting max speed: {e}")
            return False
    
    def get_navigation_status(self):
        """
        Get navigation status.
        
        Returns:
            dict: Navigation status
        """
        with self.navigation_lock:
            return {
                "active": self.navigation_active,
                "emergency_stop": self.emergency_stop,
                "goal": self.current_goal.tolist() if isinstance(self.current_goal, np.ndarray) else self.current_goal,
                "path_length": len(self.current_path) if self.current_path else 0,
                "position_tolerance": self.goal_tolerance_position,
                "orientation_tolerance": self.goal_tolerance_orientation,
                "max_linear_speed": self.max_linear_speed,
                "max_angular_speed": self.max_angular_speed
            }
    
    def explore(self, max_frontiers=5, max_distance=5.0):
        """
        Explore the environment by navigating to frontier points.
        
        Args:
            max_frontiers: Maximum number of frontiers to explore
            max_distance: Maximum distance to frontiers in meters
            
        Returns:
            bool: True if exploration started successfully, False otherwise
        """
        try:
            logger.info(f"Starting exploration: max_frontiers={max_frontiers}, max_distance={max_distance}")
            
            # Check if mapping adapter is available
            if self.mapping_adapter is None:
                logger.warning("Cannot explore: mapping adapter not available")
                return False
            
            # Find frontiers
            if not hasattr(self.mapping_adapter, 'find_frontiers'):
                logger.warning("Cannot explore: mapping adapter does not support frontier detection")
                return False
                
            frontiers = self.mapping_adapter.find_frontiers(radius=max_distance)
            if not frontiers:
                logger.info("No frontiers found for exploration")
                return False
            
            # Sort frontiers by distance to current position
            current_pose = self._get_current_pose()
            if current_pose is None:
                logger.warning("Cannot explore: current pose unknown")
                return False
                
            position, _ = current_pose
            
            sorted_frontiers = sorted(
                frontiers[:max_frontiers],
                key=lambda f: np.linalg.norm(np.array(f) - np.array(position[:2]))
            )
            
            # Navigate to closest frontier
            if sorted_frontiers:
                frontier = sorted_frontiers[0]
                return self.navigate_to(frontier)
            
            return False
        except Exception as e:
            logger.error(f"Error starting exploration: {e}")
            return False

    def _get_navigation_space(self) -> Optional[ConfigurationSpace]:
        """Helper method to get the navigation space from the mapping adapter."""
        if self.mapping_adapter is None or self.mapping_adapter.voxel_map is None:
             logger.warning("Mapping adapter or voxel map not available for navigation space.")
             return None
        if self.ros_client is None or self.ros_client.robot_model is None:
             logger.warning("ROS client or robot model not available for navigation space.")
             return None

        try:
            # Create the navigation space using the voxel map and robot model
            # Adjust parameters as needed from your config
            space = SparseVoxelMapNavigationSpace(
                voxel_map=self.mapping_adapter.voxel_map, # Pass the actual map object
                robot=self.ros_client.robot_model,
                step_size=self.parameters.get("motion_planner.step_size", 0.1),
                rotation_step_size=self.parameters.get("motion_planner.rotation_step_size", 0.1),
                dilate_frontier_size=self.parameters.get("motion_planner.frontier.dilate_frontier_size", 0),
                dilate_obstacle_size=self.parameters.get("motion_planner.frontier.dilate_obstacle_size", 0),
                grid=self.mapping_adapter.voxel_map.grid, # Access grid from map
            )
            return space
        except Exception as e:
            logger.error(f"Error creating navigation space: {e}")
            return None