"""
Segway Agent Adapter for Stretch AI

This module implements an adapter layer between the Segway robot and
Stretch AI's agent framework, handling translation between different
data formats and control paradigms.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from stretch.core.interfaces import Observations, HybridAction, PointCloud
from stretch.utils.logger import Logger

logger = Logger(__name__)


class SegwayAgentAdapter:
    """
    Adapter for integrating Segway robot with Stretch AI agent framework.
    Handles translation between different control and data representation formats.
    """
    
    def __init__(self, segway_integration, parameters, gpu_device="cuda:0"):
        """
        Initialize the Segway Agent Adapter.
        
        Args:
            segway_integration: SegwayStretchIntegration instance
            parameters: Configuration parameters
            gpu_device: GPU device to use for acceleration
        """
        self.segway = segway_integration
        self.parameters = parameters
        self.gpu_device = gpu_device
        
        # Network configuration
        self.robot_ip = parameters.get("robot_ip", "10.66.171.191")
        self.desktop_ip = parameters.get("desktop_ip", "10.66.171.131")
        self.lidar_ip = parameters.get("lidar_ip", "10.66.171.8")
        
        # Debug mode
        self.debug_mode = parameters.get("debug_mode", False)
        
        # Last received observation
        self.last_observation = None
        
        # Last processed point cloud
        self.last_point_cloud = None
        
        # LiDAR data processing settings
        self.lidar_voxel_size = parameters.get("lidar_voxel_size", 0.05)
        self.lidar_max_range = parameters.get("lidar_max_range", 10.0)
        self.lidar_min_range = parameters.get("lidar_min_range", 0.05)
        
        # Initialize adapters for different components
        self._initialize_adapters()
        
        logger.info("SegwayAgentAdapter initialized")
        
    def _initialize_adapters(self):
        """Initialize adapters for different components."""
        # Here we would initialize adapters for different components
        # such as perception, planning, etc. if needed
        pass
    
    def process_observation(self, ros_observation) -> Optional[Observations]:
        """
        Process a ROS observation into Stretch AI's Observations format.
        
        Args:
            ros_observation: ROS observation message
            
        Returns:
            Observations: Processed observation in Stretch AI format
        """
        if ros_observation is None:
            logger.warning("Received None observation")
            return None
            
        try:
            # Extract data from ROS observation
            # This is a placeholder - actual implementation depends on your ROS message structure
            
            # Create Observations instance
            observation = Observations(
                gps=np.array([0.0, 0.0]) if not hasattr(ros_observation, 'gps') else ros_observation.gps,
                compass=np.array([0.0]) if not hasattr(ros_observation, 'compass') else ros_observation.compass,
                rgb=None if not hasattr(ros_observation, 'rgb') else ros_observation.rgb,
                depth=None if not hasattr(ros_observation, 'depth') else ros_observation.depth,
                lidar_points=None if not hasattr(ros_observation, 'lidar_points') else ros_observation.lidar_points,
                lidar_timestamp=time.time() if not hasattr(ros_observation, 'lidar_timestamp') else ros_observation.lidar_timestamp,
                seq_id=0 if not hasattr(ros_observation, 'seq_id') else ros_observation.seq_id
            )
            
            # Process LiDAR data into point cloud if available
            if hasattr(ros_observation, 'lidar_points') and ros_observation.lidar_points is not None:
                # Process into PointCloud
                observation.point_cloud = self._process_lidar_to_pointcloud(
                    ros_observation.lidar_points,
                    ros_observation.lidar_timestamp if hasattr(ros_observation, 'lidar_timestamp') else time.time()
                )
                
            # Save processed observation
            self.last_observation = observation
            
            return observation
        except Exception as e:
            logger.error(f"Error processing observation: {e}")
            return None
    
    def _process_lidar_to_pointcloud(self, lidar_points, timestamp) -> Optional[PointCloud]:
        """
        Process LiDAR points into a PointCloud object.
        
        Args:
            lidar_points: LiDAR points
            timestamp: Timestamp
            
        Returns:
            PointCloud: Processed point cloud
        """
        try:
            if lidar_points is None or len(lidar_points) == 0:
                return None
                
            # Filter points by range
            valid_indices = []
            for i, point in enumerate(lidar_points):
                distance = np.linalg.norm(point)
                if self.lidar_min_range < distance < self.lidar_max_range:
                    valid_indices.append(i)
                    
            filtered_points = lidar_points[valid_indices] if valid_indices else lidar_points
            
            # Create point cloud
            point_cloud = PointCloud(
                points=filtered_points,
                frame_id="lidar",
                timestamp=timestamp,
                height=1,
                width=filtered_points.shape[0],
                is_dense=True
            )
            
            # Downsample if needed
            if self.lidar_voxel_size > 0:
                point_cloud = point_cloud.voxel_downsample(self.lidar_voxel_size)
                
            # Save processed point cloud
            self.last_point_cloud = point_cloud
            
            return point_cloud
        except Exception as e:
            logger.error(f"Error processing LiDAR points: {e}")
            return None
    
    def convert_action(self, action: HybridAction):
        """
        Convert a Stretch AI action to a Segway robot action.
        
        Args:
            action: Stretch AI action
            
        Returns:
            Segway robot action
        """
        try:
            # Extract data from HybridAction based on action type
            if action.is_discrete():
                # Handle discrete action
                return self._convert_discrete_action(action.get())
            elif action.is_navigation():
                # Handle continuous navigation action
                return self._convert_navigation_action(action.get())
            elif action.is_manipulation():
                # Handle manipulation action (if applicable)
                return self._convert_manipulation_action(action.get())
            else:
                logger.warning(f"Unknown action type: {action.action_type}")
                return None
        except Exception as e:
            logger.error(f"Error converting action: {e}")
            return None
    
    def _convert_discrete_action(self, discrete_action):
        """
        Convert a discrete action to a Segway robot action.
        
        Args:
            discrete_action: Discrete action
            
        Returns:
            Segway robot action
        """
        # Placeholder - implement based on your robot's control interface
        return discrete_action
    
    def _convert_navigation_action(self, navigation_action):
        """
        Convert a navigation action to a Segway robot action.
        
        Args:
            navigation_action: Navigation action (xyt array)
            
        Returns:
            Segway robot action
        """
        # Placeholder - implement based on your robot's control interface
        return navigation_action
    
    def _convert_manipulation_action(self, manipulation_action):
        """
        Convert a manipulation action to a Segway robot action.
        
        Args:
            manipulation_action: Manipulation action
            
        Returns:
            Segway robot action
        """
        # If the Segway doesn't have manipulation capabilities,
        # this might be a no-op or log a warning
        logger.warning("Manipulation actions not supported on this Segway robot")
        return None
    
    def get_point_cloud(self) -> Optional[PointCloud]:
        """
        Get the last processed point cloud.
        
        Returns:
            PointCloud: Last processed point cloud
        """
        return self.last_point_cloud
    
    def get_observation(self) -> Optional[Observations]:
        """
        Get the last processed observation.
        
        Returns:
            Observations: Last processed observation
        """
        return self.last_observation
    
    def transform_point_cloud(self, point_cloud: PointCloud, transform_matrix: np.ndarray) -> Optional[PointCloud]:
        """
        Transform a point cloud using a transformation matrix.
        
        Args:
            point_cloud: Point cloud to transform
            transform_matrix: 4x4 transformation matrix
            
        Returns:
            PointCloud: Transformed point cloud
        """
        try:
            if point_cloud is None:
                return None
                
            return point_cloud.transform(transform_matrix)
        except Exception as e:
            logger.error(f"Error transforming point cloud: {e}")
            return None
    
    def merge_point_clouds(self, point_clouds: List[PointCloud]) -> Optional[PointCloud]:
        """
        Merge multiple point clouds into one.
        
        Args:
            point_clouds: List of point clouds to merge
            
        Returns:
            PointCloud: Merged point cloud
        """
        try:
            if not point_clouds:
                return None
                
            # Count total points
            total_points = sum(pc.points.shape[0] for pc in point_clouds if pc is not None)
            
            # Preallocate merged points array
            merged_points = np.zeros((total_points, 3))
            
            # Merge points
            offset = 0
            for pc in point_clouds:
                if pc is None or pc.points.shape[0] == 0:
                    continue
                    
                num_points = pc.points.shape[0]
                merged_points[offset:offset + num_points] = pc.points
                offset += num_points
                
            # Create merged point cloud
            return PointCloud(
                points=merged_points,
                frame_id=point_clouds[0].frame_id,
                timestamp=point_clouds[0].timestamp,
                height=1,
                width=total_points,
                is_dense=True
            )
        except Exception as e:
            logger.error(f"Error merging point clouds: {e}")
            return None
            
    def process_ros_messages(self, ros_messages: Dict[str, Any]) -> Optional[Observations]:
        """
        Process ROS messages into a unified Observations object.
        
        Args:
            ros_messages: Dictionary of ROS messages (key: message type, value: message)
            
        Returns:
            Observations: Processed observations
        """
        try:
            # Extract individual messages
            odom_msg = ros_messages.get('odom')
            lidar_msg = ros_messages.get('lidar')
            rgb_msg = ros_messages.get('rgb')
            depth_msg = ros_messages.get('depth')
            camera_info_msg = ros_messages.get('camera_info')
            imu_msg = ros_messages.get('imu')
            battery_msg = ros_messages.get('battery')
            
            # Use the from_ros_messages method in Observations
            observation = Observations.from_ros_messages(
                odom_msg=odom_msg,
                lidar_msg=lidar_msg,
                rgb_msg=rgb_msg,
                depth_msg=depth_msg,
                camera_info_msg=camera_info_msg,
                imu_msg=imu_msg,
                battery_msg=battery_msg
            )
            
            # Save processed observation
            self.last_observation = observation
            
            return observation
        except Exception as e:
            logger.error(f"Error processing ROS messages: {e}")
            return None
            
    def check_compatibility(self, stretch_observation: Observations) -> bool:
        """
        Check if an observation is compatible with the Segway robot.
        
        Args:
            stretch_observation: Observation to check
            
        Returns:
            bool: True if compatible, False otherwise
        """
        try:
            # Check if required fields are present
            required_fields = ['gps', 'compass']
            
            for field in required_fields:
                if not hasattr(stretch_observation, field) or getattr(stretch_observation, field) is None:
                    logger.warning(f"Observation missing required field: {field}")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Error checking compatibility: {e}")
            return False
            
    def adapt_for_segway(self, stretch_action: HybridAction):
        """
        Adapt a Stretch AI action for the Segway robot.
        Depending on the Segway's capabilities, some actions might need to be adjusted.
        
        Args:
            stretch_action: Stretch AI action
            
        Returns:
            Adapted action for Segway
        """
        try:
            # Convert action based on Segway's capabilities
            if stretch_action.is_manipulation() and not self.segway.has_manipulator():
                logger.warning("Manipulation action received but Segway has no manipulator")
                # Convert to a no-op or navigation action
                return self._convert_to_safe_action(stretch_action)
                
            # Otherwise, just convert normally
            return self.convert_action(stretch_action)
        except Exception as e:
            logger.error(f"Error adapting action for Segway: {e}")
            return None
            
    def _convert_to_safe_action(self, action: HybridAction):
        """
        Convert an unsupported action to a safe action.
        
        Args:
            action: Unsupported action
            
        Returns:
            Safe action
        """
        # Default to stopping the robot
        logger.info("Converting unsupported action to safe stop action")
        
        # Placeholder - implement based on your robot's control interface
        return None  # No-op