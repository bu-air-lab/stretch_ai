"""
Segway Robot Agent for Stretch AI

This module implements a RobotAgent for the Segway robot, integrating with
Stretch AI's perception, mapping, navigation, and motion control frameworks.
"""

import os
import time
import numpy as np
from threading import Thread, Event
from typing import Dict, List, Optional, Tuple, Any, Union

from stretch.core.robot import AbstractRobotClient
from stretch.core.parameters import Parameters
from stretch.agent.robot_agent import RobotAgent
from stretch.perception.wrapper import OvmmPerception
from stretch.mapping.voxel import SparseVoxelMap
from stretch.core.interfaces import Observations, HybridAction, PointCloud
from stretch.motion.base.space import SE2
from stretch.core.segway_ros_client import SegwayROSClient
from stretch.core.segway_stretch_integration import SegwayStretchIntegration
from stretch.perception.segway_perception_adapter import SegwayPerceptionAdapter
from stretch.mapping.segway_mapping_adapter import SegwayMappingAdapter
from stretch.navigation.segway_navigation_adapter import SegwayNavigationAdapter
from stretch.motion.segway_motion_adapter import SegwayMotionAdapter
from stretch.agent.segway_agent_adapter import SegwayAgentAdapter
from stretch.utils.logger import Logger

logger = Logger(__name__)


class SegwayRobotClient(AbstractRobotClient):
    """
    Adapter class that makes the Segway robot compatible with the 
    AbstractRobotClient interface required by Stretch AI.
    """
    
    def __init__(self, ros_client, config):
        """
        Initialize the Segway Robot Client.
        
        Args:
            ros_client: SegwayROSClient instance
            config: Configuration dictionary
        """
        self.ros_client = ros_client
        self.config = config
        self.running = True
        self._rerun = None
        self._last_motion_failed = False
        self._connection_retry_count = 0
        self._max_connection_retries = config.get("connection_retries", 5)
        self._connection_retry_interval = config.get("connection_retry_interval", 2.0)
        
        # Store IP addresses for easier debugging
        self.robot_ip = config.get("robot_ip", "10.66.171.191")
        self.desktop_ip = config.get("desktop_ip", "10.66.171.131")
        self.lidar_ip = config.get("lidar_ip", "10.66.171.8")
        
        # Initialize connection to ROS client
        self._initialize_connection()
        
        # Motion adapter for more advanced control
        self.motion_adapter = None
        
        # Navigation/manipulation mode tracking
        self._in_manipulation_mode = False

    def start(self):
        """Start the robot client and establish connections."""
        try:
            logger.info("Starting SegwayRobotClient")
            
            # Check if we're connected to ROS
            if not self.ros_client.is_connected():
                logger.warning("ROS client not connected, attempting connection...")
                success = self._initialize_connection()
                if not success:
                    logger.error("Failed to initialize connection to ROS")
                    return False
            
            logger.info("SegwayRobotClient started successfully")
            return True
        except Exception as e:
            logger.error(f"Error starting robot client: {e}")
            return False

    def _initialize_connection(self):
        """Initialize connection to ROS client with retry logic."""
        while self._connection_retry_count < self._max_connection_retries:
            try:
                success = self.ros_client.connect()
                if success:
                    logger.info(f"Successfully connected to Segway robot at {self.robot_ip}")
                    return True
                else:
                    self._connection_retry_count += 1
                    logger.warning(f"Failed to connect to Segway robot. Retry {self._connection_retry_count}/{self._max_connection_retries}")
                    time.sleep(self._connection_retry_interval)
            except Exception as e:
                self._connection_retry_count += 1
                logger.error(f"Error connecting to Segway robot: {e}. Retry {self._connection_retry_count}/{self._max_connection_retries}")
                time.sleep(self._connection_retry_interval)
                
        logger.error(f"Failed to connect to Segway robot after {self._max_connection_retries} attempts")
        return False
        
    def set_motion_adapter(self, motion_adapter):
        """Set motion adapter for advanced control."""
        self.motion_adapter = motion_adapter
        
    def get_robot_model(self):
        """Return the robot model."""
        return self.ros_client.get_robot_model()
        
    def get_base_pose(self):
        """Get the current base pose of the robot."""
        return self.ros_client.get_pose()

    def move_base_to(self, pose, relative=False, blocking=True, verbose=False, timeout=30.0):
        """
        Move the robot base to the specified pose.
        
        Args:
            pose: Target pose [x, y, theta]
            relative: Whether pose is relative to current position
            blocking: Whether to wait for motion completion
            verbose: Whether to print verbose details
            timeout: Timeout for blocking motion
            
        Returns:
            bool: Success status
        """
        try:
            if verbose:
                logger.info(f"Moving base to pose {pose}, relative={relative}, blocking={blocking}")
            
            # Use the ros_client to execute the movement
            result = self.ros_client.move_to(pose, relative=relative, blocking=blocking, 
                                            verbose=verbose, timeout=timeout)
            
            # Track success/failure
            self._last_motion_failed = not result
            return result
            
        except Exception as e:
            logger.error(f"Error in move_base_to: {e}")
            import traceback
            traceback.print_exc()
            self._last_motion_failed = True
            return False
            
    def move_to(self, pose, relative=False, blocking=True, verbose=False, timeout=30.0):
        """Move the robot to a specified pose."""
        try:
            # Convert pose to Twist message
            twist = Twist()
            
            if relative:
                # Handle relative motion
                if len(pose) >= 1:
                    twist.linear.x = pose[0]
                if len(pose) >= 2:
                    twist.linear.y = pose[1]
                if len(pose) >= 3:
                    twist.angular.z = pose[2]
            else:
                # For absolute motion, need to calculate difference from current pose
                current_pose = self.get_pose()
                if current_pose is None:
                    return False
                    
                # Calculate direction vector
                if len(pose) >= 2:
                    dx = pose[0] - current_pose[0]
                    dy = pose[1] - current_pose[1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Set forward velocity proportional to distance
                    twist.linear.x = min(0.3, distance)  # Cap at 0.3 m/s
                    
                # Calculate angular difference
                if len(pose) >= 3:
                    target_theta = pose[2]
                    current_theta = current_pose[2]
                    
                    # Calculate shortest angle difference
                    angle_diff = np.arctan2(np.sin(target_theta - current_theta), 
                                        np.cos(target_theta - current_theta))
                    
                    # Set angular velocity proportional to angle difference
                    twist.angular.z = min(0.5, max(-0.5, angle_diff))  # Cap at ±0.5 rad/s
            
            # Publish the twist message
            self.cmd_vel_pub.publish(twist)
            
            # If blocking, wait until the motion is complete
            if blocking:
                start_time = time.time()
                rate = rospy.Rate(10)  # 10 Hz
                
                while time.time() - start_time < timeout:
                    # Check if we've reached the goal
                    if self.at_goal(pose):
                        return True
                        
                    rate.sleep()
                    
                # If we get here, we've timed out
                return False
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Error in move_to: {e}")
            return False
                
    def get_observation(self):
        """Get the current observation from the robot."""
        try:
            obs = self.ros_client.get_observation()
            
            # Additional processing for SICK LiDAR data if needed
            if obs is not None and hasattr(obs, 'lidar_points') and obs.lidar_points is not None:
                # Process LiDAR data if needed
                # For example, transform to the right coordinate frame
                if obs.point_cloud is None and hasattr(obs, 'process_lidar_to_pointcloud'):
                    obs.process_lidar_to_pointcloud()
                    
            return obs
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            return None
        
    def execute_trajectory(self, trajectory, pos_err_threshold=0.1, rot_err_threshold=0.1):
        """Execute a trajectory (list of poses)."""
        try:
            # Use motion adapter if available for smooth trajectory following
            if self.motion_adapter is not None:
                # Format trajectory correctly for motion adapter
                formatted_trajectory = []
                for pose in trajectory:
                    if isinstance(pose, list) or isinstance(pose, tuple) or isinstance(pose, np.ndarray):
                        formatted_trajectory.append(pose)
                    elif hasattr(pose, 'state'):
                        # Handle Stretch AI planner output
                        formatted_trajectory.append(pose.state)
                    else:
                        logger.warning(f"Unknown trajectory point format: {type(pose)}")
                        return False
                
                # Follow trajectory
                result = self.motion_adapter.follow_trajectory(formatted_trajectory)
                
                # Wait for trajectory completion
                if result:
                    while not self.motion_adapter.is_motion_complete():
                        time.sleep(0.1)
                
                self._last_motion_failed = not result
                return result
            else:
                # Fall back to step-by-step execution
                success = True
                for pose in trajectory:
                    if hasattr(pose, 'state'):
                        # Handle Stretch AI planner output
                        pose = pose.state
                    
                    result = self.move_base_to(pose, relative=False, blocking=True)
                    if not result:
                        success = False
                        self._last_motion_failed = True
                        break
                
                return success
        except Exception as e:
            logger.error(f"Error executing trajectory: {e}")
            self._last_motion_failed = True
            return False
        
    def last_motion_failed(self):
        """Check if the last motion command failed."""
        return self._last_motion_failed
        
    def stop(self):
        """Stop the robot client."""
        self.running = False
        try:
            self.ros_client.stop()
        except Exception as e:
            logger.error(f"Error stopping ROS client: {e}")
        
    def move_to_nav_posture(self):
        """Move the robot to navigation posture."""
        # For Segway, this might just ensure we're in the right mode
        return self.switch_to_navigation_mode()
      
    def move_to_manip_posture(self):
        """Move the robot to manipulation posture."""
        # For Segway, this might position the robot for manipulation
        return self.switch_to_manipulation_mode()
        
    def switch_to_navigation_mode(self):
        """Switch the robot to navigation mode."""
        try:
            # Don't call a method that doesn't exist on the ROS client
            # Instead, implement the functionality here
            logger.info("Switching to navigation mode for Segway")
            self._in_manipulation_mode = False
            
            # Configure any navigation-specific settings here
            # For example, set motion parameters, adjust safety settings, etc.
            
            return True
        except Exception as e:
            logger.error(f"Error switching to navigation mode: {e}")
            return False
        
    def switch_to_manipulation_mode(self):
        """Switch the robot to manipulation mode."""
        try:
            # Don't call a method that doesn't exist on the ROS client
            # Instead, implement the functionality here
            logger.info("Switching to manipulation mode for Segway")
            self._in_manipulation_mode = True
            
            # Configure any manipulation-specific settings here
            # For example, reduce motion speed, adjust safety settings, etc.
            
            return True
        except Exception as e:
            logger.error(f"Error switching to manipulation mode: {e}")
            return False
        
    def in_manipulation_mode(self):
        """Check if the robot is in manipulation mode."""
        # Return the internal state tracker instead of calling a non-existent method
        return self._in_manipulation_mode
        
    def open_gripper(self):
        """Open the gripper if the robot has one."""
        # May not be applicable for Segway without a gripper
        logger.info("Open gripper command (stub implementation for Segway)")
        return True
        
    def at_goal(self):
        """Check if the robot has reached its goal."""
        # Use motion adapter if available
        if self.motion_adapter is not None:
            return self.motion_adapter.is_motion_complete()
        
        # Otherwise use simple check
        return not self._last_motion_failed
        
    def get_pose_graph(self):
        """Get the pose graph from the robot."""
        try:
            return self.ros_client.get_pose_graph() if hasattr(self.ros_client, 'get_pose_graph') else []
        except Exception as e:
            logger.error(f"Error getting pose graph: {e}")
            return []
        
    def head_to(self, pan, tilt, blocking=True):
        """Move the robot's head (if it has one)."""
        # May not be applicable for Segway without a head
        logger.debug(f"Head_to called with pan={pan}, tilt={tilt} - not applicable for Segway")
        return True
        
    def say(self, text):
        """Have the robot say something (if it has audio capabilities)."""
        logger.info(f"Robot says: {text}")
        return True
        
    def say_sync(self, text):
        """Have the robot say something and wait until it's done."""
        return self.say(text)
    
    # Add the missing abstract methods
    
    def load_map(self, map_path):
        """
        Load a map from a file.
        
        Args:
            map_path: Path to the map file
            
        Returns:
            bool: Success status
        """
        logger.info(f"Loading map from {map_path}")
        try:
            # Try to use the ROS client if it has a load_map method
            if hasattr(self.ros_client, 'load_map'):
                return self.ros_client.load_map(map_path)
            
            # Implementation can be expanded based on Segway capabilities
            # For now, just return success
            return True
        except Exception as e:
            logger.error(f"Error loading map from {map_path}: {e}")
            return False
    
    def save_map(self, map_path):
        """
        Save the current map to a file.
        
        Args:
            map_path: Path to save the map
            
        Returns:
            bool: Success status
        """
        logger.info(f"Saving map to {map_path}")
        try:
            # Try to use the ROS client if it has a save_map method
            if hasattr(self.ros_client, 'save_map'):
                return self.ros_client.save_map(map_path)
            
            # Implementation can be expanded based on Segway capabilities
            # For now, just return success
            return True
        except Exception as e:
            logger.error(f"Error saving map to {map_path}: {e}")
            return False
    
    def reset(self):
        """
        Reset the robot client state.
        
        Returns:
            bool: Success status
        """
        logger.info("Resetting robot client")
        try:
            # Stop any ongoing movement
            if self.motion_adapter is not None:
                self.motion_adapter.stop_motion()
            
            # Reset flags
            self._last_motion_failed = False
            self._in_manipulation_mode = False
            
            # Try to reset the ROS client if it has a reset method
            if hasattr(self.ros_client, 'reset'):
                self.ros_client.reset()
            
            return True
        except Exception as e:
            logger.error(f"Error resetting robot client: {e}")
            return False


class SegwayRobotAgent(RobotAgent):
    """
    RobotAgent implementation for the Segway robot, integrating with
    Stretch AI's perception, mapping, navigation, and motion control frameworks.
    Optimized for ROS 1 to ROS 2 bridge with RTX 4090 GPU acceleration.
    """
    
    def __init__(
        self,
        config_path=None,
        semantic_sensor=None,
        voxel_map=None,
        show_instances_detected=True,
        use_instance_memory=True,
        enable_realtime_updates=True,
        create_semantic_sensor=True,
        obs_sub_port=4450,
        gpu_device="cuda:0",
        debug_mode=False,
        continuous_visualization=False,
        use_scene_graph=True
    ):
        """
        Initialize the Segway Robot Agent.
        
        Args:
            config_path: Path to the configuration file
            semantic_sensor: Optional OvmmPerception instance
            voxel_map: Optional SparseVoxelMap instance
            show_instances_detected: Whether to show detected instances
            use_instance_memory: Whether to use instance memory
            enable_realtime_updates: Whether to enable realtime updates
            create_semantic_sensor: Whether to create a semantic sensor
            obs_sub_port: Port for observation subscription
            gpu_device: GPU device to use for perception and mapping
            debug_mode: Whether to enable additional debugging output
        """

        # Set basic instance variables
        self.continuous_visualization = continuous_visualization
        self.visualization_thread = None
        self.visualization_running = False
        self.debug_mode = debug_mode
        self.gpu_device = gpu_device
        self.use_scene_graph = use_scene_graph
        # load configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config',
                'segway_config.yaml'
            )
            
        logger.info(f"Initializing SegwayRobotAgent with config: {config_path}")
        self.parameters = self._load_config(config_path)
        
        # Get network configuration
        self.robot_ip = self.parameters.get("network.robot_ip", "10.66.171.191")
        self.desktop_ip = self.parameters.get("network.desktop_ip", "10.66.171.131")
        self.lidar_ip = self.parameters.get("network.lidar_ip", "10.66.171.8")
        
        # Initialize ROS client
        self.ros_client = SegwayROSClient(
            config_path=config_path,
            robot_ip=self.robot_ip,
            desktop_ip=self.desktop_ip
        )
        
        # Initialize scene graph
        self.scene_graph = None
        if self.use_scene_graph:
            try:
                from stretch.mapping.scene_graph import SceneGraph
                # Create scene graph with empty instances list initially
                self.scene_graph = SceneGraph(
                    parameters=self.parameters,
                    instances=[]  # Start with empty instances list
                )
                logger.info("Scene graph initialized")
            except Exception as e:
                logger.error(f"Error initializing scene graph: {e}")
                self.use_scene_graph = False
    
        # Initialize robot client and adapters
        self.robot_client = SegwayRobotClient(self.ros_client, self.parameters)
        self.segway = SegwayStretchIntegration(self.ros_client, self.parameters)
        self.segway_adapter = SegwayAgentAdapter(self.segway, self.parameters)
        self.perception_adapter = SegwayPerceptionAdapter(self.parameters, gpu_device=self.gpu_device)
        self.mapping_adapter = SegwayMappingAdapter(self.parameters, self.ros_client,self, gpu_device=self.gpu_device)
        self.navigation_adapter = SegwayNavigationAdapter(self.parameters, self.ros_client, mapping_adapter=self.mapping_adapter)
        self.motion_adapter = SegwayMotionAdapter(self.parameters, self.ros_client)
        
        # Set motion adapter in robot client
        self.robot_client.set_motion_adapter(self.motion_adapter)
        
        #  ONLY NOW set up encoder and pass it to adapters
        self.encoder = self._setup_encoder()
        if hasattr(self, 'mapping_adapter') and self.mapping_adapter is not None and self.encoder is not None:
            try:
                if hasattr(self.mapping_adapter, 'set_encoder'):
                    self.mapping_adapter.set_encoder(self.encoder)
                    logger.info("Encoder set via set_encoder method")
                elif hasattr(self.mapping_adapter, 'instance_map') and self.mapping_adapter.instance_map is not None:
                    self.mapping_adapter.instance_map.encoder = self.encoder
                    logger.info("Encoder set directly on instance_map")
            except Exception as e:
                logger.error(f"Warning: Could not set encoder: {e}")
        
        # Override semantic sensor creation if needed
        if create_semantic_sensor and semantic_sensor is None:
            logger.info("Creating semantic sensor")
            semantic_sensor = self._create_semantic_sensor()
        
        if self.encoder is not None:
            logger.info("Using encoder for instance memory")
            voxel_map = self._create_voxel_map(self.parameters)
        else:
            logger.warning("No encoder available - creating voxel map without instance memory")
            # Create voxel map without instance memory
            voxel_map = SparseVoxelMap(
                resolution=parameters.get("mapping.voxel_size", 0.05),
                device=self.gpu_device,
                use_instance_memory=False
            )
        
        # Call parent constructor with our robot client
        try:
            logger.info("Initializing parent RobotAgent class")
            super().__init__(
                robot=self.robot_client,
                parameters=self.parameters,
                semantic_sensor=semantic_sensor,
                voxel_map=voxel_map,
                show_instances_detected=show_instances_detected,
                use_instance_memory=use_instance_memory,
                enable_realtime_updates=enable_realtime_updates,
                create_semantic_sensor=False,  # We handle this ourselves
                obs_sub_port=obs_sub_port,
            )
            logger.info("Parent RobotAgent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing parent RobotAgent: {e}")
            raise
        
        # Set encoder explicitly if needed
        self.encoder_name = self.parameters.get("encoder", "siglip")
        
        # Setup update thread
        self.update_thread_stop = Event()
        self.update_thread = Thread(target=self._update_loop)
        self.update_thread.daemon = True
        
        # Setup monitoring thread
        self.monitor_thread_stop = Event()
        self.monitor_thread = Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        
        # Connection status
        self.connection_status = {
            "robot": False,
            "lidar": False,
            "camera": False,
            "bridge": False
        }
        
        # Start the adapters
        self._start_adapters()
        
        # Start components if not using parent's threading
        if not enable_realtime_updates:
            logger.info("Starting update thread")
            self.update_thread.start()
            
        # Always start monitoring thread
        logger.info("Starting monitoring thread")
        self.monitor_thread.start()
        
        logger.info("SegwayRobotAgent initialization complete")
    
    def _start_adapters(self):
        """Start all adapter components."""
        try:
            # Start perception adapter
            if self.perception_adapter is not None:
                logger.info("Starting perception adapter")
                self.perception_adapter.start()
            
            # Start mapping adapter
            if self.mapping_adapter is not None:
                logger.info("Starting mapping adapter")
                self.mapping_adapter.start()
            
            # Start navigation adapter
            if self.navigation_adapter is not None:
                logger.info("Starting navigation adapter")
                self.navigation_adapter.start()
            
            # Start motion adapter
            if self.motion_adapter is not None:
                logger.info("Starting motion adapter")
                self.motion_adapter.start()
        except Exception as e:
            logger.error(f"Error starting adapters: {e}")
    
    def start(self, visualize_map_at_start=False):
            """Start the agent with enhanced real-time mapping."""
            result = super().start(visualize_map_at_start)
            
            # Start continuous visualization if enabled
            if self.continuous_visualization:
                vis_rate = self.parameters.get("visualization.update_rate", 1.0)
                self.start_visualization(update_rate=vis_rate)
            
            # Start mapping adapter if not already started
            if self.mapping_adapter is not None and hasattr(self.mapping_adapter, 'running') and not self.mapping_adapter.running:
                self.mapping_adapter.start()
            
            # Start perception adapter if not already started
            if self.perception_adapter is not None and hasattr(self.perception_adapter, 'running') and not self.perception_adapter.running:
                self.perception_adapter.start()
            
            # Configure instance tracking parameters for real-time operation
            if self.mapping_adapter is not None and hasattr(self.mapping_adapter, 'instance_map'):
                # Set more responsive thresholds for live tracking
                if hasattr(self.mapping_adapter.instance_map, 'view_matching_config'):
                    self.mapping_adapter.instance_map.view_matching_config.min_similarity_thresh = 0.3
                    self.mapping_adapter.instance_map.view_matching_config.visual_similarity_weight = 0.7
                    self.mapping_adapter.instance_map.view_matching_config.box_overlap_weight = 0.3
            
            logger.info("SegwayRobotAgent started with real-time mapping enabled")
            return result

    def _create_semantic_sensor(self):
        """Create a semantic sensor optimized for the RTX 4090 GPU."""
        try:
            logger.info(f"Creating semantic sensor on GPU device: {self.gpu_device}")
            
            # Get the device ID from the GPU device string
            device_id = 0  # Default to 0
            if self.gpu_device and ":" in self.gpu_device:
                try:
                    device_id = int(self.gpu_device.split(":")[-1])
                except ValueError:
                    logger.warning(f"Could not parse device ID from {self.gpu_device}, using default 0")
            
            # Create a minimal set of parameters if needed
            if not hasattr(self, 'parameters') or self.parameters is None:
                self.parameters = Parameters()
                
            # Set any required detection parameters
            self.parameters.set("detection/module", "detic")
            self.parameters.set("detection/confidence_threshold", 0.3)
            self.parameters.set("detection/use_detic_viz", False)
            
            # Get category map file from parameters or set default
            category_map_file = self.parameters.get("detection/category_map_file", "example_cat_map.json")
            
            # Create the OvmmPerception object with the correct parameters
            return OvmmPerception(
                parameters=self.parameters,
                gpu_device_id=device_id,
                verbose=self.debug_mode,
                module_kwargs={},
                category_map_file=category_map_file
            )
        except Exception as e:
            logger.error(f"Error creating semantic sensor: {e}")
            return None
    
    def _create_voxel_map(self, parameters):
        """Create a voxel map optimized for the RTX 4090 GPU."""
        try:
            logger.info(f"Creating voxel map on GPU device: {self.gpu_device}")
            
            # Ensure use_instance_memory is True
            return SparseVoxelMap(
                resolution=parameters.get("mapping.voxel_size", 0.05),
                device=self.gpu_device,
                use_instance_memory=True,  # <-- Set this to True
                encoder=self.encoder  # Make sure encoder is passed
            )
        except Exception as e:
            logger.error(f"Error creating voxel map: {e}")
            return None
    
    def _load_config(self, config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            # Create empty Parameters object with kwargs instead of positional arguments
            return Parameters()
        
        # Use Stretch AI's parameter loading mechanism
        try:
            return Parameters.from_yaml(config_path)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            # Create empty Parameters object with kwargs instead of positional arguments
            return Parameters()

    def start_visualization(self, update_rate=1.0):
        """Start continuous visualization of mapping and objects."""
        if self.visualization_thread is not None:
            logger.warning("Visualization already running")
            return
        
        logger.info(f"Starting visualization thread with update rate {update_rate}Hz")
        self.visualization_running = True
        
        def _visualization_loop():
            while self.visualization_running and self.is_running():
                try:
                    voxel_map = self.get_voxel_map()
                    if voxel_map is not None:
                        # Show map with robot position, instances, and relationships
                        voxel_map.show(
                            orig=np.zeros(3),
                            xyt=self.robot.get_base_pose(),
                            footprint=self.robot.get_robot_model().get_footprint(),
                            instances=self.semantic_sensor is not None
                        )
                except Exception as e:
                    logger.error(f"Error in visualization loop: {e}")
                
                time.sleep(1.0 / update_rate)
        
        self.visualization_thread = threading.Thread(target=_visualization_loop)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        logger.info("Visualization thread started")

    def stop_visualization(self):
        """Stop the visualization thread."""
        if not self.visualization_running:
            logger.warning("Visualization not running")
            return
        
        logger.info("Stopping visualization thread")
        self.visualization_running = False
        
        if self.visualization_thread:
            self.visualization_thread.join(timeout=2.0)
            self.visualization_thread = None
            
    def _setup_encoder(self):
        """Initialize visual encoder for object recognition."""
        try:
            from stretch.perception.encoders.clip_encoder import ClipEncoder
            
            # Get encoder type from parameters
            encoder_name = self.parameters.get("encoder", "clip")
            logger.info(f"Setting up {encoder_name} encoder on {self.gpu_device}")
            
            # Create encoder
            encoder = ClipEncoder(version="ViT-L/14@336px", device=self.gpu_device)
            
            logger.info("Visual encoder initialized successfully")
            return encoder
        except Exception as e:
            logger.error(f"Error setting up encoder: {e}")
            return None

    def update_map_with_pose_graph(self, verbose=False):
        """Override parent method to handle scene graph properly in uncertain environment."""
        try:
            # Get observation nodes from pose graph
            self._obs_history_lock.acquire()
            matched_obs = []

            for obs in self.obs_history:
                if obs.is_pose_graph_node:
                    matched_obs.append(obs)

            self._obs_history_lock.release()

            # Update voxel map with available observations
            with self._voxel_map_lock:
                self.voxel_map.reset()
                added = 0
                for obs in matched_obs:
                    if obs.is_pose_graph_node:
                        self.voxel_map.add_obs(obs)
                        added += 1

            # Update visualization if observations exist
            if len(self.get_voxel_map().observations) > 0:
                self.update_rerun()

            # Scene graph handling - ONLY process if it exists
            # Don't try to initialize it here - that should happen elsewhere
            if self.use_scene_graph and self.scene_graph is not None:
                try:
                    self._update_scene_graph()
                    self.scene_graph.get_relationships()
                except Exception as e:
                    logger.error(f"Error updating scene graph relationships: {e}")
                    
            # Even if scene graph is None, it's fine - we'll build it during exploration
            if verbose:
                logger.info(f"Updated map with {added} observations" + 
                            (", scene graph updated" if (self.use_scene_graph and self.scene_graph is not None) else ""))
                
        except Exception as e:
            logger.error(f"Error in update_map_with_pose_graph: {e}")

    def toggle_live_mapping(self, enable=True, visualize=False):
        """Toggle live mapping on or off."""
        if enable:
            logger.info("Enabling live mapping")
            
            # Ensure adapters are started
            if self.mapping_adapter is not None and hasattr(self.mapping_adapter, 'start'):
                self.mapping_adapter.start()
            
            if self.perception_adapter is not None and hasattr(self.perception_adapter, 'start'):
                self.perception_adapter.start()
            
            # Ensure update thread is running
            if self._realtime_updates or (hasattr(self, 'update_thread') and self.update_thread.is_alive()):
                logger.info("Update thread already running")
            else:
                self.update_thread_stop.clear()
                self.update_thread = threading.Thread(target=self._update_loop)
                self.update_thread.daemon = True
                self.update_thread.start()
                logger.info("Started update thread for live mapping")
            
            # Start visualization if requested
            if visualize and not self.visualization_running:
                self.start_visualization()
        else:
            logger.info("Disabling live mapping")
            
            # Stop update thread
            if not self._realtime_updates:
                self.update_thread_stop.set()
            
            # Stop visualization
            if self.visualization_running:
                self.stop_visualization()
        
        return True

    def _update_loop(self):
        """Background thread for continuous map updates."""
        logger.info("Update loop started")
        
        update_rate = self.parameters.get("mapping.update_rate", 10.0)  # 10Hz default
        update_interval = 1.0 / update_rate
        
        while not self.update_thread_stop.is_set() and self.is_running():
            start_time = time.time()
            
            try:
                # Get the latest observation
                obs = self.robot.get_observation()
                if obs is not None:
                    # Process through perception first if available
                    if self.perception_adapter is not None:
                        try:
                            obs = self.perception_adapter.update_perception(obs)
                        except Exception as e:
                            logger.error(f"Error in perception processing: {e}")
                    
                    # Update mapping with full observation
                    if self.mapping_adapter is not None:
                        try:
                            # Add camera calibration data to observation
                            if not hasattr(obs, 'camera_K') or obs.camera_K is None:
                                if hasattr(self.ros_client, 'get_camera_intrinsics'):
                                    obs.camera_K = self.ros_client.get_camera_intrinsics()
                            
                            # Add camera pose to observation
                            if not hasattr(obs, 'camera_pose') or obs.camera_pose is None:
                                obs.camera_pose = self.mapping_adapter.get_current_pose_matrix()
                            
                            # Update map with observation
                            self.mapping_adapter.update_map_from_observation(obs)
                        except Exception as e:
                            logger.error(f"Error updating mapping: {e}")
                    
                    # Update scene graph with latest instance data
                    self._update_scene_graph()
                    
                # Calculate remaining time to maintain update rate
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(0.1)  # Sleep briefly on error

    def _monitor_loop(self):
        """Background thread for monitoring system health."""
        logger.info("Monitor loop started")
        # Define a recency threshold (e.g., 5 seconds)
        recency_threshold = self.parameters.get("monitor_recency_threshold", 5.0)
        while not self.monitor_thread_stop.is_set() and self.is_running():
            try:
                current_time = time.time()

                # Check robot connection
                self.connection_status["robot"] = self.ros_client.is_connected()

                # Check LiDAR connection (based on timestamp recency)
                lidar_ts = self.ros_client.get_latest_lidar_data_timestamp()
                self.connection_status["lidar"] = lidar_ts is not None and (current_time - lidar_ts <= recency_threshold)

                # Check camera connection (based on timestamp recency)
                camera_ts = self.ros_client.get_latest_camera_data_timestamp()
                self.connection_status["camera"] = camera_ts is not None and (current_time - camera_ts <= recency_threshold)

                # Check bridge connection
                bridge_status = self.ros_client.check_bridge_status() if hasattr(self.ros_client, 'check_bridge_status') else False
                self.connection_status["bridge"] = bridge_status

                if self.debug_mode:
                    logger.debug(f"Connection status: {self.connection_status} (Lidar TS: {lidar_ts}, Cam TS: {camera_ts}, Now: {current_time})")

                # Attempt to recover any failed connections
                if not all(self.connection_status.values()):
                    missing = [k for k, v in self.connection_status.items() if not v]
                    logger.warning(f"Missing connections: {missing}")

                    # Only try recovery if robot connection is the issue or bridge is down
                    if "robot" in missing:
                        logger.info("Attempting to reconnect ROS client...")
                        self.ros_client.reconnect()

                    if "bridge" in missing and self.ros_client.use_bridge:
                        logger.info("Attempting to restart ROS bridge...")
                        self.ros_client.restart_bridge() if hasattr(self.ros_client, 'restart_bridge') else None
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

            # Sleep before next check
            time.sleep(self.parameters.get("monitor_interval", 5.0))
    
    def shutdown(self):
        """Shutdown the agent with improved cleanup."""
        # Stop visualization first
        if hasattr(self, 'visualization_running') and self.visualization_running:
            self.stop_visualization()
        
        # Rest of existing shutdown code
        logger.info("Shutting down SegwayRobotAgent")
        self.update_thread_stop.set()
        self.monitor_thread_stop.set()
        
        # Make sure we stop the ROS bridge cleanly first to avoid restart
        if hasattr(self, 'ros_client') and self.ros_client is not None:
            logger.info("Shutting down ROS client and bridge")
            if hasattr(self.ros_client, 'stop_bridge'):
                self.ros_client.stop_bridge()
            time.sleep(0.5)  # Give bridge time to shut down
        
        # Stop motion first to ensure robot stops moving
        if hasattr(self, 'motion_adapter') and self.motion_adapter is not None:
            logger.info("Stopping motion adapter")
            self.motion_adapter.stop()
        
        # Stop remaining components
        if hasattr(self, 'navigation_adapter') and self.navigation_adapter is not None:
            logger.info("Stopping navigation adapter")
            self.navigation_adapter.stop()
        
        if hasattr(self, 'perception_adapter') and self.perception_adapter is not None:
            logger.info("Stopping perception adapter")
            self.perception_adapter.stop()
                
        if hasattr(self, 'mapping_adapter') and self.mapping_adapter is not None:
            logger.info("Stopping mapping adapter")
            self.mapping_adapter.stop()
        
        # Call parent shutdown
        super().shutdown()
        logger.info("SegwayRobotAgent shutdown complete")
    
    #========================#
    # Extended/Overridden Operations #
    #========================#
    
    def rotate_in_place(
        self,
        steps=None,
        visualize=False,
        verbose=False,
        full_sweep=True,
        audio_feedback=False
    ):
        """
        Override to customize for Segway robot. 
        Rotates the robot in place, taking observations of the surroundings.
        """
        if steps is None or steps <= 0:
            steps = self.parameters.get("agent.in_place_rotation_steps", 8)
            
        logger.info(f"Rotating in place with {steps} steps")
        if audio_feedback:
            self.robot.say("Rotating in place")
            
        # Get current pose
        x, y, theta = self.robot.get_base_pose()
        
        # Use motion adapter for smoother rotation if available
        if self.motion_adapter is not None:
            step_size = 2 * np.pi / steps
            
            for i in range(steps):
                logger.info(f"Rotation step {i+1}/{steps}")
                # Calculate absolute target orientation
                target_theta = theta + (i+1) * step_size
                
                # Normalize to [-pi, pi]
                while target_theta > np.pi:
                    target_theta -= 2 * np.pi
                while target_theta < -np.pi:
                    target_theta += 2 * np.pi
                
                # Create target pose (same position, new orientation)
                target_pose = [x, y, target_theta]
                
                # Move to the pose
                self.motion_adapter.move_to_pose(target_pose)
                
                # Wait for motion to complete
                while not self.motion_adapter.is_motion_complete():
                    time.sleep(0.1)
                
                # Pause briefly to allow sensors to capture data
                time.sleep(self.parameters.get("rotation_pause_time", 0.5))
                
                if not self._realtime_updates:
                    self.update()
                    
                if visualize:
                    self.get_voxel_map().show(
                        orig=np.zeros(3),
                        xyt=self.robot.get_base_pose(),
                        footprint=self.robot.get_robot_model().get_footprint(),
                        instances=self.semantic_sensor is not None
                    )
        else:
            # Fall back to basic rotation
            step_size = 2 * np.pi / steps
            
            for i in range(steps):
                target_theta = theta + (i * step_size)
                logger.info(f"Rotating to theta: {target_theta:.2f} ({i+1}/{steps})")
                self.robot.move_base_to([x, y, target_theta], relative=False, blocking=True)
                
                # Pause briefly to allow sensors to capture data
                time.sleep(self.parameters.get("rotation_pause_time", 0.5))
                
                if not self._realtime_updates:
                    self.update()
                    
                if visualize:
                    self.get_voxel_map().show(
                        orig=np.zeros(3),
                        xyt=self.robot.get_base_pose(),
                        footprint=self.robot.get_robot_model().get_footprint(),
                        instances=self.semantic_sensor is not None
                    )
                
        return True
    
    def _update_scene_graph(self):
        """Update scene graph with latest instance data."""
        if not self.use_scene_graph or self.scene_graph is None:
            return
            
        try:
            # Get instances from mapping adapter
            instances = []
            if self.mapping_adapter is not None and hasattr(self.mapping_adapter, 'get_detected_objects'):
                instances = self.mapping_adapter.get_detected_objects()
            
            # Update scene graph with instances
            self.scene_graph.update(instances)
        except Exception as e:
            logger.error(f"Error updating scene graph: {e}")

    def update(self, visualize_map=False, debug_instances=False, move_head=None, tilt=-1 * np.pi / 4):
        """
        Update the agent's internal map and perception.
        Customized for Segway which doesn't have a movable head.
        """
        if self._realtime_updates:
            return
            
        # For Segway, we ignore the head movement parameters since it doesn't have a head
        logger.info("Updating agent's internal map and perception")
        
        obs = None
        attempts = 0
        max_attempts = 5
        
        while obs is None and attempts < max_attempts:
            obs = self.robot.get_observation()
            if obs is None:
                attempts += 1
                logger.warning(f"Failed to get observation. Attempt {attempts}/{max_attempts}")
                time.sleep(0.2)
                
        if obs is None:
            logger.error("Failed to get observation after maximum attempts")
            return
                
        # Add to observation history
        self.obs_history.append(obs)
        self.obs_count += 1
            
        # Optionally do semantic prediction
        if self.semantic_sensor is not None:
            try:
                logger.info("Running semantic prediction")
                obs = self.semantic_sensor.predict(obs)
            except Exception as e:
                logger.error(f"Error in semantic prediction: {e}")
                
        # Add to voxel map
        try:
            logger.info("Updating voxel map")
            self.get_voxel_map().add_obs(obs)
        except Exception as e:
            logger.error(f"Error updating voxel map: {e}")
            
        # Update scene graph if used
        if self.use_scene_graph:
            try:
                logger.info("Updating scene graph")
                self._update_scene_graph()
                self.scene_graph.get_relationships()
            except Exception as e:
                logger.error(f"Error updating scene graph: {e}")
                
        # Update rerun if needed
        if self.robot._rerun and self.update_rerun_every_time:
            try:
                logger.info("Updating rerun")
                self.update_rerun()
            except Exception as e:
                logger.error(f"Error updating rerun: {e}")
                
        # Visualize map if requested
        if visualize_map:
            try:
                logger.info("Visualizing map")
                self.get_voxel_map().show(
                    orig=np.zeros(3),
                    xyt=self.robot.get_base_pose(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                    instances=self.semantic_sensor is not None and debug_instances
                )
            except Exception as e:
                logger.error(f"Error visualizing map: {e}")
    
    def detect_objects(self):
        """
        Detect objects in the current scene using the perception adapter.
        
        Returns:
            list: List of detected objects
        """
        try:
            # Use perception adapter if available
            if self.perception_adapter is not None:
                logger.info("Detecting objects with perception adapter")
                return self.perception_adapter.detect_objects()
            
            # Fall back to semantic sensor
            if self.semantic_sensor is not None:
                logger.info("Detecting objects with semantic sensor")
                # Get current observation
                obs = self.robot.get_observation()
                if obs is not None:
                    return self.semantic_sensor.predict(obs)
            
            logger.warning("No perception components available for object detection")
            return []
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return []
    
    def get_rgb_image(self):
        """
        Get the latest RGB image from the perception adapter.
        
        Returns:
            np.ndarray: RGB image
        """
        try:
            # Use perception adapter if available
            if self.perception_adapter is not None:
                return self.perception_adapter.get_rgb_image()
            
            # Fall back to direct observation
            obs = self.robot.get_observation()
            if obs is not None and hasattr(obs, 'rgb'):
                return obs.rgb
            
            return None
        except Exception as e:
            logger.error(f"Error getting RGB image: {e}")
            return None
    
    def get_depth_image(self):
        """
        Get the latest depth image from the perception adapter.
        
        Returns:
            np.ndarray: Depth image
        """
        try:
            # Use perception adapter if available
            if self.perception_adapter is not None:
                return self.perception_adapter.get_depth_image()
            
            # Fall back to direct observation
            obs = self.robot.get_observation()
            if obs is not None and hasattr(obs, 'depth'):
                return obs.depth
            
            return None
        except Exception as e:
            logger.error(f"Error getting depth image: {e}")
            return None
    
    def plan_path_to_goal(self, x, y, use_rrt=False):
        """
        Plan a path to a goal position using the navigation adapter.
        
        Args:
            x: Goal X coordinate
            y: Goal Y coordinate
            use_rrt: If True, use RRT planner, otherwise use A*
            
        Returns:
            list: List of waypoints [[x, y], ...] or None if no path found
        """
        try:
            logger.info(f"Planning path to goal: ({x}, {y})")
            
            # Use navigation adapter if available
            if self.navigation_adapter is not None:
                path = self.navigation_adapter.plan_path([x, y], use_rrt)
                if path:
                    return path
            
            # Fall back to mapping adapter
            if self.mapping_adapter is not None:
                path = self.mapping_adapter.plan_path([x, y], use_rrt)
                if path:
                    return self.mapping_adapter.simplify_path(path)
            
            logger.warning("No path planning components available")
            return None
        except Exception as e:
            logger.error(f"Error planning path to goal: {e}")
            return None
    
    def navigate_to_goal(self, x, y, theta=None, use_rrt=False):
        """
        Navigate to a goal position.
        
        Args:
            x: Goal X coordinate
            y: Goal Y coordinate
            theta: Goal orientation (optional)
            use_rrt: If True, use RRT planner, otherwise use A*
            
        Returns:
            bool: True if navigation started successfully, False otherwise
        """
        try:
            logger.info(f"Navigating to goal: ({x}, {y}, {theta})")
            
            # Construct goal
            goal = [x, y]
            if theta is not None:
                goal.append(theta)
            
            # Use navigation adapter if available
            if self.navigation_adapter is not None:
                return self.navigation_adapter.navigate_to(goal, use_rrt)
            
            # Fall back to basic trajectory following
            path = self.plan_path_to_goal(x, y, use_rrt)
            if not path:
                logger.warning(f"Cannot navigate: no path found to ({x}, {y})")
                return False
            
            # Add orientation to final waypoint if specified
            if theta is not None and path:
                path[-1] = [path[-1][0], path[-1][1], theta]
            
            # Execute trajectory
            return self.robot.execute_trajectory(path)
        except Exception as e:
            logger.error(f"Error navigating to goal: {e}")
            return False
    
    def navigate_to_object(self, category, approach_distance=0.5):
        """
        Navigate to an object of the specified category.
        
        Args:
            category: Object category to navigate to
            approach_distance: Distance to keep from the object in meters
            
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        try:
            logger.info(f"Navigating to object of category: {category}")
            # Find objects of the specified category
            objects = self.find_object_by_category(category)
            if not objects:
                logger.info(f"No objects of category '{category}' found")
                return False
            
            # Use the first object found
            obj = objects[0]
            logger.info(f"Found object: {obj}")
            
            # Extract object position
            if not hasattr(obj, 'position') or obj.position is None:
                logger.info(f"Object '{category}' has no position information")
                return False
            
            # Get current position
            current_position, _ = self.get_current_pose()
            if current_position is None:
                logger.info("Cannot get current position")
                return False
            
            # Calculate direction vector
            direction = np.array(obj.position[:2]) - np.array(current_position[:2])
            direction_norm = direction / np.linalg.norm(direction)
            
            # Calculate approach position
            approach_x = obj.position[0] - direction_norm[0] * approach_distance
            approach_y = obj.position[1] - direction_norm[1] * approach_distance
            
            # Compute heading toward object
            heading = np.arctan2(
                obj.position[1] - approach_y,
                obj.position[0] - approach_x
            )
            
            logger.info(f"Approach position: ({approach_x}, {approach_y}), heading: {heading}")
            
            # Navigate to approach position
            return self.navigate_to_goal(approach_x, approach_y, heading)
        except Exception as e:
            logger.error(f"Error navigating to object: {e}")
            return False
    
    def execute_operation(self, operation_name, **kwargs):
        """
        Execute a named operation with the given parameters.
        
        Args:
            operation_name: Name of the operation to execute
            **kwargs: Operation-specific parameters
            
        Returns:
            bool: True if operation was successful, False otherwise
        """
        try:
            logger.info(f"Executing operation: {operation_name}")
            
            if operation_name == "move_forward":
                distance = kwargs.get("distance", 0.5)
                current_pose = self.robot.get_base_pose()
                target_pose = [
                    current_pose[0] + distance * np.cos(current_pose[2]),
                    current_pose[1] + distance * np.sin(current_pose[2]),
                    current_pose[2]
                ]
                # Use motion adapter if available
                if self.motion_adapter is not None:
                    return self.motion_adapter.move_to_pose(target_pose)
                else:
                    return self.robot.move_base_to(target_pose, blocking=True)
                
            elif operation_name == "rotate_in_place":
                angle_rad = kwargs.get("angle_rad", np.pi/2)
                # Use motion adapter if available
                if self.motion_adapter is not None:
                    return self.motion_adapter.rotate_in_place(angle_rad)
                else:
                    current_pose = self.robot.get_base_pose()
                    target_pose = [
                        current_pose[0],
                        current_pose[1],
                        current_pose[2] + angle_rad
                    ]
                    return self.robot.move_base_to(target_pose, blocking=True)
                
            elif operation_name == "navigate":
                target_x = kwargs.get("target_x", 0.0)
                target_y = kwargs.get("target_y", 0.0)
                target_theta = kwargs.get("target_theta", None)
                # Use navigation adapter if available
                if self.navigation_adapter is not None:
                    goal = [target_x, target_y]
                    if target_theta is not None:
                        goal.append(target_theta)
                    return self.navigation_adapter.navigate_to(goal)
                else:
                    return self.navigate_to_goal(target_x, target_y, target_theta)
                
            elif operation_name == "scan_environment":
                # New operation to perform a complete scan of the environment
                steps = kwargs.get("steps", 8)
                update_map = kwargs.get("update_map", True)
                visualize = kwargs.get("visualize", False)
                
                logger.info(f"Scanning environment with {steps} rotations")
                result = self.rotate_in_place(steps=steps, visualize=visualize)
                
                if result and update_map:
                    self.update(visualize_map=visualize)
                    
                return result
                
            elif operation_name == "explore":
                # New operation to explore unknown areas
                frontier_radius = kwargs.get("frontier_radius", 3.0)
                max_iterations = kwargs.get("max_iterations", 5)
                
                # Use navigation adapter if available
                if self.navigation_adapter is not None:
                    return self.navigation_adapter.explore(max_iterations, frontier_radius)
                
                logger.info(f"Exploring environment with {max_iterations} iterations")
                
                for i in range(max_iterations):
                    logger.info(f"Exploration iteration {i+1}/{max_iterations}")
                    
                    # Find frontier points
                    frontier_points = self.mapping_adapter.find_frontiers(
                        radius=frontier_radius
                    ) if hasattr(self.mapping_adapter, 'find_frontiers') else None
                    
                    if not frontier_points or len(frontier_points) == 0:
                        logger.info("No more frontiers to explore")
                        break
                        
                    # Choose the closest frontier point
                    current_pose = self.robot.get_base_pose()
                    closest_point = min(
                        frontier_points,
                        key=lambda p: np.linalg.norm(np.array([p[0], p[1]]) - np.array([current_pose[0], current_pose[1]]))
                    )
                    
                    logger.info(f"Navigating to frontier at {closest_point}")
                    
                    # Navigate to the frontier
                    result = self.navigate_to_goal(closest_point[0], closest_point[1])
                    
                    if not result:
                        logger.warning(f"Failed to reach frontier at {closest_point}")
                        continue
                        
                    # Scan the environment at this frontier
                    self.execute_operation("scan_environment")
                
                return True
                
            elif operation_name == "stop":
                # Stop all motion
                if self.motion_adapter is not None:
                    return self.motion_adapter.stop_motion()
                elif self.navigation_adapter is not None:
                    return self.navigation_adapter.stop_navigation()
                else:
                    self.robot.move_base_to([0, 0, 0], relative=True, blocking=False)  # Send zero velocity
                    return True
                
            else:
                logger.warning(f"Unknown operation: {operation_name}")
                return False
        except Exception as e:
            logger.error(f"Error executing operation {operation_name}: {e}")
            return False
    
    def find_object_by_category(self, category):
        """
        Find objects of a specific category in the scene.
        
        Args:
            category: Object category to find
            
        Returns:
            list: List of detected objects matching the category
        """
        try:
            logger.info(f"Finding objects of category: {category}")
            objects = self.detect_objects()
            matches = [obj for obj in objects if hasattr(obj, 'category') and obj.category.lower() == category.lower()]
            logger.info(f"Found {len(matches)} objects of category '{category}'")
            return matches
        except Exception as e:
            logger.error(f"Error finding objects by category: {e}")
            return []
    
    def get_current_pose(self):
        """
        Get the current robot pose in the map frame.
        
        Returns:
            tuple: (position, orientation) where position is [x, y, z] and 
                  orientation is [x, y, z, w] quaternion
        """
        try:
            # Use navigation adapter if available
            if self.navigation_adapter is not None:
                pose = self.navigation_adapter._get_current_pose()
                if pose is not None:
                    return pose
            
            # Fall back to mapping adapter
            if self.mapping_adapter is not None:
                return self.mapping_adapter.get_robot_pose()
            
            # Fall back to basic pose from ROS client
            xyt = self.robot.get_base_pose()
            if xyt is not None:
                x, y, theta = xyt
                position = np.array([x, y, 0.0])
                
                # Convert theta to quaternion
                qx = 0.0
                qy = 0.0
                qz = np.sin(theta / 2)
                qw = np.cos(theta / 2)
                orientation = np.array([qx, qy, qz, qw])
                
                return position, orientation
            
            return None, None
        except Exception as e:
            logger.error(f"Error getting current pose: {e}")
            return None, None
    
    def caption_scene(self):
        """
        Generate a caption for the current scene.
        
        Returns:
            str: Caption for the scene
        """
        try:
            # Use perception adapter if available
            if self.perception_adapter is not None:
                return self.perception_adapter.caption_image()
            
            return "Scene captioning not available"
        except Exception as e:
            logger.error(f"Error captioning scene: {e}")
            return "Error generating caption"
    
    def get_nearest_obstacle_distance(self):
        """
        Get the distance to the nearest obstacle.
        
        Returns:
            float: Distance to the nearest obstacle in meters
        """
        try:
            # Use mapping adapter if available
            if self.mapping_adapter is not None:
                return self.mapping_adapter.get_nearest_obstacle_distance()
            
            return float('inf')  # Return infinity if no mapping adapter
        except Exception as e:
            logger.error(f"Error getting nearest obstacle distance: {e}")
            return float('inf')
    
    def is_path_clear(self, x, y):
        """
        Check if a straight path to a point is clear of obstacles.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            
        Returns:
            bool: True if path is clear, False otherwise
        """
        try:
            # Use mapping adapter if available
            if self.mapping_adapter is not None:
                position, _ = self.get_current_pose()
                if position is None:
                    logger.warning("Cannot get current position")
                    return False
                
                start = [position[0], position[1]]
                end = [x, y]
                
                logger.info(f"Checking if path is clear from {start} to {end}")
                
                return self.mapping_adapter.is_path_clear(start, end)
            
            return True  # Assume path is clear if no mapping adapter
        except Exception as e:
            logger.error(f"Error checking if path is clear: {e}")
            return False
    
    def reset_map(self):
        """Reset the map."""
        try:
            logger.info("Resetting map")
            
            # Reset mapping adapter
            if self.mapping_adapter is not None:
                self.mapping_adapter.reset_map()
            
            # Reset voxel map
            voxel_map = self.get_voxel_map()
            if voxel_map is not None:
                voxel_map.reset()
                
            # Reset object plans
            self.reset_object_plans()
                
            logger.info("Map reset complete")
            return True
        except Exception as e:
            logger.error(f"Error resetting map: {e}")
            return False
            
    def check_system_status(self):
        """
        Check the status of various system components.
        
        Returns:
            dict: Status information for various components
        """
        status = {
            "connection": self.connection_status.copy(),
            "robot": {
                "battery": None,
                "sensors": None
            },
            "perception": {
                "available": self.perception_adapter is not None,
                "semantic_sensor": self.semantic_sensor is not None
            },
            "mapping": {
                "available": self.mapping_adapter is not None,
                "voxel_map": self.get_voxel_map() is not None
            },
            "navigation": {
                "available": self.navigation_adapter is not None,
                "is_navigating": self.navigation_adapter.is_navigating() if self.navigation_adapter is not None else False,
                "emergency_stop": self.navigation_adapter.is_emergency_stopped() if self.navigation_adapter is not None else False
            },
            "motion": {
                "available": self.motion_adapter is not None,
                "is_moving": not self.motion_adapter.is_motion_complete() if self.motion_adapter is not None else False
            }
        }
        
        # Get battery status if available
        try:
            if hasattr(self.ros_client, 'get_battery_status'):
                status["robot"]["battery"] = self.ros_client.get_battery_status()
        except Exception:
            pass
            
        # Get sensor status if available
        try:
            if hasattr(self.ros_client, 'get_sensor_status'):
                status["robot"]["sensors"] = self.ros_client.get_sensor_status()
        except Exception:
            pass
            
        return status
        
    def get_localization_status(self):
        """
        Get the status of the localization system.
        
        Returns:
            dict: Information about the current localization status
        """
        try:
            position, orientation = self.get_current_pose()
            
            # Get confidence from mapping adapter
            confidence = None
            is_lost = False
            if self.mapping_adapter is not None:
                confidence = self.mapping_adapter.get_localization_confidence()
                is_lost = self.mapping_adapter.is_robot_lost()
            
            return {
                "position": position,
                "orientation": orientation,
                "confidence": confidence,
                "is_lost": is_lost
            }
        except Exception as e:
            logger.error(f"Error getting localization status: {e}")
            return {
                "position": None,
                "orientation": None,
                "confidence": None,
                "is_lost": True
            }
    
    def process_hybrid_action(self, action):
        """
        Process a hybrid action.
        
        Args:
            action: HybridAction to process
            
        Returns:
            bool: True if action processed successfully, False otherwise
        """
        try:
            # Use motion adapter if available
            if self.motion_adapter is not None:
                return self.motion_adapter.convert_action(action)
            
            # Fall back to basic processing
            if isinstance(action, HybridAction):
                if action.is_discrete():
                    discrete_action = action.get()
                    # Process discrete action
                    if discrete_action == 0:  # STOP
                        self.robot.move_base_to([0, 0, 0], relative=True, blocking=False)
                        return True
                    elif discrete_action == 1:  # MOVE_FORWARD
                        self.execute_operation("move_forward")
                        return True
                    elif discrete_action == 2:  # TURN_LEFT
                        self.execute_operation("rotate_in_place", angle_rad=np.pi/2)
                        return True
                    elif discrete_action == 3:  # TURN_RIGHT
                        self.execute_operation("rotate_in_place", angle_rad=-np.pi/2)
                        return True
                elif action.is_navigation():
                    xyt = action.get()
                    self.navigate_to_goal(xyt[0], xyt[1], xyt[2] if len(xyt) > 2 else None)
                    return True
            
            logger.warning(f"Unsupported action: {action}")
            return False
        except Exception as e:
            logger.error(f"Error processing hybrid action: {e}")
            return False