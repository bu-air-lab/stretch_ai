"""
Segway ROS Client for Stretch AI

This module implements a ROS client for the Segway robot, handling the 
communication with ROS topics for control and sensing. It supports bridging
between ROS 1 (Melodic) on the Segway and ROS 2 (Humble) on the external
computer with RTX 4090.
"""

import os
import time
import numpy as np
import threading
import yaml
import subprocess
from typing import List, Optional, Tuple, Union, Dict, Any
import traceback

# Import ROS 2 libraries - used on the external computer with RTX 4090
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image, CameraInfo, PointCloud2, BatteryState, Imu
from std_msgs.msg import Header, String
from tf2_ros import TransformListener, Buffer
import tf2_ros

# Import interface classes from Stretch AI
from stretch.core.interfaces import Observations, PointCloud
from stretch.utils.logger import Logger
from stretch.motion import Footprint

# Import other necessary libraries
import torch
import logging

# Set up logger
logger = Logger(__name__)

# Try to import cv_bridge for image conversion
try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    logger.warning("cv_bridge not available, image conversion may be limited")
    CV_BRIDGE_AVAILABLE = False


# Define CameraIntrinsics class first
class CameraIntrinsics:
    """
    Camera intrinsics parameters class.
    
    Stores the intrinsic parameters of a camera including focal lengths,
    principal point, and the full intrinsics matrix.
    """
    
    def __init__(self, width: int, height: int, fx: float, fy: float, 
                 cx: float, cy: float, K: Optional[np.ndarray] = None):
        """
        Initialize camera intrinsics.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            fx: Focal length in x direction (pixels)
            fy: Focal length in y direction (pixels)
            cx: Principal point x coordinate (pixels)
            cy: Principal point y coordinate (pixels)
            K: Optional 3x3 camera intrinsics matrix
        """
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        # If K is not provided, construct it from fx, fy, cx, cy
        if K is None:
            self.K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            self.K = K
            
    def __repr__(self):
        """String representation"""
        return (f"CameraIntrinsics(width={self.width}, height={self.height}, "
                f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f})")
                
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'width': self.width,
            'height': self.height,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'K': self.K.tolist() if self.K is not None else None
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary"""
        return cls(
            width=data['width'],
            height=data['height'],
            fx=data['fx'],
            fy=data['fy'],
            cx=data['cx'],
            cy=data['cy'],
            K=np.array(data['K']) if data.get('K') is not None else None
        )
        
    @classmethod
    def from_camera_info(cls, camera_info_msg):
        """Create from ROS CameraInfo message"""
        K = np.array(camera_info_msg.k).reshape(3, 3)
        return cls(
            width=camera_info_msg.width,
            height=camera_info_msg.height,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            K=K
        )


# Define DensePointCloud class next
class DensePointCloud:
    """
    Dense point cloud created from depth image and camera intrinsics.
    
    This class creates a 3D point cloud from a depth image using camera
    intrinsics parameters. It can also transform the points to the world
    coordinate frame if a camera pose is provided.
    """
    
    def __init__(self, 
                 depth: torch.Tensor, 
                 intrinsics: CameraIntrinsics, 
                 camera_pose: Optional[np.ndarray] = None, 
                 max_depth: float = 5.0, 
                 sampling_step: int = 8):
        """
        Initialize and create a point cloud.
        
        Args:
            depth: Depth image tensor (H, W)
            intrinsics: Camera intrinsics parameters
            camera_pose: 4x4 transformation matrix for camera to world coordinates
            max_depth: Maximum valid depth value
            sampling_step: Step size for sampling pixels (higher = fewer points)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"Initializing DensePointCloud with depth shape {depth.shape}, sampling_step={sampling_step}")
        self.points = None
        self.colors = None

        # Store the valid depth mask for later use
        self.valid_depth_mask = (depth > 0) & (depth <= max_depth)
        
        try:
            points_cam = self._create_points(depth, intrinsics, max_depth, sampling_step)
            if points_cam is not None and points_cam.shape[0] > 0:
                # Optionally transform to world frame if pose is provided
                if camera_pose is not None:
                    self.logger.debug("Applying camera pose to point cloud.")
                    # Convert numpy pose to tensor if needed
                    if not isinstance(camera_pose, torch.Tensor):
                        cam_to_world = torch.from_numpy(camera_pose).float().to(points_cam.device)
                    else:
                        cam_to_world = camera_pose
                        
                    # Add homogeneous coordinate
                    points_cam_h = torch.cat([points_cam, torch.ones(points_cam.shape[0], 1, device=points_cam.device)], dim=1)
                    # Transform
                    points_world_h = torch.matmul(cam_to_world, points_cam_h.T).T
                    self.points = points_world_h[:, :3] # Remove homogeneous coordinate
                else:
                    self.logger.debug("No camera pose provided, points remain in camera frame.")
                    self.points = points_cam # Points are in camera frame
            else:
                 self.logger.warning("No valid points generated from depth image.")
                 self.points = torch.zeros((0, 3), dtype=torch.float32)  # Empty point cloud

        except Exception as e:
             self.logger.error(f"Error during DensePointCloud initialization: {e}")
             traceback.print_exc()
             self.points = torch.zeros((0, 3), dtype=torch.float32)  # Empty point cloud on error

    def _create_points(self, depth: torch.Tensor, intrinsics: CameraIntrinsics, max_depth: float, sampling_step: int) -> Optional[torch.Tensor]:
        """
        Create 3D points from depth image.
        
        Args:
            depth: Depth image tensor (H, W)
            intrinsics: Camera intrinsics parameters
            max_depth: Maximum valid depth value
            sampling_step: Step size for sampling pixels
            
        Returns:
            torch.Tensor: 3D points in camera frame, or None if no valid points
        """
        # Get valid depth mask
        valid_depth_mask = (depth > 0) & (depth <= max_depth)
        
        if not torch.any(valid_depth_mask):
            self.logger.warning("No valid depth points in image.")
            return None

        # Get sampled pixel coordinates
        h, w = depth.shape
        # Use slicing for sampling
        sampled_v = torch.arange(0, h, sampling_step, device=depth.device)
        sampled_u = torch.arange(0, w, sampling_step, device=depth.device)
        grid_v, grid_u = torch.meshgrid(sampled_v, sampled_u, indexing='ij')

        # Flatten and get depth values at sampled locations
        v_flat = grid_v.flatten()
        u_flat = grid_u.flatten()
        depth_sampled = depth[v_flat, u_flat]

        # Filter based on max_depth and validity (e.g., > 0) BEFORE projection
        valid_mask = (depth_sampled > 0) & (depth_sampled <= max_depth)
        v_valid = v_flat[valid_mask]
        u_valid = u_flat[valid_mask]
        depth_valid = depth_sampled[valid_mask]

        num_valid_points = depth_valid.shape[0]
        self.logger.debug(f"Sampled {v_flat.shape[0]} points, found {num_valid_points} valid depth points after filtering.")

        if num_valid_points == 0:
            self.logger.warning("No valid points found after depth filtering in _create_points.")
            return None

        # Back-project valid points to camera coordinates
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy

        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid # Depth is the z-coordinate in the camera frame

        points_camera_frame = torch.stack((x_cam, y_cam, z_cam), dim=-1)
        self.logger.debug(f"Generated {points_camera_frame.shape[0]} points in camera frame.")

        return points_camera_frame

    def transform(self, transform_matrix):
        """
        Transform the point cloud using a 4x4 transformation matrix.
        
        Args:
            transform_matrix: 4x4 transformation matrix (numpy array or torch tensor)
            
        Returns:
            self: Returns self for method chaining
        """
        if self.points is None or self.points.shape[0] == 0:
            self.logger.warning("No points to transform")
            return self
            
        try:
            # Convert numpy matrix to torch tensor if needed
            if isinstance(transform_matrix, np.ndarray):
                transform_tensor = torch.from_numpy(transform_matrix).float().to(self.points.device)
            else:
                transform_tensor = transform_matrix
                
            # Add homogeneous coordinate to points
            points_h = torch.cat([
                self.points, 
                torch.ones(self.points.shape[0], 1, device=self.points.device)
            ], dim=1)
            
            # Transform points
            transformed_points_h = torch.matmul(transform_tensor, points_h.T).T
            
            # Remove homogeneous coordinate
            self.points = transformed_points_h[:, :3]
            
            self.logger.debug(f"Transformed {self.points.shape[0]} points with a 4x4 matrix")
            return self
            
        except Exception as e:
            self.logger.error(f"Error transforming point cloud: {e}")
            traceback.print_exc()
            return self


class SegwayRobotModel:
    """
    Robot model for the Segway, implementing the interface
    required by Stretch AI.
    """
    
    def __init__(self, config):
        """
        Initialize the Segway robot model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Segway physical parameters
        self.wheel_base = config.get('wheel_base', 0.5)  # meters
        self.width = config.get('robot_width', 0.5)  # meters
        self.length = config.get('robot_length', 0.7)  # meters
        self.height = config.get('robot_height', 1.0)  # meters
        
        # Robot footprint for planning
        self.footprint_radius = config.get('footprint_radius', 0.35)  # meters
        
    def get_footprint(self):
        """
        Get the robot's footprint for collision checking.
        
        Returns:
            Footprint: Footprint object for collision checking
        """
        return Footprint(
            length=self.length,
            width=self.width,
            length_offset=0.0,
            width_offset=0.0
        )
    
    def get_dof_limits(self):
        """
        Get the limits of the robot's degrees of freedom.
        
        Returns:
            tuple: (lower_limits, upper_limits)
        """
        # For a simple differential drive robot
        # Limits for x, y, theta
        lower_limits = np.array([-np.inf, -np.inf, -np.inf])
        upper_limits = np.array([np.inf, np.inf, np.inf])
        
        return lower_limits, upper_limits
    
    def get_dof_names(self):
        """
        Get the names of the robot's degrees of freedom.
        
        Returns:
            list: List of DOF names
        """
        return ['base_x', 'base_y', 'base_theta']


class SegwayROSClient:
    """
    ROS client for interfacing with the Segway robot through ROS topics.
    Supports bridging between ROS 1 (Melodic) on the Segway and ROS 2 (Humble)
    on the external computer with RTX 4090.
    """
    
    def __init__(self, 
                 config_path=None,
                 robot_ip=None,
                 desktop_ip=None,
                 node_name='segway_client'):
        """
        Initialize the Segway ROS client.
        
        Args:
            config_path: Path to configuration file
            robot_ip: IP address of the Segway robot (ROS 1)
            desktop_ip: IP address of the external computer (ROS 2)
            node_name: Name for the ROS node
        """
        self.logger = Logger(__name__)
        
        # Initialize the configuration
        self.config = self._load_config(config_path)
        
        # Set sensor parameters
        self.max_depth = self.config.get("sensors", {}).get("max_depth", 5.0)  # meters
        
        # Set IP addresses
        self.robot_ip = robot_ip or self.config.get("network", {}).get("robot_ip", "10.66.171.191")
        self.desktop_ip = desktop_ip or self.config.get("network", {}).get("desktop_ip", "10.66.171.131")
        self.lidar_ip = self.config.get("network", {}).get("lidar_ip", "10.66.171.8")
        
        # Bridge configuration
        self.use_bridge = self.config.get("ros_bridge", {}).get("use_ros1_to_ros2_bridge", True)
        self.bridge_port = self.config.get("network", {}).get("bridge_port", 9090)
        self.bridge_process = None
        
        # Get topic names from config
        self._setup_topics()
        
        # Initialize robot model
        self.robot_model = SegwayRobotModel(self.config)
        
        # Initialize ROS node
        self.node_name = node_name
        self.node = None
        self.executor = None
        self.ros_thread = None
        self.running = False
        
        # Data storage
        self.current_pose_xyt = [0.0, 0.0, 0.0]  # x, y, theta
        self.last_odom = None
        self.last_scan = None
        self.last_rgb = None
        self.last_depth = None
        self.last_camera_info = None
        self.last_point_cloud_msg = None # Store the raw msg
        self.last_imu = None
        self.last_battery = None
        
        # Bridge status
        self.bridge_status = False
        self.bridge_last_checked = 0
        self.bridge_check_interval = 5.0  # seconds
        
        # Connection status
        self.connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = self.config.get("network", {}).get("connection_retries", 5)
        
        # For thread safety
        self.lock = threading.Lock()
        
        # For CV bridge
        self.cv_bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
        
        # TF Listener setup
        self.tf_buffer = None
        self.tf_listener = None
        
        # Start ROS node if auto_start is True
        if self.config.get("auto_start", True):
            self.connect()
    
    def _load_config(self, config_path):
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        if config_path is None:
            logger.warning("No config path provided, using default configuration")
            return {}
            
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            else:
                logger.warning(f"Config file {config_path} not found, using default configuration")
                return {}
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    def _setup_topics(self):
        """Set up ROS topic names from configuration."""
        ros_bridge_config = self.config.get("ros_bridge", {})
        topics = ros_bridge_config.get("topics", {})
        
        # ROS 1 topics (on Segway)
        self.segway_cmd_vel_topic = topics.get("segway_cmd_vel", "/segway/cmd_vel")
        self.segway_odom_topic = topics.get("segway_odom", "/segway/odom")
        self.segway_scan_topic = topics.get("segway_scan", "/segway/scan")
        self.segway_rgb_topic = topics.get("segway_rgb", "/camera/color/image_raw")
        self.segway_depth_topic = topics.get("segway_depth", "/camera/depth/image_rect_raw")
        self.segway_camera_info_topic = topics.get("segway_camera_info", "/camera/color/camera_info")
        self.segway_imu_topic = topics.get("segway_imu", "/segway/imu/data")
        self.segway_battery_topic = topics.get("segway_battery", "/segway/battery")
        
        # ROS 2 topics (on RTX 4090 desktop)
        self.cmd_vel_topic = topics.get("ros2_cmd_vel", "/cmd_vel")
        self.odom_topic = topics.get("ros2_odom", "/segway/odom")
        self.scan_topic = topics.get("ros2_scan", "/scan")
        self.rgb_topic = topics.get("ros2_rgb", "/camera/color/image_raw")
        self.depth_topic = topics.get("ros2_depth", "/camera/depth/image_rect_raw")
        self.camera_info_topic = topics.get("ros2_camera_info", "/camera/color/camera_info")
        self.point_cloud_topic = topics.get("ros2_point_cloud", "/lidar_cloud")
        self.imu_topic = topics.get("ros2_imu", "/imu")
        self.battery_topic = topics.get("ros2_battery", "/segway/connection_status")
        
        # Bridge status topic
        self.bridge_status_topic = topics.get("bridge_status", "/ros_bridge/status")
    
    def connect(self):
        """
        Connect to ROS and start the ROS node.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        # Check if already connected
        if self.connected:
            logger.info("Already connected to ROS")
            return True
            
        # Track connection attempts
        self.connection_attempts += 1
        
        try:
            # Start the ROS 1 to ROS 2 bridge if needed
            if self.use_bridge:
                bridge_success = self._start_bridge()
                if not bridge_success:
                    logger.error("Failed to start ROS bridge")
                    return False
            
            # Initialize ROS 2
            if not rclpy.ok():
                rclpy.init()
            
            # Create node
            self.node = rclpy.create_node(self.node_name)
            
            # List available topics for debugging
            self._list_topics()

            # Set up QoS profiles
            sensor_qos = QoSProfile(
                depth=10,
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE
            )
            
            # Create publishers
            self.cmd_vel_pub = self.node.create_publisher(
                Twist, 
                self.cmd_vel_topic, 
                10
            )
            
            # Create subscribers
            self.odom_sub = self.node.create_subscription(
                Odometry,
                self.odom_topic,
                self._odom_callback,
                10
            )
            
            self.scan_sub = self.node.create_subscription(
                LaserScan,
                self.scan_topic,
                self._scan_callback,
                sensor_qos
            )
            
            self.rgb_sub = self.node.create_subscription(
                Image,
                self.rgb_topic,
                self._rgb_callback,
                sensor_qos
            )
            
            self.depth_sub = self.node.create_subscription(
                Image,
                self.depth_topic,
                self._depth_callback,
                sensor_qos
            )
            
            self.camera_info_sub = self.node.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self._camera_info_callback,
                10
            )
            
            self.point_cloud_sub = self.node.create_subscription(
                PointCloud2,
                self.point_cloud_topic,
                self._point_cloud_callback,
                sensor_qos
            )
            
            self.imu_sub = self.node.create_subscription(
                Imu,  # Correct message type for IMU data
                self.imu_topic,
                self._imu_callback,
                sensor_qos
            )
            
            self.battery_sub = self.node.create_subscription(
                BatteryState,
                self.battery_topic,
                self._battery_callback,
                10
            )
            
            # Bridge status subscriber
            if self.use_bridge:
                self.bridge_status_sub = self.node.create_subscription(
                    String,
                    self.bridge_status_topic,
                    self._bridge_status_callback,
                    10
                )
            
            # Set up TF listener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self.node)
            
            # Create executor and start ROS thread
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            
            self.running = True
            self.connected = True
            self.ros_thread = threading.Thread(target=self._ros_thread_func)
            self.ros_thread.daemon = True
            self.ros_thread.start()
            
            logger.info(f"Connected to ROS with node {self.node_name}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to ROS: {e}")
            
            # Clean up if needed
            self._cleanup_ros()
            
            # Check if we should retry
            if self.connection_attempts < self.max_connection_attempts:
                retry_interval = self.config.get("network", {}).get("connection_retry_interval", 2.0)
                logger.info(f"Retrying connection in {retry_interval} seconds (attempt {self.connection_attempts}/{self.max_connection_attempts})")
                time.sleep(retry_interval)
                return self.connect()
            else:
                logger.error(f"Failed to connect after {self.max_connection_attempts} attempts")
                return False
    
    def reconnect(self):
        """
        Reconnect to ROS after a failure.
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        logger.info("Attempting to reconnect to ROS")
        
        # Reset connection attempts to allow retry loop
        self.connection_attempts = 0
        
        # Stop ROS processing
        self.stop()
        
        # Wait a bit before reconnecting
        time.sleep(2.0)
        
        # Try to connect again
        return self.connect()
    
    def _start_bridge(self):
        """
        Start the ROS 1 to ROS 2 bridge.
        
        Returns:
            bool: True if bridge started successfully, False otherwise
        """
        try:
            logger.info(f"Starting ROS 1 to ROS 2 bridge at {self.robot_ip}:{self.bridge_port}")
            
            # Check if bridge is already running
            if self.bridge_process is not None:
                if self.bridge_process.poll() is None:
                    # Process is still running
                    logger.info("Bridge already running")
                    return True
                else:
                    # Process has terminated
                    logger.warning(f"Bridge process terminated with code {self.bridge_process.returncode}")
                    self.bridge_process = None
            
            # Get bridge path - use the direct path from environment
            bridge_path = os.environ.get('ROS_BRIDGE_PATH', 
                '/home/aoloo/ros_bridge/ros-humble-ros1-bridge/install/ros1_bridge/lib/ros1_bridge/dynamic_bridge')
            
            # Use the wrapper script for proper library path
            wrapper_script = os.path.expanduser("~/segway_stretch_ws/run_bridge.sh")
            
            # Set up environment variables
            env = os.environ.copy()
            env['ROS_MASTER_URI'] = f"http://{self.robot_ip}:11311"
            env['ROS_IP'] = self.desktop_ip
            
            # Build command with wrapper script and node name
            cmd = [
                wrapper_script,
                bridge_path,
                "--ros-args", "-r", "__node:=dynamic_bridge_segway_stretch",
                "--",
                "--bridge-all-topics",
                f"--ros1-host={self.robot_ip}",
                f"--ros2-host={self.desktop_ip}"
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Start the process with the modified environment
            self.bridge_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Wait for bridge to initialize
            logger.info("Waiting for bridge to initialize (15 seconds)...")
            time.sleep(15.0)
            
            # Check if the bridge is running
            if self.bridge_process.poll() is None:
                logger.info("Bridge started successfully")
                self.bridge_status = True
                
                # We should list topics after bridge is up to confirm it's working
                if self.node:
                    self._list_topics()
                    
                return True
            else:
                stderr_output = self.bridge_process.stderr.read() if self.bridge_process.stderr else "No error output"
                stdout_output = self.bridge_process.stdout.read() if self.bridge_process.stdout else "No stdout output"
                logger.error(f"Bridge failed to start: {stderr_output}")
                logger.error(f"Bridge stdout: {stdout_output}")
                self.bridge_process = None
                self.bridge_status = False
                
        except Exception as e:
            logger.error(f"Error starting ROS bridge: {e}")
            self.bridge_status = False
            return False

    def _list_topics(self):
        """List all available ROS topics and log them."""
        try:
            logger.info("=== Listing all available ROS topics ===")
            
            # Need to wait for node to be fully initialized
            time.sleep(2.0)
            
            # Get the list of topics
            topic_names_and_types = self.node.get_topic_names_and_types()
            
            # Sort and display
            for topic_name, topic_types in sorted(topic_names_and_types):
                type_str = ', '.join(topic_types)
                logger.info(f"  • {topic_name} [{type_str}]")
                
            return True
        except Exception as e:
            logger.error(f"Error listing topics: {e}")
            return False

    def restart_bridge(self):
        """
        Restart the ROS 1 to ROS 2 bridge.
        
        Returns:
            bool: True if bridge restarted successfully, False otherwise
        """
        try:
            logger.info("Restarting ROS bridge")
            
            # Stop the bridge if it's running
            if self.bridge_process is not None:
                logger.info("Stopping existing bridge process")
                try:
                    self.bridge_process.terminate()
                    self.bridge_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.warning("Bridge process did not terminate, killing it")
                    self.bridge_process.kill()
                    self.bridge_process.wait()
                self.bridge_process = None
            
            # Start the bridge again
            return self._start_bridge()
            
        except Exception as e:
            logger.error(f"Error restarting ROS bridge: {e}")
            self.bridge_status = False
            return False

    def check_bridge_status(self):
        """
        Check if the ROS bridge is running.
        
        Returns:
            bool: True if bridge is running, False otherwise
        """
        if not self.use_bridge:
            return True
            
        current_time = time.time()
        
        # Only check periodically to avoid overhead
        if current_time - self.bridge_last_checked < self.bridge_check_interval:
            return self.bridge_status
            
        self.bridge_last_checked = current_time
        
        try:
            # Check if bridge process is running
            if self.bridge_process is not None and self.bridge_process.poll() is None:
                self.bridge_status = True
                return True
            else:
                self.bridge_status = False
                return False
        except Exception as e:
            logger.error(f"Error checking bridge status: {e}")
            self.bridge_status = False
            return False

    def _bridge_status_callback(self, msg):
        """Callback for bridge status messages."""
        # Update bridge status based on message
        # This assumes the bridge publishes a status message
        self.bridge_status = (msg.data == "running")

    def stop(self):
        """Stop the ROS client and cleanup resources."""
        if not self.running:
            return
            
        logger.info("Stopping SegwayROSClient")
        
        # Stop the robot first
        self.stop_robot()
        
        # Stop ROS processing
        self.running = False
        self.connected = False
        
        # Clean up ROS resources
        self._cleanup_ros()
        
        # Stop the bridge if it's running
        if self.use_bridge and self.bridge_process is not None:
            try:
                logger.info("Stopping ROS bridge")
                self.bridge_process.terminate()
                self.bridge_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Bridge process did not terminate, killing it")
                self.bridge_process.kill()
                self.bridge_process.wait()
            except Exception as e:
                logger.error(f"Error stopping bridge: {e}")
            self.bridge_process = None
            self.bridge_status = False
    
    def _cleanup_ros(self):
        """Clean up ROS resources."""
        if self.ros_thread:
            self.ros_thread.join(timeout=1.0)
            self.ros_thread = None
            
        if self.executor:
            self.executor.shutdown()
            self.executor = None
            
        if self.node:
            self.node.destroy_node()
            self.node = None
            
        # Don't shutdown rclpy here as it might be used by other components
    
    def _ros_thread_func(self):
        """Main function for the ROS thread."""
        logger.info("ROS thread started")
        while self.running:
            try:
                if self.executor:
                    self.executor.spin_once(timeout_sec=0.1)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in ROS thread: {e}")
                time.sleep(0.1)
        logger.info("ROS thread stopped")
    
    def _odom_callback(self, msg):
        """Callback for odometry messages."""
        with self.lock:
            self.last_odom = msg
            
            # Extract pose from odometry
            quaternion = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            
            # Convert quaternion to euler angles
            yaw = np.arctan2(2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
                            1.0 - 2.0 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2]))
            
            self.current_pose_xyt = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                yaw
            ]
            
    def _scan_callback(self, msg):
        """Callback for laser scan messages."""
        with self.lock:
            # Log the first message received
            if self.last_scan is None:
                logger.info(f"✓ Received first LIDAR scan message on {self.scan_topic}")
            self.last_scan = msg
            
    def _rgb_callback(self, msg):
        """Callback for RGB image messages."""
        with self.lock:
            # Log the first message received
            if self.last_rgb is None:
                logger.info(f"✓ Received first RGB image on {self.rgb_topic}")
            self.last_rgb = msg
                
    def _depth_callback(self, msg):
        """Callback for depth image messages."""
        with self.lock:
            # Log the first message received
            if self.last_depth is None:
                logger.info(f"✓ Received first depth image on {self.depth_topic}")
            self.last_depth = msg
            
    def _camera_info_callback(self, msg):
        """Callback for camera info messages."""
        with self.lock:
            # Log the first message received
            if self.last_camera_info is None:
                logger.info(f"✓ Received first camera info on {self.camera_info_topic}")
            self.last_camera_info = msg
            
    def _point_cloud_callback(self, msg):
        """Callback for point cloud messages."""
        with self.lock:
            # Log the first message received
            if self.last_point_cloud_msg is None:
                logger.info(f"✓ Received first PointCloud2 message on {self.point_cloud_topic}")
            self.last_point_cloud_msg = msg
            
            # Process the point cloud immediately to avoid the "not implemented" message
            try:
                pc = self.process_point_cloud2(msg)
                if pc is not None:
                    logger.debug(f"Processed PointCloud2 with {pc.points.shape[0]} points")
            except Exception as e:
                logger.error(f"Error in immediate point cloud processing: {e}")

    def process_point_cloud2(self, pc2_msg):
        """
        Process a PointCloud2 message into a usable point cloud.
        
        Args:
            pc2_msg: PointCloud2 message
            
        Returns:
            PointCloud: Processed point cloud or None if processing failed
        """
        if pc2_msg is None:
            return None
        
        try:
            # Try different methods of extracting points from PointCloud2
            
            # Method 1: Using sensor_msgs_py if available
            try:
                from sensor_msgs_py import point_cloud2
                pc_data = list(point_cloud2.read_points(
                    pc2_msg, 
                    field_names=['x', 'y', 'z'],
                    skip_nans=True
                ))
                
                if not pc_data:
                    logger.warning("No valid points in PointCloud2 message")
                    return None
                    
                # Extract x, y, z values from structured array and create a new array
                # This is the key fix for the structured array issue
                points = np.array([[p[0], p[1], p[2]] for p in pc_data], dtype=np.float32)
                
                # Create PointCloud object
                timestamp = pc2_msg.header.stamp.sec + pc2_msg.header.stamp.nanosec * 1e-9
                point_cloud = PointCloud(
                    points=points,
                    frame_id=pc2_msg.header.frame_id,
                    timestamp=timestamp
                )
                
                logger.debug(f"Processed PointCloud2 with {points.shape[0]} points")
                return point_cloud
                
            except ImportError:
                logger.debug("sensor_msgs_py not available, trying alternative method")
            
            # Method 2: Manual extraction using struct
            try:
                import struct
                
                # Get field offsets
                x_offset = None
                y_offset = None
                z_offset = None
                
                for field in pc2_msg.fields:
                    if field.name == 'x':
                        x_offset = field.offset
                    elif field.name == 'y':
                        y_offset = field.offset
                    elif field.name == 'z':
                        z_offset = field.offset
                
                if any(offset is None for offset in [x_offset, y_offset, z_offset]):
                    logger.warning("Could not find XYZ offsets in PointCloud2 message")
                    return None
                
                # Extract points
                points = []
                point_step = pc2_msg.point_step
                data = pc2_msg.data
                is_bigendian = pc2_msg.is_bigendian
                
                # Format string for unpacking
                format_str = '>f' if is_bigendian else '<f'
                
                # Sample every 5th point to reduce processing time
                sample_rate = 5  
                num_points = len(data) // point_step
                
                for i in range(0, num_points, sample_rate):
                    idx = i * point_step
                    
                    # Check if we have enough data for this point
                    if idx + max(x_offset, y_offset, z_offset) + 4 > len(data):
                        continue
                    
                    # Extract coordinates
                    x = struct.unpack(format_str, data[idx + x_offset:idx + x_offset + 4])[0]
                    y = struct.unpack(format_str, data[idx + y_offset:idx + y_offset + 4])[0]
                    z = struct.unpack(format_str, data[idx + z_offset:idx + z_offset + 4])[0]
                    
                    # Skip invalid points
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        points.append([x, y, z])
                
                if not points:
                    logger.warning("No valid points extracted from PointCloud2 message")
                    return None
                
                # Convert to numpy array
                points = np.array(points, dtype=np.float32)
                
                # Create PointCloud object
                timestamp = pc2_msg.header.stamp.sec + pc2_msg.header.stamp.nanosec * 1e-9
                point_cloud = PointCloud(
                    points=points,
                    frame_id=pc2_msg.header.frame_id,
                    timestamp=timestamp
                )
                
                logger.debug(f"Processed PointCloud2 with {points.shape[0]} points")
                return point_cloud
                
            except Exception as e:
                logger.warning(f"Manual PointCloud2 extraction failed: {e}")
                
            logger.warning("All PointCloud2 processing methods failed, falling back to LaserScan")
            return None
            
        except Exception as e:
            logger.error(f"Error processing PointCloud2 message: {e}")
            return None

    def _imu_callback(self, msg):
        """Callback for IMU messages."""
        with self.lock:
            # Log the first message received
            if self.last_imu is None:
                logger.info(f"✓ Received first IMU message on {self.imu_topic}")
            self.last_imu = msg

    def _battery_callback(self, msg):
        """Callback for battery messages."""
        with self.lock:
            self.last_battery = msg
    
    def is_connected(self):
        """
        Check if the client is connected to ROS.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected and self.running
    
    def get_robot_model(self):
        """Get the robot model."""
        return self.robot_model
            
    def get_pose(self):
        """Get the current pose of the robot as [x, y, theta]."""
        with self.lock:
            return self.current_pose_xyt.copy() if self.current_pose_xyt is not None else [0.0, 0.0, 0.0]
            
    def _get_camera_pose(self, target_frame="segway/base_link", source_frame="camera_color_optical_frame", timeout_sec=0.5):
        """Get the camera pose relative to the target frame."""
        if self.tf_buffer is None or self.tf_listener is None:
            self.logger.warning("TF listener not initialized.")
            return None
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                now,
                timeout=rclpy.duration.Duration(seconds=timeout_sec)
            )
            # Convert TransformStamped to a 4x4 numpy matrix
            trans = transform.transform.translation
            rot = transform.transform.rotation
            position = np.array([trans.x, trans.y, trans.z])
            orientation = np.array([rot.x, rot.y, rot.z, rot.w]) # x, y, z, w

            # Convert quaternion to rotation matrix
            x, y, z, w = orientation
            rotation_matrix = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
            ])

            # Create 4x4 transformation matrix
            camera_pose_matrix = np.eye(4)
            camera_pose_matrix[:3, :3] = rotation_matrix
            camera_pose_matrix[:3, 3] = position

            self.logger.debug(f"Successfully looked up transform from {source_frame} to {target_frame}")
            return camera_pose_matrix

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e:
            self.logger.warning(f"Could not get transform from {source_frame} to {target_frame}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting transform: {e}")
            return None


    def move_to(self, pose, relative=False, blocking=True, verbose=False, timeout=30.0):
        """
        Move the robot to a specified pose.
        
        Args:
            pose: Target pose [x, y, theta]
            relative: Whether the pose is relative to current pose
            blocking: Whether to block until the robot reaches the pose
            verbose: Whether to print debug info
            timeout: Timeout in seconds for blocking movements
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if connected
        if not self.is_connected():
            logger.error("Cannot move: not connected to ROS")
            return False
            
        # Calculate target pose
        if relative:
            current_pose = self.get_pose()
            # For relative movements, transform the pose
            # From robot frame to global frame
            cos_theta = np.cos(current_pose[2])
            sin_theta = np.sin(current_pose[2])
            
            dx = pose[0]
            dy = pose[1]
            
            target_x = current_pose[0] + dx * cos_theta - dy * sin_theta
            target_y = current_pose[1] + dx * sin_theta + dy * cos_theta
            target_theta = current_pose[2] + pose[2]
            
            target_pose = [target_x, target_y, target_theta]
        else:
            target_pose = pose
            
        if verbose:
            logger.info(f"Moving to pose: {target_pose}")
            
        # Simple implementation of movement using velocity commands
        # A more sophisticated implementation would use a motion planning library
        
        if blocking:
            return self._move_to_blocking(target_pose, timeout, verbose)
        else:
            return self._move_to_non_blocking(target_pose)
            
    def _move_to_blocking(self, target_pose, timeout, verbose):
        """
        Move to target pose and block until reached or timeout.
        
        Args:
            target_pose: Target pose [x, y, theta]
            timeout: Timeout in seconds
            verbose: Whether to print debug info
            
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        rate = 0.1  # seconds
        
        while time.time() - start_time < timeout:
            # Check if still connected
            if not self.is_connected():
                logger.error("Connection lost during movement")
                return False
                
            current_pose = self.get_pose()
            
            # Calculate distance and angle to target
            dx = target_pose[0] - current_pose[0]
            dy = target_pose[1] - current_pose[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Calculate angle to target
            target_angle = np.arctan2(dy, dx)
            angle_diff = self._normalize_angle(target_angle - current_pose[2])
            
            # Calculate final heading difference
            final_angle_diff = self._normalize_angle(target_pose[2] - current_pose[2])
            
            if verbose:
                logger.debug(f"Current: {current_pose}, Target: {target_pose}")
                logger.debug(f"Distance: {distance}, Angle diff: {angle_diff}, Final angle diff: {final_angle_diff}")
                
            # Check if we've reached the target
            position_tolerance = self.config.get("navigation", {}).get("goal_tolerance_position", 0.1)  # meters
            angle_tolerance = self.config.get("navigation", {}).get("goal_tolerance_orientation", 0.1)  # radians
            
            if distance < position_tolerance and abs(final_angle_diff) < angle_tolerance:
                # Stop the robot
                self.stop_robot()
                if verbose:
                    logger.info("Reached target pose")
                return True
                
            # Determine if we need to rotate to face the target or move towards it
            if abs(angle_diff) > angle_tolerance and distance > position_tolerance:
                # Rotate to face the target
                angular_speed = self.config.get("robot", {}).get("max_angular_speed", 0.5)  # rad/s
                
                if angle_diff > 0:
                    angular_vel = min(angular_speed, abs(angle_diff))
                else:
                    angular_vel = -min(angular_speed, abs(angle_diff))
                    
                self.send_velocity_command(0.0, angular_vel)
                
            elif distance > position_tolerance:
                # Move towards the target
                linear_speed = self.config.get("robot", {}).get("max_linear_speed", 0.5)  # m/s
                linear_vel = min(linear_speed, distance)
                
                # Keep adjusting orientation
                if abs(angle_diff) > angle_tolerance / 2:
                    angular_vel = 0.3 * angle_diff
                else:
                    angular_vel = 0.0
                    
                self.send_velocity_command(linear_vel, angular_vel)
                
            else:
                # We're at the right position, just correct the final orientation
                angular_speed = self.config.get("robot", {}).get("max_angular_speed", 0.5)  # rad/s
                
                if final_angle_diff > 0:
                    angular_vel = min(angular_speed, abs(final_angle_diff))
                else:
                    angular_vel = -min(angular_speed, abs(final_angle_diff))
                    
                self.send_velocity_command(0.0, angular_vel)
                
            # Sleep before the next control cycle
            time.sleep(rate)
            
        # Timeout reached, stop the robot
        self.stop_robot()
        logger.warning(f"Timeout reached after {timeout} seconds")
        return False
        
    def _move_to_non_blocking(self, target_pose):
        """
        Start moving to target pose without blocking.
        
        Args:
            target_pose: Target pose [x, y, theta]
            
        Returns:
            bool: True if command was sent successfully
        """
        # Check if connected
        if not self.is_connected():
            logger.error("Cannot move: not connected to ROS")
            return False
            
        current_pose = self.get_pose()
        
        # Calculate direction to target
        dx = target_pose[0] - current_pose[0]
        dy = target_pose[1] - current_pose[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate angle to target
        target_angle = np.arctan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - current_pose[2])
        
        # Calculate velocities
        linear_speed = self.config.get("robot", {}).get("max_linear_speed", 0.5)  # m/s
        angular_speed = self.config.get("robot", {}).get("max_angular_speed", 0.5)  # rad/s
        
        linear_vel = min(linear_speed, distance)
        
        if abs(angle_diff) > 0.1:
            # If angle difference is significant, first rotate
            angular_vel = angular_speed if angle_diff > 0 else -angular_speed
            linear_vel = 0.0
        else:
            # Otherwise, combine linear and angular motion
            angular_vel = angle_diff
            
        self.send_velocity_command(linear_vel, angular_vel)
        return True
        
    def send_velocity_command(self, linear_velocity, angular_velocity):
        """
        Send a velocity command to the robot.
        
        Args:
            linear_velocity: Linear velocity in m/s
            angular_velocity: Angular velocity in rad/s
        """
        # Check if connected
        if not self.is_connected():
            logger.error("Cannot send velocity command: not connected to ROS")
            return
            
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity
        
        try:
            self.cmd_vel_pub.publish(cmd)
        except Exception as e:
            logger.error(f"Error sending velocity command: {e}")
        
    def stop_robot(self):
        """Stop the robot by sending zero velocity commands."""
        self.send_velocity_command(0.0, 0.0)
        
    def _normalize_angle(self, angle):
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
        
    def get_observation(self) -> Optional[Observations]:
        """
        Retrieves the latest synchronized observation data including RGB, depth,
        intrinsics, LiDAR, IMU, and odometry. Returns a standard Observations object.

        Returns:
            Optional[Observations]: An observation object containing
            sensor data, or None if data is not ready or valid.
        """
        self.logger.debug("Entering get_observation()")
        cv_rgb = None
        cv_depth = None
        camera_K = None
        camera_pose_matrix = None
        current_pose_xyt = None
        odom_msg = None
        scan_msg = None
        imu_msg = None
        battery_msg = None
        point_cloud_msg = None # Raw PointCloud2 msg

        with self.lock:
            self.logger.debug("Acquired lock in get_observation()")
            # Check if critical data is available
            if not all([self.last_rgb, self.last_depth, self.last_camera_info, self.last_odom]):
                self.logger.debug("Critical sensor data (RGB, Depth, CameraInfo, Odom) not fully available yet.")
                return None
            self.logger.debug("Critical sensor data available")

            # Copy data to prevent race conditions outside the lock
            rgb_msg = self.last_rgb
            depth_msg = self.last_depth
            camera_info_msg = self.last_camera_info
            odom_msg = self.last_odom
            # Optional data
            scan_msg = self.last_scan
            imu_msg = self.last_imu
            battery_msg = self.last_battery
            point_cloud_msg = self.last_point_cloud_msg
            current_pose_xyt = self.current_pose_xyt.copy()


        # --- Process Camera Data ---
        if self.cv_bridge is None:
            self.logger.error("CV Bridge not available for image conversion")
            return None

        self.logger.debug("Starting image and intrinsics processing")
        try:
            # Convert ROS Image messages to OpenCV images
            cv_rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

            # Handle depth encoding
            if depth_msg.encoding == '16UC1':
                cv_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                cv_depth = cv_depth.astype(np.float32) / 1000.0 # mm to meters
            elif depth_msg.encoding == '32FC1':
                 cv_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            else:
                 self.logger.error(f"Unsupported depth encoding: {depth_msg.encoding}")
                 return None

            # Extract camera intrinsics
            camera_K = np.array(camera_info_msg.k).reshape(3, 3)

        except Exception as e:
            self.logger.error(f"CV Bridge conversion or intrinsics extraction failed: {e}")
            traceback.print_exc()
            return None
        self.logger.debug("Image and intrinsics processing successful")

        # --- Get Camera Pose ---
        self.logger.debug("Attempting to get camera pose via TF lookup")
        camera_pose_matrix = self._get_camera_pose()
        if camera_pose_matrix is None:
            self.logger.warning("Failed to get camera pose from TF.")
            # Decide if this is critical. For now, we'll allow observations without it.
            # return None # Uncomment if camera pose is strictly required

        # --- Process LiDAR/PointCloud Data ---
        processed_point_cloud = None

        if point_cloud_msg is not None:
            self.logger.debug("Processing PointCloud2 message")
            try:
                processed_point_cloud = self.process_point_cloud2(point_cloud_msg)
                if processed_point_cloud is not None:
                    lidar_points = processed_point_cloud.points
                    lidar_timestamp = processed_point_cloud.timestamp
                    self.logger.debug(f"Processed PointCloud2 into {lidar_points.shape[0]} points")
                else:
                    self.logger.warning("PointCloud2 processing failed")
            except Exception as e:
                self.logger.error(f"Error processing PointCloud2 message: {e}")
                processed_point_cloud = None

        # Fall back to LaserScan if PointCloud2 processing failed
        if processed_point_cloud is None and scan_msg is not None:
            self.logger.debug("Processing LaserScan message")
            try:
                ranges = np.array(scan_msg.ranges)
                angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
                valid = np.isfinite(ranges) & (ranges > scan_msg.range_min) & (ranges < scan_msg.range_max)
                x = ranges[valid] * np.cos(angles[valid])
                y = ranges[valid] * np.sin(angles[valid])
                z = np.zeros_like(x) # LaserScan is 2D
                lidar_points = np.column_stack((x, y, z))
                lidar_timestamp = scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec * 1e-9
                processed_point_cloud = PointCloud(
                    points=lidar_points,
                    frame_id=scan_msg.header.frame_id,
                    timestamp=lidar_timestamp,
                )
                self.logger.debug(f"Processed LaserScan into {lidar_points.shape[0]} points.")
            except Exception as e:
                self.logger.error(f"Error processing LaserScan message: {e}")
                lidar_points = None
                lidar_timestamp = None
                processed_point_cloud = None


        # --- Extract Other Sensor Data ---
        wheel_odometry = None
        if odom_msg is not None:
             wheel_odometry = np.array([
                 odom_msg.twist.twist.linear.x,
                 odom_msg.twist.twist.linear.y,
                 odom_msg.twist.twist.angular.z
             ])

        imu_data_dict = None
        if imu_msg is not None:
            try:
                imu_data_dict = {
                    'orientation': np.array([
                        imu_msg.orientation.x,
                        imu_msg.orientation.y,
                        imu_msg.orientation.z,
                        imu_msg.orientation.w
                    ]),
                    'angular_velocity': np.array([
                        imu_msg.angular_velocity.x,
                        imu_msg.angular_velocity.y,
                        imu_msg.angular_velocity.z
                    ]),
                    'linear_acceleration': np.array([
                        imu_msg.linear_acceleration.x,
                        imu_msg.linear_acceleration.y,
                        imu_msg.linear_acceleration.z
                    ])
                }
            except Exception as e:
                self.logger.error(f"Error processing IMU data: {e}")
                imu_data_dict = None

        battery_status_dict = None
        if battery_msg is not None:
             battery_status_dict = {
                 'voltage': battery_msg.voltage,
                 'current': battery_msg.current,
                 'percentage': battery_msg.percentage,
                 'power_supply_status': battery_msg.power_supply_status
             }

        # --- Build Standard Observations Object ---
        self.logger.debug("Building standard Observations object")
        try:
            # Ensure current_pose_xyt is valid
            if current_pose_xyt is None:
                self.logger.error("Cannot build Observations: current_pose_xyt is None")
                return None

            # GPS and Compass derived from odometry pose
            gps = np.array([current_pose_xyt[0], current_pose_xyt[1]])
            compass = np.array([current_pose_xyt[2]])

            observation = Observations(
                # Pose
                gps=gps,
                compass=compass,
                # Camera
                rgb=cv_rgb, # HxWx3 numpy array
                depth=cv_depth, # HxW numpy array (meters)
                camera_K=camera_K, # 3x3 numpy array
                camera_pose=camera_pose_matrix, # 4x4 numpy array (or None)
                # Lidar
                lidar_points=lidar_points, # Nx3 numpy array (or None)
                lidar_timestamp=lidar_timestamp, # float (or None)
                point_cloud=processed_point_cloud, # stretch.core.interfaces.PointCloud (or None)
                # Segway specific
                wheel_odometry=wheel_odometry, # 3-element numpy array (or None)
                imu_data=imu_data_dict, # dict (or None)
                battery_status=battery_status_dict, # dict (or None)
                # Other fields (defaulting to None or appropriate value)
                xyz=None, # Can be computed later if needed
                semantic=None,
                instance=None,
                ee_rgb=None,
                ee_depth=None,
                ee_xyz=None,
                ee_semantic=None,
                ee_camera_K=None,
                ee_camera_pose=None,
                ee_pose=None,
                third_person_image=None,
                joint=None,
                joint_velocities=None,
                relative_resting_position=None,
                is_holding=None,
                task_observations=None,
                seq_id=rgb_msg.header.stamp.sec, # Use timestamp as seq_id proxy
                is_simulation=False,
                is_pose_graph_node=False,
                pose_graph_timestamp=None,
                initial_pose_graph_gps=None,
                initial_pose_graph_compass=None,
            )
            self.logger.debug("Standard Observations object created successfully")
            return observation
        except Exception as e:
             self.logger.error(f"Failed to create Observations object: {e}")
             traceback.print_exc()
             return None

    def get_latest_lidar_data_timestamp(self) -> Optional[float]:
        """
        Returns the timestamp of the last received LiDAR data (Scan or PointCloud2).
        
        Returns:
            float: Timestamp in seconds or None if no data received
        """
        try:
            lidar_timestamp = None
            
            # Check scan message
            if hasattr(self, 'last_scan') and self.last_scan is not None:
                scan_timestamp = self.last_scan.header.stamp.sec + self.last_scan.header.stamp.nanosec * 1e-9
                lidar_timestamp = scan_timestamp
                
            # Check point cloud message (if available and more recent)
            if hasattr(self, 'last_point_cloud_msg') and self.last_point_cloud_msg is not None:
                pc_timestamp = self.last_point_cloud_msg.header.stamp.sec + self.last_point_cloud_msg.header.stamp.nanosec * 1e-9
                
                # Use point cloud timestamp if it's more recent or if scan timestamp is not available
                if lidar_timestamp is None or pc_timestamp > lidar_timestamp:
                    lidar_timestamp = pc_timestamp
                    
            return lidar_timestamp
        except Exception as e:
            print(f"Error getting lidar timestamp: {e}")
            return None

    def get_latest_camera_data_timestamp(self) -> Optional[float]:
        """
        Returns the timestamp of the last received camera data (RGB, Depth, or CameraInfo).
        
        Returns:
            float: Timestamp in seconds or None if no data received
        """
        try:
            timestamps = []
            
            # Check RGB message
            if hasattr(self, 'last_rgb') and self.last_rgb is not None:
                rgb_timestamp = self.last_rgb.header.stamp.sec + self.last_rgb.header.stamp.nanosec * 1e-9
                timestamps.append(rgb_timestamp)
                
            # Check depth message
            if hasattr(self, 'last_depth') and self.last_depth is not None:
                depth_timestamp = self.last_depth.header.stamp.sec + self.last_depth.header.stamp.nanosec * 1e-9
                timestamps.append(depth_timestamp)
                
            # Check camera info message
            if hasattr(self, 'last_camera_info') and self.last_camera_info is not None:
                info_timestamp = self.last_camera_info.header.stamp.sec + self.last_camera_info.header.stamp.nanosec * 1e-9
                timestamps.append(info_timestamp)
                
            return max(timestamps) if timestamps else None
        except Exception as e:
            print(f"Error getting camera timestamp: {e}")
            return None