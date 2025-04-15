"""
Segway Perception Adapter for Stretch AI

This module implements an adapter for integrating the Segway robot's
sensors with Stretch AI's advanced perception capabilities.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import queue
import torch

from stretch.core.interfaces import Observations, PointCloud
from stretch.perception.constants import COCO_CLASSES, CUSTOM_CLASSES
from stretch.perception.wrapper import OvmmPerception
from stretch.perception.detection.detic.detic_perception import DeticPerception
from stretch.perception.detection.owl.owl_perception import OwlPerception
from stretch.perception.captioners.blip_captioner import BlipCaptioner
from stretch.perception.detection.mobile_sam import MobileSAM
from stretch.utils.logger import Logger

logger = Logger(__name__)


class SegwayPerceptionAdapter:
    """
    Adapter for integrating Segway robot sensors with Stretch AI's
    perception framework. Optimized for RTX 4090 GPU acceleration.
    """
    
    def __init__(self, parameters, gpu_device="cuda:0"):
        """
        Initialize the Segway Perception Adapter.
        
        Args:
            parameters: Configuration parameters
            gpu_device: GPU device to use for acceleration
        """
        self.parameters = parameters
        self.gpu_device = gpu_device
        self.perception_config = parameters.get("perception", {})
        
        # Default perception mode
        self.perception_mode = self.perception_config.get("mode", "detic")
        
        # Object detection threshold
        self.confidence_threshold = self.perception_config.get("confidence_threshold", 0.3)
        
        # Enhanced classes for detection
        self.custom_classes = self.perception_config.get("custom_classes", []) + CUSTOM_CLASSES
        
        # Latest perception data
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_point_cloud = None
        self.latest_detections = []
        self.latest_caption = None
        
        # For thread safety
        self.perception_lock = threading.Lock()
        self.perception_thread = None
        self.perception_queue = queue.Queue(maxsize=2)  # Buffer size of 2
        self.running = False
        
        # Initialize perception models
        self._initialize_perception_models()
        
        logger.info("SegwayPerceptionAdapter initialized")


    def _filter_kwargs(self, cls, kwargs):
        """
        Filter keyword arguments to only include those accepted by the class constructor.
        
        Args:
            cls: Class to check arguments for
            kwargs: Dictionary of keyword arguments
            
        Returns:
            dict: Filtered keyword arguments dictionary
        """
        import inspect
        
        try:
            # Get the signature of the __init__ method
            sig = inspect.signature(cls.__init__)
            # Get the parameter names
            valid_params = list(sig.parameters.keys())
            # Remove 'self'
            if 'self' in valid_params:
                valid_params.remove('self')
            
            # Filter the kwargs
            filtered = {k: v for k, v in kwargs.items() if k in valid_params}
            
            # Log removed parameters
            removed = set(kwargs.keys()) - set(filtered.keys())
            if removed:
                logger.warning(f"Removed incompatible parameters for {cls.__name__}: {removed}")
                
            return filtered
        except Exception as e:
            logger.error(f"Error filtering kwargs for {cls.__name__}: {e}")
            return kwargs

        
    def _initialize_perception_models(self):
        """Initialize perception models based on configuration."""
        try:
            logger.info(f"Initializing perception models on device: {self.gpu_device}")
            
            # Main perception model
            if self.perception_mode == "detic":
                self.perception_model = self._initialize_detic()
            elif self.perception_mode == "owl":
                self.perception_model = self._initialize_owl()
            else:
                # Use OVMM as default for flexibility
                self.perception_model = self._initialize_ovmm()
                
            # Captioner model
            self.captioner = self._initialize_captioner()
            
            # Segmentation model
            if self.perception_config.get("use_segmentation", True):
                self.segmentation_model = self._initialize_segmentation()
            else:
                self.segmentation_model = None
                
            logger.info("Perception models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing perception models: {e}")
            self.perception_model = None
            self.captioner = None
            self.segmentation_model = None
    
    def _initialize_detic(self):
        """Initialize Detic object detection model."""
        try:
            logger.info("Initializing Detic with minimal parameters")
            # Try with no parameters first
            return DeticPerception()
        except Exception as e:
            logger.warning(f"Error initializing Detic with no parameters: {e}")
            # Try with just threshold
            try:
                return DeticPerception(threshold=self.confidence_threshold)
            except Exception as e:
                logger.error(f"Error initializing Detic: {e}")
                return None
    
    def _initialize_owl(self):
        """Initialize OWL-ViT object detection model."""
        try:
            config = {
                "confidence_threshold": self.confidence_threshold,
                "device": self.gpu_device,
                "classes": self.custom_classes if self.custom_classes else COCO_CLASSES,
                "cache_dir": self.perception_config.get("cache_dir", None),
                "use_tensorrt": self.perception_config.get("use_tensorrt", False)
            }
            
            logger.info(f"Initializing OWL-ViT with config: {config}")
            return OwlPerception(**config)
        except Exception as e:
            logger.error(f"Error initializing OWL-ViT: {e}")
            return None
    
    def _initialize_ovmm(self):
        """Initialize OVMM perception model."""
        try:
            config = {
                "model_type": self.perception_config.get("model_type", "ovmm"),
                "device": self.gpu_device,
                "max_batch_size": self.perception_config.get("max_batch_size", 4),
                "use_tensorrt": self.perception_config.get("use_tensorrt", True),
                "precision": self.perception_config.get("precision", "fp16"),
                "cache_dir": self.perception_config.get("cache_dir", "~/.cache/stretch/models")
            }
            
            logger.info(f"Initializing OVMM with config: {config}")
            return OvmmPerception(**config)
        except Exception as e:
            logger.error(f"Error initializing OVMM: {e}")
            return None
    
    def _initialize_captioner(self):
        """Initialize image captioning model."""
        try:
            captioner_config = self.perception_config.get("captioner", {})
            captioner_type = captioner_config.get("type", "blip")
            
            if captioner_type == "blip":
                try:
                    from stretch.perception.captioning.blip import BlipCaptioner
                    
                    # Only pass parameters that BlipCaptioner accepts
                    logger.info(f"Initializing BLIP captioner with device: {self.gpu_device}")
                    
                    # Filter parameters to only include those the BlipCaptioner constructor accepts
                    safe_params = {
                        'device': self.gpu_device,
                        'use_tensorrt': captioner_config.get("use_tensorrt", False)
                    }
                    
                    # Remove cache_dir if present in captioner_config
                    if "cache_dir" in captioner_config:
                        logger.info(f"Ignoring cache_dir parameter - not supported by BlipCaptioner")
                    
                    self.captioner = BlipCaptioner(**safe_params)
                    logger.info("BLIP captioner initialized successfully")
                except ImportError:
                    logger.error("Failed to import BlipCaptioner. BLIP captioning will be unavailable.")
                except Exception as e:
                    logger.error(f"Error initializing captioner: {e}")
                    return None
            else:
                logger.warning(f"Captioner type {captioner_type} not supported, using BLIP")
                return BlipCaptioner(device=self.gpu_device)
        except Exception as e:
            logger.error(f"Error initializing captioner: {e}")
            return None
    
    def _initialize_segmentation(self):
        """Initialize segmentation model."""
        try:
            # Use the correct absolute path
            weights_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "src/stretch_ai/src/stretch/perception/detection/weights/mobile_sam.pt"
            )
            
            logger.info(f"Initializing MobileSAM with weights path: {weights_path}")
            
            # Try with no parameters first
            return MobileSAM()
        except Exception as e:
            logger.error(f"Error initializing segmentation model: {e}")
            return None

    def start(self):
        """Start the perception adapter."""
        if self.running:
            logger.warning("SegwayPerceptionAdapter already running")
            return
            
        self.running = True
        self.perception_thread = threading.Thread(target=self._perception_thread_func)
        self.perception_thread.daemon = True
        self.perception_thread.start()
        
        logger.info("SegwayPerceptionAdapter started")
    
    def stop(self):
        """Stop the perception adapter."""
        self.running = False
        if self.perception_thread:
            self.perception_thread.join(timeout=2.0)
            self.perception_thread = None
            
        logger.info("SegwayPerceptionAdapter stopped")
    
    def _perception_thread_func(self):
        """Background thread for continuous perception processing."""
        logger.info("Perception thread started")
        
        while self.running:
            try:
                # Get item from queue with timeout
                item = None
                try:
                    item = self.perception_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the observation
                if isinstance(item, Observations):
                    self._process_observation(item)
                else:
                    logger.warning(f"Unknown item type in perception queue: {type(item)}")
                
                # Mark item as processed
                self.perception_queue.task_done()
            except Exception as e:
                logger.error(f"Error in perception thread: {e}")
        
        logger.info("Perception thread stopped")
    
    def update_perception(self, observation):
        """
        Update perception with new observation.
        
        Args:
            observation: New observation
        """
        if not self.running:
            # If not running in background mode, process immediately
            return self._process_observation(observation)
            
        # Otherwise, add to queue for background processing
        try:
            # Put with a timeout to avoid blocking
            self.perception_queue.put(observation, timeout=0.1)
        except queue.Full:
            logger.warning("Perception queue full, dropping observation")
    
    def _process_observation(self, observation):
        """
        Process an observation for perception.
        
        Args:
            observation: Observation to process
        """
        if observation is None:
            logger.warning("Received None observation in process_observation")
            return None
            
        # Get RGB and depth from observation
        rgb = None
        depth = None
        point_cloud = None
        
        if hasattr(observation, 'rgb') and observation.rgb is not None:
            rgb = observation.rgb
        
        if hasattr(observation, 'depth') and observation.depth is not None:
            depth = observation.depth
        
        if hasattr(observation, 'point_cloud') and observation.point_cloud is not None:
            point_cloud = observation.point_cloud
        elif hasattr(observation, 'lidar_points') and observation.lidar_points is not None:
            # Create point cloud from lidar points
            if hasattr(observation, 'process_lidar_to_pointcloud'):
                point_cloud = observation.process_lidar_to_pointcloud()
        
        # Update latest data
        with self.perception_lock:
            self.latest_rgb = rgb
            self.latest_depth = depth
            self.latest_point_cloud = point_cloud
        
        # Only run object detection if we have RGB
        if rgb is not None and self.perception_model is not None:
            self._run_object_detection(rgb)
        
        # Run captioning if we have RGB
        if rgb is not None and self.captioner is not None:
            self._run_captioning(rgb)
        
        return observation
    
    def _run_object_detection(self, rgb):
        """
        Run object detection on an RGB image.
        
        Args:
            rgb: RGB image
        """
        try:
            logger.debug("Running object detection")
            
            results = self.perception_model.predict(rgb)
            
            if results:
                # Filter by confidence threshold
                filtered_results = [
                    r for r in results 
                    if hasattr(r, 'confidence') and r.confidence >= self.confidence_threshold
                ]
                
                with self.perception_lock:
                    self.latest_detections = filtered_results
                    
                logger.debug(f"Detected {len(filtered_results)} objects")
            else:
                logger.debug("No objects detected")
                with self.perception_lock:
                    self.latest_detections = []
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
    
    def _run_captioning(self, rgb):
        """
        Run image captioning on an RGB image.
        
        Args:
            rgb: RGB image
        """
        try:
            logger.debug("Running image captioning")
            
            caption = self.captioner.predict(rgb)
            
            with self.perception_lock:
                self.latest_caption = caption
                
            logger.debug(f"Generated caption: {caption}")
        except Exception as e:
            logger.error(f"Error in image captioning: {e}")
    
    def detect_objects(self):
        """
        Get latest object detections.
        
        Returns:
            list: List of detected objects
        """
        with self.perception_lock:
            return self.latest_detections.copy() if self.latest_detections else []
    
    def get_rgb_image(self):
        """
        Get latest RGB image.
        
        Returns:
            np.ndarray: RGB image
        """
        with self.perception_lock:
            return self.latest_rgb
    
    def get_depth_image(self):
        """
        Get latest depth image.
        
        Returns:
            np.ndarray: Depth image
        """
        with self.perception_lock:
            return self.latest_depth
    
    def get_point_cloud(self):
        """
        Get the latest point cloud data from various possible sources.
        
        Returns:
            PointCloud: A valid point cloud object or None if not available
        """
        try:
            # Check if we already have processed point cloud
            if hasattr(self, '_last_point_cloud') and self._last_point_cloud is not None:
                return self._last_point_cloud
            
            # Try to get from ROS client
            if hasattr(self, 'ros_client') and self.ros_client is not None:
                # Get latest observation
                obs = self.ros_client.get_observation()
                
                if obs is not None:
                    # First try point_cloud attribute if available
                    if hasattr(obs, 'point_cloud') and obs.point_cloud is not None:
                        self._last_point_cloud = obs.point_cloud
                        return obs.point_cloud
                        
                    # Next try raw PointCloud2 processing
                    if hasattr(self.ros_client, 'last_point_cloud_msg') and self.ros_client.last_point_cloud_msg is not None:
                        if hasattr(self.ros_client, 'process_point_cloud2'):
                            point_cloud = self.ros_client.process_point_cloud2(self.ros_client.last_point_cloud_msg)
                            if point_cloud is not None:
                                self._last_point_cloud = point_cloud
                                return point_cloud
                            
                    # Next try LaserScan processing
                    if hasattr(self.ros_client, 'last_scan') and self.ros_client.last_scan is not None:
                        # Process LaserScan to point cloud
                        point_cloud = self._create_point_cloud_from_laser_scan(self.ros_client.last_scan)
                        if point_cloud is not None:
                            self._last_point_cloud = point_cloud
                            return point_cloud
                            
                    # Finally try existing lidar_points attribute
                    if hasattr(obs, 'lidar_points') and obs.lidar_points is not None:
                        point_cloud = PointCloud(
                            points=obs.lidar_points,
                            frame_id='base_link',
                            timestamp=time.time()
                        )
                        self._last_point_cloud = point_cloud
                        return point_cloud
            
            logger.warning("No point cloud data available from any source")
            return None
        except Exception as e:
            logger.error(f"Error getting point cloud: {e}")
            return None
        
    def _create_point_cloud_from_laser_scan(self, scan_msg):
        """
        Create a point cloud from a LaserScan message.
        
        Args:
            scan_msg: LaserScan message
            
        Returns:
            PointCloud: Point cloud object or None on failure
        """
        try:
            ranges = np.array(scan_msg.ranges)
            angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
            
            # Filter valid ranges
            valid = np.isfinite(ranges)
            if hasattr(scan_msg, 'range_min') and hasattr(scan_msg, 'range_max'):
                valid = valid & (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
                
            x = ranges[valid] * np.cos(angles[valid])
            y = ranges[valid] * np.sin(angles[valid])
            z = np.zeros_like(x)  # LaserScan is 2D
            
            points = np.column_stack((x, y, z))
            
            # Create point cloud
            point_cloud = PointCloud(
                points=points,
                frame_id=scan_msg.header.frame_id,
                timestamp=scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec * 1e-9
            )
            
            return point_cloud
        except Exception as e:
            logger.error(f"Error creating point cloud from laser scan: {e}")
            return None
    
    def caption_image(self):
        """
        Get latest image caption.
        
        Returns:
            str: Image caption
        """
        with self.perception_lock:
            return self.latest_caption if self.latest_caption else "No caption available"
    
    def segment_foreground(self, rgb, box=None):
        """
        Segment foreground objects in an RGB image.
        
        Args:
            rgb: RGB image
            box: Optional bounding box to focus segmentation
            
        Returns:
            Wrapped segmentation mask(s) with get_rotated_mask method
        """
        if self.segmentation_model is None or rgb is None:
            return None
            
        try:
            masks = self.segmentation_model.predict(rgb, box)
            return self._wrap_masks(masks)
        except Exception as e:
            logger.error(f"Error in segmentation: {e}")
            return None
    
    def get_objects_3d_positions(self, rgb=None, depth=None, detections=None):
        """
        Get 3D positions of detected objects.
        
        Args:
            rgb: Optional RGB image (uses latest if None)
            depth: Optional depth image (uses latest if None)
            detections: Optional detections (uses latest if None)
            
        Returns:
            dict: Mapping from object ID to 3D position
        """
        if rgb is None:
            rgb = self.get_rgb_image()
            
        if depth is None:
            depth = self.get_depth_image()
            
        if detections is None:
            detections = self.detect_objects()
            
        if rgb is None or depth is None or not detections:
            return {}
            
        try:
            # Ensure depth and RGB have same dimensions
            if rgb.shape[:2] != depth.shape[:2]:
                logger.warning(f"RGB shape {rgb.shape[:2]} and depth shape {depth.shape[:2]} mismatch")
                return {}
                
            # Get camera intrinsics (placeholder - actual implementation depends on your setup)
            # Should be obtained from camera_info or calibration data
            fx, fy = 500.0, 500.0  # Placeholder values
            cx, cy = rgb.shape[1] / 2, rgb.shape[0] / 2  # Center of image
            
            object_positions = {}
            
            for i, detection in enumerate(detections):
                if hasattr(detection, 'bbox'):
                    # Get bounding box
                    x1, y1, x2, y2 = detection.bbox
                    
                    # Convert to integers and ensure within image bounds
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(rgb.shape[1] - 1, int(x2))
                    y2 = min(rgb.shape[0] - 1, int(y2))
                    
                    # Calculate center point of bbox
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Get depth at center point
                    center_depth = depth[center_y, center_x]
                    
                    # Skip if depth is invalid
                    if center_depth <= 0 or not np.isfinite(center_depth):
                        continue
                    
                    # Convert to meters if needed (depends on your depth format)
                    depth_meters = center_depth / 1000.0 if center_depth > 1000 else center_depth
                    
                    # Calculate 3D position
                    x = (center_x - cx) * depth_meters / fx
                    y = (center_y - cy) * depth_meters / fy
                    z = depth_meters
                    
                    # Store position
                    object_positions[i] = (x, y, z)
                    
                    # Add 3D position to detection if it has such attribute
                    if hasattr(detection, 'position'):
                        detection.position = (x, y, z)
            
            return object_positions
        except Exception as e:
            logger.error(f"Error getting object 3D positions: {e}")
            return {}
    
    def project_point_cloud_to_image(self, point_cloud, camera_matrix, image_shape):
        """
        Project a point cloud to an image.
        
        Args:
            point_cloud: Point cloud
            camera_matrix: Camera matrix
            image_shape: Image shape (height, width)
            
        Returns:
            np.ndarray: Projected points (N, 2)
            np.ndarray: Depths (N)
        """
        if point_cloud is None or not hasattr(point_cloud, 'points') or point_cloud.points.shape[0] == 0:
            return None, None
            
        try:
            # Extract camera intrinsics
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            # Extract points
            points = point_cloud.points
            
            # Project points to image
            # [X, Y, Z] -> [x, y, depth]
            z = points[:, 2]
            x = points[:, 0] / z * fx + cx
            y = points[:, 1] / z * fy + cy
            
            # Filter points outside image
            valid = (x >= 0) & (x < image_shape[1]) & (y >= 0) & (y < image_shape[0]) & (z > 0)
            
            # Return projected points and depths
            return np.column_stack((x[valid], y[valid])), z[valid]
        except Exception as e:
            logger.error(f"Error projecting point cloud to image: {e}")
            return None, None
    
    def segment_objects_in_point_cloud(self, point_cloud, rgb=None, depth=None):
        """
        Segment objects in a point cloud using RGB and depth data.
        
        Args:
            point_cloud: Point cloud
            rgb: RGB image
            depth: Depth image
            
        Returns:
            dict: Mapping from object ID to segmented point cloud
        """
        if point_cloud is None or point_cloud.points.shape[0] == 0:
            return {}
            
        # Use latest RGB and depth if not provided
        if rgb is None:
            rgb = self.get_rgb_image()
            
        if depth is None:
            depth = self.get_depth_image()
            
        if rgb is None or depth is None:
            return {}
            
        try:
            # Get detections
            detections = self.detect_objects()
            
            if not detections:
                return {}
                
            # Get camera matrix (placeholder - actual implementation depends on your setup)
            camera_matrix = np.array([
                [500.0, 0, rgb.shape[1] / 2],
                [0, 500.0, rgb.shape[0] / 2],
                [0, 0, 1]
            ])
            
            # Project point cloud to image
            projected_points, depths = self.project_point_cloud_to_image(
                point_cloud, camera_matrix, rgb.shape[:2]
            )
            
            if projected_points is None or projected_points.shape[0] == 0:
                return {}
                
            # Segment objects
            segmented_clouds = {}
            
            for i, detection in enumerate(detections):
                if hasattr(detection, 'bbox'):
                    # Get bounding box
                    x1, y1, x2, y2 = detection.bbox
                    
                    # Convert to integers and ensure within image bounds
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(rgb.shape[1] - 1, int(x2))
                    y2 = min(rgb.shape[0] - 1, int(y2))
                    
                    # Find points inside bounding box
                    inside_bbox = (
                        (projected_points[:, 0] >= x1) &
                        (projected_points[:, 0] <= x2) &
                        (projected_points[:, 1] >= y1) &
                        (projected_points[:, 1] <= y2)
                    )
                    
                    # Get indices of points inside bounding box
                    inside_indices = np.where(inside_bbox)[0]
                    
                    if inside_indices.shape[0] > 0:
                        # Create new point cloud with points inside bounding box
                        segmented_cloud = PointCloud(
                            points=point_cloud.points[inside_indices],
                            colors=point_cloud.colors[inside_indices] if point_cloud.colors is not None else None,
                            intensities=point_cloud.intensities[inside_indices] if point_cloud.intensities is not None else None,
                            frame_id=point_cloud.frame_id,
                            timestamp=point_cloud.timestamp,
                            height=1,
                            width=inside_indices.shape[0],
                            is_dense=True
                        )
                        
                        segmented_clouds[i] = segmented_cloud
            
            return segmented_clouds
        except Exception as e:
            logger.error(f"Error segmenting objects in point cloud: {e}")
            return {}

    def process_point_cloud(self, point_cloud_or_msg):
        """
        Process different types of point clouds into a standard format.
        
        Args:
            point_cloud_or_msg: Point cloud object or message of various types
            
        Returns:
            PointCloud: Standardized point cloud object or None on failure
        """
        try:
            # If None, return None
            if point_cloud_or_msg is None:
                return None
            
            # Case 1: Already a PointCloud from stretch.core.interfaces
            if isinstance(point_cloud_or_msg, PointCloud):
                return point_cloud_or_msg
            
            # Case 2: ROS PointCloud2 message
            if hasattr(point_cloud_or_msg, 'fields') and hasattr(point_cloud_or_msg, 'data'):
                # Try to extract points using sensor_msgs_py if available
                try:
                    from sensor_msgs_py import point_cloud2
                    points_list = list(point_cloud2.read_points(
                        point_cloud_or_msg, 
                        field_names=('x', 'y', 'z'),
                        skip_nans=True
                    ))
                    
                    if not points_list:
                        logger.warning("No valid points in PointCloud2 message")
                        return None
                        
                    # Convert to numpy array
                    points = np.array(points_list, dtype=np.float32)
                    
                    # Create standard PointCloud
                    return PointCloud(
                        points=points,
                        frame_id=point_cloud_or_msg.header.frame_id,
                        timestamp=point_cloud_or_msg.header.stamp.sec + point_cloud_or_msg.header.stamp.nanosec * 1e-9
                    )
                except ImportError:
                    logger.warning("sensor_msgs_py not available for PointCloud2 processing")
                    # Try manual processing...
                    # (Similar to what we did in SegwayROSClient._point_cloud_callback)
                    
            # Case 3: Try to access a 'points' attribute
            if hasattr(point_cloud_or_msg, 'points') and point_cloud_or_msg.points is not None:
                points = point_cloud_or_msg.points
                
                # Convert points to numpy array if needed
                if isinstance(points, torch.Tensor):
                    points = points.detach().cpu().numpy()
                elif isinstance(points, list):
                    points = np.array(points, dtype=np.float32)
                    
                # Create PointCloud
                return PointCloud(
                    points=points,
                    frame_id=getattr(point_cloud_or_msg, 'frame_id', 'map'),
                    timestamp=getattr(point_cloud_or_msg, 'timestamp', time.time())
                )
                
            # Case 4: NumPy array
            if isinstance(point_cloud_or_msg, np.ndarray):
                # Ensure it has the right shape (Nx3)
                if len(point_cloud_or_msg.shape) == 2 and point_cloud_or_msg.shape[1] >= 3:
                    return PointCloud(
                        points=point_cloud_or_msg[:, :3],  # Take first 3 columns as XYZ
                        frame_id='map',
                        timestamp=time.time()
                    )
            
            logger.warning(f"Unsupported point cloud type: {type(point_cloud_or_msg)}")
            return None
        except Exception as e:
            logger.error(f"Error processing point cloud: {e}")
            return None