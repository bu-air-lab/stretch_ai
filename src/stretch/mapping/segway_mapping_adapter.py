"""
Segway Mapping Adapter for Stretch AI

This module implements an adapter for integrating the Segway robot's
mapping capabilities with Stretch AI's advanced mapping framework.
Supports voxel-based mapping, instance tracking, scene graph representation,
and path planning optimized for RTX 4090 GPU acceleration.
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
import queue
import traceback
import cv2
import torch

from stretch.core.interfaces import Observations, PointCloud
from stretch.perception.segway_perception_adapter import SegwayPerceptionAdapter
from stretch.core.segway_ros_client import SegwayROSClient, CameraIntrinsics
from stretch.motion.algo import AStar,RRT
from stretch.motion.algo.rrt_connect import RRTConnect
from stretch.mapping.instance.instance_map import InstanceMemory
from stretch.mapping.voxel import SparseVoxelMap
from stretch.mapping.scene_graph.scene_graph import SceneGraph
from stretch.mapping.grid.grid import OccupancyGrid
from stretch.utils.logger import Logger

logger = Logger(__name__)


class SegwayMappingAdapter:
    """
    Adapter for integrating Segway robot's mapping capabilities with
    Stretch AI's advanced mapping framework. Optimized for RTX 4090 GPU acceleration.
    """
    
    def __init__(self, parameters, ros_client: SegwayROSClient, perception_adapter: SegwayPerceptionAdapter, gpu_device="cuda:0", encoder=None):
        """
        Initialize the Segway Mapping Adapter.
        
        Args:
            parameters: Configuration parameters
            ros_client: SegwayROSClient instance
            perception_adapter: SegwayPerceptionAdapter instance
            gpu_device: GPU device to use for acceleration
            encoder: Optional encoder instance (e.g., CLIP) for InstanceMemory
        """
        self.parameters = parameters
        self.ros_client = ros_client
        self.perception_adapter = perception_adapter
        self.gpu_device = gpu_device
        self.mapping_config = parameters.get("mapping", {})
        
        # Voxel map settings
        self.voxel_size = self.mapping_config.get("voxel_size", 0.05)
        self.max_voxels = self.mapping_config.get("max_voxels", 1000000)
        self.use_color = self.mapping_config.get("use_color", True)
        self.use_semantic = self.mapping_config.get("use_semantic", True)
        self.use_instance = self.mapping_config.get("use_instance", True)
        
        # Occupancy grid settings
        self.grid_resolution = self.mapping_config.get("grid_resolution", 0.05)
        self.grid_size = self.mapping_config.get("grid_size", [20.0, 20.0])  # Size in meters
        self.use_grid_map = self.mapping_config.get("use_grid_map", True)
        
        # Instance map settings
        self.use_instance_map = self.mapping_config.get("use_instance_map", True)
        self.instance_retention_time = self.mapping_config.get("instance_retention_time", 300.0)  # In seconds
        
        # Scene graph settings
        self.use_scene_graph = self.mapping_config.get("use_scene_graph", True)
        self.relation_threshold = self.mapping_config.get("relation_threshold", 1.5)  # In meters
        
        # Localization settings
        self.localization_method = self.mapping_config.get("localization_method", "odom")
        self.use_icp = self.mapping_config.get("use_icp", True)
        self.icp_max_iterations = self.mapping_config.get("icp_max_iterations", 50)
        self.icp_max_correspondence_distance = self.mapping_config.get("icp_max_correspondence_distance", 0.1)
        
        # Path planning settings
        self.planning_method = self.mapping_config.get("planning_method", "astar")
        self.obstacle_inflation = self.mapping_config.get("obstacle_inflation", 0.3)
        self.path_simplification = self.mapping_config.get("path_simplification", True)
        
        # Latest pose data
        self.latest_pose = (np.zeros(3), np.array([0, 0, 0, 1]))  # (position, orientation)
        self.pose_confidence = 1.0
        self.is_lost = False
        
        # For thread safety
        self.mapping_lock = threading.Lock()
        self.mapping_thread = None
        self.mapping_queue = queue.Queue(maxsize=2)  # Buffer size of 2
        self.running = False
        
        # Initialize mapping components
        self._initialize_mapping_components(encoder=encoder)
        
        logger.info("SegwayMappingAdapter initialized")
        
    def _initialize_mapping_components(self, encoder=None):
        """Initialize all mapping components."""
        try:
            logger.info(f"Initializing mapping components on device: {self.gpu_device}")
            
            # Create voxel map
            self.voxel_map = self._initialize_voxel_map(self.gpu_device)
            
            # Create occupancy grid
            self.grid_map = self._initialize_grid_map() if self.use_grid_map else None
            
            # Create instance map
            self.instance_map = self._initialize_instance_map(encoder=encoder) if self.use_instance_map else None
            
            # Create scene graph
            self.scene_graph = self._initialize_scene_graph() if self.use_scene_graph else None
            
            # Create planners
            self.astar_planner = None
            self.rrt_planner = None
            
            logger.info("Mapping components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing mapping components: {e}")
            self.voxel_map = None
            self.grid_map = None
            self.instance_map = None
            self.scene_graph = None
            self.astar_planner = None
            self.rrt_planner = None
    
    def _initialize_voxel_map(self, device):
        """Initialize the voxel map."""
        try:
            logger.info(f"Initializing voxel map with resolution: {self.voxel_size} on device {device}")
            
            self.voxel_map = SparseVoxelMap(
                resolution=self.voxel_size,
                device=device
            )
            
            logger.info("Voxel map initialized successfully")
            return self.voxel_map
        except Exception as e:
            logger.error(f"Error initializing voxel map: {e}")
            return None
    
    def _initialize_grid_map(self):
        """Initialize the 2D occupancy grid map."""
        try:
            logger.info(f"Initializing occupancy grid with resolution: {self.grid_resolution}")
            
            # Calculate grid dimensions
            width = int(self.grid_size[0] / self.grid_resolution)
            height = int(self.grid_size[1] / self.grid_resolution)
            
            # Create occupancy grid
            grid_map = OccupancyGrid(
                width=width,
                height=height,
                resolution=self.grid_resolution
            )
            
            logger.info("Occupancy grid initialized successfully")
            return grid_map
        except Exception as e:
            logger.error(f"Error initializing occupancy grid: {e}")
            return None
    
    def _initialize_instance_map(self, encoder=None):
        """Initialize instance map with optimized parameters for live tracking."""
        try:
            instance_map = InstanceMemory(
                num_envs=1,
                du_scale=1,
                instance_association="bbox_iou",
                iou_threshold=0.2,  # Lower threshold for better association across frames
                global_box_nms_thresh=0.3,
                min_pixels_for_instance_view=50,
                min_percent_for_instance_view=0.0005,  # More lenient for partial views
                min_instance_vol=1e-8,  # Smaller minimum volume
                min_instance_points=20,  # Fewer required points
                encoder=encoder,
                use_visual_feat=True  # Always use visual features when available
            )
            return instance_map
        except Exception as e:
            logger.error(f"Error initializing instance map: {e}")
            return None
        
    def _initialize_scene_graph(self):
        """Initialize the scene graph for representing relationships."""
        try:
            logger.info("Initializing scene graph")
            
            # Get instances for initialization (empty list to start)
            instances = []
            if self.instance_map is not None and hasattr(self.instance_map, 'get_instances'):
                instances = self.instance_map.get_instances()
            
            # Create scene graph with voxel map provided directly
            scene_graph = SceneGraph(
                parameters=self.parameters, 
                instances=instances  
            )
            
            
            logger.info("Scene graph initialized successfully")
            return scene_graph
        except Exception as e:
            logger.error(f"Error initializing scene graph: {e}")
            return None

    def update_map_with_pose_graph(self):
        """Update the map using the pose graph."""
        # Add null check for scene graph
        if not hasattr(self, 'scene_graph') or self.scene_graph is None:
            logger.warning("Scene graph not available, skipping relationship update")
            return
        
        # Now safe to call methods
        self.scene_graph.get_relationships()

    def update_scene_graph(self):
        """Update scene graph with latest detected objects for real-time relationship analysis."""
        if self.scene_graph is None or self.instance_map is None:
            return False
            
        try:
            # Get latest instances
            instances = self.instance_map.get_instances()
            
            # Skip if no instances
            if not instances:
                return False
                
            # Update scene graph with latest instances
            self.scene_graph.update(instances)
            
            # Get relationships (for debugging)
            relationships = self.scene_graph.get_relationships()
            if len(relationships) > 0:
                logger.debug(f"Found {len(relationships)} spatial relationships")
                
            return True
        except Exception as e:
            logger.error(f"Error updating scene graph: {e}")
            return False
            
    def create_instances_from_semantic(self, observation):
        """Create instances from semantic segmentation and point cloud."""
        if not hasattr(observation, 'semantic') or observation.semantic is None:
            logger.error("No semantic data in observation")
            return False
        
        if not hasattr(observation, 'point_cloud') or observation.point_cloud is None:
            logger.error("No point cloud data in observation")
            return False
        
        try:
            import torch
            from stretch.mapping.instance import Instance
            
            # Get semantic segmentation
            semantic = observation.semantic
            
            # Get unique object IDs (excluding background=0)
            unique_ids = np.unique(semantic)
            unique_ids = unique_ids[unique_ids > 0]  # Exclude background
            
            logger.info(f"Found {len(unique_ids)} unique object IDs in semantic segmentation")
            
            # Process each unique object ID
            instances_created = 0
            for obj_id in unique_ids:
                # Create mask for this object
                mask = (semantic == obj_id)
                
                # Skip if mask is too small (< 1% of image)
                if np.sum(mask) < (semantic.shape[0] * semantic.shape[1] * 0.01):
                    continue
                    
                # Get points for this object from point cloud
                points = []
                
                # Get 3D points for this mask (direct pixel-to-point)
                # You'll need to adapt this based on how your point cloud is structured
                h, w = mask.shape
                for y in range(h):
                    for x in range(w):
                        if mask[y, x]:
                            # Try to get corresponding 3D point
                            if hasattr(observation, 'depth') and observation.depth is not None:
                                # Use depth to create 3D point
                                depth = observation.depth[y, x]
                                
                                # Skip invalid depths
                                if np.isnan(depth) or np.isinf(depth) or depth <= 0:
                                    continue
                                
                                # Use camera matrix to project to 3D
                                if hasattr(observation, 'camera_K') and observation.camera_K is not None:
                                    fx = observation.camera_K[0, 0]
                                    fy = observation.camera_K[1, 1]
                                    cx = observation.camera_K[0, 2]
                                    cy = observation.camera_K[1, 2]
                                    
                                    # Project to 3D
                                    X = (x - cx) * depth / fx
                                    Y = (y - cy) * depth / fy
                                    Z = depth
                                    
                                    points.append([X, Y, Z])
                
                # Skip if we don't have enough points (need at least 3 for a bounding box)
                if len(points) < 10:
                    logger.warning(f"Not enough valid 3D points for object {obj_id}, skipping")
                    continue
                    
                # Convert to tensor
                points_tensor = torch.tensor(points)
                
                # Create instance
                instance = Instance(global_id=int(obj_id))
                instance.points = points_tensor
                
                # Set category ID based on obj_id
                instance.category_id = int(obj_id)
                
                # Set category name (you might want a mapping here)
                category_names = {
                    1: "shelf", 
                    2: "box", 
                    3: "furniture",
                    4: "container",
                    5: "wall_fixture",
                    # Add more as needed
                }
                instance.category_name = category_names.get(int(obj_id), f"object_{obj_id}")
                
                # Set score
                instance.score = torch.tensor(0.95)
                
                # Compute bounds
                min_coords = torch.min(points_tensor, dim=0)[0]
                max_coords = torch.max(points_tensor, dim=0)[0]
                
                # Set bounds [x_min, x_max], [y_min, y_max], [z_min, z_max]
                instance.bounds = torch.stack([
                    torch.tensor([min_coords[0].item(), max_coords[0].item()]),
                    torch.tensor([min_coords[1].item(), max_coords[1].item()]),
                    torch.tensor([min_coords[2].item(), max_coords[2].item()])
                ])
                
                # Add instance to instance map
                self.instance_map.add_instance(instance)
                instances_created += 1
                
            logger.info(f"Created {instances_created} instances from semantic segmentation")
            return instances_created > 0
        
        except Exception as e:
            logger.error(f"Error creating instances from semantic: {e}")
            traceback.print_exc()
            return False

    def start(self):
        """Start the mapping adapter."""
        if self.running:
            logger.warning("SegwayMappingAdapter already running")
            return
            
        self.running = True
        self.mapping_thread = threading.Thread(target=self._mapping_thread_func)
        self.mapping_thread.daemon = True
        self.mapping_thread.start()

        # Set voxel map in scene graph if available
        #if self.scene_graph is not None and hasattr(self.scene_graph, 'set_voxel_map'):
        #    self.scene_graph.set_voxel_map(self.voxel_map)
        #    logger.info("Set voxel map in scene graph")
        
        logger.info("SegwayMappingAdapter started")
    
    def stop(self):
        """Stop the mapping adapter."""
        self.running = False
        if self.mapping_thread:
            self.mapping_thread.join(timeout=2.0)
            self.mapping_thread = None
            
        logger.info("SegwayMappingAdapter stopped")
    
    def _mapping_thread_func(self):
        """Background thread for continuous mapping processing."""
        logger.info("Mapping thread started")
        
        while self.running:
            try:
                # Even if queue is empty, periodically check for new sensor data
                latest_obs = self.ros_client.get_observation() 
                if latest_obs is not None:
                    self.update_map_from_observation(latest_obs)
                    
                # Process queue items as well
                item = None
                try:
                    item = self.mapping_queue.get(timeout=0.05)  # Short timeout for responsiveness
                    # Process the item...
                    self.mapping_queue.task_done()
                except queue.Empty:
                    pass  # No items in queue, continue with next sensor check
                    
            except Exception as e:
                logger.error(f"Error in mapping thread: {e}")
                
            # Short sleep for high update rate (20Hz)
            time.sleep(0.05)
        
        logger.info("Mapping thread stopped")
    
    def fix_instance_classes(self, observation):
        """Add instance_classes if missing to prevent 'not defined' error."""
        if not hasattr(observation, 'instance_classes') or observation.instance_classes is None:
            # Create the attribute if it doesn't exist
            observation.instance_classes = []
            
            # Try to populate from instances if available
            if hasattr(observation, 'instance_seg') and observation.instance_seg:
                # Add a default class ID for each mask
                observation.instance_classes = [1] * len(observation.instance_seg)
        
        return observation

    def update_map_from_observation(self, observation):
        """Update the map from an observation."""
        if observation is None:
            logger.error("Cannot update map from None observation")
            return False
            
        voxel_updated = False
        instance_updated = False
        
        # Update voxel map if available
        if self.voxel_map is not None:
            try:
                # Ensure camera_pose is available
                if not hasattr(observation, 'camera_pose') or observation.camera_pose is None:
                    # Try to get current pose from mapping system
                    camera_pose = self.get_current_pose_matrix()
                    if camera_pose is not None:
                        # Attach camera pose to observation
                        observation.camera_pose = camera_pose
                    else:
                        logger.error("Cannot update voxel map: camera_pose is missing and cannot be inferred")
                        return False
                
                # Make sure RGB is available for instance mapping
                if not hasattr(observation, 'rgb') or observation.rgb is None:
                    logger.warning("RGB data missing from observation")
                    if hasattr(observation, 'xyz') and observation.xyz is not None:
                        # Create placeholder RGB if we have point cloud data
                        observation.rgb = np.ones_like(observation.xyz) * 128  # Gray placeholder
                
                # Now update the voxel map
                if hasattr(self.voxel_map, 'add_obs'):
                    self.voxel_map.add_obs(observation)
                    voxel_updated = True
                elif hasattr(self.voxel_map, 'add'):
                    # Extract necessary data from observation
                    xyz = getattr(observation, 'xyz', None)
                    rgb = getattr(observation, 'rgb', None)
                    depth = getattr(observation, 'depth', None)
                    camera_K = getattr(observation, 'camera_K', None)
                    
                    # Add to voxel map
                    self.voxel_map.add(
                        camera_pose=observation.camera_pose,
                        rgb=rgb,
                        xyz=xyz,
                        depth=depth,
                        camera_K=camera_K
                    )
                    voxel_updated = True
                else:
                    logger.warning("Voxel map has no suitable method for updating")
            except Exception as e:
                logger.error(f"Error updating voxel map: {e}")
        
        # Update instance map if available
        if self.instance_map is not None and hasattr(observation, 'rgb') and observation.rgb is not None:
            try:
                # Process instances with available data
                instance_seg = None
                
                # Look for instance segmentation in various places
                if hasattr(observation, 'instance_segmentation') and observation.instance_segmentation is not None:
                    instance_seg = observation.instance_segmentation
                elif hasattr(observation, 'instance') and observation.instance is not None:
                    instance_seg = observation.instance
                elif hasattr(observation, 'detic_instance_map') and observation.detic_instance_map is not None:
                    instance_seg = observation.detic_instance_map
                
                # If we have instance segmentation, update the instance map
                if instance_seg is not None:
                    self.instance_map.process_instances_for_env(
                        env_id=0,
                        instance_seg=instance_seg,
                        point_cloud=observation.xyz if hasattr(observation, 'xyz') else None,
                        image=observation.rgb,
                        camera_K=observation.camera_K if hasattr(observation, 'camera_K') else None,
                        camera_pose=observation.camera_pose
                    )
                    instance_updated = True
            except Exception as e:
                logger.error(f"Error updating instance map: {e}")
                
        return voxel_updated or instance_updated
    
    def update_map_from_point_cloud(self, point_cloud):
        """Update the map from a point cloud."""
        try:
            if not hasattr(self, 'voxel_map') or self.voxel_map is None:
                return False
            
            # Extract points from different point cloud formats
            points = None
            
            if point_cloud is None:
                logger.warning("Cannot update map: point cloud is None")
                return False
            
            if hasattr(point_cloud, 'points') and point_cloud.points is not None:
                # Standard PointCloud format
                points = point_cloud.points
            elif hasattr(point_cloud, 'data') and point_cloud.data is not None:
                # Alternative attribute name
                points = point_cloud.data
            
            # Convert torch tensor to numpy if needed
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            
            # Get current pose for camera pose estimation
            camera_pose = self.get_current_pose_matrix()
            
            # Add points to voxel map using the appropriate method
            if hasattr(self.voxel_map, 'add'):
                # Use add method 
                self.voxel_map.add(
                    camera_pose=camera_pose,
                    xyz=torch.from_numpy(points).float() if isinstance(points, np.ndarray) else points,
                    rgb=torch.ones_like(points) * 255 if points is not None else None  # Default white if no color
                )
            elif hasattr(self.voxel_map, 'add_obs'):
                # Create minimal observation object
                from stretch.core.interfaces import Observations
                obs = Observations()
                obs.xyz = points
                obs.camera_pose = camera_pose
                obs.rgb = np.ones_like(points) * 255 if isinstance(points, np.ndarray) else torch.ones_like(points) * 255
                
                self.voxel_map.add_obs(obs)
            else:
                logger.error("Voxel map has neither 'add' nor 'add_obs' method")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error updating voxel map from point cloud: {e}")
            return False

    def optimize_point_cloud(self, point_cloud, max_points=50000):
        """Optimize point cloud for real-time processing.
        
        Args:
            point_cloud: Input point cloud
            max_points: Maximum number of points to process
            
        Returns:
            torch.Tensor: Optimized point cloud
        """
        if point_cloud is None or point_cloud.shape[0] <= max_points:
            return point_cloud
            
        # Downsample if too many points
        stride = point_cloud.shape[0] // max_points
        return point_cloud[::stride, :]
        
    def _update_occupancy_grid(self, points_2d):
        """
        Update occupancy grid with 2D points.
        
        Args:
            points_2d: Nx2 array of 2D points
        """
        try:
            # Get grid parameters
            resolution = getattr(self.occupancy_grid, 'resolution', 0.05)
            width = getattr(self.occupancy_grid, 'width', 500)
            height = getattr(self.occupancy_grid, 'height', 500)
            origin_x = getattr(self.occupancy_grid, 'origin_x', -12.5)
            origin_y = getattr(self.occupancy_grid, 'origin_y', -12.5)
            
            # Convert to grid coordinates
            grid_x = np.floor((points_2d[:, 0] - origin_x) / resolution).astype(np.int32)
            grid_y = np.floor((points_2d[:, 1] - origin_y) / resolution).astype(np.int32)
            
            # Filter valid coordinates
            valid = (grid_x >= 0) & (grid_x < width) & (grid_y >= 0) & (grid_y < height)
            grid_x = grid_x[valid]
            grid_y = grid_y[valid]
            
            # Different ways to update the grid
            if hasattr(self.occupancy_grid, 'update_cells'):
                # Preferred method
                cells = list(zip(grid_x, grid_y))
                self.occupancy_grid.update_cells(cells, 1.0)
            elif hasattr(self.occupancy_grid, 'set_cells'):
                # Alternative batch method
                cells = list(zip(grid_x, grid_y))
                self.occupancy_grid.set_cells(cells, 1.0)
            elif hasattr(self.occupancy_grid, 'set_cell'):
                # Individual cell setting
                for x, y in zip(grid_x, grid_y):
                    self.occupancy_grid.set_cell(x, y, 1.0)
            elif hasattr(self.occupancy_grid, 'grid') and isinstance(self.occupancy_grid.grid, np.ndarray):
                # Direct grid access
                self.occupancy_grid.grid[grid_y, grid_x] = 1.0
            else:
                logger.warning("No suitable method found to update occupancy grid")
        except Exception as e:
            logger.error(f"Error in _update_occupancy_grid: {e}")

    def set_encoder(self, encoder):
        """Set visual encoder for instance memory to improve object recognition.
        
        Args:
            encoder: Visual encoder (e.g., CLIP) for feature extraction
        """
        if self.instance_map is not None:
            self.instance_map.encoder = encoder
            logger.info("Set encoder for instance memory")
            
            # Update view matching config to use visual features
            if hasattr(self.instance_map, 'view_matching_config'):
                self.instance_map.view_matching_config.visual_similarity_weight = 0.7
                self.instance_map.view_matching_config.box_overlap_weight = 0.3
                self.instance_map.use_visual_feat = True
                logger.info("Updated view matching config to use visual features")

    def get_camera_calibration(self):
        """Get current camera calibration data needed for point cloud projection.
        
        Returns:
            dict: Camera calibration data with intrinsics and extrinsics
        """
        try:
            if hasattr(self.ros_client, 'get_camera_intrinsics'):
                intrinsics = self.ros_client.get_camera_intrinsics()
                pose = self.get_current_pose_matrix()
                
                return {
                    'intrinsics': intrinsics,
                    'extrinsics': pose,
                    'width': 640,  # Add your camera resolution
                    'height': 480
                }
            return None
        except Exception as e:
            logger.error(f"Error getting camera calibration: {e}")
            return None

    def debug_object_detection_pipeline(self, observation):
        """
        Debug tool to trace object detection and mapping pipeline.
        
        Args:
            observation: Observation to process
        """
        logger.info("=== Starting Object Detection Debug ===")
        
        # Step 1: Check input observation
        logger.info("Step 1: Checking input observation")
        if observation is None:
            logger.error("Observation is None!")
            return
        
        # Check for RGB image
        if not hasattr(observation, 'rgb') or observation.rgb is None:
            logger.error("No RGB image in observation!")
        else:
            logger.info(f"RGB image shape: {observation.rgb.shape}")
        
        # Check for depth image
        if not hasattr(observation, 'depth') or observation.depth is None:
            logger.error("No depth image in observation!")
        else:
            logger.info(f"Depth image shape: {observation.depth.shape}")
        
        # Step 2: Process through perception
        logger.info("Step 2: Processing through perception")
        # Assumes a perception adapter with predict method
        try:
            perceived_obs = self.perception_adapter.update_perception(observation)
            if hasattr(perceived_obs, 'instance_segmentation') and perceived_obs.instance_segmentation is not None:
                logger.info(f"Instance segmentation present. Unique instances: {len(np.unique(perceived_obs.instance_segmentation))}")
            else:
                logger.error("No instance segmentation produced by perception!")
        except Exception as e:
            logger.error(f"Error in perception processing: {e}")
        
        # Step 3: Process through instance mapping
        logger.info("Step 3: Processing through instance mapping")
        initial_instance_count = len(self.instance_map.get_instances())
        logger.info(f"Initial instance count: {initial_instance_count}")
        
        try:
            # Process the observation through the instance map
            if hasattr(self.instance_map, 'process_instances_for_env'):
                # Use the internal process_instances_for_env method to get detailed logs
                env_id = 0  # Default environment ID
                instance_seg = perceived_obs.instance_segmentation
                point_cloud = observation.point_cloud if hasattr(observation, 'point_cloud') else None
                image = observation.rgb
                
                self.instance_map.process_instances_for_env(
                    env_id=env_id,
                    instance_seg=instance_seg,
                    point_cloud=point_cloud,
                    image=image,
                    verbose=True  # Enable verbose logging
                )
                
                # Check the unprocessed views
                unprocessed_views = self.instance_map.get_unprocessed_instances_per_env(env_id)
                logger.info(f"Unprocessed views count: {len(unprocessed_views)}")
                
                # Trigger instance association
                self.instance_map.associate_instances_to_memory(debug=True)
            else:
                # Use the regular update method if process_instances_for_env is not available
                self.mapping_adapter.update_map_from_observation(perceived_obs)
            
            # Check how many instances we have now
            final_instance_count = len(self.instance_map.get_instances())
            logger.info(f"Final instance count: {final_instance_count}")
            logger.info(f"Added {final_instance_count - initial_instance_count} new instances")
            
            # Print out details of instances
            instances = self.instance_map.get_instances()
            for i, instance in enumerate(instances):
                if hasattr(instance, 'category_id') and instance.category_id is not None:
                    logger.info(f"Instance {i}: Category {instance.category_id}, Bounds: {instance.bounds}")
                else:
                    logger.info(f"Instance {i}: No category assigned")
        
        except Exception as e:
            logger.error(f"Error in instance mapping: {e}")
        
        # Step 4: Process through scene graph
        logger.info("Step 4: Processing through scene graph")
        if self.scene_graph is not None:
            try:
                # Get instances from the instance map
                instances = self.instance_map.get_instances()
                
                # Check if voxel map is set in scene graph
                if hasattr(self.scene_graph, 'voxel_map') and self.scene_graph.voxel_map is None:
                    logger.error("Voxel map not set in scene graph!")
                    logger.info("Attempting to set voxel map in scene graph")
                    self.scene_graph.set_voxel_map(self.voxel_map)
                
                # Update scene graph with instances
                initial_node_count = len(self.scene_graph.nodes)
                self.scene_graph.update(instances)
                final_node_count = len(self.scene_graph.nodes)
                
                logger.info(f"Scene graph nodes before: {initial_node_count}, after: {final_node_count}")
                logger.info(f"Added {final_node_count - initial_node_count} new nodes to scene graph")
                
                # Get relationships from scene graph
                relationships = self.scene_graph.get_relationships()
                logger.info(f"Total relationships: {len(relationships)}")
                
                # Print out details of relationships
                for rel in relationships[:10]:  # Print first 10 to avoid flooding logs
                    logger.info(f"Relationship: {rel}")
                    
            except Exception as e:
                logger.error(f"Error in scene graph processing: {e}")
        else:
            logger.error("Scene graph is None!")
        
        logger.info("=== Object Detection Debug Complete ===")

    def _update_grid_from_point_cloud(self, point_cloud):
        """Update occupancy grid from point cloud."""
        try:
            if not hasattr(self, 'grid_map') or self.grid_map is None:
                logger.debug("No grid map available for update")
                return False
            
            # Extract points from various point cloud formats
            points = None
            if hasattr(point_cloud, 'points') and point_cloud.points is not None:
                points = point_cloud.points
            elif isinstance(point_cloud, np.ndarray):
                points = point_cloud
            elif hasattr(point_cloud, 'to_numpy'):
                points = point_cloud.to_numpy()
            
            if points is None or points.shape[0] == 0:
                logger.debug("No points available for grid update")
                return False
            
            # Convert to numpy if it's a torch tensor
            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()
            
            # Get only 2D coordinates (x,y) for occupancy grid
            points_2d = points[:, :2]
            
            # Try different grid update methods
            updated = False
            
            # Method 1: Try update_from_points
            if hasattr(self.grid_map, 'update_from_points'):
                try:
                    self.grid_map.update_from_points(points_2d)
                    updated = True
                    logger.debug(f"Updated grid using update_from_points with {points.shape[0]} points")
                except Exception as e:
                    logger.debug(f"Error using update_from_points: {e}")
            
            # Method 2: Try mark_occupied or set_occupied
            if not updated and (hasattr(self.grid_map, 'mark_occupied') or hasattr(self.grid_map, 'set_occupied')):
                mark_method = getattr(self.grid_map, 'mark_occupied', getattr(self.grid_map, 'set_occupied', None))
                if mark_method:
                    try:
                        # Get grid resolution
                        resolution = getattr(self.grid_map, 'resolution', 0.05)
                        
                        # Convert to grid coordinates
                        for point in points_2d:
                            x, y = point
                            grid_x = int(x / resolution)
                            grid_y = int(y / resolution)
                            try:
                                mark_method(grid_x, grid_y)
                            except Exception:
                                # Try with different argument order or signature
                                try:
                                    mark_method(grid_x, grid_y, 1.0)  # Some implementations expect a value
                                except Exception:
                                    pass
                        
                        updated = True
                        logger.debug(f"Updated grid using mark_occupied with {points.shape[0]} points")
                    except Exception as e:
                        logger.debug(f"Error using mark_occupied: {e}")
            
            # Method 3: Try set_cell
            if not updated and hasattr(self.grid_map, 'set_cell'):
                try:
                    resolution = getattr(self.grid_map, 'resolution', 0.05)
                    for point in points_2d:
                        x, y = point
                        grid_x = int(x / resolution)
                        grid_y = int(y / resolution)
                        self.grid_map.set_cell(grid_x, grid_y, 1.0)  # Occupied = 1.0
                    
                    updated = True
                    logger.debug(f"Updated grid using set_cell with {points.shape[0]} points")
                except Exception as e:
                    logger.debug(f"Error using set_cell: {e}")
            
            # Method 4: Direct update of grid data array
            if not updated and hasattr(self.grid_map, 'data') and isinstance(self.grid_map.data, np.ndarray):
                try:
                    resolution = getattr(self.grid_map, 'resolution', 0.05)
                    origin_x = getattr(self.grid_map, 'origin_x', 0.0)
                    origin_y = getattr(self.grid_map, 'origin_y', 0.0)
                    
                    for point in points_2d:
                        x, y = point
                        grid_x = int((x - origin_x) / resolution)
                        grid_y = int((y - origin_y) / resolution)
                        
                        if 0 <= grid_x < self.grid_map.data.shape[1] and 0 <= grid_y < self.grid_map.data.shape[0]:
                            self.grid_map.data[grid_y, grid_x] = 1.0  # Occupied = 1.0
                    
                    updated = True
                    logger.debug(f"Updated grid directly via data array with {points.shape[0]} points")
                except Exception as e:
                    logger.debug(f"Error updating grid directly: {e}")
            
            if updated:
                logger.debug(f"Successfully updated grid with {points.shape[0]} points")
            else:
                logger.warning(f"Could not update grid map - no compatible update method found")
            
            return updated
        except Exception as e:
            logger.error(f"Error updating grid from point cloud: {e}")
            return False
    
    def get_current_pose_matrix(self):
        """
        Get the current pose as a 4x4 transformation matrix.
        
        Returns:
            np.ndarray: 4x4 transformation matrix or None if not available
        """
        try:
            if self.ros_client is None:
                return None
                
            # Try to get pose from ROS client
            pose = self.ros_client.get_pose()
            if pose is None:
                return None
                
            # Convert pose to 4x4 matrix
            x, y, theta = pose
            
            # Create transformation matrix
            import numpy as np
            transform = np.eye(4)
            
            # Set rotation matrix (rotation around Z axis)
            transform[0, 0] = np.cos(theta)
            transform[0, 1] = -np.sin(theta)
            transform[1, 0] = np.sin(theta)
            transform[1, 1] = np.cos(theta)
            
            # Set translation
            transform[0, 3] = x
            transform[1, 3] = y
            
            return transform
        except Exception as e:
            logger.error(f"Error getting current pose matrix: {e}")
            return None

    def _quaternion_to_rotation_matrix(self, q):
        """
        Convert a quaternion to a rotation matrix.
        
        Args:
            q: Quaternion [x, y, z, w]
            
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        x, y, z, w = q
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            x /= norm
            y /= norm
            z /= norm
            w /= norm
        
        # Calculate rotation matrix
        xx = x * x
        xy = x * y
        xz = x * z
        xw = x * w
        
        yy = y * y
        yz = y * z
        yw = y * w
        
        zz = z * z
        zw = z * w
        
        rotation_matrix = np.array([
            [1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)],
            [2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)],
            [2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)]
        ])
        
        return rotation_matrix
    
    def get_robot_pose(self):
        """
        Get the current robot pose.
        
        Returns:
            tuple: (position, orientation) where position is [x, y, z] and 
                  orientation is [x, y, z, w] quaternion
        """
        try:
            # First, try to get pose from ROS client
            xyt = self.ros_client.get_pose()
            
            if xyt is not None:
                x, y, theta = xyt
                
                # Convert 2D pose to 3D pose
                position = np.array([x, y, 0.0])
                
                # Convert theta to quaternion (rotation around z-axis)
                orientation = np.array([
                    0.0,
                    0.0,
                    np.sin(theta / 2),
                    np.cos(theta / 2)
                ])
                
                # Update latest pose
                with self.mapping_lock:
                    self.latest_pose = (position, orientation)
                    # Assume high confidence for odometry
                    self.pose_confidence = 0.9
                    self.is_lost = False
                
                return position, orientation
            else:
                # If ROS client doesn't provide pose, return latest pose
                with self.mapping_lock:
                    self.pose_confidence = max(0.0, self.pose_confidence - 0.1)
                    if self.pose_confidence < 0.3:
                        self.is_lost = True
                    return self.latest_pose
        except Exception as e:
            logger.error(f"Error getting robot pose: {e}")
            with self.mapping_lock:
                self.pose_confidence = 0.0
                self.is_lost = True
                return self.latest_pose
    
    def get_localization_confidence(self):
        """
        Get the confidence in the current localization.
        
        Returns:
            float: Confidence value between 0 and 1
        """
        with self.mapping_lock:
            return self.pose_confidence
    
    def is_robot_lost(self):
        """
        Check if the robot is lost.
        
        Returns:
            bool: True if robot is lost, False otherwise
        """
        with self.mapping_lock:
            return self.is_lost
    
    def reset_map(self):
        """Reset all maps."""
        try:
            logger.info("Resetting maps")
            
            # Reset voxel map
            if self.voxel_map is not None:
                self.voxel_map.reset()
                logger.debug("Reset voxel map")
                
            # Reset occupancy grid
            if self.grid_map is not None:
                self.grid_map.reset()
                logger.debug("Reset occupancy grid")
                
            # Reset instance map
            if self.instance_map is not None:
                self.instance_map.reset()
                logger.debug("Reset instance map")
                
            # Reset scene graph
            if self.scene_graph is not None:
                self.scene_graph.reset()
                logger.debug("Reset scene graph")
                
            # Reset pose data
            with self.mapping_lock:
                self.latest_pose = (np.zeros(3), np.array([0, 0, 0, 1]))
                self.pose_confidence = 1.0
                self.is_lost = False
                
            logger.info("Map reset complete")
        except Exception as e:
            logger.error(f"Error resetting maps: {e}")
    
    def plan_path(self, goal, use_rrt=False):
        """
        Plan a path to a goal position.
        
        Args:
            goal: Goal position [x, y] or [x, y, z]
            use_rrt: If True, use RRT planner, otherwise use A*
            
        Returns:
            list: List of waypoints [[x, y], ...] or None if no path found
        """
        try:
            logger.info(f"Planning path to goal: {goal}")
            
            # Get current position
            position, _ = self.get_robot_pose()
            
            # Convert goal to 2D if needed
            goal_2d = goal[:2] if len(goal) > 2 else goal
            start_2d = position[:2]
            
            # Get obstacles from occupancy grid or voxel map
            if self.grid_map is not None:
                obstacles = self.grid_map.get_occupied_cells()
            elif self.voxel_map is not None:
                obstacles = self._get_2d_obstacles_from_voxel_map()
            else:
                obstacles = []
            
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
                    path = self._plan_path_rrt(start_2d, goal_2d)
            else:
                # Use A* planner
                if self.astar_planner is not None:
                    path = self.astar_planner.plan(
                        start=start_2d,
                        goal=goal_2d,
                        obstacles=obstacles
                    )
                else:
                    path = self._plan_path_astar(start_2d, goal_2d)
                
            # Return None if no path found
            if path is None or len(path) == 0:
                logger.warning(f"No path found to goal: {goal}")
                return None
                
            # Simplify path if enabled
            if self.path_simplification:
                path = self.simplify_path(path)
                
            logger.info(f"Path planned with {len(path)} waypoints")
            return path
        except Exception as e:
            logger.error(f"Error planning path: {e}")
            return None
    
    def _get_2d_obstacles_from_voxel_map(self):
        """
        Extract 2D obstacles from 3D voxel map.
        
        Returns:
            list: List of obstacle coordinates [[x, y], ...]
        """
        if self.voxel_map is None:
            return []
            
        try:
            # Get occupied voxels from voxel map
            voxels = self.voxel_map.get_occupied_voxels()
            
            if voxels is None or len(voxels) == 0:
                return []
                
            # Convert 3D voxels to 2D obstacle points
            obstacles = []
            for voxel in voxels:
                # Check if voxel is at relevant height (e.g., between 0.1m and 0.5m)
                x, y, z = voxel
                if 0.1 <= z <= 0.5:
                    obstacles.append([x, y])
                    
            return obstacles
        except Exception as e:
            logger.error(f"Error getting 2D obstacles from voxel map: {e}")
            return []
    
    def _plan_path_astar(self, start, goal):
        """
        Plan a path using A* algorithm.
        
        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            
        Returns:
            list: List of waypoints [[x, y], ...] or None if no path found
        """
        # This is a simplified implementation - in practice, use the
        # actual AStar implementation from Stretch AI
        
        # Check if direct path is clear
        if self.is_path_clear(start, goal):
            return [start, goal]
            
        # Otherwise, create a simple path with intermediate waypoints
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
    
    def _plan_path_rrt(self, start, goal):
        """
        Plan a path using RRT algorithm.
        
        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            
        Returns:
            list: List of waypoints [[x, y], ...] or None if no path found
        """
        # This is a simplified implementation - in practice, use the
        # actual RRT implementation from Stretch AI
        
        # Check if direct path is clear
        if self.is_path_clear(start, goal):
            return [start, goal]
            
        # Otherwise, create a simple path with random intermediate waypoints
        direction = np.array(goal) - np.array(start)
        distance = np.linalg.norm(direction)
        
        # Determine number of waypoints based on distance
        num_waypoints = max(2, int(distance / 0.5))
        
        # Create waypoints with some randomness
        np.random.seed(0)  # For reproducibility
        waypoints = [start]
        
        for i in range(1, num_waypoints):
            t = i / num_waypoints
            
            # Add some randomness perpendicular to path
            perpendicular = np.array([-direction[1], direction[0]])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            
            random_offset = np.random.normal(0, 0.2) * perpendicular
            
            waypoint = start + t * direction + random_offset
            waypoints.append(waypoint.tolist())
            
        waypoints.append(goal)
            
        return waypoints
    
    def simplify_path(self, path):
        """
        Simplify a path by removing unnecessary waypoints.
        
        Args:
            path: List of waypoints [[x, y], ...]
            
        Returns:
            list: Simplified list of waypoints
        """
        if path is None or len(path) <= 2:
            return path
            
        try:
            logger.debug(f"Simplifying path with {len(path)} waypoints")
            
            simplified_path = [path[0]]
            
            i = 0
            while i < len(path) - 1:
                # Find the furthest waypoint that has a clear path from current waypoint
                furthest = i + 1
                
                for j in range(i + 2, len(path)):
                    if self.is_path_clear(path[i], path[j]):
                        furthest = j
                
                # Add furthest waypoint to simplified path
                simplified_path.append(path[furthest])
                
                # Move to furthest waypoint
                i = furthest
                
            logger.debug(f"Simplified path to {len(simplified_path)} waypoints")
            return simplified_path
        except Exception as e:
            logger.error(f"Error simplifying path: {e}")
            return path
    
    def is_path_clear(self, start, end):
        """
        Check if a straight path between two points is clear of obstacles.
        
        Args:
            start: Start position [x, y] or [x, y, z]
            end: End position [x, y] or [x, y, z]
            
        Returns:
            bool: True if path is clear, False otherwise
        """
        try:
            # Convert to 2D if needed
            start_2d = start[:2] if len(start) > 2 else start
            end_2d = end[:2] if len(end) > 2 else end
            
            # Check with grid map if available
            if self.grid_map is not None:
                return self.grid_map.is_path_clear(start_2d, end_2d)
                
            # Check with voxel map if available
            if self.voxel_map is not None:
                return self._check_path_clear_with_voxel_map(start_2d, end_2d)
                
            # If no map available, assume path is clear
            return True
        except Exception as e:
            logger.error(f"Error checking if path is clear: {e}")
            return False
    
    def _check_path_clear_with_voxel_map(self, start, end):
        """
        Check if a path is clear using the voxel map.
        
        Args:
            start: Start position [x, y]
            end: End position [x, y]
            
        Returns:
            bool: True if path is clear, False otherwise
        """
        if self.voxel_map is None:
            return True
            
        try:
            # Convert to numpy arrays
            start_np = np.array(start)
            end_np = np.array(end)
            
            # Calculate direction and distance
            direction = end_np - start_np
            distance = np.linalg.norm(direction)
            
            if distance < 1e-6:
                # Start and end are the same point
                return True
                
            # Normalize direction
            direction = direction / distance
            
            # Check for obstacles along the path
            step_size = 0.1  # Check every 10cm
            num_steps = int(distance / step_size) + 1
            
            for i in range(num_steps + 1):
                t = i * step_size if i < num_steps else distance
                point_2d = start_np + t * direction
                
                # Create 3D point with relevant heights
                for height in [0.1, 0.3, 0.5]:  # Check at different heights
                    point_3d = np.array([point_2d[0], point_2d[1], height])
                    
                    # Check for obstacle at this point
                    if self.voxel_map.is_occupied(point_3d):
                        return False
                    
            return True
        except Exception as e:
            logger.error(f"Error checking path clear with voxel map: {e}")
            return False
    
    def is_obstacle_at(self, point):
        """
        Check if there is an obstacle at a given position.
        
        Args:
            point: Position to check [x, y] or [x, y, z]
            
        Returns:
            bool: True if obstacle present, False otherwise
        """
        try:
            # Check with grid map if available and point is 2D
            if self.grid_map is not None and len(point) == 2:
                return self.grid_map.is_occupied(point[0], point[1])
                
            # Check with voxel map if available
            if self.voxel_map is not None:
                # Convert to 3D if needed
                point_3d = np.array([point[0], point[1], 0.3]) if len(point) == 2 else np.array(point)
                return self.voxel_map.is_occupied(point_3d)
                
            # If no map available, assume no obstacle
            return False
        except Exception as e:
            logger.error(f"Error checking for obstacle: {e}")
            return False
    
    def get_nearest_obstacle_distance(self):
        """
        Get the distance to the nearest obstacle.
        
        Returns:
            float: Distance to nearest obstacle in meters
        """
        try:
            # Get current position
            position, _ = self.get_robot_pose()
            position_2d = position[:2]
            
            # Check with grid map if available
            if self.grid_map is not None:
                return self.grid_map.get_nearest_obstacle_distance(position_2d)
                
            # Check with voxel map if available
            if self.voxel_map is not None:
                return self._get_nearest_obstacle_distance_voxel(position)
                
            # If no map available, assume large distance
            return float('inf')
        except Exception as e:
            logger.error(f"Error getting nearest obstacle distance: {e}")
            return float('inf')
    
    def _get_nearest_obstacle_distance_voxel(self, position):
        """
        Get nearest obstacle distance using voxel map.
        
        Args:
            position: Robot position [x, y, z]
            
        Returns:
            float: Distance to nearest obstacle in meters
        """
        if self.voxel_map is None:
            return float('inf')
            
        try:
            # Search for nearest obstacle
            min_distance = float('inf')
            search_radius = 5.0  # Search within 5m
            
            # For simplicity, we'll check in a grid around the robot
            grid_size = 0.1  # 10cm grid
            for dx in np.arange(-search_radius, search_radius + grid_size, grid_size):
                for dy in np.arange(-search_radius, search_radius + grid_size, grid_size):
                    for dz in [0.1, 0.3, 0.5]:  # Check at different heights
                        point = [position[0] + dx, position[1] + dy, position[2] + dz]
                        
                        if self.voxel_map.is_occupied(point):
                            distance = np.sqrt(dx*dx + dy*dy)
                            min_distance = min(min_distance, distance)
            
            return min_distance
        except Exception as e:
            logger.error(f"Error getting nearest obstacle distance from voxel map: {e}")
            return float('inf')
    
    def find_frontiers(self, radius=3.0):
        """
        Find frontier points for exploration.
        
        Args:
            radius: Search radius in meters
            
        Returns:
            list: List of frontier points [[x, y], ...]
        """
        try:
            # Get current position
            position, _ = self.get_robot_pose()
            position_2d = position[:2]
            
            # Check with grid map if available
            if self.grid_map is not None:
                return self.grid_map.find_frontiers(position_2d, radius)
                
            # Use voxel map if available
            if self.voxel_map is not None:
                return self._find_frontiers_voxel(position_2d, radius)
                
            # If no map available, return empty list
            return []
        except Exception as e:
            logger.error(f"Error finding frontiers: {e}")
            return []
    
    def _find_frontiers_voxel(self, position, radius):
        """
        Find frontiers using voxel map.
        
        Args:
            position: Robot position [x, y]
            radius: Search radius in meters
            
        Returns:
            list: List of frontier points [[x, y], ...]
        """
        if self.voxel_map is None:
            return []
            
        try:
            # This is a placeholder implementation - replace with actual implementation
            # using Stretch AI's frontier detection or your own algorithm
            
            # For now, we'll just return some points around the robot
            frontiers = []
            
            # Generate some directions
            num_directions = 8
            for i in range(num_directions):
                angle = 2 * np.pi * i / num_directions
                
                # Create point at edge of radius
                x = position[0] + radius * np.cos(angle)
                y = position[1] + radius * np.sin(angle)
                
                # Only add if not an obstacle
                if not self.is_obstacle_at([x, y]):
                    frontiers.append([x, y])
            
            return frontiers
        except Exception as e:
            logger.error(f"Error finding frontiers with voxel map: {e}")
            return []
    
    def get_map_as_image(self, resolution=0.05, width=500, height=500):
        """
        Get a 2D representation of the map as an image.
        
        Args:
            resolution: Resolution of the image in meters per pixel
            width: Width of the image in pixels
            height: Height of the image in pixels
            
        Returns:
            np.ndarray: Map image
        """
        try:
            # Use grid map if available
            if self.grid_map is not None:
                return self.grid_map.as_image(resolution, width, height)
                
            # Use voxel map if available
            if self.voxel_map is not None:
                return self._voxel_map_as_image(resolution, width, height)
                
            # If no map available, return empty image
            return np.zeros((height, width), dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error getting map as image: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def _voxel_map_as_image(self, resolution, width, height):
        """
        Convert voxel map to 2D image.
        
        Args:
            resolution: Resolution of the image in meters per pixel
            width: Width of the image in pixels
            height: Height of the image in pixels
            
        Returns:
            np.ndarray: Map image
        """
        if self.voxel_map is None:
            return np.zeros((height, width), dtype=np.uint8)
            
        try:
            # Create empty image
            image = np.zeros((height, width), dtype=np.uint8)
            
            # Get voxels from map
            voxels = self.voxel_map.get_occupied_voxels()
            
            if voxels is None or len(voxels) == 0:
                return image
                
            # Calculate center of image
            center_x = width // 2
            center_y = height // 2
            
            # Draw voxels on image
            for voxel in voxels:
                x, y, z = voxel
                
                # Only use voxels at relevant height
                if 0.1 <= z <= 0.5:
                    # Convert world coordinates to image coordinates
                    img_x = int(center_x + x / resolution)
                    img_y = int(center_y - y / resolution)  # Y is inverted in image
                    
                    # Check if within image bounds
                    if 0 <= img_x < width and 0 <= img_y < height:
                        # Set pixel to occupied
                        image[img_y, img_x] = 255
            
            return image
        except Exception as e:
            logger.error(f"Error converting voxel map to image: {e}")
            return np.zeros((height, width), dtype=np.uint8)
    
    def get_detected_objects(self):
        """
        Get list of detected objects from instance map.
        
        Returns:
            list: List of detected objects
        """
        if self.instance_map is None:
            return []
            
        try:
            return self.instance_map.get_instances()
        except Exception as e:
            logger.error(f"Error getting detected objects: {e}")
            return []
    
    def get_scene_graph(self):
        """
        Get the scene graph.
        
        Returns:
            SceneGraph: Scene graph instance
        """
        return self.scene_graph

    def create_semantic_segmentation(self, observation):
        """Create semantic segmentation from RGB image for research."""
        if not hasattr(observation, 'rgb') or observation.rgb is None:
            logger.error("No RGB data in observation")
            return False
        
        try:
            # Get RGB image
            rgb = observation.rgb
            
            # Create empty semantic map
            semantic = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.int32)
            
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            
            # Simple edge detection to find object boundaries
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to create regions
            kernel = np.ones((5,5), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours in the edge image
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size to remove noise
            min_contour_area = 1000  # Adjust based on your image size
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
            
            # Create labeled regions for each contour
            for i, contour in enumerate(valid_contours, start=1):
                # Create a mask for this contour
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                semantic[mask > 0] = i
            
            # If no contours were found, try color thresholding for major objects
            if len(valid_contours) == 0:
                # Try to segment the black shelving unit
                dark_mask = (gray < 60)  # Dark regions
                # Label connected components
                num_labels, labels = cv2.connectedComponents(dark_mask.astype(np.uint8))
                
                # Add each significant component as a semantic region
                for label in range(1, num_labels):
                    if np.sum(labels == label) > min_contour_area:
                        semantic[labels == label] = label
            
            # Save the semantic segmentation
            observation.semantic = semantic
            
            # Log how many objects were found
            unique_labels = np.unique(semantic)
            num_objects = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
            
            logger.info(f"Created semantic segmentation with {num_objects} objects")
            return True
        
        except Exception as e:
            logger.error(f"Error creating semantic segmentation: {e}")
            traceback.print_exc()
            return False

    def enhance_segmentation_with_edges(self, observation):
        """(Experimental) Enhance semantic segmentation using edge detection."""
        # This method refines an *existing* semantic map or creates one from edges.
        logger.warning("Using experimental enhance_segmentation_with_edges.")
        if not hasattr(observation, 'rgb') or observation.rgb is None:
            logger.error("No RGB data for edge enhancement.")
            return False
        # Ensure semantic attribute exists or can be created
        if not hasattr(observation, 'semantic'):
             setattr(observation, 'semantic', None) # Create if missing

        try:
            # --- Start of the try block ---
            # Ensure this import is inside the try block if cv2 might not be available
            # If cv2 is a core dependency, it can be imported at the top of the file.
            # import cv2 # Already imported at the top

            rgb = observation.rgb
            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
                logger.error(f"Invalid RGB format for edge enhancement: type {type(rgb)}, shape {getattr(rgb, 'shape', 'N/A')}")
                return False # Return False on error inside try

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            if observation.semantic is not None:
                # Refine existing semantic map
                semantic = observation.semantic
                if not isinstance(semantic, np.ndarray) or semantic.shape[:2] != gray.shape:
                     logger.error("Invalid existing semantic map for refinement.")
                     return False # Return False on error inside try

                kernel = np.ones((3,3), np.uint8)
                dilated_edges = cv2.dilate(edges, kernel, iterations=1)

                # Set edge pixels to background (0)
                semantic[dilated_edges > 0] = 0
                logger.info("Refined existing semantic segmentation with edges.")
            else:
                # Create semantic map from edges if none exists
                logger.info("No existing semantic map, creating one from edges.")
                kernel = np.ones((5,5), np.uint8)
                dilated_edges = cv2.dilate(edges, kernel, iterations=2)
                regions = 255 - dilated_edges
                num_labels, labels = cv2.connectedComponents(regions)
                semantic = np.zeros_like(labels, dtype=np.int32)
                min_region_size = 500

                for label in range(1, num_labels):
                    if np.sum(labels == label) > min_region_size:
                        semantic[labels == label] = label # Assign unique IDs
                observation.semantic = semantic
                num_objects = len(np.unique(semantic)) - (1 if 0 in np.unique(semantic) else 0)
                logger.info(f"Created new semantic segmentation from edges with {num_objects} objects.")

            return True # Return True on success inside try
        except Exception as e:
            logger.error(f"Error enhancing segmentation with edges: {e}")
            traceback.print_exc() # Add traceback for detailed error info
            return False
            
    def _prepare_instance_seg_map(self, instance_seg):
        """
        Prepare instance segmentation map for InstanceMemory.
        
        Args:
            instance_seg: Instance segmentation data
            
        Returns:
            torch.Tensor: Prepared instance map tensor
        """
        try:
            # Handle different input formats
            if instance_seg is None:
                logger.warning("instance_seg is None in _prepare_instance_seg_map")
                return None
                
            # Convert numpy array to tensor if needed
            if isinstance(instance_seg, np.ndarray):
                instance_seg_tensor = torch.from_numpy(instance_seg).long()
            elif isinstance(instance_seg, torch.Tensor):
                instance_seg_tensor = instance_seg.long()
            else:
                logger.warning(f"Unsupported instance_seg type: {type(instance_seg)}")
                return None
                
            # Check tensor shape and adjust if needed
            if instance_seg_tensor.dim() == 3:
                if instance_seg_tensor.shape[0] != 1:
                    # This might be a tensor of multiple instance masks [N, H, W]
                    # Convert to single tensor with instance IDs
                    instance_map = torch.zeros(instance_seg_tensor.shape[1:], dtype=torch.long)
                    for i in range(instance_seg_tensor.shape[0]):
                        instance_map[instance_seg_tensor[i] > 0] = i + 1  # Assign instance ID (1-indexed)
                    instance_seg_tensor = instance_map
                else:
                    # If [1, H, W], squeeze to [H, W]
                    instance_seg_tensor = instance_seg_tensor.squeeze(0)
                    
            # Move to specified device if available
            if self.gpu_device is not None:
                instance_seg_tensor = instance_seg_tensor.to(self.gpu_device)
                
            return instance_seg_tensor
            
        except Exception as e:
            logger.error(f"Error preparing instance segmentation map: {e}")
            return None
            
    def _prepare_optional_tensor(self, data):
        """
        Prepare optional tensor for InstanceMemory.
        
        Args:
            data: Data to convert to tensor
            
        Returns:
            torch.Tensor or None: Prepared tensor
        """
        if data is None:
            return None
            
        try:
            # Convert to tensor if needed
            if isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
            elif isinstance(data, torch.Tensor):
                tensor = data
            elif isinstance(data, list):
                tensor = torch.tensor(data)
            else:
                logger.warning(f"Unsupported data type for tensor conversion: {type(data)}")
                return None
                
            # Move to specified device if available
            if self.gpu_device is not None:
                tensor = tensor.to(self.gpu_device)
                
            return tensor
            
        except Exception as e:
            logger.error(f"Error preparing optional tensor: {e}")
            return None

    def _update_grid_from_voxel(self):
        """Update occupancy grid from voxel map."""
        try:
            if not hasattr(self, 'voxel_map') or self.voxel_map is None:
                return False
            
            if not hasattr(self, 'grid_map') or self.grid_map is None:
                return False
            
            # Get voxels from voxel map
            voxels = self.voxel_map.get_occupied_voxels()
            
            if voxels.shape[0] == 0:
                return False
            
            # Extract 2D points from 3D voxels (x,y)
            points_2d = voxels[:, :2]
            
            # Use the point cloud update method
            return self._update_grid_from_point_cloud(points_2d)
        except Exception as e:
            logger.error(f"Error in _update_grid_from_voxel: {e}")
            return False

    def match_pose_graph_loop(self, observation, pose_graph_nodes, max_distance=None):
        """
        Match the current observation to pose graph nodes.
        Handles different timestamp attribute formats.
        
        Args:
            observation: Observation object
            pose_graph_nodes: List of pose graph nodes
            max_distance: Maximum distance for matching
            
        Returns:
            tuple: (matched_node, distance)
        """
        try:
            # Get timestamp from observation with multiple fallbacks
            observation_timestamp = None
            
            # Try different timestamp attributes
            if hasattr(observation, 'timestamp'):
                observation_timestamp = observation.timestamp
            elif hasattr(observation, 'seq_id'):
                observation_timestamp = observation.seq_id  # Use seq_id as timestamp
            elif hasattr(observation, 'header') and hasattr(observation.header, 'stamp'):
                # ROS header format
                observation_timestamp = observation.header.stamp.sec + observation.header.stamp.nanosec * 1e-9
            
            if observation_timestamp is None:
                # If no timestamp found, use current time as fallback
                observation_timestamp = time.time()
                logger.warning("No timestamp found in observation, using current time")
            
            # Rest of the matching logic...
            best_match = None
            best_distance = float('inf') if max_distance is None else max_distance
            
            for node in pose_graph_nodes:
                # Get node timestamp with fallbacks
                node_timestamp = None
                
                if hasattr(node, 'timestamp'):
                    node_timestamp = node.timestamp
                elif hasattr(node, 'seq_id'):
                    node_timestamp = node.seq_id
                elif hasattr(node, 'header') and hasattr(node.header, 'stamp'):
                    node_timestamp = node.header.stamp.sec + node.header.stamp.nanosec * 1e-9
                
                if node_timestamp is None:
                    # Skip nodes without timestamp
                    continue
                
                # Calculate time distance
                time_distance = abs(observation_timestamp - node_timestamp)
                
                # If within time threshold, check spatial distance
                if time_distance < best_distance:
                    # Calculate spatial distance
                    spatial_distance = self._calculate_spatial_distance(observation, node)
                    
                    # Update best match if better
                    if spatial_distance < best_distance:
                        best_match = node
                        best_distance = spatial_distance
            
            return best_match, best_distance
        except Exception as e:
            logger.error(f"Error in match_pose_graph_loop: {e}")
            traceback.print_exc()
            return None, float('inf')

    def start_visualization(self, update_rate=1.0):
        """Start a thread for continuous visualization of the mapping process.
        
        Args:
            update_rate: Visualization update rate in Hz
        """
        if not hasattr(self, 'visualization_thread'):
            self.visualization_running = True
            
            def visualization_loop():
                while self.visualization_running:
                    try:
                        if self.voxel_map is not None:
                            # Get current robot pose
                            robot_pose = self.ros_client.get_pose()
                            
                            # Show map with instances
                            self.voxel_map.show(
                                xyt=robot_pose,
                                instances=True,
                                footprint=self.ros_client.get_robot_model().get_footprint()
                            )
                    except Exception as e:
                        logger.error(f"Error in visualization: {e}")
                    
                    time.sleep(1.0 / update_rate)
            
            self.visualization_thread = threading.Thread(target=visualization_loop)
            self.visualization_thread.daemon = True
            self.visualization_thread.start()
            logger.info("Started real-time visualization")

    def _calculate_spatial_distance(self, obs1, obs2):
        """
        Calculate spatial distance between two observations.
        
        Args:
            obs1: First observation
            obs2: Second observation
            
        Returns:
            float: Distance
        """
        # Try to get position from GPS
        if hasattr(obs1, 'gps') and hasattr(obs2, 'gps'):
            return np.linalg.norm(obs1.gps - obs2.gps)
        
        # Try to get position from pose
        if hasattr(obs1, 'pose') and hasattr(obs2, 'pose'):
            return np.linalg.norm(obs1.pose[:2] - obs2.pose[:2])
        
        # Default large distance if no position available
        return float('inf')