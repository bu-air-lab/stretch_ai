# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GeneralTaskState(Enum):
    NOT_STARTED = 0
    PREPPING = 1
    DOING_TASK = 2
    IDLE = 3
    STOP = 4


class Action:
    """Controls."""


class DiscreteNavigationAction(Action, Enum):
    """Discrete navigation controls."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PICK_OBJECT = 4
    PLACE_OBJECT = 5
    NAVIGATION_MODE = 6
    MANIPULATION_MODE = 7
    POST_NAV_MODE = 8
    # Arm extension to a fixed position and height
    EXTEND_ARM = 9
    EMPTY_ACTION = 10
    # Simulation only actions
    SNAP_OBJECT = 11
    DESNAP_OBJECT = 12
    # Discrete gripper commands
    OPEN_GRIPPER = 13
    CLOSE_GRIPPER = 14


class ContinuousNavigationAction(Action):
    xyt: np.ndarray

    def __init__(self, xyt: np.ndarray):
        if not len(xyt) == 3:
            raise RuntimeError("continuous navigation action space has 3 dimensions, x y and theta")
        self.xyt = xyt

    def __str__(self):
        return f"xyt={self.xyt}"


class ContinuousFullBodyAction:
    xyt: np.ndarray
    joints: np.ndarray

    def __init__(self, joints: np.ndarray, xyt: np.ndarray = None):
        """Create full-body continuous action"""
        if xyt is not None and not len(xyt) == 3:
            raise RuntimeError("continuous navigation action space has 3 dimensions, x y and theta")
        self.xyt = xyt
        # Joint states in robot action format
        self.joints = joints


class ContinuousEndEffectorAction:
    pos: np.ndarray
    ori: np.ndarray
    g: np.ndarray
    num_actions: int

    def __init__(
        self,
        pos: np.ndarray = None,
        ori: np.ndarray = None,
        g: np.ndarray = None,
    ):
        """Create end-effector continuous action; moves to 6D pose and activates gripper"""
        if (
            pos is not None
            and ori is not None
            and g is not None
            and not (pos.shape[1] + ori.shape[1] + g.shape[1]) == 8
        ):
            raise RuntimeError(
                "continuous end-effector action space has 8 dimensions: pos=3, ori=4, gripper=1"
            )
        self.pos = pos
        self.ori = ori
        self.g = g
        self.num_actions = pos.shape[0]


class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS_NAVIGATION = 1
    CONTINUOUS_MANIPULATION = 2
    CONTINUOUS_EE_MANIPULATION = 3


class HybridAction(Action):
    """Convenience for supporting multiple action types - provides handling to make sure we have the right class at any particular time"""

    action_type: ActionType
    action: Action

    def __init__(
        self,
        action=None,
        xyt: np.ndarray = None,
        joints: np.ndarray = None,
        pos: np.ndarray = None,
        ori: np.ndarray = None,
        gripper: np.ndarray = None,
    ):
        """Make sure that we were passed a useful generic action here. Process it into something useful."""
        if action is not None:
            if isinstance(action, HybridAction):
                self.action_type = action.action_type
            if isinstance(action, DiscreteNavigationAction):
                self.action_type = ActionType.DISCRETE
            elif isinstance(action, ContinuousNavigationAction):
                self.action_type = ActionType.CONTINUOUS_NAVIGATION
            elif isinstance(action, ContinuousEndEffectorAction):
                self.action_type = ActionType.CONTINUOUS_EE_MANIPULATION
            else:
                self.action_type = ActionType.CONTINUOUS_MANIPULATION
        elif joints is not None:
            self.action_type = ActionType.CONTINUOUS_MANIPULATION
            action = ContinuousFullBodyAction(joints, xyt)
        elif xyt is not None:
            self.action_type = ActionType.CONTINUOUS_NAVIGATION
            action = ContinuousNavigationAction(xyt)
        elif pos is not None:
            self.action_type = ActionType.CONTINUOUS_EE_MANIPULATION
            action = ContinuousEndEffectorAction(pos, ori, gripper)
        else:
            raise RuntimeError("Cannot create HybridAction without any action!")
        if isinstance(action, HybridAction):
            # TODO: should we copy like this?
            self.action_type = action.action_type
            action = action.action
            # But more likely this was a mistake so let's actually throw an error
            raise RuntimeError("Do not pass a HybridAction when creating another HybridAction!")
        self.action = action

    def is_discrete(self):
        """Let environment know if we need to handle a discrete action"""
        return self.action_type == ActionType.DISCRETE

    def is_navigation(self):
        return self.action_type == ActionType.CONTINUOUS_NAVIGATION

    def is_manipulation(self):
        return self.action_type in [
            ActionType.CONTINUOUS_MANIPULATION,
            ActionType.CONTINUOUS_EE_MANIPULATION,
        ]

    def get(self):
        """Extract continuous component of the command and return it."""
        if self.action_type == ActionType.DISCRETE:
            return self.action
        elif self.action_type == ActionType.CONTINUOUS_NAVIGATION:
            return self.action.xyt
        elif self.action_type == ActionType.CONTINUOUS_EE_MANIPULATION:
            return self.action.pos, self.action.ori, self.action.g
        else:
            # Extract both the joints and the waypoint target
            return self.action.joints, self.action.xyt


@dataclass
class Pose:
    position: np.ndarray
    orientation: np.ndarray


@dataclass
class PointCloud:
    """Point cloud data structure for LiDAR and depth processing.
    
    This class is designed to be compatible with ROS PointCloud2 messages
    while providing convenient conversion methods for use with Stretch AI's
    perception pipeline.
    """
    # Core point cloud data
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    colors: Optional[np.ndarray] = None
    intensities: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    
    # Metadata
    frame_id: str = ""
    timestamp: float = 0.0
    height: int = 1
    width: int = 0
    is_dense: bool = True
    
    # Additional fields that might be in the point cloud
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize width based on points if not explicitly set."""
        if self.width == 0 and self.points is not None:
            self.width = self.points.shape[0]
    
    @classmethod
    def from_ros_pointcloud2(cls, pc2_msg, field_names=None):
        """Convert ROS PointCloud2 message to this format.
        
        Args:
            pc2_msg: A ROS PointCloud2 message
            field_names: List of field names to extract (default: ['x', 'y', 'z'])
        
        Returns:
            PointCloud: A new PointCloud instance
        """
        # This is just a placeholder - actual implementation would use
        # methods from sensor_msgs.point_cloud2 or similar libraries
        # depending on how you're interfacing with ROS
        
        # Example implementation layout:
        # 1. Extract fields from the message
        # 2. Create numpy arrays for points, colors, etc.
        # 3. Return a new PointCloud object
        
        # For a real implementation, consider using:
        # from sensor_msgs_py import point_cloud2
        # or the appropriate ROS2 equivalent
        
        points = np.zeros((0, 3))  # Placeholder
        
        return cls(
            points=points,
            frame_id=pc2_msg.header.frame_id,
            timestamp=pc2_msg.header.stamp.sec + pc2_msg.header.stamp.nanosec * 1e-9,
            height=pc2_msg.height,
            width=pc2_msg.width,
            is_dense=pc2_msg.is_dense
        )
    
    @classmethod
    def from_depth_image(cls, depth_image: np.ndarray, camera_K: np.ndarray, 
                        rgb_image: Optional[np.ndarray] = None, 
                        frame_id: str = "", timestamp: float = 0.0):
        """Convert depth image to point cloud.
        
        Args:
            depth_image: Depth image (H, W) in meters
            camera_K: Camera intrinsics matrix (3, 3)
            rgb_image: Optional RGB image (H, W, 3)
            frame_id: Frame ID
            timestamp: Timestamp
            
        Returns:
            PointCloud: A new PointCloud instance
        """
        # Get the camera intrinsics
        fx, fy = camera_K[0, 0], camera_K[1, 1]
        cx, cy = camera_K[0, 2], camera_K[1, 2]
        
        # Get the image size
        h, w = depth_image.shape
        
        # Create the grid
        x = np.arange(w)
        y = np.arange(h)
        xv, yv = np.meshgrid(x, y)
        
        # Compute 3D points
        z = depth_image.flatten()
        mask = np.isfinite(z) & (z > 0)
        
        x = ((xv.flatten() - cx) * z / fx)[mask]
        y = ((yv.flatten() - cy) * z / fy)[mask]
        z = z[mask]
        
        points = np.column_stack((x, y, z))
        
        # Extract colors if RGB image is provided
        colors = None
        if rgb_image is not None:
            colors = rgb_image.reshape(-1, 3)[mask]
        
        return cls(
            points=points,
            colors=colors,
            frame_id=frame_id,
            timestamp=timestamp,
            height=1,
            width=points.shape[0],
            is_dense=True
        )
    
    def transform(self, transform_matrix: np.ndarray) -> 'PointCloud':
        """Transform point cloud using a 4x4 transformation matrix.
        
        Args:
            transform_matrix: 4x4 transformation matrix
            
        Returns:
            PointCloud: Transformed point cloud
        """
        if self.points.shape[0] == 0:
            return self
            
        # Create homogeneous coordinates
        homogeneous_points = np.hstack((self.points, np.ones((self.points.shape[0], 1))))
        
        # Apply transformation
        transformed_points = np.dot(homogeneous_points, transform_matrix.T)[:, :3]
        
        # Transform normals if they exist
        transformed_normals = None
        if self.normals is not None:
            # For normals, we use only the rotation part
            rotation_matrix = transform_matrix[:3, :3]
            transformed_normals = np.dot(self.normals, rotation_matrix.T)
        
        # Create a new point cloud with transformed points
        return PointCloud(
            points=transformed_points,
            colors=self.colors,
            intensities=self.intensities,
            normals=transformed_normals,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            height=self.height,
            width=self.width,
            is_dense=self.is_dense,
            fields=self.fields
        )
    
    def voxel_downsample(self, voxel_size: float) -> 'PointCloud':
        """Downsample point cloud using voxel grid.
        
        Args:
            voxel_size: Size of voxel
            
        Returns:
            PointCloud: Downsampled point cloud
        """
        if self.points.shape[0] == 0:
            return self
            
        # Compute voxel indices for each point
        voxel_indices = np.floor(self.points / voxel_size).astype(int)
        
        # Create a dictionary to store points for each voxel
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            idx_tuple = tuple(idx)
            if idx_tuple in voxel_dict:
                voxel_dict[idx_tuple].append(i)
            else:
                voxel_dict[idx_tuple] = [i]
        
        # Compute the average of points in each voxel
        downsampled_points = []
        downsampled_colors = [] if self.colors is not None else None
        downsampled_intensities = [] if self.intensities is not None else None
        downsampled_normals = [] if self.normals is not None else None
        
        for indices in voxel_dict.values():
            downsampled_points.append(np.mean(self.points[indices], axis=0))
            
            if self.colors is not None:
                downsampled_colors.append(np.mean(self.colors[indices], axis=0))
            
            if self.intensities is not None:
                downsampled_intensities.append(np.mean(self.intensities[indices], axis=0))
                
            if self.normals is not None:
                normals = np.mean(self.normals[indices], axis=0)
                downsampled_normals.append(normals / np.linalg.norm(normals))
        
        # Convert lists to numpy arrays
        downsampled_points = np.array(downsampled_points)
        downsampled_colors = np.array(downsampled_colors) if downsampled_colors else None
        downsampled_intensities = np.array(downsampled_intensities) if downsampled_intensities else None
        downsampled_normals = np.array(downsampled_normals) if downsampled_normals else None
        
        return PointCloud(
            points=downsampled_points,
            colors=downsampled_colors,
            intensities=downsampled_intensities,
            normals=downsampled_normals,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            height=1,
            width=downsampled_points.shape[0],
            is_dense=self.is_dense,
            fields=self.fields
        )
    
    def to_numpy(self) -> np.ndarray:
        """Return points as numpy array."""
        return self.points
        
    def to_ros_pointcloud2(self):
        """Convert to ROS PointCloud2 message.
        
        This is a placeholder - implementation depends on your ROS setup
        and which ROS version (1 or 2) you're using to create the message.
        
        Returns:
            PointCloud2: A ROS PointCloud2 message
        """
        # Example implementation for ROS1:
        # from sensor_msgs.msg import PointCloud2, PointField
        # import sensor_msgs.point_cloud2 as pc2
        # 
        # fields = [
        #     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        # ]
        # 
        # if self.colors is not None:
        #     fields.extend([
        #         PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
        #         PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
        #         PointField(name='b', offset=14, datatype=PointField.UINT8, count=1)
        #     ])
        # 
        # return pc2.create_cloud(
        #     Header(frame_id=self.frame_id),
        #     fields,
        #     self.points
        # )
        
        # For now, we'll just return None as a placeholder
        return None


@dataclass
class Observations:
    """Sensor observations."""

    # --------------------------------------------------------
    # Typed observations
    # --------------------------------------------------------

    # Joint states
    # joint_positions: np.ndarray

    # Pose
    # TODO: add these instead of gps + compass
    # base_pose: Pose
    # ee_pose: Pose

    # Pose
    gps: np.ndarray  # (x, y) where positive x is forward, positive y is translation to left in meters
    compass: np.ndarray  # positive theta is rotation to left in radians - consistent with robot

    # Camera
    rgb: np.ndarray  # (camera_height, camera_width, 3) in [0, 255]
    depth: np.ndarray  # (camera_height, camera_width) in meters
    xyz: Optional[np.ndarray] = None  # (camera_height, camera_width, 3) in camera coordinates
    semantic: Optional[
        np.ndarray
    ] = None  # (camera_height, camera_width) in [0, num_sem_categories - 1]
    camera_K: Optional[np.ndarray] = None  # (3, 3) camera intrinsics matrix

    # Pose of the camera in world coordinates
    camera_pose: Optional[np.ndarray] = None

    # End effector camera
    ee_rgb: Optional[np.ndarray] = None  # (camera_height, camera_width, 3) in [0, 255]
    ee_depth: Optional[np.ndarray] = None  # (camera_height, camera_width) in meters
    ee_xyz: Optional[np.ndarray] = None  # (camera_height, camera_width, 3) in camera coordinates
    ee_semantic: Optional[
        np.ndarray
    ] = None  # (camera_height, camera_width) in [0, num_sem_categories - 1]
    ee_camera_K: Optional[np.ndarray] = None  # (3, 3) camera intrinsics matrix

    # Pose of the end effector camera in world coordinates
    ee_camera_pose: Optional[np.ndarray] = None

    # Pose of the end effector grasp center in world coordinates
    ee_pose: Optional[np.ndarray] = None

    # Instance IDs per observation frame
    # Size: (camera_height, camera_width)
    # Range: 0 to max int
    instance: Optional[np.ndarray] = None

    # Optional third-person view from simulation
    third_person_image: Optional[np.ndarray] = None

    # lidar
    lidar_points: Optional[np.ndarray] = None
    lidar_timestamp: Optional[int] = None
    
    # Point cloud processed from lidar data
    point_cloud: Optional[PointCloud] = None

    # Proprioreception
    joint: Optional[np.ndarray] = None  # joint positions of the robot
    joint_velocities: Optional[np.ndarray] = None  # joint velocities of the robot
    relative_resting_position: Optional[
        np.ndarray
    ] = None  # end-effector position relative to the desired resting position
    is_holding: Optional[np.ndarray] = None  # whether the agent is holding the object
    # --------------------------------------------------------
    # Untyped task-specific observations
    # --------------------------------------------------------

    task_observations: Optional[Dict[str, Any]] = None

    # Sequence number - which message was this?
    seq_id: int = -1

    # True if in simulation
    is_simulation: bool = False

    # True if matched with a pose graph node
    is_pose_graph_node: bool = False

    # Timestamp of matched pose graph node
    pose_graph_timestamp: Optional[int] = None

    # Initial pose graph pose. GPS and compass.
    initial_pose_graph_gps: Optional[np.ndarray] = None
    initial_pose_graph_compass: Optional[np.ndarray] = None
    
    # Segway-specific observations
    wheel_odometry: Optional[np.ndarray] = None  # Wheel odometry from the Segway base
    imu_data: Optional[Dict[str, np.ndarray]] = None  # IMU data from the Segway
    battery_status: Optional[Dict[str, Any]] = None  # Battery status

    def compute_xyz(self, scaling: float = 1e-3) -> Optional[np.ndarray]:
        """Compute xyz from depth and camera intrinsics."""
        if self.depth is not None and self.camera_K is not None:
            self.xyz = self.depth_to_xyz(self.depth * scaling, self.camera_K)
        return self.xyz

    def compute_ee_xyz(self, scaling: float = 1e-3) -> Optional[np.ndarray]:
        """Compute xyz from depth and camera intrinsics."""
        if self.ee_depth is not None and self.ee_camera_K is not None:
            self.ee_xyz = self.depth_to_xyz(self.ee_depth * scaling, self.ee_camera_K)
        return self.ee_xyz

    def depth_to_xyz(self, depth, camera_K) -> np.ndarray:
        """Convert depth image to xyz point cloud."""
        # Get the camera intrinsics
        fx, fy, cx, cy = camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]
        # Get the image size
        h, w = depth.shape
        # Create the grid
        x = np.tile(np.arange(w), (h, 1))
        y = np.tile(np.arange(h).reshape(-1, 1), (1, w))
        # Compute the xyz
        x = (x - cx) * depth / fx
        y = (y - cy) * depth / fy
        return np.stack([x, y, depth], axis=-1)

    def get_ee_xyz_in_world_frame(self, scaling: float = 1.0) -> Optional[np.ndarray]:
        """Get the end effector xyz in world frame."""
        if self.ee_xyz is None:
            self.compute_ee_xyz(scaling=scaling)
        if self.ee_xyz is not None and self.ee_camera_pose is not None:
            return self.transform_points(self.ee_xyz, self.ee_camera_pose)
        return None

    def get_xyz_in_world_frame(self, scaling: float = 1.0) -> Optional[np.ndarray]:
        """Get the xyz in world frame.

        Args:
            scaling: scaling factor for xyz"""
        if self.xyz is None:
            self.compute_xyz(scaling=scaling)
        if self.xyz is not None and self.camera_pose is not None:
            return self.transform_points(self.xyz, self.camera_pose)
        return None

    def transform_points(self, points: np.ndarray, pose: np.ndarray):
        """Transform points to world frame.

        Args:
            points: points in camera frame
            pose: pose of the camera"""
        assert points.shape[-1] == 3, "Points should be in 3D"
        assert pose.shape == (4, 4), "Pose should be a 4x4 matrix"
        return np.dot(points, pose[:3, :3].T) + pose[:3, 3]
        
    def process_lidar_to_pointcloud(self) -> Optional[PointCloud]:
        """Process lidar points to create a point cloud."""
        if self.lidar_points is None:
            return None
            
        # Create a point cloud from the lidar points
        self.point_cloud = PointCloud(
            points=self.lidar_points,
            intensities=None,  # Add if your LiDAR provides intensity values
            frame_id="lidar",
            timestamp=self.lidar_timestamp if self.lidar_timestamp is not None else 0.0,
            height=1,
            width=self.lidar_points.shape[0],
            is_dense=True
        )
        
        return self.point_cloud

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observations":
        """Create observations from dictionary."""
        return cls(
            gps=data.get("gps"),
            compass=data.get("compass"),
            rgb=data.get("rgb"),
            depth=data.get("depth"),
            xyz=data.get("xyz"),
            semantic=data.get("semantic"),
            camera_K=data.get("camera_K"),
            camera_pose=data.get("camera_pose"),
            ee_rgb=data.get("ee_rgb"),
            ee_depth=data.get("ee_depth"),
            ee_xyz=data.get("ee_xyz"),
            ee_semantic=data.get("ee_semantic"),
            ee_camera_K=data.get("ee_camera_K"),
            ee_camera_pose=data.get("ee_camera_pose"),
            ee_pose=data.get("ee_pose"),
            instance=data.get("instance"),
            third_person_image=data.get("third_person_image"),
            lidar_points=data.get("lidar_points"),
            lidar_timestamp=data.get("lidar_timestamp"),
            point_cloud=data.get("point_cloud"),
            joint=data.get("joint"),
            relative_resting_position=data.get("relative_resting_position"),
            is_holding=data.get("is_holding"),
            task_observations=data.get("task_observations"),
            seq_id=data.get("seq_id"),
            is_simulation=data.get("is_simulation"),
            is_pose_graph_node=data.get("is_pose_graph_node"),
            pose_graph_timestamp=data.get("pose_graph_timestamp"),
            initial_pose_graph_gps=data.get("initial_pose_graph_gps"),
            initial_pose_graph_compass=data.get("initial_pose_graph_compass"),
            wheel_odometry=data.get("wheel_odometry"),
            imu_data=data.get("imu_data"),
            battery_status=data.get("battery_status"),
        )
        
    @classmethod
    def from_ros_messages(cls, 
                          odom_msg=None, 
                          lidar_msg=None, 
                          rgb_msg=None, 
                          depth_msg=None,
                          camera_info_msg=None, 
                          imu_msg=None,
                          battery_msg=None):
        """Create observations from ROS messages.
        
        This method is specific to the Segway robot integration and handles
        converting ROS message data into the Observations format.
        
        Args:
            odom_msg: ROS odometry message
            lidar_msg: ROS LaserScan/PointCloud2 message
            rgb_msg: ROS Image message for RGB
            depth_msg: ROS Image message for depth
            camera_info_msg: ROS CameraInfo message
            imu_msg: ROS IMU message
            battery_msg: ROS BatteryState message
            
        Returns:
            Observations: A new Observations instance
        """
        # This is a placeholder implementation - you'll need to fill in
        # the actual ROS message conversions based on your robot setup
        
        # Example odometry conversion
        gps = np.array([0.0, 0.0])
        compass = np.array([0.0])
        wheel_odometry = None
        
        if odom_msg is not None:
            # Extract position and orientation from odometry
            gps = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
            
            # Convert quaternion to yaw angle
            # Using a simplified conversion for this example
            qx = odom_msg.pose.pose.orientation.x
            qy = odom_msg.pose.pose.orientation.y
            qz = odom_msg.pose.pose.orientation.z
            qw = odom_msg.pose.pose.orientation.w
            
            # Simplified quaternion to Euler conversion for yaw only
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            compass = np.array([np.arctan2(siny_cosp, cosy_cosp)])
            
            # Store wheel odometry data
            wheel_odometry = np.array([
                odom_msg.twist.twist.linear.x,
                odom_msg.twist.twist.linear.y,
                odom_msg.twist.twist.angular.z
            ])
        
        # Example camera info conversion
        camera_K = None
        if camera_info_msg is not None:
            camera_K = np.array(camera_info_msg.K).reshape(3, 3)
        
        # Example LiDAR conversion
        lidar_points = None
        lidar_timestamp = None
        point_cloud = None
        
        if lidar_msg is not None:
            # Convert based on message type - this is just an example
            # For LaserScan messages (common in ROS 1):
            if hasattr(lidar_msg, 'ranges'):
                # Convert range/bearing to Cartesian points
                ranges = np.array(lidar_msg.ranges)
                angles = np.arange(
                    lidar_msg.angle_min,
                    lidar_msg.angle_max + lidar_msg.angle_increment,
                    lidar_msg.angle_increment
                )
                
                # Filter out invalid readings
                valid = np.isfinite(ranges) & (ranges > 0) & (ranges < lidar_msg.range_max)
                filtered_ranges = ranges[valid]
                filtered_angles = angles[valid]
                
                # Convert to Cartesian
                x = filtered_ranges * np.cos(filtered_angles)
                y = filtered_ranges * np.sin(filtered_angles)
                z = np.zeros_like(x)
                
                lidar_points = np.column_stack((x, y, z))
                lidar_timestamp = lidar_msg.header.stamp.sec + lidar_msg.header.stamp.nanosec * 1e-9
                
                # Create a point cloud
                point_cloud = PointCloud(
                    points=lidar_points,
                    frame_id=lidar_msg.header.frame_id,
                    timestamp=lidar_timestamp,
                    height=1,
                    width=lidar_points.shape[0],
                    is_dense=True
                )
        
        # Example IMU conversion
        imu_data = None
        if imu_msg is not None:
            imu_data = {
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
        
        # Example battery conversion
        battery_status = None
        if battery_msg is not None:
            battery_status = {
                'voltage': battery_msg.voltage,
                'current': battery_msg.current,
                'percentage': battery_msg.percentage,
                'power_supply_status': battery_msg.power_supply_status
            }
        
        return cls(
            gps=gps,
            compass=compass,
            rgb=None,  # You'd need to convert the RGB image message here
            depth=None,  # You'd need to convert the depth image message here
            camera_K=camera_K,
            lidar_points=lidar_points,
            lidar_timestamp=lidar_timestamp,
            point_cloud=point_cloud,
            wheel_odometry=wheel_odometry,
            imu_data=imu_data,
            battery_status=battery_status,
            seq_id=0,  # You might want to use a sequence number from one of the messages
            is_simulation=False
        )