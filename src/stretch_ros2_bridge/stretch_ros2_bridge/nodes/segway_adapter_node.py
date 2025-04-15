#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sys
import signal
import argparse
import numpy as np
from threading import Lock

# Import your Segway client
from stretch.agent.segway_ros_client import SegwayRosClient

# Camera and perception imports
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

class SegwayAdapterNode(Node):
    """
    Adapter node that bridges between Stretch AI and the Segway robot.
    This node handles initialization of the SegwayRosClient and
    exposes its functionality to the Stretch AI framework.
    
    Includes support for cameras and perception for mapping and navigation.
    """
    
    def __init__(self, args=None):
        super().__init__('segway_adapter_node')
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Segway Adapter Node')
        parser.add_argument('--cmd-vel', type=str, default='/cmd_vel',
                           help='Command velocity topic')
        parser.add_argument('--odom', type=str, default='/odom',
                           help='Odometry topic')
        parser.add_argument('--joint-states', type=str, default='',
                           help='Joint states topic (leave empty to disable arm)')
        parser.add_argument('--trajectory-action', type=str, default='',
                           help='Trajectory action name (leave empty to disable arm)')
        parser.add_argument('--use-camera', type=str, default='true',
                           help='Enable camera support (true/false)')
        parser.add_argument('--camera-type', type=str, default='realsense',
                           choices=['realsense', 'kinect', 'zed', 'other'],
                           help='Type of depth camera used')
        parser.add_argument('--rgb-topic', type=str, default='/camera/color/image_raw',
                           help='RGB camera topic')
        parser.add_argument('--depth-topic', type=str, default='/camera/depth/image_rect_raw',
                           help='Depth camera topic')
        parser.add_argument('--camera-info-topic', type=str, default='/camera/color/camera_info',
                           help='Camera info topic')
        parser.add_argument('--pointcloud-topic', type=str, default='/camera/depth/points',
                           help='Point cloud topic')
        parser.add_argument('--rate', type=int, default=20,
                           help='Spin rate in Hz')
        
        # Parse args
        parsed_args, _ = parser.parse_known_args(args)
        
        # Store camera settings
        self.use_camera = parsed_args.use_camera.lower() == 'true'
        self.camera_type = parsed_args.camera_type
        self.rgb_topic = parsed_args.rgb_topic
        self.depth_topic = parsed_args.depth_topic
        self.camera_info_topic = parsed_args.camera_info_topic
        self.pointcloud_topic = parsed_args.pointcloud_topic
        
        # Log the configuration
        self.get_logger().info('Starting Segway Adapter with:')
        self.get_logger().info(f'  - cmd_vel: {parsed_args.cmd_vel}')
        self.get_logger().info(f'  - odom: {parsed_args.odom}')
        self.get_logger().info(f'  - joint_states: {parsed_args.joint_states}')
        self.get_logger().info(f'  - trajectory_action: {parsed_args.trajectory_action}')
        self.get_logger().info(f'  - use_camera: {self.use_camera}')
        if self.use_camera:
            self.get_logger().info(f'  - camera_type: {self.camera_type}')
            self.get_logger().info(f'  - rgb_topic: {self.rgb_topic}')
            self.get_logger().info(f'  - depth_topic: {self.depth_topic}')
        self.get_logger().info(f'  - rate: {parsed_args.rate}')
        
        # Define UR5e joint names if arm support is enabled
        ur_joints = None
        if parsed_args.joint_states and parsed_args.trajectory_action:
            ur_joints = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint"
            ]
            self.get_logger().info(f'  - arm joints: {ur_joints}')
        
        # Create the Segway client
        self.segway_client = SegwayRosClient(
            node_name="segway_client_embedded",  # Use a different name to avoid conflicts
            cmd_vel_topic=parsed_args.cmd_vel,
            odom_topic=parsed_args.odom,
            manip_joint_states_topic=parsed_args.joint_states,
            manip_trajectory_action=parsed_args.trajectory_action,
            manip_joint_names=ur_joints,
            gripper_joint_name="",  # Add gripper support if needed
            spin_rate_hz=parsed_args.rate
        )
        
        # Initialize camera-related components if enabled
        if self.use_camera:
            self._setup_camera_components()
        
        # Create transform broadcasters for TF tree
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Broadcast static transforms for camera mount
        if self.use_camera:
            self._broadcast_camera_transforms()
        
        # Start the client
        self.segway_client.start()
        self.get_logger().info('Segway client started')
        
        # Create timer for health checks
        self._health_timer = self.create_timer(1.0, self._health_check)
        
        # Create timer for TF broadcasts (if needed)
        if self.use_camera:
            self._tf_timer = self.create_timer(0.1, self._publish_camera_tf)
        
        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_camera_components(self):
        """Set up camera subscribers and publishers."""
        # Initialize CV bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Data storage
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_camera_info = None
        self.latest_pointcloud = None
        self.camera_locks = {
            'rgb': Lock(),
            'depth': Lock(),
            'info': Lock(),
            'pointcloud': Lock()
        }
        
        # Define camera frame IDs based on camera type
        if self.camera_type == 'realsense':
            self.camera_frame_id = 'camera_link'
            self.rgb_frame_id = 'camera_color_optical_frame'
            self.depth_frame_id = 'camera_depth_optical_frame'
        elif self.camera_type == 'kinect':
            self.camera_frame_id = 'camera_base'
            self.rgb_frame_id = 'rgb_camera_link'
            self.depth_frame_id = 'depth_camera_link'
        elif self.camera_type == 'zed':
            self.camera_frame_id = 'zed_camera_center'
            self.rgb_frame_id = 'zed_left_camera_frame'
            self.depth_frame_id = 'zed_left_camera_frame'
        else:
            self.camera_frame_id = 'camera_link'
            self.rgb_frame_id = 'camera_rgb_optical_frame'
            self.depth_frame_id = 'camera_depth_optical_frame'
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self._rgb_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self._depth_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            10
        )
        
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self._pointcloud_callback,
            10
        )
        
        # Republishers (to standardize topic names for Stretch AI)
        self.rgb_pub = self.create_publisher(
            Image,
            '/segway/camera/color/image_raw',
            10
        )
        
        self.depth_pub = self.create_publisher(
            Image,
            '/segway/camera/depth/image_rect',
            10
        )
        
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            '/segway/camera/color/camera_info',
            10
        )
        
        self.pointcloud_pub = self.create_publisher(
            PointCloud2,
            '/segway/camera/depth/points',
            10
        )
        
        self.get_logger().info("Camera components initialized")
    
    def _rgb_callback(self, msg):
        """Handle RGB image messages."""
        try:
            # Store the latest RGB image
            with self.camera_locks['rgb']:
                self.latest_rgb_image = msg
            
            # Republish with standardized frame ID and topic
            if self.rgb_pub.get_subscription_count() > 0:
                # Only republish if someone is listening
                new_msg = Image()
                new_msg.header = msg.header
                new_msg.header.frame_id = self.rgb_frame_id
                new_msg.height = msg.height
                new_msg.width = msg.width
                new_msg.encoding = msg.encoding
                new_msg.is_bigendian = msg.is_bigendian
                new_msg.step = msg.step
                new_msg.data = msg.data
                self.rgb_pub.publish(new_msg)
        except Exception as e:
            self.get_logger().error(f"Error in RGB callback: {e}")
    
    def _depth_callback(self, msg):
        """Handle depth image messages."""
        try:
            # Store the latest depth image
            with self.camera_locks['depth']:
                self.latest_depth_image = msg
            
            # Republish with standardized frame ID and topic
            if self.depth_pub.get_subscription_count() > 0:
                new_msg = Image()
                new_msg.header = msg.header
                new_msg.header.frame_id = self.depth_frame_id
                new_msg.height = msg.height
                new_msg.width = msg.width
                new_msg.encoding = msg.encoding
                new_msg.is_bigendian = msg.is_bigendian
                new_msg.step = msg.step
                new_msg.data = msg.data
                self.depth_pub.publish(new_msg)
        except Exception as e:
            self.get_logger().error(f"Error in depth callback: {e}")
    
    def _camera_info_callback(self, msg):
        """Handle camera info messages."""
        try:
            # Store the latest camera info
            with self.camera_locks['info']:
                self.latest_camera_info = msg
            
            # Republish with standardized frame ID and topic
            if self.camera_info_pub.get_subscription_count() > 0:
                new_msg = CameraInfo()
                new_msg.header = msg.header
                new_msg.header.frame_id = self.rgb_frame_id
                new_msg.height = msg.height
                new_msg.width = msg.width
                new_msg.distortion_model = msg.distortion_model
                new_msg.d = msg.d
                new_msg.k = msg.k
                new_msg.r = msg.r
                new_msg.p = msg.p
                new_msg.binning_x = msg.binning_x
                new_msg.binning_y = msg.binning_y
                new_msg.roi = msg.roi
                self.camera_info_pub.publish(new_msg)
        except Exception as e:
            self.get_logger().error(f"Error in camera info callback: {e}")
    
    def _pointcloud_callback(self, msg):
        """Handle point cloud messages."""
        try:
            # Store the latest point cloud
            with self.camera_locks['pointcloud']:
                self.latest_pointcloud = msg
            
            # Republish with standardized frame ID and topic
            if self.pointcloud_pub.get_subscription_count() > 0:
                new_msg = PointCloud2()
                new_msg.header = msg.header
                new_msg.header.frame_id = self.depth_frame_id
                new_msg.height = msg.height
                new_msg.width = msg.width
                new_msg.fields = msg.fields
                new_msg.is_bigendian = msg.is_bigendian
                new_msg.point_step = msg.point_step
                new_msg.row_step = msg.row_step
                new_msg.data = msg.data
                new_msg.is_dense = msg.is_dense
                self.pointcloud_pub.publish(new_msg)
        except Exception as e:
            self.get_logger().error(f"Error in point cloud callback: {e}")
    
    def _broadcast_camera_transforms(self):
        """Publish static transforms for camera mounting."""
        # Define the transform from base_link to camera_link
        # This defines where the camera is mounted on the Segway robot
        
        # Default values - should be configured based on actual mounting position
        camera_x = 0.2  # 20cm forward from base_link
        camera_y = 0.0  # centered
        camera_z = 0.8  # 80cm above base_link
        camera_roll = 0.0
        camera_pitch = 0.0
        camera_yaw = 0.0
        
        # Create and publish the static transform
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "base_link"
        tf.child_frame_id = self.camera_frame_id
        
        # Set translation
        tf.transform.translation.x = camera_x
        tf.transform.translation.y = camera_y
        tf.transform.translation.z = camera_z
        
        # Set rotation (quaternion)
        from tf_transformations import quaternion_from_euler
        q = quaternion_from_euler(camera_roll, camera_pitch, camera_yaw)
        tf.transform.rotation.x = q[0]
        tf.transform.rotation.y = q[1]
        tf.transform.rotation.z = q[2]
        tf.transform.rotation.w = q[3]
        
        # Broadcast the static transform
        self.static_tf_broadcaster.sendTransform(tf)
        
        # Additional transforms depend on camera type
        if self.camera_type == 'realsense':
            # Add optical frame transforms for RGB and depth cameras
            self._broadcast_optical_frames()
    
    def _broadcast_optical_frames(self):
        """Broadcast optical frame transforms based on camera type."""
        if self.camera_type == 'realsense':
            # RGB optical frame relative to camera_link
            rgb_tf = TransformStamped()
            rgb_tf.header.stamp = self.get_clock().now().to_msg()
            rgb_tf.header.frame_id = self.camera_frame_id
            rgb_tf.child_frame_id = self.rgb_frame_id
            
            # For RealSense, RGB camera is slightly offset
            rgb_tf.transform.translation.x = 0.0
            rgb_tf.transform.translation.y = -0.014  # slight offset to the left
            rgb_tf.transform.translation.z = 0.0
            
            # Rotate to optical frame convention (-90 around X, then 90 around Z)
            q = quaternion_from_euler(-1.5708, 0.0, 1.5708)
            rgb_tf.transform.rotation.x = q[0]
            rgb_tf.transform.rotation.y = q[1]
            rgb_tf.transform.rotation.z = q[2]
            rgb_tf.transform.rotation.w = q[3]
            
            # Depth optical frame
            depth_tf = TransformStamped()
            depth_tf.header.stamp = self.get_clock().now().to_msg()
            depth_tf.header.frame_id = self.camera_frame_id
            depth_tf.child_frame_id = self.depth_frame_id
            
            # For RealSense, depth camera has a different offset
            depth_tf.transform.translation.x = 0.0
            depth_tf.transform.translation.y = 0.0
            depth_tf.transform.translation.z = 0.0
            
            # Same rotation as RGB
            depth_tf.transform.rotation.x = q[0]
            depth_tf.transform.rotation.y = q[1]
            depth_tf.transform.rotation.z = q[2]
            depth_tf.transform.rotation.w = q[3]
            
            # Broadcast both transforms
            self.static_tf_broadcaster.sendTransform([rgb_tf, depth_tf])
    
    def _publish_camera_tf(self):
        """
        Update dynamic camera transform if needed.
        This would be used if the camera is mounted on a moving part like the arm.
        """
        # Currently just a placeholder - implement if the camera moves relative to base_link
        pass
    
    def _health_check(self):
        """Periodic health check to monitor system status."""
        try:
            # Get current pose
            pose = self.segway_client.get_base_pose()
            # Log basic status every 10 seconds (approx)
            if hasattr(self, '_health_counter'):
                self._health_counter += 1
            else:
                self._health_counter = 0
                
            if self._health_counter % 10 == 0:
                self.get_logger().info(f"Segway status: pos=({pose[0]:.2f}, {pose[1]:.2f}), "
                                      f"yaw={pose[2]:.2f}, moving={self.segway_client.is_moving()}")
                
                # Add camera status if enabled
                if self.use_camera:
                    rgb_status = "Available" if self.latest_rgb_image else "Not received"
                    depth_status = "Available" if self.latest_depth_image else "Not received"
                    self.get_logger().info(f"Camera status: RGB: {rgb_status}, Depth: {depth_status}")
        except Exception as e:
            self.get_logger().error(f"Health check error: {e}")
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        self.get_logger().info(f"Received signal {sig}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def shutdown(self):
        """Shutdown the Segway client and this node."""
        if hasattr(self, 'segway_client'):
            self.segway_client.stop()  # Stop movement for safety
            self.segway_client.shutdown()
        self.get_logger().info("Segway adapter node shutdown complete")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Create and start the adapter node
        adapter = SegwayAdapterNode(args=args)
        
        # Run the node
        rclpy.spin(adapter)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in segway_adapter_node: {e}", file=sys.stderr)
    finally:
        # Make sure to perform cleanup
        rclpy.shutdown()


if __name__ == '__main__':
    main()