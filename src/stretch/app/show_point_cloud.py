#!/usr/bin/env python3

import time
import os
import click
import numpy as np
from cv_bridge import CvBridge
import open3d as o3d  # We'll use Open3D directly instead of the wrapper

from stretch.core.segway_ros_client import SegwayROSClient

@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
def main(robot_ip: str = "10.66.171.191"):
    """Display the point cloud from the depth camera."""
    # Use absolute path for config files
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "config", "segway_config.yaml")
    print(f"Using config path: {config_path}")
    
    # Initialize robot
    robot = SegwayROSClient(robot_ip=robot_ip, config_path=config_path)
    print("Starting the robot connection...")
    robot.connect()
    print("Robot connected")
    
    # Main loop
    try:
        print("Waiting for camera and point cloud data...")
        while True:
            # Get latest camera data and point cloud
            camera_data = robot.get_latest_camera_data()
            point_cloud_data = robot.last_point_cloud
            
            if camera_data is None:
                print("Waiting for camera data...")
                time.sleep(0.5)
                continue
                
            if point_cloud_data is None:
                print("Waiting for point cloud data...")
                time.sleep(0.5)
                continue
                
            try:
                # Extract XYZ coordinates from point cloud
                from sensor_msgs_py import point_cloud2
                pc_points = list(point_cloud2.read_points(point_cloud_data, 
                                                         field_names=('x', 'y', 'z'),
                                                         skip_nans=True))
                
                # Convert to numpy array without reshaping
                xyz = np.array(pc_points)
                
                # Print shape and sample points for debugging
                print(f"Point cloud shape: {xyz.shape}")
                if len(xyz) > 0:
                    print(f"First point: {xyz[0]}")
                    print(f"Sample points type: {type(xyz[0])}")
                
                # Handle different possible formats
                if len(xyz.shape) == 1 and hasattr(xyz[0], 'dtype') and xyz[0].dtype.names is not None:
                    # We have a structured array with named fields
                    points_list = [(p['x'], p['y'], p['z']) for p in xyz]
                    xyz = np.array(points_list)
                    print(f"Converted from structured array, new shape: {xyz.shape}")
                
                if xyz.size > 0 and len(xyz.shape) == 2 and xyz.shape[1] == 3:
                    print(f"Displaying point cloud with {xyz.shape[0]} points")
                    
                    # Create colors
                    rgb_colors = np.zeros((xyz.shape[0], 3))
                    
                    # Calculate normalized position for colors
                    x_min, x_max = xyz[:, 0].min(), xyz[:, 0].max()
                    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
                    z_min, z_max = xyz[:, 2].min(), xyz[:, 2].max()
                    
                    # Assign colors based on position
                    rgb_colors[:, 0] = (xyz[:, 0] - x_min) / (x_max - x_min + 1e-6)
                    rgb_colors[:, 1] = (xyz[:, 1] - y_min) / (y_max - y_min + 1e-6)
                    rgb_colors[:, 2] = (xyz[:, 2] - z_min) / (z_max - z_min + 1e-6)
                    
                    # Display point cloud using Open3D directly
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz)
                    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
                    
                    # Create coordinate frame
                    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
                    
                    # Visualize
                    o3d.visualization.draw_geometries([pcd, coord_frame])
                    print("Point cloud visualization complete")
                    break
                else:
                    print(f"Invalid point cloud data format: shape={xyz.shape}")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error processing point cloud: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping robot...")
        robot.stop()
        print("Done!")


if __name__ == "__main__":
    main()