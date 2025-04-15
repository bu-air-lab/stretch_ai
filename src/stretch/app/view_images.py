#!/usr/bin/env python3

import time
import os
import sys

import click
import cv2
import numpy as np
from cv_bridge import CvBridge

import stretch.utils.logger as logger
from stretch.core.segway_ros_client import SegwayROSClient
from stretch.perception import create_semantic_sensor
from stretch.core import get_parameters

@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--run_semantic_segmentation", "--segment", "-s", is_flag=True, 
              help="Run semantic segmentation on images")
def main(robot_ip: str = "10.66.171.191", run_semantic_segmentation: bool = False):
    # Use absolute path for config file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "config", "segway_config.yaml")
    print(f"Using config path: {config_path}")
    
    # Initialize robot
    robot = SegwayROSClient(robot_ip=robot_ip, config_path=config_path)
    print("Starting the robot connection...")
    robot.connect()
    print("Robot connected")
    
    # Set up CV bridge
    bridge = CvBridge()
    
    # Load parameters and initialize semantic sensor if needed
    if run_semantic_segmentation:
        try:

            param_path = "default_planner.yaml"
            print(f"Loading parameters from: {param_path}")
            
            parameters = get_parameters(param_path)
            print("Creating semantic sensor...")
            semantic_sensor = create_semantic_sensor(
                parameters,
                device_id=0,
                verbose=True,
            )
            print("Semantic sensor created successfully")
        except Exception as e:
            print(f"ERROR creating semantic sensor: {e}")
            import traceback
            traceback.print_exc()
            run_semantic_segmentation = False
            semantic_sensor = None
    else:
        semantic_sensor = None
    
    # Initialize variables
    colors = {}
    print("Starting image display loop...")
    
    # Main loop
    try:
        while True:
            # Get latest camera data
            camera_data = robot.get_latest_camera_data()
            if camera_data is None:
                print("Waiting for camera data...")
                time.sleep(0.5)
                continue
            
            # Ensure we have both RGB and depth data before proceeding
            if not ('rgb' in camera_data and camera_data['rgb'] is not None and 
                   'depth' in camera_data and camera_data['depth'] is not None):
                print("Waiting for both RGB and depth data...")
                time.sleep(0.5)
                continue
                
            # Convert messages to OpenCV images
            rgb_msg = camera_data['rgb']
            depth_msg = camera_data['depth']
            rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Display RGB image
            rgb_display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow("RGB Camera", rgb_display)
            
            # Display depth image
            depth_viz = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_viz = depth_viz.astype(np.uint8)
            depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            cv2.imshow("Depth Camera", depth_viz)
            
            # Run semantic segmentation if enabled
            if run_semantic_segmentation and semantic_sensor is not None:
                try:
                    # Create a complete observation object with all required attributes
                    class SimpleObs:
                        def __init__(self):
                            self.rgb = None
                            self.depth = None
                            self.ee_rgb = None  # Not available on Segway
                            self.ee_depth = None  # Not available on Segway
                            self.camera_K = None
                            # Add task_observations attribute that was missing
                            self.task_observations = None
                            # Add other potential attributes that might be needed
                            self.semantic = None
                            self.instance = None
                    
                    # Create and populate the observation
                    obs = SimpleObs()
                    obs.rgb = rgb_image
                    obs.depth = depth_image
                    
                    # If camera_info is available, add camera intrinsics
                    if 'info' in camera_data and camera_data['info'] is not None:
                        obs.camera_K = np.array(camera_data['info'].k).reshape(3, 3)
                    
                    # Run prediction (explicitly set ee=False for Segway)
                    print("Running semantic segmentation...")
                    seg_obs = semantic_sensor.predict(obs, ee=False)
                    
                    # Check if we have semantic segmentation results
                    if hasattr(seg_obs, 'semantic') and seg_obs.semantic is not None:
                        print(f"Semantic segmentation shape: {seg_obs.semantic.shape}")
                        
                        # Create visualization
                        semantic_vis = np.zeros(
                            (seg_obs.semantic.shape[0], seg_obs.semantic.shape[1], 3)
                        ).astype(np.uint8)
                        
                        # Determine segmentation type
                        if semantic_sensor.is_semantic():
                            segmentation = seg_obs.semantic
                        elif semantic_sensor.is_instance():
                            segmentation = seg_obs.instance
                        else:
                            raise ValueError("Unknown perception model type")
                        
                        # Color each segment
                        for cls in np.unique(segmentation):
                            if cls > 0:  # Skip background
                                if cls not in colors:
                                    colors[cls] = (np.random.rand(3) * 255).astype(np.uint8)
                                semantic_vis[segmentation == cls] = colors[cls]
                        
                        # Blend with original image
                        alpha = 0.5
                        blended = cv2.addWeighted(
                            rgb_image, alpha, semantic_vis, 1 - alpha, 0
                        )
                        
                        # Display segmentation
                        blended_display = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
                        cv2.imshow("Semantic Segmentation", blended_display)
                        print("Displayed semantic segmentation")
                except Exception as e:
                    print(f"Error in semantic segmentation: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Break if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
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