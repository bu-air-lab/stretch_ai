#!/usr/bin/env python3
"""
Scene Graph App for Segway-Stretch Integration

This module provides a command-line interface for running the scene graph
in standalone mode, similar to other Stretch AI apps.
"""

import os
import sys
import time
import argparse
import threading
from typing import List, Dict, Any, Optional

import rclpy

from stretch.core.parameters import load_parameters, Parameters
from stretch.core.segway_ros_client import SegwayROSClient
from stretch.agent.segway_robot_agent import SegwayRobotAgent
from stretch.mapping.voxel import SparseVoxelMapNavigationSpace
from stretch.mapping.scene_graph.enhanced_scene_graph import EnhancedSceneGraph
from stretch.mapping.scene_graph.scene_graph_node import SceneGraphNode
from stretch.utils.logger import Logger

logger = Logger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Scene Graph App for Segway-Stretch Integration")
    
    parser.add_argument('--config', type=str, default=None,
                      help='Path to configuration file')
    parser.add_argument('--robot-ip', type=str, default=None,
                      help='IP address of the Segway robot')
    parser.add_argument('--desktop-ip', type=str, default=None,
                      help='IP address of the RTX 4090 workstation')
    parser.add_argument('--output-dir', type=str, default='~/scene_graph_output',
                      help='Output directory for visualizations')
    parser.add_argument('--show-visualization', action='store_true',
                      help='Show visualization windows')
    parser.add_argument('--update-interval', type=float, default=1.0,
                      help='Scene graph update interval in seconds')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    
    return parser.parse_args()

def run_scene_graph(args):
    """
    Run the scene graph app.
    
    Args:
        args: Parsed command line arguments
    """
    # Load parameters
    if args.config:
        parameters = load_parameters(args.config)
    else:
        parameters = Parameters({})
    
    # Override parameters with command line arguments
    if args.robot_ip:
        parameters.set("network.robot_ip", args.robot_ip)
    if args.desktop_ip:
        parameters.set("network.desktop_ip", args.desktop_ip)
    if args.output_dir:
        parameters.set("scene_graph.output_dir", os.path.expanduser(args.output_dir))
    if args.show_visualization:
        parameters.set("scene_graph.show_visualization", True)
    if args.update_interval:
        parameters.set("scene_graph.update_interval", args.update_interval)
    if args.debug:
        parameters.set("debug", True)
    
    # Print parameters
    logger.info(f"Running with parameters:")
    logger.info(f"  Robot IP: {parameters.get('network.robot_ip', '10.66.171.191')}")
    logger.info(f"  Desktop IP: {parameters.get('network.desktop_ip', '10.66.171.131')}")
    logger.info(f"  Output directory: {parameters.get('scene_graph.output_dir', '~/scene_graph_output')}")
    logger.info(f"  Show visualization: {parameters.get('scene_graph.show_visualization', False)}")
    logger.info(f"  Update interval: {parameters.get('scene_graph.update_interval', 1.0)}")
    logger.info(f"  Debug: {parameters.get('debug', False)}")
    
    # Initialize ROS
    rclpy.init()
    
    try:
        # Create ROS client
        ros_client = SegwayROSClient(
            config_path=args.config,
            robot_ip=parameters.get("network.robot_ip", "10.66.171.191"),
            desktop_ip=parameters.get("network.desktop_ip", "10.66.171.131")
        )
        
        # Connect to ROS
        if not ros_client.connect():
            logger.error("Failed to connect to ROS")
            return False
        
        # Initialize robot agent
        agent = SegwayRobotAgent(
            config_path=args.config,
            create_semantic_sensor=True,
            debug_mode=parameters.get("debug", False),
            gpu_device="cuda:0"
        )
        
        # Get instances from agent
        instances = []
        
        # Create scene graph node
        node = SceneGraphNode(
            parameters=parameters,
            instances=instances,
            voxel_map=agent.get_voxel_map()
        )
        
        # Create thread for detecting objects and updating scene graph
        def object_detection_thread():
            """Thread for object detection and scene graph updates"""
            logger.info("Starting object detection thread")
            
            while rclpy.ok():
                try:
                    # Detect objects using the agent
                    logger.info("Detecting objects...")
                    objects = agent.detect_objects()
                    
                    # Update instances in scene graph node
                    if objects:
                        node.add_instances(objects)
                        logger.info(f"Detected {len(objects)} objects")
                    
                    # Sleep before next detection
                    time.sleep(parameters.get("scene_graph.detection_interval", 5.0))
                except Exception as e:
                    logger.error(f"Error in object detection thread: {e}")
                    time.sleep(1.0)  # Sleep on error
        
        # Start object detection thread
        detection_thread = threading.Thread(target=object_detection_thread)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Spin node
        logger.info("Starting scene graph node")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        logger.info("Caught keyboard interrupt")
    except Exception as e:
        logger.error(f"Error running scene graph: {e}")
    finally:
        # Clean up
        try:
            if 'node' in locals():
                node.shutdown()
                node.destroy_node()
            
            if 'agent' in locals():
                agent.shutdown()
            
            if 'ros_client' in locals():
                ros_client.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        rclpy.shutdown()
    
    return True

def main():
    """Main entry point"""
    args = parse_args()
    return run_scene_graph(args)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)