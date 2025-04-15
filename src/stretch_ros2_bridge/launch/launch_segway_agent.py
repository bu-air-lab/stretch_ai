#!/usr/bin/env python3
"""
Launch script for the Segway-Stretch AI integration.
This script launches the ROS bridge and the Segway robot agent.
"""

import os
import sys
import argparse
import subprocess
import time
import signal
import logging
from pathlib import Path

# Import the Segway robot agent
from stretch.agent.segway_robot_agent import SegwayRobotAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('segway_agent.log')
    ]
)
logger = logging.getLogger(__name__)

def start_ros_bridge(bridge_path=None):
    """
    Start the ROS 1-ROS 2 bridge.
    
    Args:
        bridge_path: Path to the ROS bridge executable or script
    """
    if not bridge_path:
        # Use default path from environment
        bridge_path = os.environ.get('ROS_BRIDGE_PATH', '/ros-humble-ros1-bridge/install/ros1_bridge/lib/ros1_bridge/dynamic_bridge')
    
    logger.info(f"Starting ROS bridge from: {bridge_path}")
    
    # Start the ROS bridge in a separate process
    try:
        bridge_process = subprocess.Popen(
            [bridge_path, '--bridge-all-topics'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"ROS bridge started with PID: {bridge_process.pid}")
        return bridge_process
    except Exception as e:
        logger.error(f"Failed to start ROS bridge: {e}")
        return None

def main():
    """Main function to start the Segway robot agent"""
    parser = argparse.ArgumentParser(description='Launch Segway-Stretch AI integration')
    parser.add_argument('--config', type=str, default='segway_config.yaml', help='Path to the configuration file')
    parser.add_argument('--bridge-path', type=str, default=None, help='Path to the ROS 1-ROS 2 bridge executable')
    parser.add_argument('--no-bridge', action='store_true', help='Do not start the ROS bridge (assume it is already running)')
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Start ROS bridge if needed
    bridge_process = None
    if not args.no_bridge:
        bridge_process = start_ros_bridge(args.bridge_path)
        if not bridge_process:
            logger.error("Failed to start ROS bridge. Exiting.")
            sys.exit(1)
        
        # Wait for the bridge to initialize
        logger.info("Waiting for ROS bridge to initialize...")
        time.sleep(5)
    
    # Initialize and start the Segway robot agent
    try:
        logger.info(f"Initializing Segway robot agent with config: {config_path}")
        agent = SegwayRobotAgent(config_path=config_path)
        
        # Register signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received. Stopping agent...")
            agent.shutdown()
            if bridge_process:
                logger.info("Stopping ROS bridge...")
                bridge_process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info("Segway robot agent initialized and running")
        
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Error running Segway robot agent: {e}")
    finally:
        # Ensure clean shutdown
        if 'agent' in locals():
            agent.shutdown()
        
        if bridge_process:
            logger.info("Stopping ROS bridge...")
            bridge_process.terminate()
            bridge_process.wait()
    
    logger.info("Segway robot agent shutdown complete")

if __name__ == "__main__":
    main()