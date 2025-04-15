#!/usr/bin/env python3

import click
import yaml
import signal
import threading
import time
import numpy as np
import sys
from pathlib import Path

from stretch.agent.segway_robot_agent import SegwayRobotAgent
from stretch.core.segway_ros_client import SegwayROSClient
from stretch.perception.encoders.clip_encoder import ClipEncoder
from stretch.utils.logger import Logger

logger = Logger(__name__)

# Global flag to control shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("Shutdown signal received")
    shutdown_requested = True

@click.command()
@click.option("--robot_ip", default="10.66.171.191", help="IP address of the Segway robot")
@click.option("--desktop_ip", default="10.66.171.131", help="Desktop IP address")
@click.option("--visualize", default=False, is_flag=True, help="Enable visualization")
@click.option("--scan_on_start", default=True, is_flag=True, help="Perform initial 360° scan")
@click.option("--update_rate", default=10.0, type=float, help="Map update rate in Hz")
@click.option("--vis_update_rate", default=1.0, type=float, help="Visualization update rate in Hz")
@click.option("--debug", default=True, is_flag=True, help="Enable debug logging")
def main(robot_ip, desktop_ip, visualize, scan_on_start, update_rate, vis_update_rate, debug):
    """Live Segway robot mapping and exploration."""
    
    print("Initializing live mapping system...")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Find the config directory
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent / "config"
    segway_config_path = str(config_dir / "segway_config.yaml")
    
    # Load and update configuration
    try:
        with open(segway_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update configuration with our parameters
        if 'network' not in config:
            config['network'] = {}
        config['network']['robot_ip'] = robot_ip
        config['network']['desktop_ip'] = desktop_ip
        
        # Add ROS bridge settings to prevent auto-restart
        if 'ros_bridge' not in config:
            config['ros_bridge'] = {}
        config['ros_bridge']['auto_restart'] = False  # Disable auto-restart
        
        # Save updated config
        with open(segway_config_path, 'w') as f:
            yaml.dump(config, f)
            
        print(f"Updated config with robot_ip={robot_ip}, desktop_ip={desktop_ip}")
    except Exception as e:
        print(f"Warning: Could not update config file: {e}")
    
    # First test direct connection to ROS
    print("Testing direct ROS connection...")
    try:
        ros_client = SegwayROSClient(
            config_path=segway_config_path,
            robot_ip=robot_ip,
            desktop_ip=desktop_ip,
            node_name='segway_test_client'
        )
        
        # Connect to ROS
        print("Connecting to ROS...")
        success = ros_client.connect()
        if success:
            print("✅ Successfully connected to ROS")
            
            # Test if we can get pose
            pose = ros_client.get_pose()
            print(f"Current pose: {pose}")
            
            # Send a small test motion command
            print("Sending test motion command...")
            if pose is not None:
                x, y, theta = pose
                target_theta = theta + 0.1  # Small rotation
                cmd_result = ros_client.move_to([x, y, target_theta], relative=False, blocking=True, timeout=5.0, verbose=True)
                if cmd_result:
                    print("✅ Test motion command successful")
                else:
                    print("❌ Test motion command failed - check robot emergency stop and permissions")
            else:
                print("❌ Could not get robot pose")
                
            # Stop the test client
            ros_client.stop()
        else:
            print("❌ Failed to connect to ROS")
            return
    except Exception as e:
        print(f"Error in ROS connection test: {e}")
        import traceback
        traceback.print_exc()
    
    # Create CLIP encoder for object recognition
    print("Creating CLIP encoder...")
    try:
        encoder = ClipEncoder(version="ViT-L/14@336px", device="cuda:0")
        print("Successfully created ClipEncoder")
    except Exception as e:
        print(f"Error creating ClipEncoder: {e}")
        encoder = None
        print("Warning: No encoder available, instance recognition may be limited")
    
    # Create agent variable outside the try block to access it in the finally block
    agent = None
    
    try:
        # Initialize agent
        print("Initializing SegwayRobotAgent...")
        agent = SegwayRobotAgent(
            config_path=segway_config_path,
            semantic_sensor=None,
            voxel_map=None,
            show_instances_detected=True,
            use_instance_memory=True,
            enable_realtime_updates=True,
            create_semantic_sensor=True,
            gpu_device="cuda:0",
            debug_mode=debug
        )
        
        # If you created an encoder, manually set it on the mapping adapter
        if encoder is not None and hasattr(agent, 'mapping_adapter') and agent.mapping_adapter is not None:
            if hasattr(agent.mapping_adapter, 'set_encoder'):
                agent.mapping_adapter.set_encoder(encoder)
                print("✅ Set encoder on mapping adapter")
            elif hasattr(agent.mapping_adapter, 'instance_map') and agent.mapping_adapter.instance_map is not None:
                agent.mapping_adapter.instance_map.encoder = encoder
                print("✅ Set encoder on instance map")
        
        # Start agent
        print("Starting agent...")
        agent.start()
        
        # Check adapter initializations
        if hasattr(agent, 'motion_adapter') and agent.motion_adapter is not None:
            print("✅ Motion adapter initialized")
        else:
            print("❌ Motion adapter not initialized")
            
        if hasattr(agent, 'navigation_adapter') and agent.navigation_adapter is not None:
            print("✅ Navigation adapter initialized")
        else:
            print("❌ Navigation adapter not initialized")
            
        if hasattr(agent, 'mapping_adapter') and agent.mapping_adapter is not None:
            print("✅ Mapping adapter initialized")
        else:
            print("❌ Mapping adapter not initialized")
        
        # Start visualization if requested
        if visualize:
            print("Starting visualization...")
            if hasattr(agent, 'start_visualization'):
                agent.start_visualization(update_rate=vis_update_rate)
                print("✅ Visualization started")
            else:
                print("❌ No start_visualization method found")
        
        # Perform initial scan if requested
        if scan_on_start:
            print("Performing initial 360° scan...")
            print("Using motion adapter for rotation" if (hasattr(agent, 'motion_adapter') and agent.motion_adapter is not None) else "Using basic rotation")
            
            # Debug rotation steps
            for step in range(8):
                if shutdown_requested:
                    break
                    
                print(f"Rotation step {step+1}/8...")
                current_pose = agent.robot.get_base_pose()
                target_angle = current_pose[2] + (np.pi/4)  # 45 degrees
                
                # Try the motion adapter first
                if hasattr(agent, 'motion_adapter') and agent.motion_adapter is not None:
                    target_pose = [current_pose[0], current_pose[1], target_angle]
                    result = agent.motion_adapter.move_to_pose(target_pose)
                    if result:
                        print(f"  ✅ Rotation step {step+1} successful (motion_adapter)")
                    else:
                        print(f"  ❌ Rotation step {step+1} failed (motion_adapter)")
                else:
                    # Fall back to direct command
                    result = agent.robot.move_base_to(
                        [current_pose[0], current_pose[1], target_angle],
                        relative=False,
                        blocking=True,
                        verbose=True
                    )
                    if result:
                        print(f"  ✅ Rotation step {step+1} successful (direct)")
                    else:
                        print(f"  ❌ Rotation step {step+1} failed (direct)")
                
                # Wait for a bit between steps
                time.sleep(2.0)
                
                # Force update of the map
                if hasattr(agent, 'update'):
                    agent.update(visualize_map=visualize)
                    print(f"  ✅ Map updated after step {step+1}")
        
        # Main loop with clean shutdown
        print("Live mapping active. Press Ctrl+C to exit.")

        try:
            while not shutdown_requested:
                # Process any commands or monitor system status
                if hasattr(agent, 'check_system_status'):
                    try:
                        status = agent.check_system_status()
                        if status and not all(status.get('connection', {}).values()):
                            missing = [k for k, v in status.get('connection', {}).items() if not v]
                            print(f"Warning: Missing connections: {missing}")
                    except Exception as e:
                        print(f"Error checking system status: {e}")
                
                # Update scene graph periodically if available
                if hasattr(agent, '_update_scene_graph') and hasattr(agent, 'scene_graph') and agent.scene_graph is not None:
                    try:
                        agent._update_scene_graph()
                    except Exception as e:
                        print(f"Error updating scene graph: {e}")
                
                # Sleep to avoid consuming too much CPU
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            # This is a fallback - the signal handler should catch this
            print("Keyboard interrupt detected, shutting down...")
            shutdown_requested = True
    
    finally:
        # Ensure shutdown happens correctly regardless of how we exit
        if agent is not None:
            print("Shutting down agent...")
            agent.shutdown()
            
            # Add a small delay to ensure ROS bridge has time to clean up
            time.sleep(1.0)
            
        print("Live mapping complete!")
        
        # Exit cleanly
        if shutdown_requested:
            sys.exit(0)

if __name__ == "__main__":
    main()