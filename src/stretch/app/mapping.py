#!/usr/bin/env python3

import datetime
import os
import click
import yaml
import numpy as np
from pathlib import Path

# Import the SegwayRobotAgent 
from stretch.agent.segway_robot_agent import SegwayRobotAgent
from stretch.perception import create_semantic_sensor


@click.command()
@click.option("--robot_ip", default="10.66.171.191", help="IP address of the Segway robot")
@click.option("--desktop_ip", default="10.66.171.131", help="Desktop/GPU computer IP")
@click.option("--visualize", default=False, is_flag=True)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--output-filename", default="segway_output", type=str)
@click.option("--show-intermediate-maps", default=False, is_flag=True)
@click.option("--show-final-map", default=False, is_flag=True)
@click.option("--explore-iter", default=10, type=int, help="Exploration iterations")
@click.option("--device-id", default=0, help="GPU device ID")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def main(
    robot_ip,
    desktop_ip,
    visualize,
    manual_wait,
    output_filename,
    show_intermediate_maps,
    show_final_map,
    explore_iter,
    device_id,
    debug
):
    """Segway robot mapping and exploration."""
    
    print("Initializing Segway mapping...")
    
    # Find the config directory
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent / "config"
    
    # Create config path
    segway_config_path = str(config_dir / "segway_config.yaml")
    print(f"Using Segway config: {segway_config_path}")
    
    # Load configuration directly
    try:
        with open(segway_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update IP addresses
        if 'network' not in config:
            config['network'] = {}
        config['network']['robot_ip'] = robot_ip
        config['network']['desktop_ip'] = desktop_ip
        
        # Save updated config
        with open(segway_config_path, 'w') as f:
            yaml.dump(config, f)
            
        print(f"Updated config with robot_ip={robot_ip}, desktop_ip={desktop_ip}")
    except Exception as e:
        print(f"Warning: Could not update config file: {e}")
    
    # Initialize SegwayRobotAgent with minimal parameters
    try:
        print("Creating SegwayRobotAgent...")
        agent = SegwayRobotAgent(
            config_path=segway_config_path,
            debug_mode=debug
        )
        
        # Wait for robot to initialize
        print("Waiting for robot to initialize...")
        import time
        time.sleep(3)
        
        # Start agent for exploration
        print("Starting agent...")
        agent.start(visualize_map_at_start=show_intermediate_maps)
        
        # Perform initial scan
        print("Performing initial scan...")
        agent.rotate_in_place(steps=8, visualize=show_intermediate_maps)
        
        # Run exploration
        print(f"Running {explore_iter} exploration iterations...")
        for i in range(explore_iter):
            print(f"Exploration iteration {i+1}/{explore_iter}")
            
            # Execute scan environment operation
            agent.execute_operation("scan_environment", steps=4, update_map=True, visualize=show_intermediate_maps)
            
            # Execute explore operation
            agent.execute_operation("explore", max_iterations=1, frontier_radius=3.0)
            
            # Update map
            agent.update(visualize_map=sh+ow_intermediate_maps)
            
        # Show final map if requested
        if show_final_map:
            print("Showing final map...")
            agent.get_voxel_map().show()
            
        # Save map data
        output_path = f"{output_filename}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        print(f"Saving map to {output_path}...")
        agent.get_voxel_map().write_to_pickle(output_path)
        
        # Return home
        print("Returning home...")
        agent.go_home()
        
        # Shutdown agent
        print("Shutting down agent...")
        agent.shutdown()
        
        print("Mapping complete!")
        
    except Exception as e:
        print(f"Error during mapping: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()