#!/usr/bin/env python3

import click
import yaml
from pathlib import Path
import threading
import time

from stretch.agent.segway_robot_agent import SegwayRobotAgent
from stretch.perception.encoders import create_clip_encoder

@click.command()
@click.option("--robot_ip", default="10.66.171.191", help="IP address of the Segway robot")
@click.option("--visualize", default=False, is_flag=True)
def main(robot_ip, visualize):
    """Live Segway robot mapping and exploration."""
    
    print("Initializing live mapping system...")
    
    # Find the config directory
    script_dir = Path(__file__).parent
    config_dir = script_dir.parent / "config"
    segway_config_path = str(config_dir / "segway_config.yaml")
    
    # Create CLIP encoder for object recognition
    encoder = create_clip_encoder(device="cuda:0")
    
    # Initialize agent with continuous mapping enabled
    agent = SegwayRobotAgent(
        config_path=segway_config_path,
        debug_mode=False,
        continuous_mapping=True,  # Enable continuous mapping
        encoder=encoder           # Pass encoder for object recognition
    )
    
    # Start mapping in a separate thread
    def mapping_thread():
        while agent.is_running:
            # Get latest observation
            observation = agent.get_latest_observation()
            if observation is not None:
                # Update maps with latest observation
                agent.update_maps_from_observation(observation)
            time.sleep(0.1)  # Short sleep to avoid consuming too much CPU
    
    # Initialize visualization if requested
    if visualize:
        agent.start_visualization()
    
    # Start agent
    agent.start()
    
    # Start mapping thread
    mapping_thread = threading.Thread(target=mapping_thread)
    mapping_thread.daemon = True
    mapping_thread.start()
    
    try:
        # Run until Ctrl+C is pressed
        print("Live mapping active. Press Ctrl+C to exit.")
        while True:
            if visualize:
                agent.update_visualization()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        agent.shutdown()

if __name__ == "__main__":
    main()