#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class RotationTest(Node):
    def __init__(self):
        super().__init__('rotation_test')
        
        # Create publishers for both topics to test them
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.segway_cmd_vel_pub = self.create_publisher(Twist, '/segway/cmd_vel', 10)
        
        # Give publisher time to connect
        time.sleep(2.0)
        
        self.get_logger().info("Starting rotation test...")
        self.run_test()
        
    def run_test(self):
        # Create command message
        cmd = Twist()
        
        try:
            # First send stop command to ensure robot starts from rest
            self.get_logger().info("Sending initial stop command...")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            for i in range(10):  # Send multiple stop commands to ensure robot receives it
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.2)
            
            # Test 1: Direct to /cmd_vel
            self.get_logger().info("TEST 1: Publishing directly to /cmd_vel for 30 SECONDS...")
            cmd.angular.z = 0.2
            
            # Send commands continuously for 30 seconds
            start_time = time.time()
            while time.time() - start_time < 30.0:  # Run for 30 seconds
                self.cmd_vel_pub.publish(cmd)
                if int(time.time() - start_time) % 5 == 0:  # Log every 5 seconds
                    self.get_logger().info(f"Still sending rotation command: angular.z=0.2 ({int(time.time() - start_time)} seconds elapsed)")
                time.sleep(0.2)  # Send command at 5Hz
            
            # Stop with multiple commands
            self.get_logger().info("Stopping...")
            cmd.angular.z = 0.0
            for i in range(10):  # Send multiple stop commands
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.2)
            
            # Wait longer between tests
            self.get_logger().info("Waiting between tests for 10 seconds...")
            time.sleep(10.0)
            
            # Test 2: Through /segway/cmd_vel
            self.get_logger().info("TEST 2: Publishing to /segway/cmd_vel for 30 SECONDS...")
            cmd.angular.z = -0.2
            
            # Send commands continuously for 30 seconds
            start_time = time.time()
            while time.time() - start_time < 30.0:  # Run for 30 seconds
                self.segway_cmd_vel_pub.publish(cmd)
                if int(time.time() - start_time) % 5 == 0:  # Log every 5 seconds
                    self.get_logger().info(f"Still sending rotation command: angular.z=-0.2 ({int(time.time() - start_time)} seconds elapsed)")
                time.sleep(0.2)  # Send command at 5Hz
            
            # Final stop with multiple commands
            self.get_logger().info("Test complete. Stopping robot...")
            cmd.angular.z = 0.0
            for i in range(10):  # Send multiple stop commands
                self.segway_cmd_vel_pub.publish(cmd)
                time.sleep(0.2)
            
        except Exception as e:
            self.get_logger().error(f"Error during test: {e}")
        
        # Shutdown the node
        self.get_logger().info("Test complete, shutting down...")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    test = RotationTest()
    rclpy.spin(test)
    
    # Clean up
    test.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()