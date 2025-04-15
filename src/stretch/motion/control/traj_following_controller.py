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
import threading
from typing import Callable, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from stretch.motion.utils.config import get_control_config
from stretch.motion.utils.geometry import xyt_global_to_base

DEFAULT_CFG_NAME = "traj_follower"


class TrajFollower:
    def __init__(self, cfg: Optional["DictConfig"] = None):
        if cfg is None:
            cfg = get_control_config(DEFAULT_CFG_NAME)
        else:
            self.cfg = cfg

        # Compute gain
        self.kp = cfg.k_p
        self.ki = (cfg.damp_ratio * self.kp) ** 2 / 4.0

        # Init
        self._traj_update_lock = threading.Lock()

        self._is_done = True
        self.traj = None
        self.traj_buffer: Optional[Callable[[float], Tuple[np.ndarray, np.ndarray, bool]]] = None

        self.e_int = np.zeros(3)
        self._t_prev = 0

    def update_trajectory(self, traj: Callable[[float], Tuple[np.ndarray, np.ndarray, bool]]):
        with self._traj_update_lock:
            self.traj_buffer = traj

    def is_done(self) -> bool:
        return self._is_done

    def forward(self, xyt: np.ndarray, t: float) -> Tuple[float, float]:
        """Returns velocity control command (v, w)"""
        # Check for trajectory updates
        if self.traj_buffer is not None:
            with self._traj_update_lock:
                self.traj = self.traj_buffer  # type: ignore
                self.traj_buffer = None
            self._is_done = False

        if self.traj is None:
            # Return zero velocities if no trajectory is active
            return 0.0, 0.0

        # Return zero velocites if no trajectory is active
        if self._is_done:
            return 0.0, 0.0

        # Query trajectory for desired states
        xyt_traj, dxyt_traj, done = self.traj(t)
        if done:
            self._is_done = True

        # Feedback control
        dt = t - self._t_prev
        self._t_prev = t
        v, w = self._feedback_controller(xyt_traj, dxyt_traj, xyt, dt)

        return v, w

    def _feedback_controller(
        self, xyt_des: np.ndarray, dxyt_des: np.ndarray, xyt_curr: np.ndarray, dt: float
    ) -> Tuple[float, float]:
        # Compute reference input
        u_ref = np.array([np.linalg.norm(dxyt_des[:2]), dxyt_des[2]])

        # Compute error in local frame
        e = xyt_global_to_base(xyt_des, xyt_curr)

        # Compute desired error derivative via PI control
        self.e_int = self.cfg.decay * self.e_int + e * dt
        de_des = -self.kp * e - self.ki * self.e_int

        # Compute velocity feedback commands to achieve desired error derivative
        M_u2e = np.array([[-1, e[1]], [0, -e[0]], [0, -1]])
        M_ur2e = np.array([[np.cos(e[2]), 0], [np.sin(e[2]), 0], [0, 1]])
        u_output = np.linalg.pinv(M_u2e) @ (de_des - M_ur2e @ u_ref)

        return u_output[0], u_output[1]

class TrajectoryFollowingController:
    """
    Higher-level controller for following trajectories with more convenient interface.
    Wraps the lower-level TrajFollower class.
    """
    
    def __init__(
        self,
        position_tolerance=0.1,
        orientation_tolerance=0.1,
        max_linear_speed=0.5,
        max_angular_speed=0.5
    ):
        """
        Initialize the trajectory following controller.
        
        Args:
            position_tolerance: Tolerance for position error (meters)
            orientation_tolerance: Tolerance for orientation error (radians)
            max_linear_speed: Maximum linear speed (m/s)
            max_angular_speed: Maximum angular speed (rad/s)
        """
        from omegaconf import OmegaConf
        
        # Create configuration
        cfg = OmegaConf.create({
            "k_p": 1.0,
            "damp_ratio": 0.7,
            "decay": 0.99,
            "position_tolerance": position_tolerance,
            "orientation_tolerance": orientation_tolerance,
            "max_linear_speed": max_linear_speed,
            "max_angular_speed": max_angular_speed
        })
        
        # Create traj follower
        self.follower = TrajFollower(cfg=cfg)
        
        # Store parameters
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        
        # State variables
        self.current_time = 0.0
        self.current_pose = None
        self.trajectory = None
        
    def update_pose(self, pose):
        """
        Update the controller with current pose.
        
        Args:
            pose: Current pose [x, y, theta]
        """
        self.current_pose = np.array(pose)
        
    def set_trajectory(self, trajectory):
        """
        Set the trajectory to follow.
        
        Args:
            trajectory: List of poses [[x, y, theta], ...]
        """
        # Create trajectory function for TrajFollower
        def traj_func(t):
            # Simple linear interpolation between waypoints
            if not trajectory or len(trajectory) == 0:
                return np.zeros(3), np.zeros(3), True
                
            # Convert to normalized time [0, 1]
            total_time = len(trajectory) - 1
            normalized_t = min(1.0, max(0.0, t / total_time))
            
            # Find segment
            segment_idx = int(normalized_t * total_time)
            segment_t = (normalized_t * total_time) - segment_idx
            
            # Get waypoints
            start = trajectory[segment_idx]
            end = trajectory[min(segment_idx + 1, len(trajectory) - 1)]
            
            # Interpolate position
            pos = start[:2] + segment_t * (end[:2] - start[:2])
            
            # Interpolate orientation (shortest path)
            theta_diff = end[2] - start[2]
            # Normalize to [-pi, pi]
            if theta_diff > np.pi:
                theta_diff -= 2 * np.pi
            elif theta_diff < -np.pi:
                theta_diff += 2 * np.pi
                
            theta = start[2] + segment_t * theta_diff
            
            # Position and orientation
            xyt = np.array([pos[0], pos[1], theta])
            
            # Velocity (derivative)
            velocity = (end[:2] - start[:2]) / total_time
            angular_velocity = theta_diff / total_time
            dxyt = np.array([velocity[0], velocity[1], angular_velocity])
            
            # Done when we reach the end
            done = (segment_idx >= total_time - 0.01)
            
            return xyt, dxyt, done
            
        # Set the trajectory function
        self.trajectory = trajectory
        self.follower.update_trajectory(traj_func)
        self.current_time = 0.0
        
    def is_done(self):
        """
        Check if trajectory following is complete.
        
        Returns:
            bool: True if completed
        """
        return self.follower.is_done()
        
    def compute_velocity(self, dt=0.1):
        """
        Compute velocity commands.
        
        Args:
            dt: Time since last update (seconds)
            
        Returns:
            Tuple of (linear_velocity, angular_velocity)
        """
        if self.current_pose is None or self.trajectory is None:
            return 0.0, 0.0
            
        # Update time
        self.current_time += dt
        
        # Compute control
        linear, angular = self.follower.forward(self.current_pose, self.current_time)
        
        # Apply limits
        linear = max(min(linear, self.max_linear_speed), -self.max_linear_speed)
        angular = max(min(angular, self.max_angular_speed), -self.max_angular_speed)
        
        return linear, angular
        
    def stop(self):
        """
        Stop trajectory following.
        
        Returns:
            bool: True if successful
        """
        self.trajectory = None
        return True