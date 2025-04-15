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
import abc
from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from stretch.utils.geometry import normalize_ang_error


class DiffDriveVelocityController(abc.ABC):
    """
    Abstract class for differential drive robot velocity controllers.
    """

    def set_linear_error_tolerance(self, error_tol: float):
        self.lin_error_tol = error_tol

    def set_angular_error_tolerance(self, error_tol: float):
        self.ang_error_tol = error_tol

    @abc.abstractmethod
    def __call__(self, xyt_err: np.ndarray) -> Tuple[float, float, bool]:
        """Contain execution logic, predict velocities for the left and right wheels. Expected to
        return true/false if we have reached this goal and the controller will be moving no
        farther."""


class DDVelocityControlNoplan(DiffDriveVelocityController):
    """
    Control logic for differential drive robot velocity control.
    Does not plan at all, instead uses heuristics to gravitate towards the goal.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.reset_error_tolerances()
        self.reset_velocity_profile()

    def reset_velocity_profile(self):
        """Read velocity configuration info from the config"""
        self.update_velocity_profile(
            self.cfg.v_max, self.cfg.w_max, self.cfg.acc_lin, self.cfg.acc_ang
        )

    def update_velocity_profile(
        self,
        v_max: Optional[float] = None,
        w_max: Optional[float] = None,
        acc_lin: Optional[float] = None,
        acc_ang: Optional[float] = None,
    ):
        """Call controller and update velocity profile.

        Parameters:
            v_max: max linear velocity
            w_max: max rotational velocity
            acc_lin: forward acceleration
            acc_ang: rotational acceleration"""
        if v_max is not None:
            self.v_max = v_max
        if w_max is not None:
            self.w_max = w_max
        if acc_lin is not None:
            self.acc_lin = acc_lin
        if acc_ang is not None:
            self.acc_ang = acc_ang

    def reset_error_tolerances(self):
        """Reset error tolerances to default values"""
        self.lin_error_tol = self.cfg.lin_error_tol
        self.ang_error_tol = self.cfg.ang_error_tol

    @staticmethod
    def _velocity_feedback_control(x_err, a, v_max):
        """
        Computes velocity based on distance from target (trapezoidal velocity profile).
        Used for both linear and angular motion.
        """
        t = np.sqrt(2.0 * abs(x_err) / a)  # x_err = (1/2) * a * t^2
        v = min(a * t, v_max)
        return v * np.sign(x_err)

    def _turn_rate_limit(self, lin_err, heading_diff, w_max):
        """
        Compute velocity limit that prevents path from overshooting goal

        heading error decrease rate > linear error decrease rate
        (w - v * np.sin(phi) / D) / phi > v * np.cos(phi) / D
        v < (w / phi) / (np.sin(phi) / D / phi + np.cos(phi) / D)
        v < w * D / (np.sin(phi) + phi * np.cos(phi))

        (D = linear error, phi = angular error)
        """
        assert lin_err >= 0.0
        assert heading_diff >= 0.0

        if heading_diff > self.cfg.max_heading_ang:
            return 0.0
        else:
            return (
                w_max
                * lin_err
                / (np.sin(heading_diff) + heading_diff * np.cos(heading_diff) + 1e-5)
            )

    def __call__(
        self, xyt_err: np.ndarray, allow_reverse: bool = False
    ) -> Tuple[float, float, bool]:
        v_cmd = w_cmd = 0
        in_reverse = False
        done = True

        # Compute errors
        lin_err_abs = np.linalg.norm(xyt_err[0:2])
        ang_err = xyt_err[2]

        heading_err = np.arctan2(xyt_err[1], xyt_err[0])

        # Check if reverse is required
        if allow_reverse and abs(heading_err) > np.pi / 2.0:
            in_reverse = True
            heading_err = normalize_ang_error(heading_err + np.pi)

        # Go to goal XY position if not there yet
        if lin_err_abs > self.lin_error_tol:
            # Compute linear velocity -- move towards goal XY
            v_raw = self._velocity_feedback_control(lin_err_abs, self.acc_lin, self.v_max)
            v_limit = self._turn_rate_limit(
                lin_err_abs,
                abs(heading_err),
                self.w_max / 2.0,
            )
            v_cmd = np.clip(v_raw, 0.0, v_limit)

            # Compute angular velocity -- turn towards goal XY
            w_cmd = self._velocity_feedback_control(heading_err, self.acc_ang, self.w_max)
            done = False

        # Rotate to correct yaw if XY position is at goal
        elif abs(ang_err) > self.ang_error_tol:
            # Compute angular velocity -- turn to goal orientation
            w_cmd = self._velocity_feedback_control(ang_err, self.acc_ang, self.w_max)
            done = False

        if in_reverse:
            v_cmd = -v_cmd

        return v_cmd, w_cmd, done

class VelocityController:
    """
    Simplified interface for velocity control of a differential drive robot.
    Wraps the lower-level DDVelocityControlNoplan class.
    """
    
    def __init__(
        self,
        kp_linear=0.5,
        kp_angular=0.5,
        max_linear_speed=0.5,
        max_angular_speed=0.5
    ):
        """
        Initialize the velocity controller.
        
        Args:
            kp_linear: Proportional gain for linear control
            kp_angular: Proportional gain for angular control
            max_linear_speed: Maximum linear speed (m/s)
            max_angular_speed: Maximum angular speed (rad/s)
        """
        from omegaconf import OmegaConf
        
        # Create configuration for DDVelocityControlNoplan
        cfg = OmegaConf.create({
            "v_max": max_linear_speed,
            "w_max": max_angular_speed,
            "acc_lin": kp_linear,  # Use P gain as acceleration param
            "acc_ang": kp_angular,  # Use P gain as angular acceleration param
            "lin_error_tol": 0.05,  # Default position tolerance
            "ang_error_tol": 0.05,  # Default orientation tolerance
            "min_lin_error_tol": 0.01,
            "min_ang_error_tol": 0.01,
            "lin_error_ratio": 0.5,
            "ang_error_ratio": 0.5,
            "max_rev_dist": 1.0,
            "max_heading_ang": 0.5 * np.pi  # 90 degrees
        })
        
        # Create controller
        self.controller = DDVelocityControlNoplan(cfg)
        
        # Store parameters
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        
    def compute_velocity(self, error_x, error_y, error_theta):
        """
        Compute velocity commands based on position and orientation errors.
        
        Args:
            error_x: Error in x direction (meters)
            error_y: Error in y direction (meters)
            error_theta: Error in orientation (radians)
            
        Returns:
            tuple: (linear_velocity, angular_velocity)
        """
        # Create error vector
        xyt_err = np.array([error_x, error_y, error_theta])
        
        # Compute control
        v_cmd, w_cmd, _ = self.controller(xyt_err)
        
        return v_cmd, w_cmd
        
    def set_max_speed(self, linear=None, angular=None):
        """
        Set maximum speed.
        
        Args:
            linear: Maximum linear speed (m/s)
            angular: Maximum angular speed (rad/s)
        """
        self.controller.update_velocity_profile(v_max=linear, w_max=angular)
        
        if linear is not None:
            self.max_linear_speed = linear
        if angular is not None:
            self.max_angular_speed = angular
            
    def set_gains(self, kp_linear=None, kp_angular=None):
        """
        Set control gains.
        
        Args:
            kp_linear: Proportional gain for linear control
            kp_angular: Proportional gain for angular control
        """
        self.controller.update_velocity_profile(acc_lin=kp_linear, acc_ang=kp_angular)
        
        if kp_linear is not None:
            self.kp_linear = kp_linear
        if kp_angular is not None:
            self.kp_angular = kp_angular
            
    def set_tolerances(self, position_tolerance=None, orientation_tolerance=None):
        """
        Set error tolerances.
        
        Args:
            position_tolerance: Position error tolerance (meters)
            orientation_tolerance: Orientation error tolerance (radians)
        """
        if position_tolerance is not None:
            self.controller.set_linear_error_tolerance(position_tolerance)
        if orientation_tolerance is not None:
            self.controller.set_angular_error_tolerance(orientation_tolerance)