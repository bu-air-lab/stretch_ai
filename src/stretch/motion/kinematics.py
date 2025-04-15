# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import math
import os
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from stretch.core.interfaces import ContinuousFullBodyAction
from stretch.motion.constants import (
    SEGWAY_BASE_FRAME,
    SEGWAY_CAMERA_FRAME,
    SEGWAY_WIDTH,
    SEGWAY_LENGTH,
)
from stretch.motion.robot import Footprint

# Segway home configuration
SEGWAY_HOME_Q = np.array([0.0, 0.0, 0.0])  # x, y, theta

# Stores joint indices for the Segway configuration space
class SegwayIdx:
    """Index definitions for Segway robot joint configuration."""
    BASE_X = 0
    BASE_Y = 1
    BASE_THETA = 2
    
    name_to_idx = {
        "base_x": BASE_X,
        "base_y": BASE_Y,
        "base_theta": BASE_THETA,
    }
    
    @classmethod
    def get_idx(cls, name: str) -> int:
        if name in cls.name_to_idx:
            return cls.name_to_idx[name]
        else:
            raise ValueError(f"Unknown joint name: {name}")


class SegwayKinematics:
    """Define motion planning structure for the Segway robot."""
    
    DEFAULT_BASE_HEIGHT = 0.0
    
    default_step = np.array([0.1, 0.1, 0.2])  # x, y, theta
    default_tols = np.array([0.05, 0.05, 0.05])  # x, y, theta

    def __init__(
        self,
        name: str = "segway_robot",
        urdf_path: str = "",
        visualize: bool = False,
        root: str = ".",
        joint_tolerance: float = 0.01,
    ):
        """Initialize the Segway kinematics."""
        self.joint_tol = joint_tolerance
        self.name = name
        self.visualize = visualize
        
        # Segway robot has 3 DoF: x, y, theta
        self.dof = 3
        self.joints_dof = 3
        self.base_height = self.DEFAULT_BASE_HEIGHT
        
        # Ranges for joints (unlimited for planar motion)
        self.range = np.zeros((self.dof, 2))
        self.range[:, 0] = -float("inf") * np.ones(3)
        self.range[:, 1] = float("inf") * np.ones(3)
        
        # Pre-compute range values for sampling
        self._mins = self.range[:, 0]
        self._maxs = self.range[:, 1]
        self._rngs = self.range[:, 1] - self.range[:, 0]

    def get_footprint(self) -> Footprint:
        """Return footprint for the robot."""
        return Footprint(
            width=SEGWAY_WIDTH,
            length=SEGWAY_LENGTH,
            width_offset=0.0,
            length_offset=0.0
        )
        
    def get_dof(self) -> int:
        """Return degrees of freedom of the robot."""
        return self.dof
        
    def sample_uniform(self, q0=None, pos=None, radius=2.0):
        """Sample random configurations."""
        q = (np.random.random(self.dof) * self._rngs) + self._mins
        q[SegwayIdx.BASE_THETA] = np.random.random() * np.pi * 2
        
        # Set the position to sample poses
        if pos is not None:
            x, y = pos[0], pos[1]
        elif q0 is not None:
            x = q0[SegwayIdx.BASE_X]
            y = q0[SegwayIdx.BASE_Y]
        else:
            x, y = None, None
            
        # Randomly sample
        if x is not None:
            theta = np.random.random() * 2 * np.pi
            dx = radius * np.cos(theta)
            dy = radius * np.sin(theta)
            q[SegwayIdx.BASE_X] = x + dx
            q[SegwayIdx.BASE_Y] = y + dy
            
        return q
        
    def set_config(self, q):
        """Set robot configuration."""
        assert len(q) == self.dof
        # In a real implementation, this would set joint values 
        pass
        
    def interpolate(self, q0, qg, step=None, xy_tol=0.05, theta_tol=0.01):
        """Interpolate from initial to final configuration."""
        if step is None:
            step = self.default_step
            
        qi = q0.copy()
        theta0 = q0[SegwayIdx.BASE_THETA]
        thetag = qg[SegwayIdx.BASE_THETA]
        xy0 = q0[[SegwayIdx.BASE_X, SegwayIdx.BASE_Y]]
        xyg = qg[[SegwayIdx.BASE_X, SegwayIdx.BASE_Y]]
        dist = np.linalg.norm(xy0 - xyg)
        
        if dist > xy_tol:
            dx, dy = xyg - xy0
            theta = np.arctan2(dy, dx)
            for qi, ai in self.interpolate_angle(
                qi, theta0, theta, step[SegwayIdx.BASE_THETA]
            ):
                yield qi, ai
            for qi, ai in self.interpolate_xy(qi, xy0, dist, step[SegwayIdx.BASE_X]):
                yield qi, ai
        else:
            theta = theta0
            
        # Update angle
        if np.abs(thetag - theta) > theta_tol:
            for qi, ai in self.interpolate_angle(
                qi, theta, thetag, step[SegwayIdx.BASE_THETA]
            ):
                yield qi, ai
                
    def interpolate_xy(self, qi, xy0, dist, step=0.1):
        """Move forward with step to target distance."""
        x, y = xy0
        theta = qi[SegwayIdx.BASE_THETA]
        
        while dist > 0:
            qi = qi.copy()
            ai = np.zeros(self.dof)
            
            if dist > step:
                dx = step
            else:
                dx = dist
                
            dist -= dx
            x += np.cos(theta) * dx
            y += np.sin(theta) * dx
            
            qi[SegwayIdx.BASE_X] = x
            qi[SegwayIdx.BASE_Y] = y
            ai[0] = dx
            
            yield qi, ai
            
    def interpolate_angle(self, qi, theta0, thetag, step=0.1):
        """Rotate to target angle."""
        # Handle angle wrapping
        if theta0 > thetag:
            thetag2 = thetag + 2 * np.pi
        else:
            thetag2 = thetag - 2 * np.pi
            
        dist1 = np.abs(thetag - theta0)
        dist2 = np.abs(thetag2 - theta0)
        
        if dist2 < dist1:
            dist = dist2
            thetag = thetag2
        else:
            dist = dist1
            
        # Get direction
        dirn = 1.0 if thetag > theta0 else -1.0
        
        while dist > 0:
            qi = qi.copy()
            ai = np.zeros(self.dof)
            
            if dist > step:
                dtheta = step
            else:
                dtheta = dist
                
            dist -= dtheta
            ai[2] = dirn * dtheta
            qi[SegwayIdx.BASE_THETA] += dirn * dtheta
            
            yield qi, ai
            
    def create_action_from_config(self, q: np.ndarray) -> ContinuousFullBodyAction:
        """Create a default interface action from configuration."""
        xyt = np.zeros(3)
        xyt[0] = q[SegwayIdx.BASE_X]
        xyt[1] = q[SegwayIdx.BASE_Y]
        xyt[2] = q[SegwayIdx.BASE_THETA]
        return ContinuousFullBodyAction(joints=np.array([]), xyt=xyt)


# For backwards compatibility with existing code that might use HelloStretchIdx
HelloStretchIdx = SegwayIdx