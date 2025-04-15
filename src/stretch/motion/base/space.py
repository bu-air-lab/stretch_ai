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
from abc import ABC
from typing import List

import numpy as np


class Node(ABC):
    """Placeholder containing just a state."""

    def __init__(self, state):
        self.state = state


class ConfigurationSpace(ABC):
    """class defining a region over which we can sample parameters"""

    def __init__(self, dof: int, mins, maxs, step_size: float = 0.1):
        self.dof = dof
        self.step_size = step_size
        self.update_bounds(mins, maxs)
        assert self.step_size > 0.0

    def update_bounds(self, mins, maxs):
        assert len(mins) == self.dof, "mins' length must be equal to the space dof"
        assert len(maxs) == self.dof, "maxs' length must be equal to space dof"
        self.mins = mins
        self.maxs = maxs
        self.ranges = maxs - mins

    def sample(self) -> np.ndarray:
        return (np.random.random(self.dof) * self.ranges) + self.mins

    def distance(self, q0, q1) -> float:
        """Return distance between q0 and q1."""
        return float(np.linalg.norm(q0 - q1))

    def extend(self, q0, q1):
        """extend towards another configuration in this space"""
        dq = q1 - q0
        step = dq / np.linalg.norm(dq) * self.step_size
        if self.distance(q0, q1) > self.step_size:
            qi = q0 + step
            while self.distance(qi, q1) > self.step_size:
                qi = qi + step
                yield qi
        yield q1

    def closest_node_to_state(self, state, nodes: List[Node]):
        """returns closest node to a given state"""
        min_dist = float("Inf")
        min_node = None
        if nodes is None:
            return None
        for node in nodes:
            dist = self.distance(node.state, state)
            if dist < min_dist:
                min_dist = dist
                min_node = node
        return min_node


class XYT(ConfigurationSpace):
    """Space for (x, y, theta) base rotations"""

    def __init__(self, mins: np.ndarray = None, maxs: np.ndarray = None):
        """Create XYT space with some defaults"""
        if mins is None:
            mins = np.array([-10, -10, -np.pi])
        if maxs is None:
            maxs = np.array([10, 10, np.pi])
        super(XYT, self).__init__(3, mins, maxs)

    def update_bounds(self, mins, maxs):
        """Update bounds for just x and y sometimes, since that's all that will be changing"""
        if len(mins) == 3:
            super().update_bounds(mins, maxs)
        elif len(mins) == 2:
            assert len(mins) == len(maxs), "min and max bounds must match"
            # Just update x and y
            self.mins[:2] = mins
            self.maxs[:2] = maxs
            self.ranges[:2] = maxs - mins

class SE2:
    """
    Special Euclidean group SE(2) representing 2D poses (position and orientation).
    Used for representing robot positions in a 2D plane.
    """
    
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        """
        Initialize an SE(2) pose.
        
        Args:
            x: X-coordinate in meters
            y: Y-coordinate in meters
            theta: Orientation in radians
        """
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
    
    @classmethod
    def from_xyz_rpy(cls, xyz, rpy):
        """Create from position and Euler angles."""
        return cls(xyz[0], xyz[1], rpy[2])
    
    @classmethod
    def from_matrix(cls, matrix):
        """Create from homogeneous transformation matrix."""
        if matrix.shape != (3, 3):
            raise ValueError("SE2 transformation matrix must be 3x3")
        
        x = matrix[0, 2]
        y = matrix[1, 2]
        theta = np.arctan2(matrix[1, 0], matrix[0, 0])
        
        return cls(x, y, theta)
    
    def to_matrix(self):
        """Convert to homogeneous transformation matrix."""
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        
        matrix = np.array([
            [cos_theta, -sin_theta, self.x],
            [sin_theta, cos_theta, self.y],
            [0.0, 0.0, 1.0]
        ])
        
        return matrix
    
    def to_xyz_rpy(self):
        """Convert to position and Euler angles."""
        return np.array([self.x, self.y, 0.0]), np.array([0.0, 0.0, self.theta])
    
    def to_array(self):
        """Convert to numpy array [x, y, theta]."""
        return np.array([self.x, self.y, self.theta])
    
    def __mul__(self, other):
        """Compose two SE(2) transformations."""
        if not isinstance(other, SE2):
            raise TypeError("SE2 can only be multiplied with another SE2")
        
        # Matrix multiplication for composition
        matrix = self.to_matrix() @ other.to_matrix()
        return SE2.from_matrix(matrix)
    
    def inverse(self):
        """Get the inverse transformation."""
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        
        x = -self.x * cos_theta - self.y * sin_theta
        y = self.x * sin_theta - self.y * cos_theta
        theta = -self.theta
        
        return SE2(x, y, theta)
    
    def __str__(self):
        return f"SE2(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f})"