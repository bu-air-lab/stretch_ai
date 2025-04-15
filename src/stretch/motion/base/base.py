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
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from .space import ConfigurationSpace, Node

"""
This just defines the standard interface for a motion planner
"""


class PlanResult(object):
    """Stores motion plan. Can be extended."""

    def __init__(
        self,
        success,
        trajectory: Optional[List] = None,
        reason: Optional[str] = None,
        planner: Optional["Planner"] = None,
    ):
        self.success = success
        self.trajectory = trajectory
        self.reason = reason
        self.planner = planner

    def get_success(self):
        """Was the trajectory planning successful?"""
        return self.success

    def get_trajectory(self, *args, **kwargs) -> Optional[List]:
        """Return the trajectory"""
        return self.trajectory

    def get_length(self):
        """Length of a plan"""
        if not self.success:
            return 0
        return len(self.trajectory)


class Planner(ABC):
    """planner base class"""

    def __init__(self, space: ConfigurationSpace, validate_fn: Callable):
        self._space = space
        self._validate = validate_fn
        self._nodes: Optional[List[Node]] = None

    @property
    def space(self) -> ConfigurationSpace:
        return self._space

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: List[Node]):
        self._nodes = nodes

    @abstractmethod
    def plan(self, start, goal, verbose: bool = False, **kwargs) -> PlanResult:
        """returns a trajectory"""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """reset the planner"""
        raise NotImplementedError

    def validate(self, state) -> bool:
        """Check if state is valid"""
        return self._validate(state)

class MotionBase:
    """
    Base class for robot motion control.
    
    This class provides a common interface for controlling robot motion,
    independent of the specific robot platform.
    """
    
    def __init__(self):
        """Initialize the motion base controller."""
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.is_moving = False
        self.max_linear_velocity = 1.0  # m/s
        self.max_angular_velocity = 1.0  # rad/s
        self.min_linear_velocity = 0.0  # m/s
        self.min_angular_velocity = 0.0  # rad/s
        
    def set_velocity(self, linear, angular):
        """
        Set the robot's velocity.
        
        Args:
            linear: Linear velocity (m/s)
            angular: Angular velocity (rad/s)
            
        Returns:
            bool: True if successful
        """
        # Apply limits
        linear = max(min(linear, self.max_linear_velocity), -self.max_linear_velocity)
        angular = max(min(angular, self.max_angular_velocity), -self.max_angular_velocity)
        
        # Update internal state
        self.linear_velocity = linear
        self.angular_velocity = angular
        self.is_moving = (abs(linear) > 0.001 or abs(angular) > 0.001)
        
        return True
        
    def stop(self):
        """
        Stop the robot immediately.
        
        Returns:
            bool: True if successful
        """
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.is_moving = False
        return True
        
    def get_velocity(self):
        """
        Get the current velocity.
        
        Returns:
            tuple: (linear_velocity, angular_velocity)
        """
        return (self.linear_velocity, self.angular_velocity)
        
    def is_stopped(self):
        """
        Check if the robot is stopped.
        
        Returns:
            bool: True if stopped
        """
        return not self.is_moving
        
    def set_velocity_limits(self, max_linear=None, max_angular=None, min_linear=None, min_angular=None):
        """
        Set velocity limits.
        
        Args:
            max_linear: Maximum linear velocity (m/s)
            max_angular: Maximum angular velocity (rad/s)
            min_linear: Minimum linear velocity (m/s)
            min_angular: Minimum angular velocity (rad/s)
            
        Returns:
            bool: True if successful
        """
        if max_linear is not None:
            self.max_linear_velocity = max_linear
        if max_angular is not None:
            self.max_angular_velocity = max_angular
        if min_linear is not None:
            self.min_linear_velocity = min_linear
        if min_angular is not None:
            self.min_angular_velocity = min_angular
        return True
        
    def get_velocity_limits(self):
        """
        Get velocity limits.
        
        Returns:
            tuple: (max_linear, max_angular, min_linear, min_angular)
        """
        return (self.max_linear_velocity, self.max_angular_velocity, 
                self.min_linear_velocity, self.min_angular_velocity)