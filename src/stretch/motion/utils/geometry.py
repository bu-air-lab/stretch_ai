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

from typing import Iterable, Tuple

import numpy as np
import sophuspy as sp
from scipy.spatial.transform import Rotation

PI2 = 2 * np.pi


def normalize_ang_error(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + np.pi) % PI2 - np.pi


def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate the smallest difference between two angles in radians."""
    angle1 = angle1 % PI2
    angle2 = angle2 % PI2
    diff = np.abs(angle1 - angle2)
    return min(diff, PI2 - diff)


def xyt_global_to_base(XYT, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : base position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    pose_world2target = xyt2sophus(XYT)
    pose_world2base = xyt2sophus(current_pose)
    pose_base2target = pose_world2base.inverse() * pose_world2target
    return sophus2xyt(pose_base2target)


def xyt_base_to_global(out_XYT, current_pose):
    """
    Transforms the point cloud from base frame into geocentric frame
    Input:
        XYZ                     : ...x3
        current_pose            : base position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    pose_base2target = xyt2sophus(out_XYT)
    pose_world2base = xyt2sophus(current_pose)
    pose_world2target = pose_world2base * pose_base2target
    return sophus2xyt(pose_world2target)


def xyt2sophus(xyt: np.ndarray) -> sp.SE3:
    """
    Converts SE2 coordinates (x, y, rz) to an sophus SE3 pose object.
    """
    x = np.array([xyt[0], xyt[1], 0.0])
    r_mat = sp.SO3.exp([0.0, 0.0, xyt[2]]).matrix()
    return sp.SE3(r_mat, x)


def sophus2xyt(se3: sp.SE3) -> np.ndarray:
    """
    Converts an sophus SE3 pose object to SE2 coordinates (x, y, rz).
    """
    x_vec = se3.translation()
    r_vec = se3.so3().log()
    return np.array([x_vec[0], x_vec[1], r_vec[2]])


def posquat2sophus(pos: Iterable[float], quat: Iterable[float]) -> sp.SE3:
    r_mat = Rotation.from_quat(quat).as_matrix()
    return sp.SE3(r_mat, pos)


def sophus2posquat(se3: sp.SE3) -> Tuple[Iterable[float], Iterable[float]]:
    pos = se3.translation()
    quat = Rotation.from_matrix(se3.so3().matrix()).as_quat()
    return pos, quat


def interpolate_angles(start_angle: float, end_angle: float, step_size: float = 0.1) -> float:
    """Interpolate between two angles in radians with a given step size."""
    start_angle = start_angle % PI2
    end_angle = end_angle % PI2
    diff1 = (end_angle - start_angle) % PI2
    diff2 = (start_angle - end_angle) % PI2
    if diff1 <= diff2:
        direction = 1
        delta = diff1
    else:
        direction = -1
        delta = diff2
    step = min(delta, step_size) * direction
    interpolated_angle = start_angle + step
    return interpolated_angle % PI2

def transform_pose(pose, transform):
    """
    Transform a pose using a transformation.
    More general version of xyt_base_to_global and xyt_global_to_base.
    
    Args:
        pose: Pose as [x, y, theta] or (pos, quat)
        transform: Transformation as [x, y, theta] or 4x4 matrix
        
    Returns:
        Transformed pose in the same format as input
    """
    if isinstance(pose, tuple) and len(pose) == 2:
        # (pos, quat) format
        pos, quat = pose
        pose_se3 = posquat2sophus(pos, quat)
        
        if isinstance(transform, sp.SE3):
            transform_se3 = transform
        elif len(transform) == 3:
            transform_se3 = xyt2sophus(transform)
        else:
            raise ValueError("Unsupported transform format")
        
        transformed_se3 = transform_se3 * pose_se3
        return sophus2posquat(transformed_se3)
    
    elif len(pose) == 3:
        # [x, y, theta] format - use existing function
        if len(transform) == 3:
            return xyt_base_to_global(pose, transform)
        else:
            # Handle general transform case
            pose_se3 = xyt2sophus(pose)
            transform_se3 = posquat2sophus(transform[:3], transform[3:])
            transformed_se3 = transform_se3 * pose_se3
            return sophus2xyt(transformed_se3)
    
    else:
        raise ValueError("Unsupported pose format")

def get_yaw(pose_or_quat):
    """
    Extract the yaw angle from a pose or quaternion.
    
    Args:
        pose_or_quat: Pose as [x, y, theta], sp.SE3, or quaternion
        
    Returns:
        Yaw angle in radians
    """
    if isinstance(pose_or_quat, sp.SE3):
        return sophus2xyt(pose_or_quat)[2]
    elif isinstance(pose_or_quat, (list, np.ndarray)) and len(pose_or_quat) == 3:
        return pose_or_quat[2]
    elif isinstance(pose_or_quat, (list, np.ndarray)) and len(pose_or_quat) == 4:
        # Quaternion format
        quat = pose_or_quat
        rot = Rotation.from_quat(quat)
        euler = rot.as_euler('xyz')
        return euler[2]  # Yaw is the rotation around z-axis
    else:
        raise ValueError("Unsupported pose or quaternion format")

# Simple alias for normalize_ang_error
normalize_angle = normalize_ang_error

if __name__ == "__main__":
    print(interpolate_angles(4.628, 4.28))
