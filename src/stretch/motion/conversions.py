# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
from typing import Dict, List, Tuple
import numpy as np

# Import from our kinematics - use both for backward compatibility
from stretch.motion.kinematics import SegwayIdx, HelloStretchIdx

# Define indices for the Segway robot
SegwayManipIdx: Dict[str, int] = {
    "BASE_X": 0,
    "BASE_Y": 1,
    "BASE_THETA": 2,
}

# Keep the original for backward compatibility
HelloStretchManipIdx: Dict[str, int] = {
    "BASE_X": 0,
    "LIFT": 1,
    "ARM": 2,
    "WRIST_YAW": 3,
    "WRIST_PITCH": 4,
    "WRIST_ROLL": 5,
}

def delta_hab_to_position_command(cmd, pan, tilt, deltas) -> Tuple[List[float], float, float]:
    """
    Compatibility function for Stretch - for Segway this is simplified
    to just keep position values.
    """
    # For Segway, we only care about the base position
    if len(deltas) == 3:  # Segway format
        return [cmd[0] + deltas[0], cmd[1] + deltas[1], cmd[2] + deltas[2]], pan, tilt
    
    # Backward compatibility for Stretch format
    assert len(deltas) == 10
    arm = deltas[0] + deltas[1] + deltas[2] + deltas[3]
    lift = deltas[4]
    roll = deltas[5]
    pitch = deltas[6]
    yaw = deltas[7]
    positions = [
        0, # This is the robot's base x axis - not currently used
        cmd[1] + lift,
        cmd[2] + arm,
        cmd[3] + yaw,
        cmd[4] + pitch,
        cmd[5] + roll,
    ]
    pan = pan + deltas[8]
    tilt = tilt + deltas[9]
    return positions, pan, tilt

def config_to_manip_command(q):
    """
    Convert from general representation to command.
    For Segway, this is just the base position.
    """
    if len(q) == 3:  # Segway format
        return [q[SegwayIdx.BASE_X], q[SegwayIdx.BASE_Y], q[SegwayIdx.BASE_THETA]]
    
    # Backward compatibility for Stretch format
    return [
        q[HelloStretchIdx.BASE_X],
        q[HelloStretchIdx.LIFT],
        q[HelloStretchIdx.ARM],
        q[HelloStretchIdx.WRIST_YAW],
        q[HelloStretchIdx.WRIST_PITCH],
        q[HelloStretchIdx.WRIST_ROLL],
    ]

def config_to_hab(q: np.ndarray) -> np.ndarray:
    """
    Convert configuration to habitat commands.
    For Segway, this is just the base position.
    """
    if len(q) == 3:  # Segway format
        return q  # Segway just uses x, y, theta directly
    
    # Backward compatibility for Stretch format
    hab = np.zeros(10)
    hab[0] = q[HelloStretchIdx.ARM]
    hab[4] = q[HelloStretchIdx.LIFT]
    hab[5] = q[HelloStretchIdx.WRIST_ROLL]
    hab[6] = q[HelloStretchIdx.WRIST_PITCH]
    hab[7] = q[HelloStretchIdx.WRIST_YAW]
    hab[8] = q[HelloStretchIdx.HEAD_PAN]
    hab[9] = q[HelloStretchIdx.HEAD_TILT]
    return hab

def hab_to_position_command(hab_positions) -> Tuple[List[float], float, float]:
    """
    Convert habitat positions to command.
    For Segway, this is just the base position.
    """
    if len(hab_positions) == 3:  # Segway format
        return hab_positions, 0.0, 0.0  # No pan/tilt for Segway
    
    # Backward compatibility for Stretch format
    assert len(hab_positions) == 10
    arm = hab_positions[0] + hab_positions[1] + hab_positions[2] + hab_positions[3]
    lift = hab_positions[4]
    roll = hab_positions[5]
    pitch = hab_positions[6]
    yaw = hab_positions[7]
    positions = [
        0, # This is the robot's base x axis - not currently used
        lift,
        arm,
        yaw,
        pitch,
        roll,
    ]
    pan = hab_positions[8]
    tilt = hab_positions[9]
    return positions, pan, tilt

def get_manip_joint_idx(joint: str) -> int:
    """
    Get manipulation joint index.
    For Segway, we use the Segway indices.
    """
    if joint.upper() in SegwayManipIdx:
        return SegwayManipIdx[joint.upper()]
    return HelloStretchManipIdx[joint.upper()]

# Euler angle / quaternion conversion functions

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)
        
    Returns:
        quaternion: [x, y, z, w]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([x, y, z, w])

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to Euler angles.
    
    Args:
        quaternion: [x, y, z, w]
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    x, y, z, w = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])