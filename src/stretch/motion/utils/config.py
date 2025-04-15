# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import Dict, Optional

from omegaconf import OmegaConf

# Default configurations for different controllers
DEFAULT_CONFIGS = {
    "traj_follower": {
        "k_p": 1.0,
        "damp_ratio": 0.7,
        "decay": 0.99
    },
    "noplan_velocity_sim": {
        "v_max": 0.5,
        "w_max": 0.5,
        "acc_lin": 0.5,
        "acc_ang": 0.5,
        "timeout": 30.0,
        "lin_error_tol": 0.1,
        "ang_error_tol": 0.1,
        "min_lin_error_tol": 0.01, 
        "min_ang_error_tol": 0.01,
        "lin_error_ratio": 0.5,
        "ang_error_ratio": 0.5,
        "max_rev_dist": 1.0
    }
}

# Cache for loaded configurations
_config_cache: Dict[str, OmegaConf] = {}

def get_control_config(name: str, config_dir: Optional[str] = None) -> OmegaConf:
    """
    Get control configuration by name.
    
    Args:
        name: Name of the configuration to load
        config_dir: Directory to look for config files, or None to use default
        
    Returns:
        OmegaConf configuration object
    """
    global _config_cache
    
    # Check cache first
    if name in _config_cache:
        return _config_cache[name]
    
    # Try to load from file if config_dir is specified
    if config_dir is not None:
        config_path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(config_path):
            try:
                config = OmegaConf.load(config_path)
                _config_cache[name] = config
                return config
            except Exception as e:
                print(f"Error loading config from {config_path}: {e}")
    
    # Fall back to default configurations
    if name in DEFAULT_CONFIGS:
        config = OmegaConf.create(DEFAULT_CONFIGS[name])
        _config_cache[name] = config
        return config
    
    # If no default exists, create empty config
    print(f"Warning: No configuration found for '{name}', using empty config")
    config = OmegaConf.create({})
    _config_cache[name] = config
    return config