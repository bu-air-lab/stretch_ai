from dataclasses import dataclass
from typing import Tuple

@dataclass
class VoxelMapConfig:
    resolution: float = 0.05
    min_points_per_voxel: int = 3
    confidence_threshold: float = 0.3
    padding: float = 1.0

@dataclass
class MappingConfig:
    voxel_map: VoxelMapConfig = VoxelMapConfig() 