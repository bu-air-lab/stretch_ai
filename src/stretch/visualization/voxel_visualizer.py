import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..mapping.voxel_map.voxel_grid import VoxelGrid
from ..utils.logging import get_logger

logger = get_logger(__name__)

class VoxelVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
    
    def visualize_voxel_grid(self, 
                            voxel_grid: VoxelGrid,
                            show_objects: bool = True,
                            output_file: str = None):
        """Visualize voxel grid with optional object coloring"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax.clear()
        
        # Collect occupied voxels
        occupied_voxels = []
        colors = []
        
        for idx in np.ndindex(tuple(voxel_grid.dimensions)):
            cell = voxel_grid.grid[idx]
            if cell.occupied:
                occupied_voxels.append(idx)
                if show_objects and cell.semantic_class:
                    # Use hash of semantic class for consistent colors
                    colors.append(plt.cm.tab20(hash(cell.semantic_class) % 20))
                else:
                    colors.append('gray')
        
        if occupied_voxels:
            occupied_voxels = np.array(occupied_voxels) * voxel_grid.resolution
            self.ax.scatter(
                occupied_voxels[:, 0],
                occupied_voxels[:, 1],
                occupied_voxels[:, 2],
                c=colors,
                marker='s',
                alpha=0.6
            )
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved voxel visualization to {output_file}")
        else:
            plt.show() 