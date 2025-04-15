# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# This is how much memory we allocate
DEFAULT_GRID_SIZE = [1024, 1024]


class GridParams:
    """A 2d map that can be used for path planning. Maps in and out of the discrete grid."""

    def __init__(
        self,
        grid_size: Tuple[int, int],
        resolution: float,
        device: torch.device = torch.device("cpu"),
    ):

        if grid_size is not None:
            self.grid_size = [grid_size[0], grid_size[1]]
        else:
            self.grid_size = DEFAULT_GRID_SIZE

        # Track the center of the grid - (0, 0) in our coordinate system
        # We then just need to update everything when we want to track obstacles
        self.grid_origin = Tensor(self.grid_size + [0], device=device) // 2
        self.resolution = resolution
        # Used to track the offset from our observations so maps dont use too much space

        # Used for tensorized bounds checks
        self._grid_size_t = Tensor(self.grid_size, device=device)

    def xy_to_grid_coords(self, xy: torch.Tensor) -> Optional[np.ndarray]:
        """convert xy point to grid coords"""
        assert xy.shape[-1] == 2, "coords must be Nx2 or 2d array"
        # Handle conversion between world (X, Y) and grid coordinates
        if isinstance(xy, np.ndarray):
            xy = torch.from_numpy(xy).float()
        grid_xy = (xy / self.resolution) + self.grid_origin[:2]
        if torch.any(grid_xy >= self._grid_size_t) or torch.any(grid_xy < torch.zeros(2)):
            return None
        else:
            return grid_xy

    def grid_coords_to_xy(self, grid_coords: torch.Tensor) -> np.ndarray:
        """convert grid coordinate point to metric world xy point"""
        assert grid_coords.shape[-1] == 2, "grid coords must be an Nx2 or 2d array"
        return (grid_coords - self.grid_origin[:2]) * self.resolution

    def grid_coords_to_xyt(self, grid_coords: np.ndarray) -> np.ndarray:
        """convert grid coordinate point to metric world xyt point"""
        res = torch.zeros(3)
        res[:2] = self.grid_coords_to_xy(grid_coords)
        return res

    def mask_from_bounds(self, obstacles, explored, bounds: np.ndarray, debug: bool = False):
        """create mask from a set of 3d object bounds"""
        assert bounds.shape[0] == 3, "bounding boxes in xyz"
        assert bounds.shape[1] == 2, "min and max"
        assert (len(bounds.shape)) == 2, "only one bounding box"
        mins = torch.floor(self.xy_to_grid_coords(bounds[:2, 0])).long()
        maxs = torch.ceil(self.xy_to_grid_coords(bounds[:2, 1])).long()
        mask = torch.zeros_like(explored)
        mask[mins[0] : maxs[0] + 1, mins[1] : maxs[1] + 1] = True
        if debug:
            import matplotlib.pyplot as plt

            plt.imshow(obstacles.int() + explored.int() + mask.int())
        return mask

class OccupancyGrid:
    """
    A 2D occupancy grid for mapping environments.
    
    This grid stores occupancy information where each cell can be:
    - Unknown (0)
    - Free (-1)
    - Occupied (positive values, typically 100)
    
    The grid provides methods for updating occupancy based on sensor data,
    raytracing, and querying occupancy at specific coordinates.
    """
    
    def __init__(self, width, height, resolution, device=torch.device("cpu")):
        """
        Initialize a new occupancy grid.
        
        Args:
            width: Width of the grid in cells
            height: Height of the grid in cells
            resolution: Size of each cell in meters
            device: Device to store the grid on (CPU or GPU)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.device = device
        
        # Create underlying grid storage
        # Initialize all cells as unknown (0)
        self.grid = torch.zeros((width, height), dtype=torch.int8, device=device)
        
        # Track explored regions
        self.explored = torch.zeros((width, height), dtype=torch.bool, device=device)
        
        # Create GridParams for coordinate transformations
        self.params = GridParams(
            grid_size=(width, height),
            resolution=resolution,
            device=device
        )
        
        # Origin of the grid in world coordinates
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_theta = 0.0
        
    def reset(self):
        """Reset the grid to all unknown cells."""
        self.grid.fill_(0)
        self.explored.fill_(False)
        
    def update(self, x, y, value):
        """
        Update a single cell at (x, y) with the given occupancy value.
        
        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame
            value: Occupancy value (negative for free, positive for occupied)
        """
        grid_coords = self.params.xy_to_grid_coords(torch.tensor([x, y]))
        if grid_coords is None:
            return  # Outside grid bounds
        
        # Convert to integers
        gx, gy = grid_coords.long()
        
        # Update the grid
        self.grid[gx, gy] = value
        self.explored[gx, gy] = True
        
    def update_cells(self, cells, values):
        """
        Update multiple cells with occupancy values.
        
        Args:
            cells: Tensor of shape (N, 2) with grid coordinates
            values: Tensor of shape (N,) with occupancy values
        """
        # Check if cells are within bounds
        valid_mask = (cells[:, 0] >= 0) & (cells[:, 0] < self.width) & \
                    (cells[:, 1] >= 0) & (cells[:, 1] < self.height)
        
        valid_cells = cells[valid_mask].long()
        valid_values = values[valid_mask]
        
        # Update grid and explored status
        self.grid[valid_cells[:, 0], valid_cells[:, 1]] = valid_values
        self.explored[valid_cells[:, 0], valid_cells[:, 1]] = True
        
    def get_occupancy(self, x, y):
        """
        Get the occupancy value at world coordinates (x, y).
        
        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame
            
        Returns:
            Occupancy value or None if outside grid bounds
        """
        grid_coords = self.params.xy_to_grid_coords(torch.tensor([x, y]))
        if grid_coords is None:
            return None  # Outside grid bounds
        
        # Convert to integers
        gx, gy = grid_coords.long()
        
        return self.grid[gx, gy].item()
    
    def get_explored_mask(self):
        """Get the mask of explored cells."""
        return self.explored
    
    def get_occupancy_grid(self):
        """Get the full occupancy grid."""
        return self.grid
    
    def is_occupied(self, x, y, threshold=50):
        """
        Check if a cell is occupied.
        
        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame
            threshold: Threshold above which a cell is considered occupied
            
        Returns:
            True if occupied, False if free or unknown, None if outside bounds
        """
        occupancy = self.get_occupancy(x, y)
        if occupancy is None:
            return None
        
        return occupancy > threshold
    
    def is_free(self, x, y, threshold=-50):
        """
        Check if a cell is free.
        
        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame
            threshold: Threshold below which a cell is considered free
            
        Returns:
            True if free, False if occupied or unknown, None if outside bounds
        """
        occupancy = self.get_occupancy(x, y)
        if occupancy is None:
            return None
        
        return occupancy < threshold
    
    def raytrace(self, start_x, start_y, end_x, end_y, max_range=None):
        """
        Perform a raytrace from start to end coordinates.
        
        Args:
            start_x, start_y: Start point in world coordinates
            end_x, end_y: End point in world coordinates
            max_range: Maximum range for the raytrace
            
        Returns:
            List of grid cells along the ray
        """
        # Convert to grid coordinates
        start = self.params.xy_to_grid_coords(torch.tensor([start_x, start_y]))
        end = self.params.xy_to_grid_coords(torch.tensor([end_x, end_y]))
        
        if start is None or end is None:
            return []
        
        # Convert to integers
        start_x, start_y = start.long()
        end_x, end_y = end.long()
        
        # Use Bresenham's line algorithm
        cells = []
        dx = abs(end_x - start_x)
        dy = -abs(end_y - start_y)
        sx = 1 if start_x < end_x else -1
        sy = 1 if start_y < end_y else -1
        err = dx + dy
        
        x, y = start_x, start_y
        
        while True:
            cells.append((x.item(), y.item()))
            
            if x == end_x and y == end_y:
                break
                
            # Check max range if specified
            if max_range is not None:
                distance = torch.sqrt(((x - start_x)**2 + (y - start_y)**2).float()) * self.resolution
                if distance > max_range:
                    break
            
            e2 = 2 * err
            if e2 >= dy:
                if x == end_x:
                    break
                err += dy
                x += sx
            if e2 <= dx:
                if y == end_y:
                    break
                err += dx
                y += sy
        
        return cells
    
    def update_from_scan(self, pose_x, pose_y, pose_theta, ranges, angles, max_range=None, free_val=-1, occupied_val=100):
        """
        Update the grid using a laser scan.
        
        Args:
            pose_x, pose_y, pose_theta: Robot pose
            ranges: Array of range measurements
            angles: Array of angles for each measurement
            max_range: Maximum valid range
            free_val: Value to use for free cells
            occupied_val: Value to use for occupied cells
        """
        for i, (r, angle) in enumerate(zip(ranges, angles)):
            # Skip invalid measurements
            if torch.isnan(r) or r <= 0:
                continue
                
            # Skip measurements beyond max range
            if max_range is not None and r > max_range:
                r = max_range
            
            # Calculate endpoint in world coordinates
            global_angle = pose_theta + angle
            end_x = pose_x + r * torch.cos(global_angle)
            end_y = pose_y + r * torch.sin(global_angle)
            
            # Get cells along the ray
            ray_cells = self.raytrace(pose_x, pose_y, end_x, end_y, max_range)
            
            # Mark cells as free or occupied
            for j, (cell_x, cell_y) in enumerate(ray_cells):
                # Last cell is occupied if within valid range
                if j == len(ray_cells) - 1 and r < max_range:
                    self.grid[cell_x, cell_y] = occupied_val
                else:
                    # Mark as free
                    self.grid[cell_x, cell_y] = free_val
                
                # Mark as explored
                self.explored[cell_x, cell_y] = True