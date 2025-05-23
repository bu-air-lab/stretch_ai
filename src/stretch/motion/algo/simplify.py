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

import random

import numpy as np

from stretch.motion.algo.rrt import TreeNode
from stretch.motion.base import Planner, PlanResult
from stretch.motion.utils.geometry import angle_difference


class SimplifyXYT(Planner):
    """Define RRT planning problem and parameters. Holds two different trees and tries to connect them with some probabability."""

    # For floating point comparisons
    theta_tol = 1e-8
    dist_tol = 1e-8

    # Debug info
    verbose: bool = False
    remove_in_place_rotations: bool = True

    def __init__(
        self,
        planner: Planner,
        min_step: float = 0.1,
        max_step: float = 0.5,
        num_steps: int = 6,
        min_angle: float = np.deg2rad(5),
    ):
        self.min_step = min_step
        self.max_step = max_step
        self.num_steps = num_steps
        self.min_angle = min_angle
        self.planner = planner
        self.reset()

    def reset(self):
        self.nodes = None

    @property
    def space(self):
        return self.planner.space

    def _verify(self, new_nodes):
        """Check to see if new nodes are spaced enough apart and nothing is within min_dist"""
        prev_node = None
        if len(new_nodes) < 2:
            return False
        for node in new_nodes:
            if prev_node is None:
                continue
            else:
                dist = np.linalg.norm(prev_node.state[:2] - node.state[:2])
                if dist < self.min_dist:
                    return False
        return True

    def plan(self, start, goal, verbose: bool = False, **kwargs) -> PlanResult:
        """Do plan simplification"""
        self.planner.reset()
        verbose = verbose or self.verbose
        if verbose:
            print("Call internal planner")
        res = self.planner.plan(start, goal, verbose=verbose, **kwargs)
        self.nodes = self.planner.nodes
        if not res.success or len(res.trajectory) < 4:
            # Planning failed so nothing to do here
            return res

        # Is it 2d?
        assert len(start) == 2 or len(start) == 3, "must be 2d or 3d to use this code"
        is_2d = len(start) == 2

        for step in np.linspace(self.max_step, self.min_step, self.num_steps):

            # The last node we explored
            prev_node = None
            # The last node in simplified sequence
            anchor_node = None
            # Cumulative distance so far (cartesian only)
            cum_dist = 0
            # angle between last 2 waypoints to make sure we're going in the same direction
            prev_theta = None
            # New trajectory
            new_nodes = []

            for i, node in enumerate(res.trajectory):
                if verbose:
                    print()
                    print()
                    print(i + 1, "/", res.get_length())
                    print(
                        "anchor =",
                        anchor_node.state if anchor_node is not None else None,  # type: ignore
                    )
                # Set the last node in the simplified sequence
                if anchor_node is None:
                    cum_dist = 0
                    new_nodes.append(TreeNode(parent=anchor_node, state=node.state))
                    prev_node = new_nodes[-1]
                    anchor_node = new_nodes[-1]
                    prev_theta = None if is_2d else node.state[-1]
                else:
                    # Check to see if we can simplify by skipping this node, or if we should add it
                    assert prev_node is not None
                    # Get the angle we use
                    if is_2d:
                        x, y = prev_node.state[:2] - node.state[:2]
                        cur_theta = np.arctan2(y, x)
                    else:
                        cur_theta = node.state[-1]
                    # Previous theta handling
                    if prev_theta is None:
                        theta_dist = 0
                    else:
                        theta_dist = np.abs(angle_difference(prev_theta, cur_theta))
                        if verbose:
                            print(f"{prev_theta=}, {cur_theta=}, {theta_dist=}")

                    dist = np.linalg.norm(node.state[:2] - prev_node.state[:2])
                    if verbose:
                        print(node.state[-1], prev_node.state[-1])
                        print("theta dist =", theta_dist)
                        print("dist", dist)
                        print("cumulative", cum_dist)

                    added = False
                    if theta_dist < self.theta_tol:
                        if verbose:
                            print(f"{theta_dist=} < {self.theta_tol=}")
                        if cum_dist >= step:
                            # Add it to the stack
                            if verbose:
                                print("add to stack")
                            new_nodes.append(TreeNode(parent=anchor_node, state=prev_node.state))
                            added = True
                            anchor_node = prev_node
                            # Distance from previous to current node, since we're setting anchor to previous node
                            cum_dist = dist
                        else:
                            # Increment distance tracker
                            cum_dist += dist
                    else:
                        # We turned, so start again from here
                        # Check to see if we moved since the anchor node
                        if cum_dist > self.dist_tol:
                            new_nodes.append(TreeNode(parent=anchor_node, state=prev_node.state))
                            anchor_node = new_nodes[-1]
                            cum_dist = dist
                        new_nodes.append(TreeNode(parent=anchor_node, state=node.state))
                        if not is_2d:
                            if not (
                                np.abs(anchor_node.state[0] - node.state[0]) < self.dist_tol
                                and np.abs(anchor_node.state[1] - node.state[1]) < self.dist_tol
                            ):
                                raise RuntimeError(
                                    f"Trajectory inconsistent: {anchor_node.state} vs {node.state}"
                                )
                        added = True
                        anchor_node = new_nodes[-1]
                        # rotated, so there should be no cumulative distance
                        cum_dist = 0

                    # Final check to make sure the last node gets added
                    if not added and i == res.get_length() - 1:
                        if verbose:
                            print("Add final node!")
                        new_nodes.append(TreeNode(parent=anchor_node, state=node.state))
                        # We're done
                        if verbose:
                            print("===========")
                        break

                    prev_node = node
                    prev_theta = cur_theta

            if self.remove_in_place_rotations:
                new_nodes_cleaned = []
                for i, node in enumerate(new_nodes):
                    if i == 0 or i == len(new_nodes) - 1:
                        new_nodes_cleaned.append(node)
                    else:
                        xy = node.state[:2]
                        if np.allclose(xy, new_nodes[i - 1].state[:2], atol=1e-6) and np.allclose(
                            xy, new_nodes[i + 1].state[:2], atol=1e-6
                        ):
                            continue
                        new_nodes_cleaned.append(node)

                # Overwrite these
                new_nodes = new_nodes_cleaned

            # Check to make sure things are spaced out enough
            if self._verify(new_nodes_cleaned):
                if verbose:
                    print("[Simplify] !!!! DONE")
                break
            else:
                if verbose:
                    print("[Simplify] VERIFY FAILED")
                new_nodes = None

        if new_nodes is not None:
            return PlanResult(True, new_nodes, planner=self)
        else:
            return PlanResult(False, reason="simplification and verification failed!", planner=self)
def simplify_path(path, obstacles=None, obstacle_inflation=0.0, min_step=0.1, max_step=0.5, num_steps=6):
    """
    Simplify a path by reducing the number of waypoints while maintaining the overall shape.
    Uses the SimplifyXYT class internally.
    
    Args:
        path: List of waypoints [[x, y], ...] or [[x, y, theta], ...]
        obstacles: List of obstacles or None
        obstacle_inflation: Inflation radius for obstacles
        min_step: Minimum step size for simplification
        max_step: Maximum step size for simplification
        num_steps: Number of steps to try for simplification
        
    Returns:
        Simplified path as a list of waypoints
    """
    if not path or len(path) < 3:
        return path  # No simplification needed for short paths
    
    # Import necessary components - these are already imported in the main file
    # but we include them here for clarity
    from stretch.motion.base import Planner, PlanResult, Space
    import numpy as np
    
    # Create a dummy planner that just returns the provided path
    class DummyPlanner(Planner):
        def __init__(self, path_to_use, space_obj):
            self.path_to_use = path_to_use
            self.space_obj = space_obj
            self.nodes = []
            
        @property
        def space(self):
            return self.space_obj
            
        def plan(self, start, goal, **kwargs):
            # Convert path to TreeNodes
            nodes = []
            for i, waypoint in enumerate(self.path_to_use):
                parent = nodes[-1] if i > 0 else None
                nodes.append(TreeNode(parent=parent, state=np.array(waypoint)))
            
            self.nodes = nodes
            return PlanResult(True, nodes, planner=self)
            
        def reset(self):
            self.nodes = []
        
        def validate(self, state1, state2):
            """Simple collision check based on obstacles"""
            if obstacles is None or len(obstacles) == 0:
                return True
                
            # Convert states to numpy arrays
            s1 = np.array(state1[:2])
            s2 = np.array(state2[:2])
            
            # Check each obstacle
            for obs in obstacles:
                obs_pos = np.array(obs[:2] if len(obs) > 2 else obs)
                
                # Check if line segment from s1 to s2 is too close to obstacle
                # Calculate distance from point to line segment
                line_vec = s2 - s1
                line_len = np.linalg.norm(line_vec)
                if line_len == 0:
                    continue
                    
                line_unit_vec = line_vec / line_len
                obs_vec = obs_pos - s1
                proj_len = np.clip(np.dot(obs_vec, line_unit_vec), 0, line_len)
                closest_point = s1 + proj_len * line_unit_vec
                distance = np.linalg.norm(obs_pos - closest_point)
                
                if distance <= obstacle_inflation:
                    return False  # Collision detected
                    
            return True
    
    # Determine dimensionality of the path
    is_3d = len(path[0]) > 2
    
    # Create space based on path dimensionality
    if is_3d:
        space = Space(lower=[-np.inf, -np.inf, -np.inf], upper=[np.inf, np.inf, np.inf])
    else:
        space = Space(lower=[-np.inf, -np.inf], upper=[np.inf, np.inf])
    
    # Create simplifier with the dummy planner
    dummy_planner = DummyPlanner(path, space)
    simplifier = SimplifyXYT(
        dummy_planner,
        min_step=min_step,
        max_step=max_step,
        num_steps=num_steps
    )
    
    # Run simplification using the first and last points as start and goal
    result = simplifier.plan(path[0], path[-1])
    
    # Extract simplified path
    if result.success:
        simplified_path = [node.state.tolist() for node in result.trajectory]
        return simplified_path
    else:
        # Return original path if simplification fails
        return path

if __name__ == "__main__":
    from stretch.motion.algo.rrt_connect import RRTConnect
    from stretch.motion.algo.shortcut import Shortcut
    from stretch.motion.utils.simple_env import SimpleEnv

    start, goal, obs = np.array([1.0, 1.0]), np.array([9.0, 9.0]), np.array([2.0, 7.0])
    env = SimpleEnv(obs)
    planner0 = RRTConnect(env.get_space(), env.validate)
    planner1 = Shortcut(planner0)
    planner2 = SimplifyXYT(planner1, max_step=5.0)

    def eval(planner):
        random.seed(0)
        np.random.seed(0)
        res = planner.plan(start, goal, verbose=True)
        print("Success:", res.success)
        if res.success:
            print("Plan =")
            for i, n in enumerate(res.trajectory):
                print(f"\t{i} = {n.state}")
            return res.get_length(), [node.state for node in res.trajectory]
        return 0, []

    len0, plan0 = eval(planner0)
    # len0 = 0
    len1, plan1 = eval(planner1)
    # len1 = 0
    len2, plan2 = eval(planner2)
    print(f"{len0=} {len1=} {len2=}")

    if len1 > 0:
        env.show(plan1)
    if len2 > 0:
        env.show(plan2)

