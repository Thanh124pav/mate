"""
High-level assignment strategies for HRL V2.

These strategies assign targets to cameras. They can be:
1. Hard-coded (greedy, optimal, etc.)
2. Learned (using RL, optimization, etc.)
"""

import numpy as np
from typing import List, Tuple, Optional
import mate


__all__ = [
    'HighLevelAssigner',
    'GreedyDistanceAssigner',
    'GreedyCoverageAssigner',
    'MaxCoverageAssigner',
    'LearnedAssigner',
]


class HighLevelAssigner:
    """Base class for high-level target assignment strategies."""
    
    def __init__(self, num_cameras: int, num_targets: int):
        self.num_cameras = num_cameras
        self.num_targets = num_targets
    
    def assign(
        self, 
        cameras: List,
        targets: List,
        observations: np.ndarray
    ) -> np.ndarray:
        """
        Assign targets to cameras.
        
        Args:
            cameras: List of camera objects
            targets: List of target objects
            observations: Camera observations [num_cameras, obs_size]
        
        Returns:
            assignments: [num_cameras, num_targets] binary matrix
                        1 if camera i should track target j, 0 otherwise
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset assigner state (for stateful assigners)."""
        pass


class GreedyDistanceAssigner(HighLevelAssigner):
    """
    Greedy assignment based on distance.
    
    Each camera is assigned to the closest visible target.
    Simple but effective for basic tracking scenarios.
    """
    
    def __init__(self, num_cameras: int, num_targets: int, max_assignments_per_camera: int = 1):
        super().__init__(num_cameras, num_targets)
        self.max_assignments = max_assignments_per_camera
    
    def assign(
        self,
        cameras: List,
        targets: List,
        observations: np.ndarray
    ) -> np.ndarray:
        """Assign each camera to its closest visible target."""
        assignments = np.zeros((self.num_cameras, self.num_targets), dtype=np.bool8)
        
        # Get visibility masks from observations
        camera_obs_slices = mate.camera_observation_slices_of(
            self.num_cameras, self.num_targets, num_obstacles=0
        )
        target_view_mask_slice = camera_obs_slices['opponent_mask']
        
        for c in range(self.num_cameras):
            camera = cameras[c]
            visible_mask = observations[c, target_view_mask_slice].astype(np.bool8)
            
            if not visible_mask.any():
                # No visible targets, assign to closest target overall
                distances = [
                    np.linalg.norm(camera.location - target.location)
                    for target in targets
                ]
                closest_idx = np.argmin(distances)
                assignments[c, closest_idx] = True
            else:
                # Assign to closest visible target(s)
                visible_targets = np.where(visible_mask)[0]
                distances = [
                    np.linalg.norm(camera.location - targets[t].location)
                    for t in visible_targets
                ]
                
                # Sort by distance
                sorted_indices = np.argsort(distances)
                
                # Assign up to max_assignments closest targets
                for i in range(min(self.max_assignments, len(sorted_indices))):
                    target_idx = visible_targets[sorted_indices[i]]
                    assignments[c, target_idx] = True
        
        return assignments


class GreedyCoverageAssigner(HighLevelAssigner):
    """
    Greedy assignment to maximize coverage.
    
    Iteratively assign cameras to targets that need coverage most.
    Considers current coverage and camera capabilities.
    """
    
    def __init__(self, num_cameras: int, num_targets: int):
        super().__init__(num_cameras, num_targets)
    
    def assign(
        self,
        cameras: List,
        targets: List,
        observations: np.ndarray
    ) -> np.ndarray:
        """Assign cameras to maximize coverage."""
        assignments = np.zeros((self.num_cameras, self.num_targets), dtype=np.bool8)
        
        # Get current coverage count for each target
        camera_obs_slices = mate.camera_observation_slices_of(
            self.num_cameras, self.num_targets, num_obstacles=0
        )
        target_view_mask_slice = camera_obs_slices['opponent_mask']
        
        coverage_count = np.zeros(self.num_targets, dtype=np.int32)
        visibility_matrix = np.zeros((self.num_cameras, self.num_targets), dtype=np.bool8)
        
        # Build visibility matrix
        for c in range(self.num_cameras):
            visibility_matrix[c] = observations[c, target_view_mask_slice].astype(np.bool8)
            coverage_count += visibility_matrix[c]
        
        # Greedy assignment: prioritize least covered targets
        for c in range(self.num_cameras):
            visible_targets = np.where(visibility_matrix[c])[0]
            
            if len(visible_targets) == 0:
                # Assign to globally least covered target
                target_idx = np.argmin(coverage_count)
                assignments[c, target_idx] = True
            else:
                # Among visible, choose least covered
                visible_coverage = coverage_count[visible_targets]
                least_covered_idx = visible_targets[np.argmin(visible_coverage)]
                assignments[c, least_covered_idx] = True
        
        return assignments


class MaxCoverageAssigner(HighLevelAssigner):
    """
    Optimal assignment to maximize total coverage.
    
    Uses optimization to find best camera-target assignment.
    More computationally expensive but theoretically optimal.
    """
    
    def __init__(self, num_cameras: int, num_targets: int):
        super().__init__(num_cameras, num_targets)
    
    def assign(
        self,
        cameras: List,
        targets: List,
        observations: np.ndarray
    ) -> np.ndarray:
        """Find optimal assignment using greedy approximation."""
        # TODO: Can be replaced with optimal solver (Hungarian algorithm, etc.)
        # For now, use greedy coverage
        greedy = GreedyCoverageAssigner(self.num_cameras, self.num_targets)
        return greedy.assign(cameras, targets, observations)


class LearnedAssigner(HighLevelAssigner):
    """
    Learned assignment using a neural network or RL policy.
    
    This is a placeholder for future implementation where the high-level
    assignment can also be learned.
    """
    
    def __init__(
        self,
        num_cameras: int,
        num_targets: int,
        checkpoint_path: Optional[str] = None
    ):
        super().__init__(num_cameras, num_targets)
        self.checkpoint_path = checkpoint_path
        self.policy = None
        
        # TODO: Load learned policy if checkpoint provided
        if checkpoint_path is not None:
            raise NotImplementedError("Learned assigner not yet implemented")
    
    def assign(
        self,
        cameras: List,
        targets: List,
        observations: np.ndarray
    ) -> np.ndarray:
        """Use learned policy to assign targets."""
        # TODO: Implement learned assignment
        # For now, fallback to greedy
        greedy = GreedyDistanceAssigner(self.num_cameras, self.num_targets)
        return greedy.assign(cameras, targets, observations)
