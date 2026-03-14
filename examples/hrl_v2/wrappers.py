"""
Wrapper for Hierarchical Camera Control V2.

Key difference from original HRL:
- High-level: Hard-coded (greedy assignment) - can be replaced with learned policy
- Low-level: Learned (RL algorithms) - camera control to track assigned targets
"""

import re
import gym
import numpy as np
from gym import spaces
from typing import Optional, Type

import mate
from examples.hrl_v2.high_level import HighLevelAssigner, GreedyDistanceAssigner
from examples.utils import CustomMetricCallback, MetricCollector


__all__ = ['HierarchicalCameraV2']


class HierarchicalCameraV2(gym.Wrapper, metaclass=mate.WrapperMeta):
    """
    Hierarchical Camera Control Wrapper V2.
    
    Architecture:
    1. High-level: Assigns targets to cameras (hard-coded or learned)
    2. Low-level: Learns camera control to track assigned targets (RL)
    
    The low-level policy receives:
    - Camera observation
    - Assigned target(s) information
    - Goal: learn to control camera to track the assigned targets
    
    Action space: Continuous camera control [Δorientation, Δviewing_angle]
    """
    
    INFO_KEYS = {
        'raw_reward': 'sum',
        'normalized_raw_reward': 'sum',
        re.compile(r'^auxiliary_reward(\w*)$'): 'sum',
        re.compile(r'^reward_coefficient(\w*)$'): 'mean',
        'coverage_rate': 'mean',
        'real_coverage_rate': 'mean',
        'mean_transport_rate': 'last',
        'num_delivered_cargoes': 'last',
        'num_tracked': 'mean',
        'num_assigned_targets': 'mean',
        'assignment_coverage_rate': 'mean',
    }
    
    def __init__(
        self,
        env,
        assigner: Optional[HighLevelAssigner] = None,
        assigner_class: Type[HighLevelAssigner] = GreedyDistanceAssigner,
        assigner_kwargs: Optional[dict] = None,
        frame_skip: int = 1,
        include_assignment_in_obs: bool = True,
        custom_metrics: Optional[dict] = None,
    ):
        """
        Initialize HRL V2 wrapper.
        
        Args:
            env: Base environment (should be MultiCamera)
            assigner: High-level assigner instance (if None, will create from class)
            assigner_class: Class to create assigner if not provided
            assigner_kwargs: Kwargs for assigner initialization
            frame_skip: Number of low-level steps per high-level assignment
            include_assignment_in_obs: Whether to include assignment in observation
            custom_metrics: Custom metrics configuration
        """
        assert isinstance(env, mate.MultiCamera), (
            f'You should use wrapper `{self.__class__}` with wrapper `MultiCamera`. '
            f'Please wrap the environment with wrapper `MultiCamera` first. '
            f'Got env = {env}.'
        )
        
        super().__init__(env)
        
        # Initialize high-level assigner
        if assigner is None:
            assigner_kwargs = assigner_kwargs or {}
            self.assigner = assigner_class(
                num_cameras=env.num_cameras,
                num_targets=env.num_targets,
                **assigner_kwargs
            )
        else:
            self.assigner = assigner
        
        self.frame_skip = frame_skip
        self.include_assignment_in_obs = include_assignment_in_obs
        
        # Low-level action space: continuous camera control
        # Each camera controls [Δorientation, Δviewing_angle]
        single_camera_action_space = env.unwrapped.camera_action_space
        self.action_space = spaces.Tuple(
            (single_camera_action_space,) * env.num_cameras
        )
        
        # Observation space: include assignment information
        if include_assignment_in_obs:
            # Original obs + assignment vector (binary, which targets assigned)
            original_obs_space = env.observation_space.spaces[0]
            assignment_space = spaces.MultiBinary(env.num_targets)
            
            augmented_obs_space = spaces.Dict({
                'obs': original_obs_space,
                'assignment': assignment_space,
            })
            
            self.observation_space = spaces.Tuple(
                (augmented_obs_space,) * env.num_cameras
            )
        else:
            self.observation_space = env.observation_space
        
        self.last_observations = None
        self.current_assignments = None
        
        # Metrics
        self.custom_metrics = custom_metrics or CustomMetricCallback.DEFAULT_CUSTOM_METRICS
        self.custom_metrics.update({
            'num_assigned_targets': 'mean',
            'assignment_coverage_rate': 'mean',
        })
    
    def load_config(self, config=None):
        """Load new configuration."""
        self.env.load_config(config=config)
        
        # Re-initialize assigner with new dimensions if needed
        assigner_class = type(self.assigner)
        assigner_kwargs = {}
        if hasattr(self.assigner, 'max_assignments'):
            assigner_kwargs['max_assignments_per_camera'] = self.assigner.max_assignments
        
        self.assigner = assigner_class(
            num_cameras=self.env.num_cameras,
            num_targets=self.env.num_targets,
            **assigner_kwargs
        )
        
        self.__init__(
            self.env,
            assigner=self.assigner,
            frame_skip=self.frame_skip,
            include_assignment_in_obs=self.include_assignment_in_obs,
            custom_metrics=self.custom_metrics,
        )
    
    def reset(self, **kwargs):
        """Reset environment and get initial assignments."""
        self.last_observations = base_observations = self.env.reset(**kwargs)
        
        # Get initial high-level assignment
        self.current_assignments = self.assigner.assign(
            cameras=self.cameras,
            targets=self.targets,
            observations=base_observations
        )
        
        # Augment observations with assignments
        if self.include_assignment_in_obs:
            observations = self._augment_observations(base_observations)
        else:
            observations = base_observations
        
        return observations
    
    def step(self, action):
        """
        Execute low-level actions.
        
        Args:
            action: Low-level camera control actions [num_cameras, 2]
                   Each action: [Δorientation, Δviewing_angle]
        
        Returns:
            observations: Augmented with target assignments
            rewards: From environment
            dones: From environment
            infos: Augmented with assignment metrics
        """
        action = np.asarray(action, dtype=np.float64)
        action = action.reshape(self.num_cameras, 2)
        
        fragment_rewards = []
        if self.frame_skip > 1:
            metric_collectors = [MetricCollector(self.INFO_KEYS) for _ in range(self.num_cameras)]
        else:
            metric_collectors = []
        
        observations = self.last_observations
        
        # Execute frame_skip steps
        for f in range(self.frame_skip):
            # Re-assign targets at the beginning of each frame_skip period
            if f == 0:
                self.current_assignments = self.assigner.assign(
                    cameras=self.cameras,
                    targets=self.targets,
                    observations=observations
                )
            
            # Execute low-level actions
            observations, rewards, dones, infos = self.env.step(action)
            
            # Add assignment metrics
            for c in range(self.num_cameras):
                num_assigned = self.current_assignments[c].sum()
                infos[c]['num_assigned_targets'] = num_assigned
                
                # Check if assigned targets are being tracked
                camera_obs_slices = mate.camera_observation_slices_of(
                    self.num_cameras, self.num_targets, num_obstacles=0
                )
                target_view_mask_slice = camera_obs_slices['opponent_mask']
                visible_mask = observations[c, target_view_mask_slice].astype(np.bool8)
                
                assigned_and_visible = np.logical_and(
                    self.current_assignments[c], visible_mask
                ).sum()
                
                assignment_coverage = (
                    assigned_and_visible / max(1, num_assigned)
                    if num_assigned > 0 else 0.0
                )
                infos[c]['assignment_coverage_rate'] = assignment_coverage
            
            if self.frame_skip > 1:
                fragment_rewards.append(rewards)
                for collector, info in zip(metric_collectors, infos):
                    collector.add(info)
            
            if all(dones):
                break
        
        self.last_observations = observations
        
        if self.frame_skip > 1:
            rewards = np.sum(fragment_rewards, axis=0).tolist()
            for collector, info in zip(metric_collectors, infos):
                info.update(collector.collect())
        
        # Augment observations with assignments
        if self.include_assignment_in_obs:
            observations = self._augment_observations(observations)
        
        return observations, rewards, dones, infos
    
    def _augment_observations(self, base_observations: np.ndarray) -> tuple:
        """Add assignment information to observations."""
        augmented_obs = []
        for c in range(self.num_cameras):
            obs_dict = {
                'obs': base_observations[c],
                'assignment': self.current_assignments[c].astype(np.float32),
            }
            augmented_obs.append(obs_dict)
        
        return tuple(augmented_obs)
    
    @property
    def cameras(self):
        """Get camera objects from environment."""
        return self.env.unwrapped.cameras
    
    @property
    def targets(self):
        """Get target objects from environment."""
        return self.env.unwrapped.targets
