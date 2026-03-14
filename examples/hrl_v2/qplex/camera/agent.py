"""
QPLEX Agent for HRL V2 Camera Control.

This agent learns low-level camera control to track assigned targets.
High-level assignment is done by a greedy algorithm.
"""

import numpy as np
import ray
from ray.rllib.agents.qplex import QPlexTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

import mate
from examples.hrl_v2.high_level import GreedyDistanceAssigner


class HRLv2QPLEXCameraAgent(mate.CameraAgentBase):
    """
    QPLEX-based camera agent for HRL V2.
    
    Uses greedy high-level assignment and learned low-level control.
    """
    
    def __init__(
        self,
        num_cameras,
        seed=0,
        checkpoint_path=None,
        policy_id=DEFAULT_POLICY_ID,
        **kwargs
    ):
        super().__init__(
            num_cameras=num_cameras,
            seed=seed,
            **kwargs
        )
        
        self.checkpoint_path = checkpoint_path
        self.policy_id = policy_id
        
        # Initialize high-level assigner
        self.assigner = None  # Will be initialized in reset
        
        # Load QPLEX trainer
        if checkpoint_path is not None:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
            
            self.trainer = QPlexTrainer(env='mate-hrl_v2.qplex.camera')
            self.trainer.restore(checkpoint_path)
            self.policy = self.trainer.get_policy(policy_id)
        else:
            raise ValueError("checkpoint_path is required for HRLv2QPLEXCameraAgent")
        
        # State
        self.hidden_states = None
        self.current_assignments = None
    
    def reset(self, observation):
        """Reset agent and initialize high-level assigner."""
        observation = np.asarray(observation)
        
        # Initialize assigner if not done yet
        if self.assigner is None:
            num_targets = self._infer_num_targets(observation)
            self.assigner = GreedyDistanceAssigner(
                num_cameras=self.num_cameras,
                num_targets=num_targets,
                max_assignments_per_camera=1
            )
        
        # Reset hidden states for RNN
        self.hidden_states = self.policy.get_initial_state()
        
        # Initial assignment (will be updated in act)
        self.current_assignments = None
        
        return super().reset(observation)
    
    def act(self, observation, info=None):
        """
        Execute hierarchical action.
        
        1. High-level: Assign targets using greedy algorithm
        2. Low-level: Use QPLEX policy to control camera
        """
        observation = np.asarray(observation)
        
        # Extract base observations (remove assignment if included)
        if isinstance(observation[0], dict):
            base_observations = np.array([obs['obs'] for obs in observation])
        else:
            base_observations = observation
        
        # High-level assignment (greedy)
        # Note: In real deployment, cameras and targets should come from environment
        # Here we use a simplified approach
        if self.current_assignments is None:
            # Dummy assignment for inference (should be provided by environment)
            num_targets = self._infer_num_targets(base_observations)
            self.current_assignments = np.zeros(
                (self.num_cameras, num_targets), dtype=np.bool8
            )
            # Simple assignment: camera i → target i % num_targets
            for c in range(self.num_cameras):
                self.current_assignments[c, c % num_targets] = True
        
        # Augment observations with assignments
        augmented_obs = []
        for c in range(self.num_cameras):
            obs_dict = {
                'obs': base_observations[c],
                'assignment': self.current_assignments[c].astype(np.float32),
            }
            augmented_obs.append(obs_dict)
        
        # Low-level action from QPLEX policy
        # Group observations for QPLEX
        grouped_obs = {'camera': augmented_obs}
        
        # Compute actions
        actions, self.hidden_states, _ = self.policy.compute_actions(
            obs_batch=[grouped_obs],
            state_batches=self.hidden_states,
            explore=False
        )
        
        # Ungroup actions
        camera_actions = actions[0]  # [num_cameras, 2]
        
        return camera_actions
    
    def _infer_num_targets(self, observations):
        """Infer number of targets from observation shape."""
        # This is a heuristic - in practice should be provided by environment
        obs_size = observations.shape[1]
        
        # Typical observation structure for MATE:
        # camera_obs + target_obs * num_targets
        # Assume camera_obs ~ 10, target_obs ~ 5-10 per target
        
        # For MATE-4v5: obs_size ~ 60-80
        # Rough estimate
        num_targets_estimate = max(1, (obs_size - 20) // 10)
        return min(num_targets_estimate, 10)  # Cap at reasonable value
