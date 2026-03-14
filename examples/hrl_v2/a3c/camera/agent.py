"""
A3C Agent for HRL V2 Camera Control.

This agent learns low-level camera control to track assigned targets.
High-level assignment is done by a greedy algorithm.
A3C (Asynchronous Advantage Actor-Critic) trains multiple workers asynchronously.
"""

import numpy as np
import ray
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

import mate
from examples.hrl_v2.high_level import GreedyDistanceAssigner


class HRLv2A3CCameraAgent(mate.CameraAgentBase):
    """
    A3C-based camera agent for HRL V2.
    
    Uses greedy high-level assignment and learned low-level control.
    Asynchronous training for fast convergence.
    """
    
    def __init__(
        self,
        num_cameras,
        seed=0,
        checkpoint_path=None,
        policy_id='camera_policy',
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
        
        # Load A3C trainer
        if checkpoint_path is not None:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
            
            self.trainer = A3CTrainer(env='mate-hrl_v2.a3c.camera')
            self.trainer.restore(checkpoint_path)
            self.policy = self.trainer.get_policy(policy_id)
        else:
            raise ValueError("checkpoint_path is required for HRLv2A3CCameraAgent")
        
        # State
        self.lstm_states = None
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
        
        # Reset LSTM states
        self.lstm_states = {
            f'camera_{i}': self.policy.get_initial_state()
            for i in range(self.num_cameras)
        }
        
        # Initial assignment
        self.current_assignments = None
        
        return super().reset(observation)
    
    def act(self, observation, info=None):
        """
        Execute hierarchical action.
        
        1. High-level: Assign targets using greedy algorithm
        2. Low-level: Use A3C policy to control camera
        """
        observation = np.asarray(observation)
        
        # Extract base observations
        if isinstance(observation[0], dict):
            base_observations = np.array([obs['obs'] for obs in observation])
        else:
            base_observations = observation
        
        # High-level assignment (greedy)
        if self.current_assignments is None:
            num_targets = self._infer_num_targets(base_observations)
            self.current_assignments = np.zeros(
                (self.num_cameras, num_targets), dtype=np.bool8
            )
            # Simple assignment: camera i → target i % num_targets
            for c in range(self.num_cameras):
                self.current_assignments[c, c % num_targets] = True
        
        # Augment observations with assignments
        augmented_obs = {}
        for c in range(self.num_cameras):
            obs_dict = {
                'obs': base_observations[c],
                'assignment': self.current_assignments[c].astype(np.float32),
            }
            augmented_obs[f'camera_{c}'] = obs_dict
        
        # Low-level actions from A3C policy
        camera_actions = []
        new_lstm_states = {}
        
        for c in range(self.num_cameras):
            agent_id = f'camera_{c}'
            obs = augmented_obs[agent_id]
            
            # Compute action for this camera
            action, lstm_state, _ = self.policy.compute_single_action(
                obs,
                state=self.lstm_states[agent_id],
                explore=False
            )
            
            camera_actions.append(action)
            new_lstm_states[agent_id] = lstm_state
        
        self.lstm_states = new_lstm_states
        
        return np.array(camera_actions)
    
    def _infer_num_targets(self, observations):
        """Infer number of targets from observation shape."""
        obs_size = observations.shape[1]
        num_targets_estimate = max(1, (obs_size - 20) // 10)
        return min(num_targets_estimate, 10)
