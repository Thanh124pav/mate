#!/usr/bin/env python3

"""
Unit tests for HRL V2 components.
"""

import unittest
import numpy as np
import mate
from examples.hrl_v2.high_level import (
    GreedyDistanceAssigner,
    GreedyCoverageAssigner,
)
from examples.hrl_v2.wrappers import HierarchicalCameraV2


class TestHighLevelAssigners(unittest.TestCase):
    """Test high-level assignment strategies."""
    
    def setUp(self):
        """Create test environment."""
        self.env = mate.make('MultiAgentTracking-v0', config='MATE-4v5-0.yaml')
        self.num_cameras = self.env.num_cameras
        self.num_targets = self.env.num_targets
    
    def tearDown(self):
        """Clean up."""
        self.env.close()
    
    def test_greedy_distance_assigner(self):
        """Test greedy distance assignment."""
        assigner = GreedyDistanceAssigner(self.num_cameras, self.num_targets)
        
        obs = self.env.reset()
        camera_obs = obs['camera']
        
        # Split camera observations
        camera_joint_obs = mate.MultiCamera.split_camera_joint_observation(
            camera_obs, self.num_cameras
        )
        
        # Get cameras and targets
        cameras = self.env.unwrapped.cameras
        targets = self.env.unwrapped.targets
        
        # Test assignment
        assignments = assigner.assign(cameras, targets, camera_joint_obs)
        
        # Verify shape
        self.assertEqual(assignments.shape, (self.num_cameras, self.num_targets))
        
        # Verify each camera has at least one assignment
        for c in range(self.num_cameras):
            self.assertGreater(assignments[c].sum(), 0, 
                             f"Camera {c} has no assignments")
    
    def test_greedy_coverage_assigner(self):
        """Test greedy coverage assignment."""
        assigner = GreedyCoverageAssigner(self.num_cameras, self.num_targets)
        
        obs = self.env.reset()
        camera_obs = obs['camera']
        camera_joint_obs = mate.MultiCamera.split_camera_joint_observation(
            camera_obs, self.num_cameras
        )
        
        cameras = self.env.unwrapped.cameras
        targets = self.env.unwrapped.targets
        
        assignments = assigner.assign(cameras, targets, camera_joint_obs)
        
        # Verify shape
        self.assertEqual(assignments.shape, (self.num_cameras, self.num_targets))
        
        # Verify coverage (each target should ideally have some coverage)
        target_coverage = assignments.sum(axis=0)
        self.assertGreater(target_coverage.sum(), 0, "No targets covered")


class TestHierarchicalCameraV2Wrapper(unittest.TestCase):
    """Test HierarchicalCameraV2 wrapper."""
    
    def setUp(self):
        """Create test environment."""
        base_env = mate.make('MultiAgentTracking-v0', config='MATE-4v5-0.yaml')
        target_agent = mate.GreedyTargetAgent(seed=0)
        self.env = mate.MultiCamera(base_env, target_agent=target_agent)
    
    def tearDown(self):
        """Clean up."""
        self.env.close()
    
    def test_wrapper_initialization(self):
        """Test wrapper can be initialized."""
        wrapped_env = HierarchicalCameraV2(
            self.env,
            assigner_class=GreedyDistanceAssigner,
            frame_skip=5,
        )
        
        self.assertIsNotNone(wrapped_env.assigner)
        self.assertEqual(wrapped_env.frame_skip, 5)
        
        wrapped_env.close()
    
    def test_observation_augmentation(self):
        """Test observation includes assignment."""
        wrapped_env = HierarchicalCameraV2(
            self.env,
            include_assignment_in_obs=True,
        )
        
        obs = wrapped_env.reset()
        
        # Check structure
        self.assertEqual(len(obs), wrapped_env.num_cameras)
        
        # Check first observation
        self.assertIsInstance(obs[0], dict)
        self.assertIn('obs', obs[0])
        self.assertIn('assignment', obs[0])
        
        # Check shapes
        self.assertIsInstance(obs[0]['obs'], np.ndarray)
        self.assertIsInstance(obs[0]['assignment'], np.ndarray)
        self.assertEqual(len(obs[0]['assignment']), wrapped_env.num_targets)
        
        wrapped_env.close()
    
    def test_step_execution(self):
        """Test environment step works correctly."""
        wrapped_env = HierarchicalCameraV2(
            self.env,
            frame_skip=5,
        )
        
        obs = wrapped_env.reset()
        
        # Random action
        action = wrapped_env.action_space.sample()
        
        # Step
        next_obs, reward, done, info = wrapped_env.step(action)
        
        # Verify outputs
        self.assertEqual(len(next_obs), wrapped_env.num_cameras)
        self.assertEqual(len(reward), wrapped_env.num_cameras)
        self.assertEqual(len(done), wrapped_env.num_cameras)
        self.assertEqual(len(info), wrapped_env.num_cameras)
        
        # Check metrics in info
        self.assertIn('num_assigned_targets', info[0])
        
        wrapped_env.close()
    
    def test_frame_skip(self):
        """Test frame skip behavior."""
        for frame_skip in [1, 5, 10]:
            wrapped_env = HierarchicalCameraV2(
                self.env,
                frame_skip=frame_skip,
            )
            
            obs = wrapped_env.reset()
            
            # Take one step
            action = wrapped_env.action_space.sample()
            next_obs, reward, done, info = wrapped_env.step(action)
            
            # Reward should be accumulated over frame_skip steps
            # (testing implicitly by checking no errors)
            
            wrapped_env.close()
    
    def test_different_assigners(self):
        """Test different assignment strategies work."""
        assigners = [
            GreedyDistanceAssigner,
            GreedyCoverageAssigner,
        ]
        
        for assigner_class in assigners:
            wrapped_env = HierarchicalCameraV2(
                self.env,
                assigner_class=assigner_class,
            )
            
            obs = wrapped_env.reset()
            action = wrapped_env.action_space.sample()
            next_obs, reward, done, info = wrapped_env.step(action)
            
            # Just verify it doesn't crash
            self.assertIsNotNone(next_obs)
            
            wrapped_env.close()


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_full_episode(self):
        """Run a complete episode."""
        base_env = mate.make('MultiAgentTracking-v0', config='MATE-4v5-0.yaml')
        target_agent = mate.GreedyTargetAgent(seed=0)
        env = mate.MultiCamera(base_env, target_agent=target_agent)
        
        env = HierarchicalCameraV2(
            env,
            assigner_class=GreedyDistanceAssigner,
            frame_skip=5,
            include_assignment_in_obs=True,
        )
        
        obs = env.reset()
        
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            total_reward += sum(reward)
            steps += 1
            
            if all(done):
                break
        
        # Just verify episode completes without errors
        self.assertGreater(steps, 0)
        
        env.close()


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    print("Running HRL V2 Unit Tests...")
    print("=" * 60)
    run_tests()
    print("=" * 60)
    print("All tests completed!")
