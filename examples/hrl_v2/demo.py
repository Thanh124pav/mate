#!/usr/bin/env python3

"""
Demo script showing how to use HRL V2 wrapper.

This demonstrates:
1. Creating environment with HRL V2 wrapper
2. Using different assignment strategies
3. Running episodes with learned low-level control
"""

import numpy as np
import mate
from examples.hrl_v2.wrappers import HierarchicalCameraV2
from examples.hrl_v2.high_level import (
    GreedyDistanceAssigner,
    GreedyCoverageAssigner,
)


def demo_basic_usage():
    """Basic usage of HRL V2 wrapper."""
    print("=" * 60)
    print("Demo 1: Basic HRL V2 Usage")
    print("=" * 60)
    
    # Create base environment
    env = mate.make('MultiAgentTracking-v0', config='MATE-4v5-0.yaml')
    
    # Add target agent
    target_agent = mate.GreedyTargetAgent(seed=0)
    env = mate.MultiCamera(env, target_agent=target_agent)
    
    # Wrap with HRL V2
    env = HierarchicalCameraV2(
        env,
        assigner_class=GreedyDistanceAssigner,
        assigner_kwargs={'max_assignments_per_camera': 1},
        frame_skip=5,
        include_assignment_in_obs=True,
    )
    
    # Run one episode
    obs = env.reset()
    print(f"\nInitial observation structure:")
    print(f"  Number of cameras: {len(obs)}")
    print(f"  Observation type: {type(obs[0])}")
    if isinstance(obs[0], dict):
        print(f"  Keys: {obs[0].keys()}")
        print(f"  Base obs shape: {obs[0]['obs'].shape}")
        print(f"  Assignment shape: {obs[0]['assignment'].shape}")
        print(f"  Assignment: {obs[0]['assignment']}")
    
    total_reward = 0
    for step in range(100):
        # Random actions for demo
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(actions)
        total_reward += sum(reward)
        
        if done:
            break
    
    print(f"\nEpisode completed:")
    print(f"  Steps: {step + 1}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward: {total_reward / (step + 1):.2f}")
    
    env.close()


def demo_different_assigners():
    """Compare different assignment strategies."""
    print("\n" + "=" * 60)
    print("Demo 2: Comparing Assignment Strategies")
    print("=" * 60)
    
    strategies = [
        ("Greedy Distance", GreedyDistanceAssigner, {}),
        ("Greedy Coverage", GreedyCoverageAssigner, {}),
    ]
    
    results = {}
    
    for name, assigner_class, kwargs in strategies:
        print(f"\nTesting {name}...")
        
        # Create environment
        env = mate.make('MultiAgentTracking-v0', config='MATE-4v5-0.yaml')
        target_agent = mate.GreedyTargetAgent(seed=0)
        env = mate.MultiCamera(env, target_agent=target_agent)
        
        env = HierarchicalCameraV2(
            env,
            assigner_class=assigner_class,
            assigner_kwargs=kwargs,
            frame_skip=5,
        )
        
        # Run episode
        obs = env.reset()
        episode_reward = 0
        coverage_rates = []
        
        for step in range(100):
            actions = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            episode_reward += sum(reward)
            
            # Collect coverage metrics
            if 'coverage_rate' in info[0]:
                coverage_rates.append(info[0]['coverage_rate'])
            
            if done:
                break
        
        avg_coverage = np.mean(coverage_rates) if coverage_rates else 0
        
        results[name] = {
            'reward': episode_reward,
            'steps': step + 1,
            'avg_coverage': avg_coverage,
        }
        
        env.close()
    
    # Print comparison
    print("\n" + "-" * 60)
    print("Results Comparison:")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Reward':>10} {'Steps':>8} {'Coverage':>10}")
    print("-" * 60)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['reward']:>10.2f} {metrics['steps']:>8d} {metrics['avg_coverage']:>10.2%}")
    
    print("-" * 60)


def demo_observation_structure():
    """Detailed look at observation structure."""
    print("\n" + "=" * 60)
    print("Demo 3: Observation Structure Analysis")
    print("=" * 60)
    
    env = mate.make('MultiAgentTracking-v0', config='MATE-4v5-0.yaml')
    target_agent = mate.GreedyTargetAgent(seed=0)
    env = mate.MultiCamera(env, target_agent=target_agent)
    
    env = HierarchicalCameraV2(
        env,
        assigner_class=GreedyDistanceAssigner,
        include_assignment_in_obs=True,
    )
    
    obs = env.reset()
    
    print(f"\nEnvironment Info:")
    print(f"  Cameras: {env.num_cameras}")
    print(f"  Targets: {env.num_targets}")
    print(f"  Action space per camera: {env.action_space.spaces[0]}")
    
    print(f"\nObservation Structure (Camera 0):")
    obs_dict = obs[0]
    print(f"  Base observation shape: {obs_dict['obs'].shape}")
    print(f"  Assignment vector shape: {obs_dict['assignment'].shape}")
    
    print(f"\n  Assignment breakdown:")
    for t, assigned in enumerate(obs_dict['assignment']):
        status = "ASSIGNED" if assigned else "not assigned"
        print(f"    Target {t}: {status}")
    
    # Show how assignment changes
    print(f"\nAssignment over 5 steps:")
    for step in range(5):
        actions = env.action_space.sample()
        obs, _, _, _ = env.step(actions)
        
        assignment = obs[0]['assignment']
        assigned_targets = [i for i, a in enumerate(assignment) if a]
        print(f"  Step {step + 1}: Camera 0 → Targets {assigned_targets}")
    
    env.close()


def demo_frame_skip():
    """Demonstrate frame skip behavior."""
    print("\n" + "=" * 60)
    print("Demo 4: Frame Skip Behavior")
    print("=" * 60)
    
    for frame_skip in [1, 5, 10]:
        print(f"\nFrame skip = {frame_skip}")
        
        env = mate.make('MultiAgentTracking-v0', config='MATE-4v5-0.yaml')
        target_agent = mate.GreedyTargetAgent(seed=0)
        env = mate.MultiCamera(env, target_agent=target_agent)
        
        env = HierarchicalCameraV2(
            env,
            frame_skip=frame_skip,
        )
        
        obs = env.reset()
        
        # Count actual environment steps vs wrapper steps
        wrapper_steps = 0
        env_steps = 0
        
        for _ in range(20):
            actions = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            
            wrapper_steps += 1
            env_steps += frame_skip  # Each wrapper step = frame_skip env steps
            
            if done:
                break
        
        print(f"  Wrapper steps: {wrapper_steps}")
        print(f"  Actual env steps: {env_steps}")
        print(f"  Ratio: {env_steps / wrapper_steps:.1f}x")
        
        env.close()


if __name__ == '__main__':
    demo_basic_usage()
    demo_different_assigners()
    demo_observation_structure()
    demo_frame_skip()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
