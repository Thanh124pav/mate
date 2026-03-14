#!/usr/bin/env python3

r"""
HRL V2 QPLEX Camera Agent - Low-level Control Learning.

Example usage:

.. code:: bash

    # Train
    python3 -m examples.hrl_v2.qplex.camera.train
    
    # Evaluate
    python3 -m examples.hrl_v2.qplex.camera \
        --checkpoint-path examples/hrl_v2/qplex/camera/ray_results/HRLv2-QPLEX-LowLevel/latest-checkpoint

Architecture:
- High-level: Greedy target assignment (hard-coded)
- Low-level: QPLEX learned camera control
"""

import argparse
import functools
import os
import sys

import mate
from examples.hrl_v2.qplex.camera.agent import HRLv2QPLEXCameraAgent
from examples.hrl_v2.qplex.camera.train import experiment


CHECKPOINT_PATH = os.path.join(experiment.checkpoint_dir, 'latest-checkpoint')
MAX_EPISODE_STEPS = 4000


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--checkpoint-path', '--checkpoint', '--ckpt',
        type=str, metavar='PATH', default=CHECKPOINT_PATH,
        help='path to the checkpoint file',
    )
    parser.add_argument(
        '--max-episode-steps', type=int, metavar='STEP', default=MAX_EPISODE_STEPS,
        help='maximum episode steps (default: %(default)d)',
    )
    parser.add_argument(
        '--seed', type=int, metavar='SEED', default=0,
        help='the global seed (default: %(default)d)',
    )
    parser.add_argument(
        '--episodes', type=int, metavar='EPISODES', default=1,
        help='number of episodes to run (default: %(default)d)',
    )
    parser.add_argument(
        '--config', type=str, metavar='CONFIG', default='MATE-4v5-0.yaml',
        help='environment config file (default: %(default)s)',
    )
    parser.add_argument(
        '--render', action='store_true',
        help='render the environment',
    )
    
    args = parser.parse_args()
    
    # Create environment
    env = mate.make(
        'MultiAgentTracking-v0',
        config=args.config,
        max_episode_steps=args.max_episode_steps,
    )
    
    # Create camera agent
    camera_agent = HRLv2QPLEXCameraAgent(
        num_cameras=env.num_cameras,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
    )
    
    # Create target agent
    target_agent = mate.GreedyTargetAgent(seed=args.seed)
    
    # Evaluate
    print(f"Evaluating HRL V2 QPLEX agent...")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Config: {args.config}")
    print(f"Episodes: {args.episodes}")
    
    total_rewards = []
    
    for episode in range(args.episodes):
        observation = env.reset()
        camera_observation = observation['camera']
        target_observation = observation['target']
        
        camera_joint_observation = mate.MultiCamera.split_camera_joint_observation(
            camera_observation, env.num_cameras
        )
        target_joint_observation = mate.MultiTarget.split_target_joint_observation(
            target_observation, env.num_targets
        )
        
        camera_agent.reset(camera_joint_observation)
        target_agent.reset(target_joint_observation)
        
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done:
            # Get actions
            camera_action = camera_agent.act(camera_joint_observation)
            target_action = target_agent.act(target_joint_observation)
            
            # Step environment
            observation, reward, done, info = env.step({
                'camera': camera_action,
                'target': target_action,
            })
            
            camera_observation = observation['camera']
            target_observation = observation['target']
            
            camera_joint_observation = mate.MultiCamera.split_camera_joint_observation(
                camera_observation, env.num_cameras
            )
            target_joint_observation = mate.MultiTarget.split_target_joint_observation(
                target_observation, env.num_targets
            )
            
            episode_reward += reward
            step += 1
            
            if args.render:
                env.render()
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: reward = {episode_reward:.2f}, steps = {step}")
    
    print(f"\nAverage reward: {sum(total_rewards) / len(total_rewards):.2f}")
    
    env.close()


if __name__ == '__main__':
    main()
