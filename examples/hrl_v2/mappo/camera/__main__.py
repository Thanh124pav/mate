#!/usr/bin/env python3

r"""
HRL V2 MAPPO Camera Agent - Low-level Control Learning.

Example usage:

.. code:: bash

    # Train
    python3 -m examples.hrl_v2.mappo.camera.train
    
    # Evaluate
    python3 -m examples.hrl_v2.mappo.camera \
        --checkpoint-path examples/hrl_v2/mappo/camera/ray_results/HRLv2-MAPPO-LowLevel/latest-checkpoint

Architecture:
- High-level: Greedy target assignment (hard-coded)
- Low-level: MAPPO learned camera control
"""

import argparse
import os

import mate
from examples.hrl_v2.mappo.camera.agent import HRLv2MAPPOCameraAgent
from examples.hrl_v2.mappo.camera.train import experiment


CHECKPOINT_PATH = os.path.join(experiment.checkpoint_dir, 'latest-checkpoint')
MAX_EPISODE_STEPS = 4000


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--checkpoint-path', '--checkpoint', '--ckpt',
        type=str, metavar='PATH', default=CHECKPOINT_PATH,
    )
    parser.add_argument('--max-episode-steps', type=int, metavar='STEP', default=MAX_EPISODE_STEPS)
    parser.add_argument('--seed', type=int, metavar='SEED', default=0)
    parser.add_argument('--episodes', type=int, metavar='EPISODES', default=1)
    parser.add_argument('--config', type=str, metavar='CONFIG', default='MATE-4v5-0.yaml')
    parser.add_argument('--render', action='store_true')
    
    args = parser.parse_args()
    
    # Create environment
    env = mate.make(
        'MultiAgentTracking-v0',
        config=args.config,
        max_episode_steps=args.max_episode_steps,
    )
    
    # Create agents
    camera_agent = HRLv2MAPPOCameraAgent(
        num_cameras=env.num_cameras,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
    )
    target_agent = mate.GreedyTargetAgent(seed=args.seed)
    
    print(f"Evaluating HRL V2 MAPPO agent...")
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
            camera_action = camera_agent.act(camera_joint_observation)
            target_action = target_agent.act(target_joint_observation)
            
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
