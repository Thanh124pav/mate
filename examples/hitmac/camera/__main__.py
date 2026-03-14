#!/usr/bin/env python3

r"""Demo/eval script for HiTMAC-QPLEX camera agents.

Theo paper HiTMAC (NeurIPS 2020, Sec 3.3 "Training Strategy"):

  Phase 1 — Train QPLEX executors với greedy coordinator:
      python -m examples.hitmac.camera.train

  Phase 2 — Train MAPPO coordinator với scripted executor:
      python -m examples.hrl.mappo.camera.train

  Eval sau Phase 1 (executor only, greedy coordinator):
      python -m examples.hitmac.camera \
          --checkpoint-path examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint

  Eval sau cả hai phases (executor + MAPPO coordinator):
      python -m examples.hitmac.camera \
          --checkpoint-path examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint \
          --coordinator-checkpoint-path examples/hrl/mappo/camera/ray_results/HRL-MAPPO/latest-checkpoint

  Dùng mate.evaluate:
      python -m mate.evaluate --episodes 1 --render-communication \
          --camera-agent examples.hitmac.camera:HiTMACQPLEXCameraAgent \
          --camera-kwargs '{
              "checkpoint_path": "examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint",
              "coordinator_checkpoint_path": "examples/hrl/mappo/camera/ray_results/HRL-MAPPO/latest-checkpoint"
          }'
"""

import argparse
import os
import sys

import mate
from examples.hitmac.camera.agent import HiTMACQPLEXCameraAgent
from examples.hitmac.camera.train import experiment


CHECKPOINT_PATH = os.path.join(experiment.checkpoint_dir, 'latest-checkpoint')

MAX_EPISODE_STEPS = 4000


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--checkpoint-path',
        '--checkpoint',
        '--ckpt',
        type=str,
        metavar='PATH',
        default=CHECKPOINT_PATH,
        help='path to the QPLEX executor checkpoint (Phase 1 output)',
    )
    parser.add_argument(
        '--coordinator-checkpoint-path',
        '--coordinator-checkpoint',
        '--coord-ckpt',
        type=str,
        metavar='PATH',
        default=None,
        help='path to the MAPPO coordinator checkpoint (Phase 2 output); '
             'if omitted, uses greedy fallback',
    )
    parser.add_argument(
        '--max-episode-steps',
        type=int,
        metavar='STEP',
        default=MAX_EPISODE_STEPS,
        help='maximum episode steps (default: %(default)d)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        metavar='SEED',
        default=0,
        help='global random seed (default: %(default)d)',
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(
            f'Executor checkpoint ("{args.checkpoint_path}") does not exist.\n'
            f'Run the following to train first:\n'
            f'  python -m examples.hitmac.camera.train',
            file=sys.stderr,
        )
        sys.exit(1)

    # Make agents ##############################################################
    camera_agent = HiTMACQPLEXCameraAgent(
        checkpoint_path=args.checkpoint_path,
        coordinator_checkpoint_path=args.coordinator_checkpoint_path,
        seed=args.seed,
    )
    target_agent = mate.GreedyTargetAgent(seed=args.seed)

    # Make the environment #####################################################
    env_config = camera_agent.config.get('env_config', {})
    enhanced_observation_team = str(env_config.get('enhanced_observation', None)).lower()

    base_env = mate.make(
        env_config.get('env_id', 'MultiAgentTracking-v0'),
        config=env_config.get('config'),
        **env_config.get('config_overrides', {}),
    )
    base_env = mate.RenderCommunication(base_env)
    if enhanced_observation_team != 'none':
        base_env = mate.EnhancedObservation(base_env, team=enhanced_observation_team)
    env = mate.MultiCamera(base_env, target_agent=target_agent)
    print(env)

    coord_label = 'MAPPO' if args.coordinator_checkpoint_path else 'Greedy'
    print(f'Coordinator: {coord_label}')

    # Rollout ##################################################################
    camera_agents = camera_agent.spawn(env.num_cameras)

    camera_joint_observation = env.reset()
    env.render()

    mate.group_reset(camera_agents, camera_joint_observation)
    camera_infos = None

    for _ in range(args.max_episode_steps):
        camera_joint_action = mate.group_step(
            env, camera_agents, camera_joint_observation, camera_infos
        )

        results = env.step(camera_joint_action)
        camera_joint_observation, camera_team_reward, done, camera_infos = results

        env.render()
        if done:
            break


if __name__ == '__main__':
    main()
