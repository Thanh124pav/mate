#!/usr/bin/env python3

r"""Demo/eval script for HiTMAC Phase 2A — MAPPO Coordinator + QPLEX Executor.

Train Phase 1 (QPLEX executors):
    python -m examples.hitmac.camera.train --env MATE-4v8-9.yaml

Train Phase 2A (MAPPO coordinator):
    python -m examples.hitmac.coordinator.camera.train \
        --executor-checkpoint examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint

Eval:
    python -m examples.hitmac.coordinator.camera \
        --checkpoint-path examples/hitmac/coordinator/camera/ray_results/HiTMAC-Coordinator/latest-checkpoint \
        --executor-checkpoint-path examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint
"""

import argparse
import os
import sys

import mate
from examples.hitmac.coordinator.mappo.camera.agent import HiTMACCoordinatorCameraAgent
from examples.hitmac.coordinator.mappo.camera.train import experiment


CHECKPOINT_PATH = os.path.join(experiment.checkpoint_dir, 'latest-checkpoint')
MAX_EPISODE_STEPS = 4000


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--checkpoint-path', '--checkpoint', '--ckpt',
        type=str, metavar='PATH', default=CHECKPOINT_PATH,
        help='path to MAPPO coordinator checkpoint (Phase 2A output)',
    )
    parser.add_argument(
        '--executor-checkpoint-path', '--executor-checkpoint', '--exec-ckpt',
        type=str, metavar='PATH', default=None,
        help='path to QPLEX executor checkpoint (Phase 1 output); '
             'if omitted, uses geometric fallback',
    )
    parser.add_argument(
        '--max-episode-steps', type=int, metavar='STEP', default=MAX_EPISODE_STEPS,
        help='maximum episode steps (default: %(default)d)',
    )
    parser.add_argument(
        '--seed', type=int, metavar='SEED', default=0,
        help='global random seed (default: %(default)d)',
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(
            f'Coordinator checkpoint ("{args.checkpoint_path}") does not exist.\n'
            f'Run the following to train first:\n'
            f'  python -m examples.hitmac.coordinator.camera.train '
            f'--executor-checkpoint <Phase1_checkpoint>',
            file=sys.stderr,
        )
        sys.exit(1)

    # Make agent
    camera_agent = HiTMACCoordinatorCameraAgent(
        checkpoint_path=args.checkpoint_path,
        executor_checkpoint_path=args.executor_checkpoint_path,
        seed=args.seed,
    )
    target_agent = mate.GreedyTargetAgent(seed=args.seed)

    # Make environment
    env_config = camera_agent.config.get('env_config', {})
    base_env = mate.make(
        env_config.get('env_id', 'MultiAgentTracking-v0'),
        config=env_config.get('config'),
        **env_config.get('config_overrides', {}),
    )
    base_env = mate.RenderCommunication(base_env)
    if str(env_config.get('enhanced_observation', None)).lower() != 'none':
        base_env = mate.EnhancedObservation(base_env, team=env_config['enhanced_observation'])
    env = mate.MultiCamera(base_env, target_agent=target_agent)
    print(env)
    exec_label = 'QPLEX' if args.executor_checkpoint_path else 'Geometric'
    print(f'Executor: {exec_label}')

    # Rollout
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
