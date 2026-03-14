#!/usr/bin/env python3

r"""Demo/eval script for HiTMAC Phase 2B — Role-Based Coordinator.

Train Phase 1 (QPLEX executors):
    python -m examples.hitmac.camera.train --env MATE-4v8-9.yaml

Train Phase 2B (role coordinator):
    python -m examples.hitmac.role.camera.train \
        --executor-checkpoint examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint

Eval:
    python -m examples.hitmac.role.camera \
        --checkpoint-path examples/hitmac/role/camera/ray_results/HiTMAC-Role/latest-checkpoint \
        --executor-checkpoint-path examples/hitmac/camera/ray_results/HiTMAC-QPLEX/latest-checkpoint
"""

import argparse
import os
import sys

import mate
from examples.hitmac.role.camera.agent import HiTMACRoleCameraAgent
from examples.hitmac.role.camera.train import experiment


CHECKPOINT_PATH = os.path.join(experiment.checkpoint_dir, 'latest-checkpoint')
MAX_EPISODE_STEPS = 4000


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument(
        '--checkpoint-path', '--checkpoint', '--ckpt',
        type=str, metavar='PATH', default=CHECKPOINT_PATH,
        help='path to role coordinator checkpoint (Phase 2B output)',
    )
    parser.add_argument(
        '--executor-checkpoint-path', '--executor-checkpoint', '--exec-ckpt',
        type=str, metavar='PATH', default=None,
        help='path to QPLEX executor checkpoint (Phase 1 output)',
    )
    parser.add_argument(
        '--coordinator-idx', type=int, default=0, metavar='IDX',
        help='index of the coordinator camera (default: %(default)d)',
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
            f'Role coordinator checkpoint ("{args.checkpoint_path}") does not exist.\n'
            f'Run the following to train first:\n'
            f'  python -m examples.hitmac.role.camera.train '
            f'--executor-checkpoint <Phase1_checkpoint>',
            file=sys.stderr,
        )
        sys.exit(1)

    # Make coordinator agent
    camera_agent = HiTMACRoleCameraAgent(
        checkpoint_path=args.checkpoint_path,
        executor_checkpoint_path=args.executor_checkpoint_path,
        coordinator_idx=args.coordinator_idx,
        seed=args.seed,
    )
    target_agent = mate.GreedyTargetAgent(seed=args.seed)

    # Make environment (without HiTMACRoleWrapper — agents handle roles internally)
    env_config = camera_agent.config.get('env_config', {})
    base_env = mate.make(
        env_config.get('env_id', 'MultiAgentTracking-v0'),
        config=env_config.get('config'),
        **env_config.get('config_overrides', {}),
    )
    base_env = mate.RenderCommunication(base_env)
    # NOTE: EnhancedObservation not applied here — coordinator_agent already trained
    # with enhanced obs baked into the policy's obs space expectations.
    # For bare env evaluation, coordinator sees its own local obs (no enhanced).
    # For proper enhanced obs eval, would need to wrap base_env with EnhancedObservation.
    enhanced = str(env_config.get('enhanced_observation', 'camera')).lower()
    if enhanced != 'none':
        base_env = mate.EnhancedObservation(base_env, team=enhanced)
    env = mate.MultiCamera(base_env, target_agent=target_agent)
    print(env)
    print(f'Coordinator camera: {args.coordinator_idx}')

    # Spawn agents (coordinator + executors via spawn())
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
