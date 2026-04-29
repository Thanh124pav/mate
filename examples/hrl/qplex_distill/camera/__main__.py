#!/usr/bin/env python3
"""
python3 -m examples.hrl.qplex_distill.camera
"""

import argparse
import functools
import os
import sys

import mate
from examples.hrl.qplex_distill.camera.agent import HRLQPLEXDistillCameraAgent
from examples.hrl.qplex_distill.camera.train import experiment
from examples.hrl.wrappers import HierarchicalCamera

CHECKPOINT_PATH = os.path.join(experiment.checkpoint_dir, 'latest-checkpoint')
MAX_EPISODE_STEPS = 4000


def main():
    parser = argparse.ArgumentParser(prog=f'python -m {__package__}')
    parser.add_argument('--checkpoint-path', '--checkpoint', '--ckpt',
                        type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--max-episode-steps', type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        print(
            f'Checkpoint "{args.checkpoint_path}" does not exist. Train first:\n'
            f'  python -m examples.hrl.qplex_distill.camera.train',
            file=sys.stderr,
        )
        sys.exit(1)

    camera_agent = HRLQPLEXDistillCameraAgent(checkpoint_path=args.checkpoint_path)
    target_agent = mate.GreedyTargetAgent()

    env_config = camera_agent.config.get('env_config', {})
    enhanced_observation_team = str(env_config.get('enhanced_observation', None)).lower()

    base_env = mate.make(
        'MultiAgentTracking-v0',
        config=env_config.get('config'),
        **env_config.get('config_overrides', {}),
    )
    base_env = mate.RenderCommunication(base_env)
    if enhanced_observation_team != 'none':
        base_env = mate.EnhancedObservation(base_env, team=enhanced_observation_team)
    env = mate.MultiCamera(base_env, target_agent=target_agent)

    camera_agents = camera_agent.spawn(env.num_cameras)
    camera_joint_observation = env.reset()
    env.render()

    mate.group_reset(camera_agents, camera_joint_observation)
    camera_infos = None

    for i in range(args.max_episode_steps):
        camera_joint_action = mate.group_step(
            env, camera_agents, camera_joint_observation, camera_infos
        )
        selections = [
            (agent.index, agent.last_selection, agent.last_mask) for agent in camera_agents
        ]
        results = env.step(camera_joint_action)
        camera_joint_observation, _, done, camera_infos = results

        render_callback = functools.partial(
            HierarchicalCamera.render_selection_callback, selections=selections
        )
        env.render(onetime_callbacks=[render_callback])
        if done:
            break


if __name__ == '__main__':
    main()
