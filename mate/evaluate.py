#!/usr/bin/env python3

"""Evaluation script for the Multi-Agent Tracking Environment."""

import argparse
import importlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Union

import gym
import numpy as np
import tqdm
from gym.utils import colorize
from pkg_resources import parse_version

import mate


@dataclass
class Column:  # pylint: disable=missing-class-docstring,missing-function-docstring
    name: str
    width: int
    fmt: Callable[[Union[int, float]], str] = '{}'.format
    color: str = 'white'
    bold: bool = False
    highlight: bool = False
    justification: Callable[..., str] = str.rjust

    @property
    def formatter(self):
        return colorize(' {} ', color=self.color, bold=self.bold, highlight=self.highlight)

    def title(self, width=None):
        if width is None:
            width = self.width
        return self.formatter.format(self.justification(self.name, width))

    def separator(self, width=None):
        if width is None:
            width = self.width
        return self.formatter.format(self.justification(':', width, '-'))

    def format(self, value, width=None):
        if width is None:
            width = self.width
        return self.formatter.format(self.fmt(value).rjust(width))


COLUMNS = [
    Column(name='Step', fmt='{:d}'.format,
           width=6, color='red'),
    Column(name='Cargo', fmt='{:d}'.format,
           width=5, color='green'),
    Column(name='Reward', fmt='{:+.2f}'.format,
           width=8, color='yellow'),
    Column(name='Target Episode Reward', fmt='{:+.2f}'.format,
           width=21, color='blue', bold=True),
    Column(name='Step / Cargo', fmt='{:.1f}'.format,
           width=12, color='magenta'),
    Column(name='Mean Transport Rate', fmt=lambda x: f'{100.0 * x:.3f}%',
           width=19, color='cyan', bold=True),
    Column(name='Mean Coverage Rate', fmt=lambda x: f'{100.0 * x:.3f}%',
           width=18, color='red', bold=True),
    Column(name='Normalized Target Episode Reward', fmt='{:+.5f}'.format,
           width=32, color='green', bold=True),
    Column(name='FPS', fmt='{:.1f}'.format,
           width=5, color='yellow'),
]  # fmt: skip
COLUMNS = OrderedDict([(column.name, column) for column in COLUMNS])


def load_entry(entry_point):
    """Load a module attribute from given entry point."""

    mod_name, attr_name = entry_point.split(':')
    mod = importlib.import_module(mod_name)
    entry = getattr(mod, attr_name)
    return entry


class FocusBeliefRenderer:
    """Render FOCUS target occupancy beliefs as transient environment overlays."""

    COLORS = (
        (0.08, 0.34, 0.92),
        (0.88, 0.20, 0.16),
        (0.00, 0.55, 0.34),
        (0.86, 0.49, 0.02),
        (0.48, 0.22, 0.78),
        (0.00, 0.57, 0.65),
        (0.86, 0.14, 0.48),
        (0.43, 0.43, 0.10),
    )

    def __init__(self, env, sigma_scale=2.0, qmc_topk=24):
        self.env = env
        self.sigma_scale = float(sigma_scale)
        self.qmc_topk = int(qmc_topk)
        self.policy = None
        self.occupancy_model = None
        self.warned_no_model = False
        self.warned_error = False
        self.disabled = False

    def register(self):
        self.env.unwrapped.add_render_callback('focus_belief', self.callback)

    def _camera_agents(self):
        for name in ('opponent_agents_ordered', 'opponent_agents'):
            agents = getattr(self.env, name, None)
            if agents is not None:
                yield from agents

        agent = getattr(self.env, 'opponent_agent', None)
        if agent is not None:
            yield agent

    def _resolve_model(self):
        if self.occupancy_model is not None:
            return True

        for agent in self._camera_agents():
            policy = getattr(agent, 'policy', None)
            occupancy_model = getattr(policy, 'occupancy_model', None)
            if occupancy_model is not None:
                self.policy = policy
                self.occupancy_model = occupancy_model
                self.occupancy_model.eval()
                return True

        if not self.warned_no_model:
            gym.logger.warn(
                'Option --render-focus-belief was set, but no camera policy with '
                '`occupancy_model` was found. Rendering the environment without FOCUS belief.'
            )
            self.warned_no_model = True
        return False

    def _focus_config(self):
        config = getattr(self.policy, 'config', {}) or {}
        return config.get('focus', {}) or {}

    def _state_tensor(self, unwrapped):
        # FOCUS belief models are trained against the centralized normalized
        # environment state used by RLlibMultiAgentCentralizedTraining.
        state = unwrapped.state()
        state = mate.normalize_observation(state, unwrapped.state_space)
        state_dim = getattr(self.occupancy_model, 'state_dim', state.size)
        if int(state_dim) != int(state.size):
            raise ValueError(
                f'FOCUS belief model expects state_dim={state_dim}, '
                f'but environment state has size {state.size}.'
            )

        device = getattr(self.policy, 'device', None)
        if device is None:
            try:
                device = next(self.occupancy_model.parameters()).device
            except StopIteration:
                device = next(self.occupancy_model.buffers()).device

        # pylint: disable-next=import-outside-toplevel
        import torch

        return torch.as_tensor(
            state.reshape(1, 1, -1), dtype=torch.float, device=device
        )

    def _horizon_weights(self, horizon, dtype=float):
        discount = float(self._focus_config().get('horizon_discount', 0.9))
        weights = np.asarray([discount ** h for h in range(horizon)], dtype=dtype)
        return weights / max(float(weights.sum()), 1e-12)

    @staticmethod
    def _ellipse(center, radius, res=48):
        theta = np.linspace(0.0, 2.0 * np.pi, res, endpoint=False)
        return np.column_stack(
            [
                center[0] + radius[0] * np.cos(theta),
                center[1] + radius[1] * np.sin(theta),
            ]
        )

    def _gaussian_geoms(self, state_tensor, rendering):
        # pylint: disable-next=import-outside-toplevel
        import torch

        with torch.no_grad():
            mean, std = self.occupancy_model(state_tensor)

        mean = mean[0, 0].detach().cpu().numpy()
        std = std[0, 0].detach().cpu().numpy()
        horizon = mean.shape[0]
        weights = self._horizon_weights(horizon)
        geoms = []

        for h in range(horizon):
            horizon_alpha = max(0.025, 0.20 * float(weights[h]))
            for target, (center, sigma) in enumerate(zip(mean[h], std[h])):
                color = self.COLORS[target % len(self.COLORS)]
                ellipse = rendering.make_polygon(
                    self._ellipse(center, self.sigma_scale * sigma), filled=True
                )
                ellipse.set_color(*color, horizon_alpha)
                geoms.append(ellipse)

                marker = rendering.make_circle(radius=10.0, res=16, filled=True)
                marker.add_attr(rendering.Transform(translation=center))
                marker.set_color(*color, min(0.35, 0.10 + 0.50 * float(weights[h])))
                geoms.append(marker)

        return geoms

    def _qmc_probs(self, state_tensor):
        # pylint: disable-next=import-outside-toplevel
        import torch

        model = self.occupancy_model
        with torch.no_grad():
            if hasattr(model, '_params'):
                mix_logits, mean, std = model._params(state_tensor)  # pylint: disable=protected-access
                qmc = model.qmc_points.to(device=state_tensor.device, dtype=state_tensor.dtype)
                eps = float(self._focus_config().get('eps', 1e-8))
                log_mix = torch.log_softmax(mix_logits, dim=-1)
                diff = qmc.view(1, 1, 1, 1, 1, -1, 2) - mean.unsqueeze(-2)
                z = diff / (std.unsqueeze(-2) + eps)
                log_comp = (
                    -0.5 * (z ** 2).sum(dim=-1)
                    - torch.log(std.unsqueeze(-2) + eps).sum(dim=-1)
                    - np.log(2.0 * np.pi)
                )
                logits = torch.logsumexp(log_mix.unsqueeze(-1) + log_comp, dim=-2)
                probs = torch.softmax(logits, dim=-1)
            else:
                logits = model(state_tensor)
                probs = torch.softmax(logits, dim=-1)

        qmc_points = model.qmc_points.detach().cpu().numpy()
        probs = probs[0, 0].detach().cpu().numpy()
        horizon = probs.shape[0]
        weights = self._horizon_weights(horizon, dtype=probs.dtype)
        return qmc_points, np.einsum('h,hjm->jm', weights, probs)

    def _qmc_geoms(self, state_tensor, rendering):
        qmc_points, target_probs = self._qmc_probs(state_tensor)
        geoms = []

        for target, probs in enumerate(target_probs):
            color = self.COLORS[target % len(self.COLORS)]
            if probs.size == 0:
                continue

            topk = min(self.qmc_topk, probs.size)
            top_indices = np.argpartition(probs, -topk)[-topk:]
            peak = max(float(probs[top_indices].max()), 1e-12)

            for index in top_indices:
                strength = float(probs[index]) / peak
                radius = 14.0 + 30.0 * strength
                point = rendering.make_circle(radius=radius, res=18, filled=True)
                point.add_attr(rendering.Transform(translation=qmc_points[index]))
                point.set_color(*color, 0.06 + 0.30 * strength)
                geoms.append(point)

        return geoms

    def callback(self, unwrapped, mode):  # pylint: disable=unused-argument
        if self.disabled or not self._resolve_model():
            return

        try:
            # pylint: disable-next=import-outside-toplevel
            import mate.assets.pygletrendering as rendering

            state_tensor = self._state_tensor(unwrapped)
            if hasattr(self.occupancy_model, 'qmc_points'):
                geoms = self._qmc_geoms(state_tensor, rendering)
            else:
                geoms = self._gaussian_geoms(state_tensor, rendering)
        except Exception as exc:  # pragma: no cover - defensive render hook
            if not self.warned_error:
                gym.logger.warn('Disabling FOCUS belief rendering after error: %s', exc)
                self.warned_error = True
            self.disabled = True
            return

        unwrapped.viewer.onetime_geoms[:0] = geoms


def evaluate(
    env, target_agents, render=False, video_path=None
):  # pylint: disable=missing-function-docstring,too-many-locals,too-many-branches,too-many-statements
    status = {}
    if render and video_path is not None:
        # pylint: disable-next=import-outside-toplevel
        from gym.wrappers.monitoring.video_recorder import VideoRecorder

        video_path = os.path.realpath(video_path)
        print(f'Rollout video will be saved to "{video_path}".')
        print()
        recorder = VideoRecorder(env, path=video_path)
        recorder.__del__ = lambda r: r.close()
    else:
        recorder = None

    target_joint_observation = env.reset()
    mate.group_reset(target_agents, target_joint_observation)
    target_infos = None

    if render:
        if recorder is not None:
            recorder.capture_frame()
        else:
            env.render()
        time.sleep(1.0)

    headers = False
    num_cargoes = 0
    target_team_episode_reward = 0.0
    time_start = time.perf_counter()
    coverage_rates = []
    while env.episode_step < env.max_episode_steps:
        target_joint_action = mate.group_step(
            env, target_agents, target_joint_observation, target_infos
        )

        target_joint_observation, target_team_reward, done, target_infos = env.step(
            target_joint_action
        )
        coverage_rates.append(env.coverage_rate)

        num_cargoes = env.num_delivered_cargoes
        target_team_episode_reward += target_team_reward

        values = [
            env.episode_step,
            num_cargoes,
            target_team_reward,
            target_team_episode_reward,
            env.episode_step / num_cargoes if num_cargoes > 0 else np.nan,
            env.mean_transport_rate,
            np.mean(coverage_rates),
            target_team_episode_reward / env.max_target_team_episode_reward,
            env.episode_step / (time.perf_counter() - time_start),
        ]

        if num_cargoes > 0 or done:
            status = dict(zip(COLUMNS, values))

        if render:
            if not headers:
                print('|'.join(['', *map(Column.title, COLUMNS.values()), '']))
                print('|'.join(['', *map(Column.separator, COLUMNS.values()), '']))
                headers = True
            print('|'.join(['', *map(Column.format, COLUMNS.values(), values), '']))

        if render:
            if recorder is not None:
                recorder.capture_frame()
            else:
                env.render()

        if done:
            break

    if render:
        if recorder is not None:
            recorder.close()
        time.sleep(1.0)
        print()

    return status


def parse_arguments():  # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser(
        prog='python -m mate.evaluate',
        description='Evaluation script for the Multi-Agent Tracking Environment.',
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        '--help',
        '-h',
        action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.',
    )

    environment_parser = parser.add_argument_group('environment')
    environment_parser.add_argument(
        '--config',
        '--cfg',
        type=str,
        metavar='PATH',
        default=None,
        help='Path to a JSON/YAML configuration file of MultiAgentTracking.',
    )
    environment_parser.add_argument(
        '--enhanced-observation',
        type=str,
        metavar='TEAM',
        default='none',
        const='both',
        nargs='?',
        choices=['both', 'camera', 'target', 'none'],
        help=(
            "Enhance the agent's observation in the given team.\n"
            'If the argument is omitted, set for both teams.'
        ),
    )
    environment_parser.add_argument(
        '--shared-field-of-view',
        type=str,
        metavar='TEAM',
        default='none',
        const='both',
        nargs='?',
        choices=['both', 'camera', 'target', 'none'],
        help=(
            'Share the field of view among agents in the given team.\n'
            'If the argument is omitted, set for both teams.'
        ),
    )
    environment_parser.add_argument(
        '--no-communication',
        type=str,
        metavar='TEAM',
        default='none',
        const='both',
        nargs='?',
        choices=['both', 'camera', 'target', 'none'],
        help=(
            'Disable all communications for the given team.\n'
            'If the argument is omitted, set for both teams.'
        ),
    )
    environment_parser.add_argument(
        '--seed',
        type=int,
        metavar='SEED',
        default=0,
        help='Random seed for RNGs, overwrites agent arguments. (default: %(default)d)',
    )
    environment_parser.add_argument(
        '--episodes',
        type=int,
        metavar='EPISODE',
        default=20,
        help='Number of episodes to evaluate. (default: %(default)d)',
    )

    agent_parser = parser.add_argument_group('agent')
    agent_parser.add_argument(
        '--camera-agent',
        type=load_entry,
        metavar='ENTRY',
        default='mate:GreedyCameraAgent',
        help='Entry point of camera agent class.\n(default: %(default)s)',
    )
    agent_parser.add_argument(
        '--target-agent',
        type=load_entry,
        metavar='ENTRY',
        default='mate:GreedyTargetAgent',
        help='Entry point of target agent class.\n(default: %(default)s)',
    )
    agent_parser.add_argument(
        '--camera-kwargs',
        type=json.loads,
        metavar='STRING',
        default='{}',
        help=(
            'Keyword arguments of camera agents in JSON string.\n'
            "(example: '{\"discrete_levels\": 5}', default: '{}')"
        ),
    )
    agent_parser.add_argument(
        '--target-kwargs',
        type=json.loads,
        metavar='STRING',
        default='{}',
        help=(
            'Keyword arguments of target agents in JSON string.\n'
            "(example: '{\"discrete_levels\": 5}', default: '{}')"
        ),
    )
    agent_parser.add_argument(
        '--camera-discrete-levels',
        type=int,
        metavar='LEVEL',
        default=None,
        help=(
            'Levels of discrete action space for camera agents,\n'
            'continuous action space will be used if not present.'
        ),
    )
    agent_parser.add_argument(
        '--target-discrete-levels',
        type=int,
        metavar='LEVEL',
        default=None,
        help=(
            'Levels of discrete action space for camera agents,\n'
            'continuous action space will be used if not present.'
        ),
    )

    rendering_parser = parser.add_argument_group('rendering')
    rendering_parser.add_argument(
        '--no-render',
        action='store_true',
        help=(
            'Do not render the environment.\n'
            'Suppress options `--render-communication` and `--save-video`.'
        ),
    )
    rendering_parser.add_argument(
        '--render-communication',
        type=int,
        metavar='DURATION',
        default=None,
        const=20,
        nargs='?',
        help=(
            'Draw arrows for communication edges in the rendering results.\n'
            '(default duration: %(const)d)'
        ),
    )
    rendering_parser.add_argument(
        '--render-focus-belief',
        action='store_true',
        help=(
            'Overlay FOCUS/FOCUS2 future target occupancy beliefs in rendering results.\n'
            'Requires a camera agent policy with `occupancy_model`.'
        ),
    )
    rendering_parser.add_argument(
        '--save-video',
        type=str,
        metavar='PATH',
        nargs='?',
        default=argparse.SUPPRESS,
        help='Save the render video (default: "video.mp4")',
    )

    args = parser.parse_args()

    assert issubclass(args.camera_agent, mate.CameraAgentBase), (
        f'You should provide a subclass of `mate.CameraAgentBase`. '
        f'Got camera_agent = {args.camera_agent}.'
    )
    assert issubclass(args.target_agent, mate.TargetAgentBase), (
        f'You should provide a subclass of `mate.TargetAgentBase`. '
        f'Got target_agent = {args.target_agent}.'
    )
    assert (
        args.episodes > 0
    ), f'The argument `episodes` should be a positive number. Got episodes = {args.episodes}.'

    if not hasattr(args, 'save_video'):
        args.save_video = None
    elif args.save_video is None:
        args.save_video = 'video.mp4'
    if args.no_render:
        args.save_video = None
    if args.save_video is not None and parse_version(gym.__version__) < parse_version('0.18.3'):
        gym.logger.warn(
            'Video recording requires gym 0.18.3 or higher (current version: %s).', gym.__version__
        )

    if args.no_render:
        args.render_communication = False
        args.render_focus_belief = False

    args.camera_kwargs = OrderedDict(sorted(dict(args.camera_kwargs, seed=args.seed).items()))
    args.target_kwargs = OrderedDict(sorted(dict(args.target_kwargs, seed=args.seed).items()))
    args.camera_kwargs.move_to_end('seed')
    args.target_kwargs.move_to_end('seed')
    camera_kwargs_joined = ', '.join(f'{k}={v!r}' for k, v in args.camera_kwargs.items())
    target_kwargs_joined = ', '.join(f'{k}={v!r}' for k, v in args.target_kwargs.items())
    args.camera_name = '{cls.__module__}.{cls.__name__}({kwargs})'.format(
        cls=args.camera_agent, kwargs=camera_kwargs_joined
    )
    args.target_name = '{cls.__module__}.{cls.__name__}({kwargs})'.format(
        cls=args.target_agent, kwargs=target_kwargs_joined
    )

    return args


def main():  # pylint: disable=missing-function-docstring,too-many-branches,too-many-statements
    args = parse_arguments()

    mate.seed_everything(args.seed)

    camera_agent = args.camera_agent(**args.camera_kwargs)
    target_agent = args.target_agent(**args.target_kwargs)

    wrappers = []
    if args.enhanced_observation != 'none':
        wrappers.append(mate.WrapperSpec(mate.EnhancedObservation, team=args.enhanced_observation))
    if args.shared_field_of_view != 'none':
        wrappers.append(mate.WrapperSpec(mate.SharedFieldOfView, team=args.shared_field_of_view))
    if args.no_communication != 'none':
        wrappers.append(mate.WrapperSpec(mate.NoCommunication, team=args.no_communication))
    if args.render_communication is not None and args.render_communication:
        wrappers.append(
            mate.WrapperSpec(mate.RenderCommunication, duration=args.render_communication)
        )
    if args.camera_discrete_levels is not None:
        wrappers.append(mate.WrapperSpec(mate.DiscreteCamera, levels=args.camera_discrete_levels))
    if args.target_discrete_levels is not None:
        wrappers.append(mate.WrapperSpec(mate.DiscreteTarget, levels=args.target_discrete_levels))
    wrappers.append(mate.WrapperSpec(mate.MultiTarget, camera_agent=camera_agent))

    env = mate.make('MultiAgentTracking-v0', config=args.config, wrappers=wrappers)
    env.seed(args.seed)

    print(f'Environment:  {env}')
    print(f'Camera agent: {args.camera_name}')
    print(f'Target agent: {args.target_name}')

    target_agents = target_agent.spawn(env.num_targets)
    if args.render_focus_belief:
        FocusBeliefRenderer(env).register()

    keys = [
        'Step / Cargo',
        'Target Episode Reward',
        'Mean Transport Rate',
        'Mean Coverage Rate',
        'Normalized Target Episode Reward',
    ]
    statuses = OrderedDict([(key, []) for key in keys])
    initial = 0
    postfix = None

    if not args.no_render:
        print()
        try:
            status = evaluate(env, target_agents, render=True, video_path=args.save_video)
        except KeyboardInterrupt:
            pass
        else:
            for key in keys:
                statuses[key].append(status[key])
            initial = 1
            postfix = OrderedDict([
                ('MeanCoverageRate', f'{100.0 * np.mean(statuses["Mean Coverage Rate"]):.1f}%'),
                ('MeanTransportRate', f'{100.0 * np.mean(statuses["Mean Transport Rate"]):.1f}%'),
                ('NormalizedTargetEpisodeReward', f'{np.mean(statuses["Normalized Target Episode Reward"]):+.5f}'),
                ('FPS', status['FPS'])
            ])  # fmt: skip
        finally:
            if env.viewer is not None:
                env.viewer.close()
                env.viewer = None

    try:
        with tqdm.trange(
            initial,
            args.episodes,
            desc='Evaluating',
            unit='episode',
            total=args.episodes,
            initial=initial,
            postfix=postfix,
        ) as pbar:
            for _ in pbar:
                status = evaluate(env, target_agents, render=False)
                for key in keys:
                    statuses[key].append(status[key])
                pbar.set_postfix(OrderedDict([
                    ('MeanCoverageRate', f'{100.0 * np.mean(statuses["Mean Coverage Rate"]):.1f}%'),
                    ('MeanTransportRate', f'{100.0 * np.mean(statuses["Mean Transport Rate"]):.1f}%'),
                    ('NormalizedTargetEpisodeReward', f'{np.mean(statuses["Normalized Target Episode Reward"]):+.5f}'),
                    ('FPS', status['FPS'])
                ]))  # fmt: skip
    except KeyboardInterrupt:
        pass

    if len(statuses[keys[-1]]) > 0:
        # pylint: disable=consider-using-f-string
        print('| {:>32} | {:>12} |'.format('Metric', 'Mean'))
        print('| {:->32} | {:->12} |'.format(':', ':'))
        for key, values in statuses.items():
            print(
                '|{}|{}|'.format(
                    COLUMNS[key].title(width=32), COLUMNS[key].format(np.mean(values), width=12)
                )
            )
        # pylint: disable-enable=consider-using-f-string


if __name__ == '__main__':
    main()
