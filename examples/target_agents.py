import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ['PYTHONPATH'] = (
    str(REPO_ROOT)
    if not os.environ.get('PYTHONPATH')
    else f"{REPO_ROOT}:{os.environ['PYTHONPATH']}"
)

import mate
from examples.utils import EvaluationLoggingCallback, RLlibMultiCallbacks


def _load_evasive_target_agent():
    path = Path(__file__).resolve().parents[1] / 'ray/rllib/agents/qplex_wm2/evasive_target.py'
    spec = importlib.util.spec_from_file_location('_mate_evasive_target', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EvasiveTargetAgent


EvasiveTargetAgent = _load_evasive_target_agent()


def greedy_target_agent_factory():
    return mate.agents.GreedyTargetAgent(seed=0)


def evasive_target_agent_factory():
    return EvasiveTargetAgent(seed=0, avoidance_strength=0.5, avoidance_range=0.5)


TARGET_AGENT_FACTORIES = {
    'greedy': greedy_target_agent_factory,
    'evasive': evasive_target_agent_factory,
}


def target_agent_factory(name):
    if name is None:
        return None
    try:
        return TARGET_AGENT_FACTORIES[name]
    except KeyError as exc:
        choices = ', '.join(sorted(TARGET_AGENT_FACTORIES))
        raise ValueError(f'Unknown target agent {name!r}. Choose one of: {choices}.') from exc


def configure_target_agent(config, name):
    factory = target_agent_factory(name)
    if factory is None:
        return
    config.setdefault('env_config', {})['opponent_agent_factory'] = factory


def _with_evaluation_logging_callback(callbacks):
    if callbacks is None:
        return EvaluationLoggingCallback
    if callbacks is EvaluationLoggingCallback:
        return callbacks
    callback_types = getattr(callbacks, '_callback_class_list', None)
    if callback_types is None:
        callback_types = (callbacks,)
    else:
        callback_types = tuple(callback_types)
    if EvaluationLoggingCallback not in callback_types:
        callback_types = (*callback_types, EvaluationLoggingCallback)
    return RLlibMultiCallbacks(callback_types)


def configure_greedy_evaluation(config, interval):
    config['callbacks'] = _with_evaluation_logging_callback(config.get('callbacks'))
    if interval is None or interval <= 0:
        config['evaluation_interval'] = None
        return
    config['evaluation_interval'] = int(interval)
    config.setdefault('evaluation_duration', 5)
    config.setdefault('evaluation_duration_unit', 'episodes')
    config.setdefault('always_attach_evaluation_results', True)
    evaluation_config = config.setdefault('evaluation_config', {})
    evaluation_config['explore'] = False
    evaluation_env_config = evaluation_config.setdefault('env_config', {})
    evaluation_env_config['opponent_agent_factory'] = greedy_target_agent_factory
