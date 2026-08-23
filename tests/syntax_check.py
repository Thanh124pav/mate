"""Parse changed Python files without importing their runtime dependencies."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "examples/mappo/models.py",
    "examples/smpe2_focus/models.py",
    "hmvfe_mate_d/belief.py",
    "hmvfe_mate_d/config.py",
    "hmvfe_mate_d/envs.py",
    "hmvfe_mate_d/models.py",
    "hmvfe_mate_d/trainer.py",
    "hmvfe_mate_d/vector_env.py",
    "hmvfe_mate_d/__main__.py",
    "hmvfe_mate_d/ceiling_check.py",
    "ray/rllib/agents/focus_common/belief_state.py",
    "ray/rllib/agents/focus_common/adapters.py",
    "ray/rllib/agents/focus_common/responsibility_engine.py",
    "ray/rllib/agents/qmix_focus/mixers.py",
    "ray/rllib/agents/qmix_focus/qmix_policy.py",
    "ray/rllib/agents/duelmix_focus/mixers.py",
    "ray/rllib/agents/duelmix_focus/duelmix_policy.py",
    "ray/rllib/agents/ppo/ppo_torch_policy.py",
]

for relative in FILES:
    path = ROOT / relative
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"OK {relative}")
