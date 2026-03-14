cp -r ray/rllib/agents/qplex_v2/ ~/miniconda3/envs/mate/lib/python3.9/site-packages/ray/rllib/agents

python -m examples.hrl.qplex_v2.camera.train --seed 1 2 3 --project mate-4v5-0