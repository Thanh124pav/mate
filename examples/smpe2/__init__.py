"""SMPE2: State Modelling and adversarial Exploration for cooperative MARL (ICML 2025).

Builds on PPO/MAPPO with:
  - Variational belief inference (encode local obs → latent z)
  - AM filters (learn which neighbor features are informative)
  - Adversarial exploration (count-based intrinsic rewards on belief space)

Train:
    python -m examples.smpe2.camera.train
"""
