# Soft Actor-Critic (SAC) from Scratch

**Authors:** Diavolitsis Panagiotis Eleftherios 

## Project Description

This repository contains a PyTorch implementation of the Soft Actor-Critic (SAC) algorithm, developed from scratch and applied to continuous-control robotics tasks using the PyBullet environments. The implementation covers core SAC components:

- **Actor-Critic architecture** with stochastic policy and twin Q-networks
- **Maximum Entropy RL** via entropy regularization (fixed reward scale)
- **Target networks** for stabilized Q-value updates
- **Replay buffer** for off-policy learning

## Repository Structure

```
├── buffer.py          # ReplayBuffer implementation
├── networks.py        # Actor, Critic, and Value network definitions
├── sac_torch.py       # SAC Agent: training routines and learn() method
├── utils.py           # Utility functions (plotting learning curves)
├── main_sac.py        # Training & evaluation script (includes random baseline)
└── plots/             # Output directory for saved training curves
```

## Requirements

- Python 3.7+
- PyTorch
- Gym
- pybullet_envs
- NumPy
- Matplotlib

Install dependencies via pip:

```bash
pip install torch gym pybullet pybullet_envs numpy matplotlib
```

## Usage

1. **Configure training** in `main_sac.py`:
   - `train_sac`: set `True` to train SAC or `False` to skip training
   - `n_games`: number of episodes to train/evaluate
   - `env_id`: choose a Bullet environment (e.g., `HopperBulletEnv-v0`)

2. **Run training & evaluation**:

   ```bash
   python main_sac.py
   ```

   - SAC agent training curve and random baseline curve will be saved in `plots/`.

3. **Inspect results**:
   - Learning curves show running average of scores (100-episode window).

## Hyperparameters

- Learning rates: `alpha=3e-4`, `beta=3e-4`
- Discount factor: `γ=0.99`
- Target update rate: `τ=0.005`
- Replay buffer size: `1e6`
- Batch size: `256`
- Reward scaling (entropy weight): `2.0`

Adjust these in `sac_torch.py` (Agent constructor) or `main_sac.py` as needed.


