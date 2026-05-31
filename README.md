# Learning Agents Playground

A reproducible reinforcement learning playground for experimenting with classic and deep RL algorithms on Gymnasium environments.

**Author:** Huynh Mai Linh Nguyen

---

## Overview

Learning Agents Playground provides a clean, modular codebase for training and evaluating RL agents:

- **Tabular Q-Learning** on discrete environments (FrozenLake)
- **Deep Q-Network (DQN)** on continuous-state environments (CartPole)
- **Random baseline** for comparison
- **CSV logging, plotting, and a Streamlit dashboard** for visualization

The project emphasizes reproducibility through centralized seeding, YAML configs, and lightweight smoke tests.

---

## Features

| Component | Description |
|-----------|-------------|
| `RandomAgent` | Uniform random baseline |
| `QLearningAgent` | Tabular Q-learning with epsilon-greedy exploration |
| `DQNAgent` | Deep Q-Network with 128-128 MLP, replay buffer, target network |
| `make_env` | Gymnasium factory with seeding and `is_slippery` for FrozenLake |
| Training loops | CSV logs, checkpoints, reward/success/loss plots |
| Evaluation | `evaluate_agent` with success rate and reward summaries |
| Streamlit app | Visualize training runs (no training in the dashboard) |

---

## Project Structure

```
learning-agents-playground/
├── README.md
├── requirements.txt
├── pyproject.toml
├── app.py                          # Streamlit dashboard
├── src/
│   ├── agents/
│   │   ├── random_agent.py
│   │   ├── q_learning.py
│   │   └── dqn.py
│   ├── envs/
│   │   └── make_env.py
│   ├── training/
│   │   ├── train_q_learning.py
│   │   └── train_dqn.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── metrics.py
│   └── utils/
│       ├── seeding.py
│       ├── plotting.py
│       └── replay_buffer.py
├── scripts/
│   ├── train_q_learning.py
│   ├── train_dqn.py
│   ├── evaluate_agent.py
│   └── plot_results.py
├── configs/
│   ├── frozenlake_q_learning.yaml
│   └── cartpole_dqn.yaml
├── results/                        # Training outputs (CSV, plots, checkpoints)
├── assets/
└── tests/
    └── test_smoke.py
```

---

## Installation

```bash
cd learning-agents-playground
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

### Train Q-Learning on FrozenLake

```bash
python scripts/train_q_learning.py --config configs/frozenlake_q_learning.yaml --verbose
```

Outputs are saved to `results/frozenlake_qlearning/`:
- `training_log.csv`
- `rewards.png`, `success_rate.png`
- `agent.pkl`

### Train DQN on CartPole

```bash
python scripts/train_dqn.py --config configs/cartpole_dqn.yaml --verbose
```

Outputs are saved to `results/cartpole_dqn/`:
- `training_log.csv`
- `rewards.png`, `losses.png`
- `agent.pt`

### Evaluate a Trained Agent

```bash
python scripts/evaluate_agent.py \
  --agent-type qlearning \
  --checkpoint results/frozenlake_qlearning/agent.pkl \
  --env-id FrozenLake-v1 \
  --episodes 100

python scripts/evaluate_agent.py \
  --agent-type dqn \
  --checkpoint results/cartpole_dqn/agent.pt \
  --env-id CartPole-v1 \
  --episodes 100
```

### Plot Results from CSV

```bash
python scripts/plot_results.py --csv results/frozenlake_qlearning/training_log.csv
```

### Launch Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard reads from `results/` and displays reward curves, success rates, and saved plots. **Training is not performed in the app.**

---

## Configuration

### FrozenLake Q-Learning (`configs/frozenlake_q_learning.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `env_id` | `FrozenLake-v1` | Gymnasium environment |
| `is_slippery` | `true` | Stochastic vs deterministic transitions |
| `episodes` | `5000` | Training episodes |
| `learning_rate` | `0.1` | Q-learning step size (α) |
| `discount_factor` | `0.99` | Bellman discount (γ) |
| `epsilon` | `1.0` | Initial exploration rate |
| `epsilon_decay` | `0.995` | Per-episode decay |

### CartPole DQN (`configs/cartpole_dqn.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `env_id` | `CartPole-v1` | Gymnasium environment |
| `episodes` | `500` | Training episodes |
| `hidden_dim` | `128` | MLP hidden size (128-128 architecture) |
| `batch_size` | `64` | Replay mini-batch size |
| `buffer_capacity` | `100000` | Replay buffer capacity |
| `target_update_freq` | `10` | Target network sync frequency |
| `warmup_steps` | `1000` | Steps before gradient updates |

All configs support `seed`, `run_name`, and `output_dir` for reproducible experiment tracking.

---

## Algorithms

### Q-Learning

Tabular off-policy TD control. The Q-table is updated with:

```
Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
```

Exploration uses an epsilon-greedy policy that decays over episodes.

### DQN

Function approximation with a two-layer MLP (128 → 128 → |A|). Key stabilizers:

- **Experience replay** — breaks temporal correlation in mini-batches
- **Target network** — slow-moving copy for bootstrap targets
- **Epsilon-greedy exploration** — decays during training

---

## Reproducibility

Use `set_seed()` from `src/utils/seeding.py` to synchronize Python, NumPy, and environment RNGs:

```python
from src.utils.seeding import set_seed
set_seed(42)
```

Pass the same `seed` in YAML configs and CLI `--seed` overrides for consistent runs.

---

## Testing

Lightweight smoke tests verify imports, agent APIs, and short training loops:

```bash
pytest tests/test_smoke.py -v
```

Smoke tests run only a handful of episodes — not full DQN training.

---

## Results Directory

After training, each run creates a subdirectory under `results/`:

```
results/<run_name>/
├── training_log.csv
├── rewards.png
├── success_rate.png   # Q-learning
├── losses.png         # DQN
└── agent.pkl / agent.pt
```

---

## License

MIT License — see project repository for details.

---

## Author

**Huynh Mai Linh Nguyen**
