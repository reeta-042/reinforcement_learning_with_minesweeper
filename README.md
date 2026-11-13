# Reinforcement Learning with Minesweeper Grid Navigation

An experimental reinforcement learning framework investigating **curriculum transfer learning**, **schema representation**, **catastrophic forgetting**, and **recurrent working memory (DRQN / Meta-RL)** in embodied grid navigation with Minesweeper rules.

---

## Project Purpose

In standard reinforcement learning, agents often struggle with sample efficiency and partial observability (POMDPs). This project investigates:
1. **Curriculum Transfer Learning**: Whether spatial sensory representations learned in simpler navigation tasks accelerate learning in harder, hazard-dense environments with fewer training episodes.
2. **Schema Learning**: How value-based neural networks construct internal topographic cognitive maps ($V(s)$ = max_a Q(s, a)) to navigate safe corridors and avoid hazards.
3. **Catastrophic Forgetting**: Measuring backward retention when an agent adapts to new environment rules.
4. **Recurrent Working Memory (DRQN / Meta-RL)**: How augmenting Q-learning with an internal LSTM belief state enables sequential in-context reasoning, zero-shot inference, and one-shot hazard adaptation with frozen weights.

---

## What It Does & How It Does It

### What It Does
An embodied agent starts on the left side of a grid and must reach a goal column on the far right without detonating hidden mines or falling into black hole traps. The agent uses numerical clues ($0$ to $8$) on revealed safe tiles to infer safe paths under partial visibility (fog of war).

### How It Works
- **State Representation (7-Channel Tensor)**:
  Encodes agent position, revealed numbers, unrevealed fog, flags, safe sidewalks, goal zone, and normalized remaining steps.
- **Feedforward Architecture (DQN)**:
  Uses a 2D Conv2D spatial backbone, adaptive average pooling to canonical $(6, 8)$ coordinates, and an egocentric 5-cell neighborhood gathering slice (`center`, `up`, `down`, `left`, `right`) to achieve **dimension invariance** across different grid sizes ($8 \times 6 \leftrightarrow 12 \times 10$).
- **Recurrent Architecture (DRQN)**:
  Combines the Conv2D spatial backbone with an LSTM recurrent memory core conditioned on the transition context $[s_t, a_{t-1}, r_{t-1}, d_{t-1}]$ and trained via sequential Backpropagation Through Time (BPTT).
- **Curriculum Transfer Protocol**:
  Transfers visual feature weights into successive environments, purges old experience replay buffers to prevent negative transfer, and resets exploration ($\epsilon \to 1.0$) to discover new hazard distributions.

---

## 📓 Notebook Curriculum & Evaluation Metrics

The experimental suite spans five modular Jupyter notebooks:

### [Notebook 1: Environment 1 — Static Mines with Sidewalks](notebooks/01_env1_training_and_evaluation.ipynb)
- **Environment**: $8 \times 6$ grid, 6 static hidden mines, safe sidewalk corridors along edges.
- **General Metrics Tracked**: Success / Win Rate %, Rolling Total Reward, Episode Length (Steps to Goal), Double DQN vs. Vanilla Q-Network stability, Retention Benchmark across dynamic start positions.

### [Notebook 2: Environment 2 — Static Mines without Sidewalks](notebooks/02_env2_no_sidewalks_fixed_mines.ipynb)
- **Environment**: $8 \times 6$ grid, 6 static hidden mines, sidewalks removed (forcing interior navigation).
- **General Metrics Tracked**: Scratch vs. Transferred (Env 1 $\to$ 2) Win Rate %, Rolling Reward Progression, Path Length / Steps to Finish, Training Wall-Clock Time, Catastrophic Forgetting retention curve on Env 1, State-Value Potential Field $V(s)$.

### [Notebook 3: Environment 3 — Dynamic Random Mines](notebooks/03_env3_random_mines_dynamic_reasoning.ipynb)
- **Environment**: $8 \times 6$ grid, 6 mines randomized on every episode reset (eliminates path memorization).
- **General Metrics Tracked**: Scratch vs. Transferred (Env 1+2 $\to$ 3) Win Rate %, Rolling Reward, Steps to Goal, Wall-Clock Training Time & Speedup factor, Zero-Shot Generalization on novel mine layouts, Value Field $V(s)$ under stochastic hazards.

### [Notebook 4: Environment 4 — Scaled Grid + Black Holes](notebooks/04_env4_bigger_grid_black_holes.ipynb)
- **Environment**: $12 \times 10$ grid (120 cells), 15 randomized mines, 4 black holes (3-step entrapment, step penalty).
- **General Metrics Tracked**: Scratch vs. Transferred (Env 1+2+3 $\to$ 4) Win Rate %, Rolling Reward, Steps to Goal, Wall-Clock Training Time, Cross-Grid Dimension Transfer stability, Black Hole Entrapment/Escape dynamics.

### [Notebook 5: Recurrent Memory & In-Context Meta-RL (DRQN)](notebooks/05_recurrent_memory_evaluation.ipynb)
- **Environment**: Evaluations across Environment 3 (dynamic mines) and Environment 4 (black holes).
- **General Metrics Tracked**: Recurrent DRQN vs. Feedforward DQN Win Rate % and variance reduction, Scratch vs. Transferred DRQN sample efficiency, Zero-Shot generalization on 50 unseen layouts with frozen weights, One-Shot in-memory hazard avoidance across trials, Recurrent Schema Potential Field $V(s)$.

---

## Repository Structure

```
reinforcement_learning_with_minesweeper/
├── .gitignore                           # Git ignore rules (filters caches, checkpoints, logs)
├── README.md                            # Project overview and documentation
├── requirements.txt                     # Python dependencies
├── environment/
│   ├── grid_nav_env.py                  # Gymnasium GridNavEnv (v1 to v4)
│   └── __init__.py                      # Environment registration
├── models/
│   ├── grid_dqn_agent.py                # Feedforward QNetwork & GridNavDQNAgent
│   └── grid_drqn_agent.py               # Recurrent DRQNNetwork & GridNavDRQNAgent
└── notebooks/
    ├── 01_env1_training_and_evaluation.ipynb
    ├── 02_env2_no_sidewalks_fixed_mines.ipynb
    ├── 03_env3_random_mines_dynamic_reasoning.ipynb
    ├── 04_env4_bigger_grid_black_holes.ipynb
    ├── 05_recurrent_memory_evaluation.ipynb
    └── visualizer.py                    # Plotting, trajectories, heatmaps, and metrics
```

### Running the Notebooks
Launch JupyterLab or Jupyter Notebook:
```bash
jupyter lab
```
Open any of the curriculum notebooks (`01` through `05`) in the `notebooks/` directory to run or inspect experiments.
