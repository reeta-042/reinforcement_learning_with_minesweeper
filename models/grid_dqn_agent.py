"""
DQN Agent for GridNavMinesweeper Environments.

Implements Deep Q-Learning with:
- Multi-channel grid state representation
- Experience Replay Memory
- Target Network with soft or periodic hard updates
- Epsilon-greedy exploration with decay tracking
- Action masking / invalid action penalties
- Detailed metrics collection for logging and visualization
"""

import random
import time
from collections import deque
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment.grid_nav_env import (
    MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT,
    FLAG_UP, FLAG_DOWN, FLAG_LEFT, FLAG_RIGHT,
    ACTION_NAMES, UNREVEALED, FLAGGED_VAL, SIDEWALK, GOAL_CELL
)


# ─────────────────────────────────────────────────────────────
# 1. State Encoder
# ─────────────────────────────────────────────────────────────
class GridStateEncoder:
    """
    Converts GridNavEnv observation dict into a multi-channel PyTorch tensor.
    
    Channels:
      0: Agent Position (1.0 at agent_pos, 0.0 elsewhere)
      1: Revealed Numbers (normalized count 0..8 / 8.0, 0 for unrevealed/flagged)
      2: Unrevealed Mask (1.0 where cell is unrevealed)
      3: Flagged Mask (1.0 where cell is flagged)
      4: Sidewalk Mask (1.0 for sidewalk cells)
      5: Goal Mask (1.0 for goal cells)
      6: Normalized Steps Remaining (broadcast scalar across grid)
    Total channels = 7
    """
    NUM_CHANNELS = 7

    @staticmethod
    def encode(obs: Dict[str, Any], max_steps: int = 100) -> np.ndarray:
        board = obs["board"]                     # (H, W)
        cell_types = obs["cell_types"]           # (H, W)
        agent_pos = obs["agent_pos"]             # (2,) [row, col]
        steps_left = float(obs["steps_remaining"][0]) / max(1.0, float(max_steps))

        H, W = board.shape
        state = np.zeros((GridStateEncoder.NUM_CHANNELS, H, W), dtype=np.float32)

        # Ch 0: Agent position
        r, c = int(agent_pos[0]), int(agent_pos[1])
        if 0 <= r < H and 0 <= c < W:
            state[0, r, c] = 1.0

        # Ch 1: Revealed numbers
        revealed_nums = np.clip(board, 0, 8).astype(np.float32) / 8.0
        # Zero out unrevealed/flagged
        revealed_nums[board < 0] = 0.0
        state[1] = revealed_nums

        # Ch 2: Unrevealed mask
        state[2] = (board == UNREVEALED).astype(np.float32)

        # Ch 3: Flagged mask
        state[3] = (board == FLAGGED_VAL).astype(np.float32)

        # Ch 4: Sidewalk
        state[4] = (cell_types == SIDEWALK).astype(np.float32)

        # Ch 5: Goal
        state[5] = (cell_types == GOAL_CELL).astype(np.float32)

        # Ch 6: Steps remaining
        state[6] = np.full((H, W), steps_left, dtype=np.float32)

        return state


# ─────────────────────────────────────────────────────────────
# 2. DQN Neural Network
# ─────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """Convolutional + Egocentric + Fully Connected Q-Network for Grid Navigation.

    Uses adaptive pooling to a canonical (6, 8) spatial map plus exact 5-cell egocentric
    neighborhood features centered on the agent so the same weights transfer cleanly
    across different grid dimensions (e.g., 8x6 <-> 12x10).
    """

    def __init__(self, in_channels: int = 7, height: int = 6, width: int = 8, num_actions: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool_h = 6
        self.pool_w = 8
        pooled_size = 32 * self.pool_h * self.pool_w
        ego_size = 32 * 5  # center + UP, DOWN, LEFT, RIGHT neighbors

        self.fc = nn.Sequential(
            nn.Linear(pooled_size + ego_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        features = self.conv(x)
        pooled = F.adaptive_avg_pool2d(features, (self.pool_h, self.pool_w))
        pooled_flat = pooled.reshape(features.size(0), -1)

        agent_mask = x[:, 0:1, :, :]
        padded_feat = F.pad(features, (1, 1, 1, 1))
        H, W = features.shape[2], features.shape[3]
        ego_slices = []
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            r0 = 1 + dr
            c0 = 1 + dc
            shifted = padded_feat[:, :, r0:r0 + H, c0:c0 + W]
            ego_slices.append((shifted * agent_mask).sum(dim=(2, 3)))
        ego_flat = torch.cat(ego_slices, dim=1)

        combined = torch.cat([pooled_flat, ego_flat], dim=1)
        q_values = self.fc(combined)
        return q_values


# ─────────────────────────────────────────────────────────────
# 3. Replay Buffer
# ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        # When transfer memory mixes grid sizes (e.g. 8x6 and 12x10), keep a uniform shape per batch.
        target_shape = batch[0][0].shape
        if any(t[0].shape != target_shape for t in batch):
            same_shape = [t for t in self.buffer if t[0].shape == target_shape]
            if len(same_shape) >= batch_size:
                batch = random.sample(same_shape, batch_size)
            else:
                batch = random.choices(same_shape, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────
# 4. GridNav DQN Agent
# ─────────────────────────────────────────────────────────────
class GridNavDQNAgent:
    def __init__(
        self,
        grid_size: Tuple[int, int] = (8, 6),   # (width, height)
        num_actions: int = 4,
        lr: float = 5e-4,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 20000,
        batch_size: int = 64,
        target_update_freq: int = 10,          # in episodes
        device: Optional[str] = None,
    ):
        self.grid_width, self.grid_height = grid_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_net = QNetwork(
            in_channels=GridStateEncoder.NUM_CHANNELS,
            height=self.grid_height,
            width=self.grid_width,
            num_actions=self.num_actions
        ).to(self.device)

        self.target_net = QNetwork(
            in_channels=GridStateEncoder.NUM_CHANNELS,
            height=self.grid_height,
            width=self.grid_width,
            num_actions=self.num_actions
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Tracking exploration / exploitation
        self.total_actions = 0
        self.random_actions = 0
        self.greedy_actions = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False, valid_actions: Optional[List[int]] = None) -> Tuple[int, bool]:
        """
        Returns (action, is_exploration), optionally masked by valid_actions.
        """
        self.total_actions += 1

        if not evaluate and random.random() < self.epsilon:
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = random.randrange(self.num_actions)
            self.random_actions += 1
            return action, True
        else:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            self.q_net.eval()
            with torch.no_grad():
                q_vals = self.q_net(state_tensor)[0]
                if valid_actions:
                    mask = torch.full((self.num_actions,), -1e9, device=self.device)
                    mask[valid_actions] = 0.0
                    q_vals = q_vals + mask
                action = int(torch.argmax(q_vals).item())
            self.greedy_actions += 1
            return action, False

    def update(self) -> Optional[float]:
        """Runs one mini-batch gradient descent step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        # Current Q
        self.q_net.train()
        q_current = self.q_net(states_t).gather(1, actions_t)

        # Target Q (Double DQN or Standard DQN)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            q_target = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def clear_buffer(self):
        """Clears experience replay buffer to prevent negative transfer / replay buffer poisoning."""
        self.replay_buffer.buffer.clear()

    def freeze_backbone(self):
        """Freezes Conv2D spatial backbone for adapter/head fine-tuning."""
        for param in self.q_net.conv.parameters():
            param.requires_grad = False
        for param in self.target_net.conv.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes Conv2D spatial backbone for full end-to-end tuning."""
        for param in self.q_net.conv.parameters():
            param.requires_grad = True
        for param in self.target_net.conv.parameters():
            param.requires_grad = True

    def save(self, path: str, save_buffer: bool = True):
        data = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        if save_buffer:
            data["replay_buffer"] = list(self.replay_buffer.buffer)
        torch.save(data, path)

    def load(self, path: str, load_buffer: bool = True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        if load_buffer and "replay_buffer" in checkpoint:
            self.replay_buffer.buffer = deque(checkpoint["replay_buffer"], maxlen=self.replay_buffer.capacity)
            print(f"Loaded {len(self.replay_buffer)} stored experiences from previous environment!")


# ─────────────────────────────────────────────────────────────
# 5. Vanilla Q-Network (QN) Agent (No Replay Buffer, No Target Net)
# ─────────────────────────────────────────────────────────────
class VanillaQNAgent:
    """
    Standard / Vanilla Q-Network (QN) Agent for benchmark comparison against DQN.
    
    Key differences from DQN:
    1. NO Replay Buffer: Updates online immediately on each step (correlated transitions).
    2. NO Target Network: Evaluates target using the same moving Q-network weights.
    3. Uses identical CNN backbone to provide an apples-to-apples comparison against DQN.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (8, 6),
        num_actions: int = 4,
        lr: float = 5e-4,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.985,
        device: Optional[str] = None,
    ):
        self.grid_width, self.grid_height = grid_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Single network (no target network)
        self.q_net = QNetwork(
            in_channels=GridStateEncoder.NUM_CHANNELS,
            height=self.grid_height,
            width=self.grid_width,
            num_actions=self.num_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[int, bool]:
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.num_actions), True
        else:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            self.q_net.eval()
            with torch.no_grad():
                q_vals = self.q_net(state_tensor)
                action = int(torch.argmax(q_vals, dim=1).item())
            return action, False

    def update_online(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Optional[float]:
        """Online single-step update: NO replay buffer, NO separate target net."""
        self.q_net.train()

        s_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        ns_t = torch.from_numpy(next_state).unsqueeze(0).to(self.device)

        # Current Q
        q_current = self.q_net(s_t)[0, action]

        # Target Q (using the SAME moving network)
        with torch.no_grad():
            next_max_q = self.q_net(ns_t).max(1)[0].item()
            q_target = reward + (0.0 if done else self.gamma * next_max_q)

        loss = self.loss_fn(q_current, torch.tensor(q_target, dtype=torch.float32, device=self.device))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_qn_agent(
    env,
    agent: VanillaQNAgent,
    num_episodes: int = 200,
    verbose_freq: int = 50,
) -> Dict[str, Any]:
    """
    Trains a Vanilla QN Agent online without replay memory or target network.
    """
    metrics = {
        "episodes": [],
        "rewards": [],
        "rolling_rewards": [],
        "wins": [],
        "rolling_win_rate": [],
        "steps": [],
        "epsilons": [],
        "avg_loss": [],
        "cumulative_time": [],
    }

    t_start = time.time()
    rolling_window = 25

    print(f"Starting Vanilla QN (Online) Training for {num_episodes} episodes...")

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = GridStateEncoder.encode(obs, env.unwrapped.max_steps)

        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_losses = []
        ep_won = 0

        while not done:
            ep_steps += 1
            action, _ = agent.select_action(state)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            next_state = GridStateEncoder.encode(next_obs, env.unwrapped.max_steps)

            # Online update on every step
            loss = agent.update_online(state, action, reward, next_state, done)
            if loss is not None:
                ep_losses.append(loss)

            state = next_state
            ep_reward += reward

            if step_info.get("won"):
                ep_won = 1

        agent.decay_epsilon()

        cum_time = time.time() - t_start
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        metrics["episodes"].append(ep)
        metrics["rewards"].append(ep_reward)
        metrics["wins"].append(ep_won)
        metrics["steps"].append(ep_steps)
        metrics["epsilons"].append(agent.epsilon)
        metrics["avg_loss"].append(avg_loss)
        metrics["cumulative_time"].append(cum_time)

        recent_wins = metrics["wins"][-rolling_window:]
        recent_rewards = metrics["rewards"][-rolling_window:]
        rolling_win_rate = (sum(recent_wins) / len(recent_wins)) * 100.0
        rolling_reward = float(np.mean(recent_rewards))
        metrics["rolling_win_rate"].append(rolling_win_rate)
        metrics["rolling_rewards"].append(rolling_reward)

        if ep % verbose_freq == 0 or ep == num_episodes:
            print(f"Vanilla QN Ep {ep:3d} | Win Rate: {rolling_win_rate:5.1f}% | Ep Reward: {ep_reward:+6.1f}")

    return metrics


def train_dqn_agent(
    env,
    agent: GridNavDQNAgent,
    num_episodes: int = 250,
    update_freq: int = 4,
    target_update_freq: int = 10,
    warmup_steps: int = 64,
    verbose_freq: int = 25,
) -> Dict[str, Any]:
    """
    Trains a GridNavDQNAgent and collects comprehensive metrics for visualization.
    """
    metrics = {
        "episodes": [],
        "rewards": [],
        "rolling_rewards": [],
        "wins": [],
        "rolling_win_rate": [],
        "steps": [],
        "epsilons": [],
        "exploration_rate": [],       # % random actions taken this episode
        "avg_loss": [],
        "wall_hits": [],
        "mine_hits": [],
        "timeouts": [],
        "durations": [],              # seconds per episode
        "cumulative_time": [],        # elapsed seconds from start
        "first_win_episode": None,
        "best_winning_history": None,
    }

    total_steps = 0
    t_start = time.time()
    rolling_window = 25
    best_winning_history = None
    first_win_episode = None

    print(f"Starting DQN Training on {env.unwrapped.__class__.__name__} for {num_episodes} episodes...")
    print(f"{'═'*75}")
    print(f"{'Ep':>5} | {'Reward':>8} | {'Win% (25)':>10} | {'Steps':>6} | {'Epsilon':>8} | {'Explor%':>8} | {'Time (s)':>8}")
    print(f"{'─'*75}")

    for ep in range(1, num_episodes + 1):
        ep_t0 = time.time()
        obs, info = env.reset()
        state = GridStateEncoder.encode(obs, env.unwrapped.max_steps)

        # Track trajectory only when needed (early win tracking or periodic sample)
        # to avoid allocating millions of numpy arrays during long exploratory episodes
        should_track = (best_winning_history is None) or (ep % verbose_freq == 0)
        current_traj = []
        if should_track:
            current_traj.append({
                "step": 0,
                "agent_pos": (int(obs["agent_pos"][0]), int(obs["agent_pos"][1])),
                "board": obs["board"].copy(),
                "cell_types": obs["cell_types"].copy(),
                "action_name": "START",
                "action_id": -1,
                "reward": 0.0,
                "event": "start",
                "steps_remaining": int(obs["steps_remaining"][0]),
                "won": False,
                "mine_map": env.unwrapped._mine_map.copy() if hasattr(env.unwrapped, "_mine_map") else None,
            })

        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_losses = []
        ep_random_actions = 0
        ep_wall_hits = 0
        ep_mine_hits = 0
        ep_won = 0

        while not done:
            total_steps += 1
            ep_steps += 1

            action, is_exploration = agent.select_action(state)
            if is_exploration:
                ep_random_actions += 1

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            next_state = GridStateEncoder.encode(next_obs, env.unwrapped.max_steps)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Training step
            if total_steps >= warmup_steps and total_steps % update_freq == 0:
                loss = agent.update()
                if loss is not None:
                    ep_losses.append(loss)

            if should_track:
                current_traj.append({
                    "step": ep_steps,
                    "agent_pos": (int(next_obs["agent_pos"][0]), int(next_obs["agent_pos"][1])),
                    "board": next_obs["board"].copy(),
                    "cell_types": next_obs["cell_types"].copy(),
                    "action_name": ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"ACT_{action}",
                    "action_id": action,
                    "reward": float(reward),
                    "event": step_info.get("event", "move"),
                    "steps_remaining": int(next_obs["steps_remaining"][0]),
                    "won": bool(step_info.get("won", False)),
                    "mine_map": env.unwrapped._mine_map.copy() if hasattr(env.unwrapped, "_mine_map") else None,
                })

            state = next_state
            ep_reward += reward

            event = step_info.get("event", "")
            if event == "wall_hit":
                ep_wall_hits += 1
            elif event == "mine_hit":
                ep_mine_hits += 1

            if step_info.get("won"):
                ep_won = 1
                if first_win_episode is None:
                    first_win_episode = ep
                if should_track and (best_winning_history is None or len(current_traj) < len(best_winning_history)):
                    best_winning_history = [f.copy() for f in current_traj]

        ep_duration = time.time() - ep_t0
        cum_time = time.time() - t_start

        # Target network update
        if ep % target_update_freq == 0:
            agent.update_target_network()

        # Exploration decay
        agent.decay_epsilon()

        # Compute episode stats
        explor_pct = (ep_random_actions / max(1, ep_steps)) * 100.0
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        is_timeout = 1 if (ep_steps >= env.unwrapped.max_steps and not ep_won and ep_mine_hits == 0) else 0

        metrics["episodes"].append(ep)
        metrics["rewards"].append(ep_reward)
        metrics["wins"].append(ep_won)
        metrics["steps"].append(ep_steps)
        metrics["epsilons"].append(agent.epsilon)
        metrics["exploration_rate"].append(explor_pct)
        metrics["avg_loss"].append(avg_loss)
        metrics["wall_hits"].append(ep_wall_hits)
        metrics["mine_hits"].append(ep_mine_hits)
        metrics["timeouts"].append(is_timeout)
        metrics["durations"].append(ep_duration)
        metrics["cumulative_time"].append(cum_time)

        # Rolling statistics
        recent_wins = metrics["wins"][-rolling_window:]
        recent_rewards = metrics["rewards"][-rolling_window:]
        rolling_win_rate = (sum(recent_wins) / len(recent_wins)) * 100.0
        rolling_reward = float(np.mean(recent_rewards))
        metrics["rolling_win_rate"].append(rolling_win_rate)
        metrics["rolling_rewards"].append(rolling_reward)

        if ep % verbose_freq == 0 or ep == num_episodes:
            print(
                f"{ep:5d} | {rolling_reward:+8.1f} | {rolling_win_rate:9.1f}% | "
                f"{ep_steps:6d} | {agent.epsilon:8.3f} | {explor_pct:7.1f}% | {cum_time:8.1f}s"
            )

    metrics["best_winning_history"] = best_winning_history
    metrics["first_win_episode"] = first_win_episode
    print(f"{'═'*75}")
    print(f"Training completed in {time.time() - t_start:.2f}s!")
    if first_win_episode is not None:
        print(f"First Goal reached at Episode {first_win_episode}!")
    print(f"Final 25-episode win rate: {metrics['rolling_win_rate'][-1]:.1f}%")

    return metrics
