"""
DRQN (Deep Recurrent Q-Network) Agent for GridNavMinesweeper Environments.

Implements Recurrent Memory Q-Learning (DRQN / Meta-RL context adaptation) with:
- Multi-channel grid state representation (7 channels)
- Conv2D spatial backbone with adaptive pooling + 5-cell egocentric slice
- Sequential working memory core (nn.LSTM) integrating (spatial_feat, a_{t-1}, r_{t-1}, d_{t-1})
- Episodic / Chunked Experience Replay Buffer for BPTT (Backpropagation Through Time)
- Persistent hidden state support for Zero-Shot and One-Shot learning evaluation
- Dimension-invariant across grid sizes (8x6 <-> 12x10)
- Seamless spatial backbone transfer from feedforward QNetwork checkpoints
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
    ACTION_NAMES, UNREVEALED, FLAGGED_VAL, SIDEWALK, GOAL_CELL
)
from models.grid_dqn_agent import GridStateEncoder


# ─────────────────────────────────────────────────────────────
# 1. DRQN Neural Network Architecture
# ─────────────────────────────────────────────────────────────
class DRQNNetwork(nn.Module):
    """
    Convolutional + Egocentric + Recurrent (LSTM) Q-Network.
    
    Spatial features (1696-dim) are concatenated with the agent's historical
    transition context: previous action (4-dim one-hot), previous reward (1-dim),
    and previous done signal (1-dim), feeding an LSTM recurrent core.
    """
    def __init__(
        self,
        in_channels: int = 7,
        num_actions: int = 4,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # 1. Conv2D spatial backbone (identical layer names for seamless transfer)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool_h = 6
        self.pool_w = 8
        pooled_size = 32 * self.pool_h * self.pool_w  # 1536
        ego_size = 32 * 5                              # 160 (center + UP, DOWN, LEFT, RIGHT)
        self.spatial_dim = pooled_size + ego_size     # 1696

        # 2. Meta-RL context vector: [spatial_dim + a_{t-1} (one-hot) + r_{t-1} + d_{t-1}]
        recurrent_input_dim = self.spatial_dim + num_actions + 1 + 1

        # 3. Recurrent memory core (LSTM)
        self.lstm = nn.LSTM(
            input_size=recurrent_input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # 4. Action-Value Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts spatial features from (B, C, H, W) states.
        Uses adaptive pooling and vectorized egocentric neighborhood gathering.
        """
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

        return torch.cat([pooled_flat, ego_flat], dim=1)

    def forward(
        self,
        x_seq: torch.Tensor,
        prev_a_seq: torch.Tensor,
        prev_r_seq: torch.Tensor,
        prev_d_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Batched forward pass over sequences.
        Args:
            x_seq: (B, seq_len, C, H, W)
            prev_a_seq: (B, seq_len) int64
            prev_r_seq: (B, seq_len) float32
            prev_d_seq: (B, seq_len) float32
            hidden: Optional (h_0, c_0) tuple of shape (1, B, hidden_dim)
        Returns:
            q_vals_seq: (B, seq_len, num_actions)
            next_hidden: (h_t, c_t)
        """
        B, seq_len, C, H, W = x_seq.shape
        x_flat = x_seq.reshape(B * seq_len, C, H, W)

        spatial_feats = self.extract_spatial_features(x_flat)
        spatial_feats = spatial_feats.view(B, seq_len, self.spatial_dim)

        # Meta-RL Context encoding
        a_one_hot = F.one_hot(prev_a_seq, num_classes=self.num_actions).float()
        r_col = prev_r_seq.unsqueeze(-1).float()
        d_col = prev_d_seq.unsqueeze(-1).float()

        recurrent_in = torch.cat([spatial_feats, a_one_hot, r_col, d_col], dim=-1)

        lstm_out, next_hidden = self.lstm(recurrent_in, hidden)
        q_vals = self.fc(lstm_out)

        return q_vals, next_hidden

    def forward_step(
        self,
        x: torch.Tensor,
        prev_a: int,
        prev_r: float,
        prev_d: bool,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        agent_pos: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fast single-step online execution for action selection.
        Avoids full grid padding/elementwise multiplication.
        """
        features = self.conv(x)
        pooled = F.adaptive_avg_pool2d(features, (self.pool_h, self.pool_w)).reshape(1, -1)

        if agent_pos is None:
            idx = torch.nonzero(x[0, 0] > 0.5, as_tuple=False)
            if len(idx) > 0:
                r, c = int(idx[0, 0].item()), int(idx[0, 1].item())
            else:
                r, c = 0, 0
        else:
            r, c = agent_pos

        H, W = features.shape[2], features.shape[3]
        ego_vecs = []
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                ego_vecs.append(features[0, :, nr, nc])
            else:
                ego_vecs.append(torch.zeros(32, device=x.device))
        ego_flat = torch.cat(ego_vecs, dim=0).unsqueeze(0)
        spatial_feat = torch.cat([pooled, ego_flat], dim=1)
        
        a_oh = torch.zeros((1, self.num_actions), dtype=torch.float32, device=x.device)
        if 0 <= prev_a < self.num_actions:
            a_oh[0, prev_a] = 1.0
        r_t = torch.tensor([[prev_r]], dtype=torch.float32, device=x.device)
        d_t = torch.tensor([[1.0 if prev_d else 0.0]], dtype=torch.float32, device=x.device)

        recurrent_in = torch.cat([spatial_feat, a_oh, r_t, d_t], dim=-1).unsqueeze(1)
        lstm_out, next_hidden = self.lstm(recurrent_in, hidden)
        q_vals = self.fc(lstm_out[:, 0, :])

        return q_vals, next_hidden


# ─────────────────────────────────────────────────────────────
# 2. Episodic Chunk Replay Buffer for DRQN
# ─────────────────────────────────────────────────────────────
class EpisodeChunkReplayBuffer:
    """
    Stores full episodes and samples contiguous chunks of length seq_len
    to train the recurrent memory core via Backpropagation Through Time (BPTT).
    """
    def __init__(self, capacity_episodes: int = 1500, chunk_len: int = 8):
        self.capacity = capacity_episodes
        self.chunk_len = chunk_len
        self.episodes = deque(maxlen=capacity_episodes)
        self.current_episode = []

    def push_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        prev_a: int,
        prev_r: float,
        prev_d: bool
    ):
        """Appends a transition step to the current episode buffer."""
        self.current_episode.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "prev_a": prev_a,
            "prev_r": prev_r,
            "prev_d": prev_d
        })
        if done:
            self.end_episode()

    def end_episode(self):
        """Finalizes the current episode and commits to the replay queue."""
        if len(self.current_episode) > 0:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def sample_chunks(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Samples batch_size chunks of length self.chunk_len.
        Zero-pads sequences shorter than chunk_len.
        """
        if len(self.episodes) < 4:
            return None

        sampled_episodes = random.choices(self.episodes, k=batch_size)
        
        # When transfer memory mixes grid sizes (e.g. 8x6 and 12x10), keep a uniform shape per batch
        target_shape = sampled_episodes[0][0]["state"].shape
        valid_episodes = [ep for ep in self.episodes if ep[0]["state"].shape == target_shape]
        if len(valid_episodes) >= 4:
            sampled_episodes = random.choices(valid_episodes, k=batch_size)

        states_b = []
        actions_b = []
        rewards_b = []
        next_states_b = []
        dones_b = []
        prev_a_b = []
        prev_r_b = []
        prev_d_b = []
        next_prev_a_b = []
        next_prev_r_b = []
        next_prev_d_b = []

        C, H, W = target_shape
        L = self.chunk_len

        for ep in sampled_episodes:
            ep_len = len(ep)
            if ep_len <= L:
                start_idx = 0
                chunk = ep
            else:
                start_idx = random.randint(0, ep_len - L)
                chunk = ep[start_idx:start_idx + L]

            s_seq = [step["state"] for step in chunk]
            a_seq = [step["action"] for step in chunk]
            r_seq = [step["reward"] for step in chunk]
            ns_seq = [step["next_state"] for step in chunk]
            d_seq = [1.0 if step["done"] else 0.0 for step in chunk]
            pa_seq = [step["prev_a"] for step in chunk]
            pr_seq = [step["prev_r"] for step in chunk]
            pd_seq = [1.0 if step["prev_d"] else 0.0 for step in chunk]

            # Next step prev contexts (shift by 1)
            npa_seq = [step["action"] for step in chunk]
            npr_seq = [step["reward"] for step in chunk]
            npd_seq = [1.0 if step["done"] else 0.0 for step in chunk]

            # Zero-pad if chunk is shorter than L
            pad_len = L - len(chunk)
            if pad_len > 0:
                zero_state = np.zeros((C, H, W), dtype=np.float32)
                s_seq.extend([zero_state] * pad_len)
                ns_seq.extend([zero_state] * pad_len)
                a_seq.extend([0] * pad_len)
                r_seq.extend([0.0] * pad_len)
                d_seq.extend([1.0] * pad_len)
                pa_seq.extend([0] * pad_len)
                pr_seq.extend([0.0] * pad_len)
                pd_seq.extend([0.0] * pad_len)
                npa_seq.extend([0] * pad_len)
                npr_seq.extend([0.0] * pad_len)
                npd_seq.extend([1.0] * pad_len)

            states_b.append(np.array(s_seq, dtype=np.float32))
            actions_b.append(np.array(a_seq, dtype=np.int64))
            rewards_b.append(np.array(r_seq, dtype=np.float32))
            next_states_b.append(np.array(ns_seq, dtype=np.float32))
            dones_b.append(np.array(d_seq, dtype=np.float32))
            prev_a_b.append(np.array(pa_seq, dtype=np.int64))
            prev_r_b.append(np.array(pr_seq, dtype=np.float32))
            prev_d_b.append(np.array(pd_seq, dtype=np.float32))
            next_prev_a_b.append(np.array(npa_seq, dtype=np.int64))
            next_prev_r_b.append(np.array(npr_seq, dtype=np.float32))
            next_prev_d_b.append(np.array(npd_seq, dtype=np.float32))

        return {
            "states": torch.from_numpy(np.array(states_b)),
            "actions": torch.from_numpy(np.array(actions_b)),
            "rewards": torch.from_numpy(np.array(rewards_b)),
            "next_states": torch.from_numpy(np.array(next_states_b)),
            "dones": torch.from_numpy(np.array(dones_b)),
            "prev_a": torch.from_numpy(np.array(prev_a_b)),
            "prev_r": torch.from_numpy(np.array(prev_r_b)),
            "prev_d": torch.from_numpy(np.array(prev_d_b)),
            "next_prev_a": torch.from_numpy(np.array(next_prev_a_b)),
            "next_prev_r": torch.from_numpy(np.array(next_prev_r_b)),
            "next_prev_d": torch.from_numpy(np.array(next_prev_d_b)),
        }

    def clear(self):
        """Clears memory to prevent negative transfer / replay buffer poisoning."""
        self.episodes.clear()
        self.current_episode.clear()

    def __len__(self):
        return len(self.episodes)


# ─────────────────────────────────────────────────────────────
# 3. GridNav DRQN Agent
# ─────────────────────────────────────────────────────────────
class GridNavDRQNAgent:
    """
    Recurrent Memory Agent for POMDP Grid Navigation with Minesweeper rules.
    Maintains an internal LSTM hidden state across timesteps to track past clues,
    enabling dynamic Bayesian belief updating and One-Shot in-context adaptation.
    """
    def __init__(
        self,
        grid_size: Tuple[int, int] = (8, 6),
        num_actions: int = 4,
        lr: float = 5e-4,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        hidden_dim: int = 128,
        chunk_len: int = 8,
        batch_size: int = 32,
        buffer_capacity: int = 1500,
        device: Optional[str] = None
    ):
        self.grid_width, self.grid_height = grid_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.chunk_len = chunk_len

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        self.q_net = DRQNNetwork(
            in_channels=GridStateEncoder.NUM_CHANNELS,
            num_actions=self.num_actions,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.target_net = DRQNNetwork(
            in_channels=GridStateEncoder.NUM_CHANNELS,
            num_actions=self.num_actions,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.replay_buffer = EpisodeChunkReplayBuffer(
            capacity_episodes=buffer_capacity,
            chunk_len=chunk_len
        )

        # Working Memory Hidden State
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.prev_action: int = 0
        self.prev_reward: float = 0.0
        self.prev_done: bool = False

        # Tracking metrics
        self.total_actions = 0
        self.random_actions = 0
        self.greedy_actions = 0

    def reset_hidden(self):
        """Resets the recurrent hidden state (e.g. at episode boundaries)."""
        self.hidden = None
        self.prev_action = 0
        self.prev_reward = 0.0
        self.prev_done = False

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False,
        valid_actions: Optional[List[int]] = None,
        agent_pos: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, bool]:
        """
        Selects an action using the current state and recurrent hidden memory.
        Updates self.hidden in-place.
        """
        self.total_actions += 1

        if not evaluate and random.random() < self.epsilon:
            if valid_actions:
                action = random.choice(valid_actions)
            else:
                action = random.randrange(self.num_actions)
            # Advance hidden state even during random action to keep memory in sync
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            self.q_net.eval()
            with torch.no_grad():
                _, self.hidden = self.q_net.forward_step(
                    state_tensor,
                    self.prev_action,
                    self.prev_reward,
                    self.prev_done,
                    self.hidden,
                    agent_pos=agent_pos
                )
            self.random_actions += 1
            return action, True
        else:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            self.q_net.eval()
            with torch.no_grad():
                q_vals, self.hidden = self.q_net.forward_step(
                    state_tensor,
                    self.prev_action,
                    self.prev_reward,
                    self.prev_done,
                    self.hidden,
                    agent_pos=agent_pos
                )
                q_vals = q_vals[0]
                if valid_actions:
                    mask = torch.full((self.num_actions,), -1e9, device=self.device)
                    mask[valid_actions] = 0.0
                    q_vals = q_vals + mask
                action = int(torch.argmax(q_vals).item())
            self.greedy_actions += 1
            return action, False

    def step_memory_update(self, action: int, reward: float, done: bool):
        """Updates transition history registers for the next step."""
        self.prev_action = action
        self.prev_reward = reward
        self.prev_done = done

    def update(self) -> Optional[float]:
        """
        Executes one mini-batch BPTT update across sampled sequence chunks.
        """
        batch = self.replay_buffer.sample_chunks(self.batch_size)
        if batch is None:
            return None

        states = batch["states"].to(self.device)          # (B, L, C, H, W)
        actions = batch["actions"].to(self.device)        # (B, L)
        rewards = batch["rewards"].to(self.device)        # (B, L)
        next_states = batch["next_states"].to(self.device)# (B, L, C, H, W)
        dones = batch["dones"].to(self.device)            # (B, L)
        prev_a = batch["prev_a"].to(self.device)          # (B, L)
        prev_r = batch["prev_r"].to(self.device)          # (B, L)
        prev_d = batch["prev_d"].to(self.device)          # (B, L)
        next_prev_a = batch["next_prev_a"].to(self.device)# (B, L)
        next_prev_r = batch["next_prev_r"].to(self.device)# (B, L)
        next_prev_d = batch["next_prev_d"].to(self.device)# (B, L)

        # Current Q across the sequence
        self.q_net.train()
        q_seq, _ = self.q_net(states, prev_a, prev_r, prev_d, hidden=None)
        q_current = q_seq.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (B, L)

        # Target Q across the sequence
        with torch.no_grad():
            next_q_seq, _ = self.target_net(next_states, next_prev_a, next_prev_r, next_prev_d, hidden=None)
            max_next_q = next_q_seq.max(dim=2)[0]  # (B, L)
            q_target = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        """Synchronizes target network weights with online policy weights."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        """Decays exploration rate epsilon towards epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def clear_buffer(self):
        """Purges replay buffer to prevent negative transfer."""
        self.replay_buffer.clear()

    def freeze_backbone(self):
        """Freezes Conv2D spatial layers to preserve perceptual representations."""
        for param in self.q_net.conv.parameters():
            param.requires_grad = False
        for param in self.target_net.conv.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes Conv2D spatial layers for end-to-end tuning."""
        for param in self.q_net.conv.parameters():
            param.requires_grad = True
        for param in self.target_net.conv.parameters():
            param.requires_grad = True

    def load_spatial_backbone(self, checkpoint_path: str):
        """
        Loads the Conv2D spatial backbone from a trained QNetwork checkpoint (e.g. Env 1 or 2).
        Enables seamless sensory transfer from feedforward agents into recurrent agents.
        """
        data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = data.get("q_net", data)
        
        conv_weights = {k: v for k, v in state_dict.items() if k.startswith("conv.")}
        missing, unexpected = self.q_net.load_state_dict(conv_weights, strict=False)
        self.target_net.load_state_dict(self.q_net.state_dict())
        return conv_weights

    def save(self, path: str):
        """Saves agent weights and optimizer state."""
        data = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_actions": self.total_actions,
            "random_actions": self.random_actions,
            "greedy_actions": self.greedy_actions,
        }
        torch.save(data, path)

    def load(self, path: str):
        """Loads agent weights and target network."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(data["q_net"])
        self.target_net.load_state_dict(data["target_net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.epsilon = data.get("epsilon", self.epsilon_min)


# ─────────────────────────────────────────────────────────────
# 4. DRQN Training Runner
# ─────────────────────────────────────────────────────────────
def train_drqn_agent(
    env,
    agent: GridNavDRQNAgent,
    num_episodes: int = 2500,
    update_freq: int = 4,
    target_update_freq: int = 10,
    eval_freq: int = 50,
    eval_episodes: int = 25,
    verbose_freq: int = 200,
    win_rate_window: int = 100,
    mask_actions: bool = True,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Trains a GridNavDRQNAgent on the given environment.
    Optimized for CPU/GPU efficiency with sequence-based BPTT.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    metrics = {
        "episodes": [],
        "rewards": [],
        "rolling_rewards": [],
        "wins": [],
        "rolling_win_rate": [],
        "win_rates": [],
        "steps": [],
        "episode_rewards": [],
        "episode_lengths": [],
        "eval_win_rates": [],
        "eval_episodes": [],
        "losses": [],
        "epsilons": [],
        "cumulative_time": [],
        "wall_clock_time": 0.0,
        "first_win_episode": None,
        "best_winning_history": None
    }

    recent_wins = deque(maxlen=win_rate_window)
    recent_rewards = deque(maxlen=win_rate_window)
    start_time = time.time()
    best_eval_win = -1.0
    eval_win_rate = 0.0
    max_steps = getattr(env.unwrapped, "max_steps", 100)

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        state = GridStateEncoder.encode(obs, max_steps=max_steps)
        agent.reset_hidden()

        ep_reward = 0.0
        ep_steps = 0
        ep_loss = []
        done = False
        won = False
        traj = []

        while not done:
            valid_actions = env.unwrapped.get_valid_actions() if (mask_actions and hasattr(env.unwrapped, "get_valid_actions")) else None
            prev_a = agent.prev_action
            prev_r = agent.prev_reward
            prev_d = agent.prev_done
            agent_pos = (int(obs["agent_pos"][0]), int(obs["agent_pos"][1])) if "agent_pos" in obs else None

            action, _ = agent.select_action(state, evaluate=False, valid_actions=valid_actions, agent_pos=agent_pos)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = GridStateEncoder.encode(next_obs, max_steps=max_steps)

            # Store step in episodic replay buffer
            agent.replay_buffer.push_step(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                prev_a=prev_a,
                prev_r=prev_r,
                prev_d=prev_d
            )

            # Update agent transition history registers
            agent.step_memory_update(action=action, reward=reward, done=done)

            ep_reward += reward
            ep_steps += 1

            # Only record trajectory details on periodic log episodes to avoid memory bloat
            if ep % verbose_freq == 0:
                traj.append({
                    "step": ep_steps,
                    "action": action,
                    "action_name": ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else "UNKNOWN",
                    "reward": reward,
                    "event": info.get("event", "STEP"),
                    "board": obs["board"].copy(),
                    "agent_pos": obs["agent_pos"].copy(),
                    "steps_left": obs["steps_remaining"][0]
                })

            if bool(info.get("won", False)) or info.get("event") == "goal_reached":
                won = True

            # Periodic BPTT Gradient Update
            if ep_steps % update_freq == 0:
                loss_val = agent.update()
                if loss_val is not None:
                    ep_loss.append(loss_val)

            obs = next_obs
            state = next_state

        # Episode end cleanup
        agent.replay_buffer.end_episode()
        agent.decay_epsilon()

        if ep % target_update_freq == 0:
            agent.update_target_network()

        recent_wins.append(1 if won else 0)
        recent_rewards.append(ep_reward)
        rolling_win = (sum(recent_wins) / len(recent_wins)) * 100.0
        rolling_r = float(np.mean(recent_rewards))

        metrics["episodes"].append(ep)
        metrics["rewards"].append(ep_reward)
        metrics["rolling_rewards"].append(rolling_r)
        metrics["wins"].append(1 if won else 0)
        metrics["rolling_win_rate"].append(rolling_win)
        metrics["win_rates"].append(rolling_win)
        metrics["steps"].append(ep_steps)
        metrics["episode_rewards"].append(ep_reward)
        metrics["episode_lengths"].append(ep_steps)
        metrics["epsilons"].append(agent.epsilon)
        metrics["cumulative_time"].append(time.time() - start_time)
        if ep_loss:
            metrics["losses"].append(float(np.mean(ep_loss)))

        if won and metrics["first_win_episode"] is None:
            metrics["first_win_episode"] = ep

        if won and metrics["best_winning_history"] is None and len(traj) > 0:
            metrics["best_winning_history"] = traj

        # Evaluation runs
        if ep % eval_freq == 0 or ep == num_episodes:
            eval_wins = 0
            for _ in range(eval_episodes):
                e_obs, _ = env.reset()
                e_state = GridStateEncoder.encode(e_obs, max_steps=max_steps)
                agent.reset_hidden()
                e_done = False
                while not e_done:
                    e_valid = env.unwrapped.get_valid_actions() if (mask_actions and hasattr(env.unwrapped, "get_valid_actions")) else None
                    e_pos = (int(e_obs["agent_pos"][0]), int(e_obs["agent_pos"][1])) if "agent_pos" in e_obs else None
                    e_action, _ = agent.select_action(e_state, evaluate=True, valid_actions=e_valid, agent_pos=e_pos)
                    e_obs, e_r, e_term, e_trunc, e_info = env.step(e_action)
                    e_done = e_term or e_trunc
                    agent.step_memory_update(e_action, e_r, e_done)
                    e_state = GridStateEncoder.encode(e_obs, max_steps=max_steps)
                    if bool(e_info.get("won", False)) or e_info.get("event") == "goal_reached":
                        eval_wins += 1
                        break
            eval_win_rate = (eval_wins / eval_episodes) * 100.0
            metrics["eval_win_rates"].append(eval_win_rate)
            metrics["eval_episodes"].append(ep)
            if eval_win_rate > best_eval_win:
                best_eval_win = eval_win_rate

        if ep % verbose_freq == 0 or ep == num_episodes:
            elapsed = time.time() - start_time
            print(f"Episode {ep:5d}/{num_episodes} | "
                  f"Rolling Win: {rolling_win:5.1f}% | "
                  f"Eval Win: {eval_win_rate:5.1f}% | "
                  f"Reward: {ep_reward:6.1f} | "
                  f"Steps: {ep_steps:3d} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Time: {elapsed:5.1f}s")

    metrics["wall_clock_time"] = time.time() - start_time
    return metrics
