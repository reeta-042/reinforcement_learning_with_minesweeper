"""
Grid Navigation Minesweeper Environment

Agent navigates from START (top-left) to GOAL (right column) on a grid,
using classic minesweeper reveal/flag mechanics to avoid hidden mines.

Actions (Discrete 8):
    0-3:  Move UP / DOWN / LEFT / RIGHT  — physically step onto the cell (reveals it)
    4-7:  Flag UP / DOWN / LEFT / RIGHT  — toggle flag on that neighbour (no movement)

Difficulty levels:
    1: Sidewalks (safe edges) ON, fixed mines per seed, timer
    2: Sidewalks OFF, fixed mines per seed, timer
    3: Sidewalks OFF, random mines each episode, timer
    4: Bigger grid, mines + black holes (trap agent N steps), timer
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from typing import Optional

# ──────────── Constants ────────────

# Cell types (static per episode)
INTERIOR = 0
SIDEWALK = 1
GOAL_CELL = 2
BLACK_HOLE_CELL = 3

# Board observation values
UNREVEALED = -3
FLAGGED_VAL = -2
MINE_REVEALED = -1
# 0–8 = adjacent mine count (revealed safe cell)

# Actions
MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = 0, 1, 2, 3
FLAG_UP, FLAG_DOWN, FLAG_LEFT, FLAG_RIGHT = 4, 5, 6, 7

ACTION_NAMES = [
    "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
    "FLAG_UP", "FLAG_DOWN", "FLAG_LEFT", "FLAG_RIGHT",
]

# Direction deltas: (delta_row, delta_col) — shared by move and flag
DIR_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ──────────── Pygame palette ────────────
COLORS = {
    "bg":           (30,  30,  50),
    "grid_line":    (60,  60,  80),
    "unrevealed":   (51,  85, 170),
    "revealed":     (192, 192, 192),
    "sidewalk":     (55,  55,  55),
    "goal":         (0,  180,  60),
    "black_hole":   (80,   0, 120),
    "agent":        (0,  210, 210),
    "mine":         (220,  30, 30),
    "flag":         (255, 140,  0),
    "text":         (20,  20,  20),
    "hud_text":     (200, 200, 200),
    "num": {
        1: (0,   0, 255),
        2: (0, 128,   0),
        3: (255, 0,   0),
        4: (0,   0, 128),
        5: (128, 0,   0),
        6: (0, 128, 128),
        7: (0,   0,   0),
        8: (128, 128, 128),
    },
}


# ════════════════════════════════════════════════════════════════════════
class GridNavEnv(gym.Env):
    """Grid navigation minesweeper — see module docstring."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    def __init__(
        self,
        grid_size: tuple[int, int] = (8, 6),   # (width, height)
        num_mines: int = 6,
        difficulty: int = 1,
        max_steps: Optional[int] = None,
        sidewalks: Optional[bool] = None,
        random_mines: Optional[bool] = None,
        num_black_holes: int = 0,
        black_hole_trap_steps: int = 3,
        allow_flagging: bool = False,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        fps: Optional[int] = None,
        random_sidewalk_start: bool = False,
    ):
        super().__init__()

        self.grid_width, self.grid_height = grid_size
        self.num_mines = num_mines
        self.difficulty = difficulty
        self.render_mode = render_mode
        self._seed = seed
        self.fps = fps or self.metadata["render_fps"]
        self.random_sidewalk_start = random_sidewalk_start
        self.allow_flagging = allow_flagging

        # ── difficulty-driven defaults ──
        self.sidewalks_on = sidewalks if sidewalks is not None else (difficulty == 1)
        self.random_mines_on = random_mines if random_mines is not None else (difficulty >= 3)
        self.num_black_holes = num_black_holes if difficulty >= 4 else 0
        self.black_hole_trap_steps = black_hole_trap_steps

        # Timer
        self.max_steps = max_steps or (self.grid_width * self.grid_height * 2)

        # ── spaces (4 clean moves: UP, DOWN, LEFT, RIGHT; or 8 if flagging enabled) ──
        self.action_space = spaces.Discrete(8 if self.allow_flagging else 4)

        self.observation_space = spaces.Dict({
            "board": spaces.Box(
                low=-3, high=8,
                shape=(self.grid_height, self.grid_width),
                dtype=np.int8,
            ),
            "cell_types": spaces.Box(
                low=0, high=3,
                shape=(self.grid_height, self.grid_width),
                dtype=np.int8,
            ),
            "agent_pos": spaces.Box(
                low=np.array([0, 0], dtype=np.int32),
                high=np.array([self.grid_height - 1, self.grid_width - 1], dtype=np.int32),
                shape=(2,),
                dtype=np.int32,
            ),
            "steps_remaining": spaces.Box(
                low=0, high=self.max_steps,
                shape=(1,), dtype=np.int32,
            ),
            "trapped_remaining": spaces.Box(
                low=0,
                high=max(1, self.black_hole_trap_steps if self.num_black_holes else 0),
                shape=(1,), dtype=np.int32,
            ),
        })

        # ── rewards ──
        self.R_goal          =  100.0
        self.R_mine_hit      = -100.0
        self.R_wall_hit      =   -2.0
        self.R_safe_step     =    1.0
        self.R_timeout       =  -50.0
        self.R_step          =   -0.5   # per-step cost (encourages speed)
        self.R_correct_flag  =    2.0
        self.R_wrong_flag    =   -1.0
        self.R_invalid_flag  =   -1.0   # flag out-of-bounds / revealed cell
        self.R_revisit       =   -0.5   # stepping on already-visited cell
        self.R_black_hole    =   -2.0   # per trapped step

        # ── internal state (set in reset) ──
        self._rng: random.Random = random.Random(seed)
        self._mine_map: Optional[np.ndarray] = None
        self._black_hole_map: Optional[np.ndarray] = None
        self._adj_counts: Optional[np.ndarray] = None
        self._cell_types: Optional[np.ndarray] = None
        self._revealed: Optional[np.ndarray] = None
        self._flagged: Optional[np.ndarray] = None
        self._visited: Optional[np.ndarray] = None
        self._fixed_mine_map: Optional[np.ndarray] = None

        self.agent_row: int = 0
        self.agent_col: int = 0
        self.steps_taken: int = 0
        self.stalled_steps: int = 0
        self.trapped_steps: int = 0
        self._episode_count: int = 0

        # ── pygame ──
        self.cell_px = 60
        self._screen = None
        self._clock = None
        self._font = None
        self._hud_font = None
        self._hud_height = 40

    # ────────────────────── board helpers ──────────────────────

    def _build_cell_types(self) -> np.ndarray:
        """Create the cell-type grid (INTERIOR / SIDEWALK / GOAL / BLACK_HOLE)."""
        ct = np.full((self.grid_height, self.grid_width), INTERIOR, dtype=np.int8)

        if self.sidewalks_on:
            ct[0, :-1] = SIDEWALK                          # top row (except goal column)
            ct[self.grid_height - 1, :-1] = SIDEWALK       # bottom row (except goal column)
            ct[:, 0] = SIDEWALK                             # left column

        # Goal = entire right column
        ct[:, self.grid_width - 1] = GOAL_CELL
        return ct

    def _placeable_cells(self, exclude_mines: bool = False) -> list[tuple[int, int]]:
        """Interior cells available for mine / black-hole placement."""
        cells = []
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if self._cell_types[r, c] != INTERIOR:
                    continue
                if r == self.agent_row and c == self.agent_col:
                    continue
                if exclude_mines and self._mine_map is not None and self._mine_map[r, c]:
                    continue
                cells.append((r, c))
        return cells

    def _place_mines(self) -> np.ndarray:
        mm = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        valid = self._placeable_cells()
        n = min(self.num_mines, len(valid))
        for r, c in self._rng.sample(valid, n):
            mm[r, c] = True
        return mm

    def _place_black_holes(self) -> np.ndarray:
        bh = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        if self.num_black_holes <= 0:
            return bh
        valid = self._placeable_cells(exclude_mines=True)
        n = min(self.num_black_holes, len(valid))
        for r, c in self._rng.sample(valid, n):
            bh[r, c] = True
            self._cell_types[r, c] = BLACK_HOLE_CELL
        return bh

    def _compute_adj_counts(self) -> np.ndarray:
        """Standard 8-neighbour adjacent-mine count."""
        adj = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if self._mine_map[r, c]:
                    continue
                cnt = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                            if self._mine_map[nr, nc]:
                                cnt += 1
                adj[r, c] = cnt
        return adj

    def _flood_fill(self, row: int, col: int) -> None:
        """Iterative BFS flood-fill from a 0-cell (classic minesweeper)."""
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if not (0 <= r < self.grid_height and 0 <= c < self.grid_width):
                continue
            if self._revealed[r, c] or self._flagged[r, c]:
                continue
            if self._mine_map[r, c]:
                continue
            if self._black_hole_map is not None and self._black_hole_map[r, c]:
                continue

            self._revealed[r, c] = True

            if self._adj_counts[r, c] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        stack.append((r + dr, c + dc))

    def _pre_reveal_safe_cells(self) -> None:
        """Sidewalk, goal, and start cells are visible from the start."""
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if self._cell_types[r, c] in (SIDEWALK, GOAL_CELL):
                    self._revealed[r, c] = True
        self._revealed[self.agent_row, self.agent_col] = True

    def _reveal_all_mines(self) -> None:
        """Reveal every mine (called on game over)."""
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if self._mine_map[r, c]:
                    self._revealed[r, c] = True

    # ────────────────────── observation ──────────────────────

    def _get_board_obs(self) -> np.ndarray:
        board = np.full((self.grid_height, self.grid_width), UNREVEALED, dtype=np.int8)
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if self._flagged[r, c]:
                    board[r, c] = FLAGGED_VAL
                elif self._revealed[r, c]:
                    if self._mine_map[r, c]:
                        board[r, c] = MINE_REVEALED
                    else:
                        board[r, c] = self._adj_counts[r, c]
        return board

    def _get_obs(self) -> dict:
        return {
            "board": self._get_board_obs(),
            "cell_types": self._cell_types.copy(),
            "agent_pos": np.array([self.agent_row, self.agent_col], dtype=np.int32),
            "steps_remaining": np.array(
                [max(0, self.max_steps - self.steps_taken)], dtype=np.int32
            ),
            "trapped_remaining": np.array([self.trapped_steps], dtype=np.int32),
        }

    # ────────────────────── gym interface ──────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        self._episode_count += 1
        self.steps_taken = 0
        self.stalled_steps = 0
        self.trapped_steps = 0

        # Layout
        self._cell_types = self._build_cell_types()

        # Determine starting position (custom option, random sidewalk, or default top-left)
        if options and "start_pos" in options:
            sr, sc = options["start_pos"]
            self.agent_row = int(sr)
            self.agent_col = int(sc)
        elif self.random_sidewalk_start and self.sidewalks_on:
            # Pick any starting cell along the left sidewalk column
            self.agent_row = self._rng.randint(0, self.grid_height - 1)
            self.agent_col = 0
        else:
            self.agent_row = 0
            self.agent_col = 0

        # Mines — fixed reuse or fresh random
        if self.random_mines_on or self._fixed_mine_map is None:
            if self.random_mines_on and seed is None:
                self._rng = random.Random()          # truly random each episode if seed not specified
            self._mine_map = self._place_mines()
            if not self.random_mines_on:
                self._fixed_mine_map = self._mine_map.copy()
        else:
            self._mine_map = self._fixed_mine_map.copy()

        self._adj_counts = self._compute_adj_counts()

        # Black holes (env 4) — placed after mines, mutates cell_types
        self._black_hole_map = self._place_black_holes()

        # Visibility
        self._revealed = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self._flagged  = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self._visited  = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self._pre_reveal_safe_cells()
        self._visited[self.agent_row, self.agent_col] = True
        if self._adj_counts[self.agent_row, self.agent_col] == 0:
            self._flood_fill(self.agent_row, self.agent_col)

        obs = self._get_obs()
        info = {"episode": self._episode_count}

        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    def step(self, action):
        reward = self.R_step           # base per-step cost
        terminated = False
        truncated = False
        info: dict = {}

        self.steps_taken += 1

        # ── trapped in a black hole — skip this turn ──
        if self.trapped_steps > 0:
            self.trapped_steps -= 1
            reward += self.R_black_hole
            if self.steps_taken >= self.max_steps:
                truncated = True
                reward += self.R_timeout
            obs = self._get_obs()
            if self.render_mode == "human":
                self._render_frame()
            return obs, reward, terminated, truncated, info

        action = int(action)
        is_move = action < 4
        direction = action if is_move else action - 4
        dr, dc = DIR_DELTA[direction]
        target_r = self.agent_row + dr
        target_c = self.agent_col + dc
        in_bounds = 0 <= target_r < self.grid_height and 0 <= target_c < self.grid_width

        if is_move:
            # ───── MOVE (reveal) ─────
            if not in_bounds:
                # Wall collision
                reward += self.R_wall_hit
                info["event"] = "wall_hit"
                self.stalled_steps += 1

            elif self._flagged[target_r, target_c]:
                # Can't walk onto a flagged cell (unflag it first)
                reward += self.R_wall_hit
                info["event"] = "move_blocked_by_flag"
                self.stalled_steps += 1

            else:
                # Actually move
                self.agent_row = target_r
                self.agent_col = target_c

                if self._mine_map[target_r, target_c]:
                    # ── MINE HIT ──
                    self.stalled_steps = 0
                    self._revealed[target_r, target_c] = True
                    self._reveal_all_mines()
                    reward += self.R_mine_hit
                    terminated = True
                    info["event"] = "mine_hit"

                elif self._cell_types[target_r, target_c] == GOAL_CELL:
                    # ── GOAL REACHED ──
                    self.stalled_steps = 0
                    reward += self.R_goal
                    terminated = True
                    info["event"] = "goal_reached"
                    info["won"] = True

                elif self._visited[target_r, target_c]:
                    # ── REVISIT ──
                    self.stalled_steps += 1
                    reward += self.R_revisit
                    info["event"] = "revisit"

                else:
                    # First visit to this cell
                    self.stalled_steps = 0
                    self._visited[target_r, target_c] = True

                    if (self._black_hole_map is not None
                            and self._black_hole_map[target_r, target_c]):
                        # ── BLACK HOLE ──
                        self._revealed[target_r, target_c] = True
                        self.trapped_steps = self.black_hole_trap_steps
                        reward += self.R_black_hole
                        info["event"] = "black_hole_trapped"
                    else:
                        # ── SAFE STEP ──
                        reward += self.R_safe_step
                        self._flood_fill(target_r, target_c)
                        info["event"] = "safe_step"
        else:
            # ───── FLAG (toggle) ─────
            self.stalled_steps += 1
            if not in_bounds:
                reward += self.R_invalid_flag
                info["event"] = "flag_out_of_bounds"

            elif self._revealed[target_r, target_c]:
                reward += self.R_invalid_flag
                info["event"] = "flag_on_revealed"

            else:
                was_flagged = bool(self._flagged[target_r, target_c])
                self._flagged[target_r, target_c] = not was_flagged

                if not was_flagged:
                    # Just placed a flag
                    if self._mine_map[target_r, target_c]:
                        reward += self.R_correct_flag
                        info["event"] = "correct_flag"
                    else:
                        reward += self.R_wrong_flag
                        info["event"] = "wrong_flag"
                else:
                    # Removed a flag
                    info["event"] = "unflagged"

        # ── timeout ──
        if not terminated and self.steps_taken >= self.max_steps:
            truncated = True
            reward += self.R_timeout
            info["event"] = "timeout"

        obs = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, info

    # ────────────────────── rendering ──────────────────────

    def get_valid_actions(self) -> list[int]:
        """
        Returns a list of action IDs currently valid from the agent's position.
        Excludes moving into walls, moving into flags, and flagging already-revealed cells.
        """
        valid = []
        for a in range(self.action_space.n):
            is_move = a < 4
            direction = a if is_move else a - 4
            dr, dc = DIR_DELTA[direction]
            tr = self.agent_row + dr
            tc = self.agent_col + dc
            in_bounds = 0 <= tr < self.grid_height and 0 <= tc < self.grid_width
            if not in_bounds:
                continue
            if is_move:
                if not self._flagged[tr, tc]:
                    valid.append(a)
            else:
                if not self._revealed[tr, tc]:
                    valid.append(a)
        return valid

    def render(self):
        """Public render method required by gymnasium."""
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame_rgb()
        return None

    def _render_frame_rgb(self) -> np.ndarray:
        """Return an RGB numpy array of the current frame (for rgb_array mode)."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if self._font is None:
            self._font = pygame.font.SysFont("consolas", self.cell_px // 3)
            self._hud_font = pygame.font.SysFont("consolas", 16)
        px = self.cell_px
        win_w = self.grid_width * px
        win_h = self.grid_height * px + self._hud_height
        surf = pygame.Surface((win_w, win_h))
        self._draw_to_surface(surf)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
        )

    def _init_pygame(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(
            f"Grid Nav Minesweeper  —  Env {self.difficulty}"
        )
        win_w = self.grid_width * self.cell_px
        win_h = self.grid_height * self.cell_px + self._hud_height
        self._screen = pygame.display.set_mode((win_w, win_h))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", self.cell_px // 3)
        self._hud_font = pygame.font.SysFont("consolas", 16)

    def _render_frame(self):
        if self._screen is None:
            self._init_pygame()

        self._draw_to_surface(self._screen)
        pygame.display.flip()
        self._clock.tick(self.fps)

    def _draw_to_surface(self, surface):
        """Draw the full board onto the given pygame surface."""
        surface.fill(COLORS["bg"])
        px = self.cell_px

        for r in range(self.grid_height):
            for c in range(self.grid_width):
                rect = pygame.Rect(c * px, r * px, px, px)
                ct = int(self._cell_types[r, c])

                # ── background colour ──
                if self._flagged[r, c]:
                    bg = COLORS["flag"]
                elif not self._revealed[r, c]:
                    bg = (COLORS["black_hole"] if ct == BLACK_HOLE_CELL
                          else COLORS["unrevealed"])
                elif ct == SIDEWALK:
                    bg = COLORS["sidewalk"]
                elif ct == GOAL_CELL:
                    bg = COLORS["goal"]
                elif self._mine_map[r, c]:
                    bg = COLORS["mine"]
                else:
                    bg = COLORS["revealed"]

                pygame.draw.rect(surface, bg, rect)
                pygame.draw.rect(surface, COLORS["grid_line"], rect, 1)

                # ── cell label ──
                if self._flagged[r, c]:
                    txt = self._font.render("F", True, COLORS["text"])
                    surface.blit(txt, txt.get_rect(center=rect.center))
                elif self._revealed[r, c]:
                    if self._mine_map[r, c]:
                        txt = self._font.render("*", True, (255, 255, 255))
                        surface.blit(txt, txt.get_rect(center=rect.center))
                    else:
                        n = int(self._adj_counts[r, c])
                        if n > 0:
                            color = COLORS["num"].get(n, COLORS["text"])
                            txt = self._font.render(str(n), True, color)
                            surface.blit(txt, txt.get_rect(center=rect.center))
                else:
                    if ct == GOAL_CELL:
                        txt = self._font.render("G", True, (255, 255, 255))
                        surface.blit(txt, txt.get_rect(center=rect.center))

                # ── agent marker ──
                if r == self.agent_row and c == self.agent_col:
                    sz = px // 2
                    a_rect = pygame.Rect(0, 0, sz, sz)
                    a_rect.center = rect.center
                    pygame.draw.rect(surface, COLORS["agent"], a_rect)
                    pygame.draw.rect(surface, (255, 255, 255), a_rect, 2)

        # ── HUD bar ──
        hud_y = self.grid_height * px + 5
        parts = [
            f"Ep {self._episode_count}",
            f"Steps {self.steps_taken}/{self.max_steps}",
            f"Pos ({self.agent_row},{self.agent_col})",
        ]
        if self.trapped_steps > 0:
            parts.append(f"TRAPPED {self.trapped_steps}")
        hud_str = "  |  ".join(parts)
        txt = self._hud_font.render(hud_str, True, COLORS["hud_text"])
        surface.blit(txt, (10, hud_y))

    def close(self):
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
