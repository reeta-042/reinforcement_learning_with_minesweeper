from gymnasium.envs.registration import register
from .minesweeper_env import MinesweeperEnv
from .grid_nav_env import GridNavEnv

# ── Original minesweeper (reveal/flag all cells) ──
register(
    id='Minesweeper-v0',
    entry_point='environment.minesweeper_env:MinesweeperEnv',
)

# ── Grid navigation environments (navigate START → GOAL avoiding mines) ──
register(
    id='GridNavMinesweeper-v1',
    entry_point='environment.grid_nav_env:GridNavEnv',
    kwargs={
        'difficulty': 1,
        'grid_size': (8, 6),
        'num_mines': 6,
    },
)

register(
    id='GridNavMinesweeper-v2',
    entry_point='environment.grid_nav_env:GridNavEnv',
    kwargs={
        'difficulty': 2,
        'grid_size': (8, 6),
        'num_mines': 6,
    },
)

register(
    id='GridNavMinesweeper-v3',
    entry_point='environment.grid_nav_env:GridNavEnv',
    kwargs={
        'difficulty': 3,
        'grid_size': (8, 6),
        'num_mines': 6,
    },
)

register(
    id='GridNavMinesweeper-v4',
    entry_point='environment.grid_nav_env:GridNavEnv',
    kwargs={
        'difficulty': 4,
        'grid_size': (12, 10),
        'num_mines': 15,
        'num_black_holes': 4,
    },
)

__all__ = ['MinesweeperEnv', 'GridNavEnv']
