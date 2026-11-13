"""
Quick smoke-test for the GridNavMinesweeper environments.

Runs a few episodes with random actions and prints events.
Pass --render to open the pygame window.
Usage:
    python scripts/test_grid_nav.py
    python scripts/test_grid_nav.py --render
    python scripts/test_grid_nav.py --env GridNavMinesweeper-v2 --render
"""

import argparse
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import environment  # triggers registration
from environment.grid_nav_env import ACTION_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="GridNavMinesweeper-v1")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--check", action="store_true",
                        help="Run gymnasium env checker first")
    args = parser.parse_args()

    render_mode = "human" if args.render else None

    env = gym.make(args.env, render_mode=render_mode,
                   seed=args.seed, fps=args.fps)

    # Optional: run the gymnasium environment checker
    if args.check:
        print("Running gymnasium env checker …")
        check_env(env.unwrapped)
        print("✓ Environment passed check_env!\n")

    wins = 0
    mine_hits = 0
    timeouts = 0

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        print(f"\n{'═'*60}")
        print(f"Episode {ep}  |  Agent starts at {obs['agent_pos']}")
        print(f"Board shape: {obs['board'].shape}  |  "
              f"Steps budget: {obs['steps_remaining'][0]}")
        print(f"{'═'*60}")

        done = False
        step_num = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_num += 1

            event = info.get("event", "")
            if event or done:
                print(f"  step {step_num:3d}  {ACTION_NAMES[action]:>22s}  "
                      f"r={reward:+7.2f}  pos=({obs['agent_pos'][0]},{obs['agent_pos'][1]})  "
                      f"event={event}")

        # Tally outcome
        if info.get("won"):
            wins += 1
            outcome = "🏆 WIN"
        elif info.get("event") == "mine_hit":
            mine_hits += 1
            outcome = "💥 MINE HIT"
        else:
            timeouts += 1
            outcome = "⏰ TIMEOUT"
        print(f"  → {outcome}  (steps: {step_num})")

    print(f"\n{'─'*60}")
    print(f"Summary over {args.episodes} episodes:")
    print(f"  Wins: {wins}  |  Mine hits: {mine_hits}  |  Timeouts: {timeouts}")

    env.close()


if __name__ == "__main__":
    main()

