"""Train a new PPO neural network to land in Gymnasium LunarLander-v3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from reward_wrapper import RewardShapingWrapper

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"
VALID_MODES = ("standard", "time_penalty", "hover_bonus")


def model_path_for(mode: str) -> Path:
    """Return the saved model path, including Stable-Baselines3's .zip suffix."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")
    return MODELS_DIR / f"lander_{mode}.zip"


def make_training_env(mode: str, render_mode: str | None = None):
    """Create the game and optionally put our reward wrapper around it."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")

    env = gym.make("LunarLander-v3", render_mode=render_mode)
    if mode != "standard":
        env = RewardShapingWrapper(env, mode=mode)
    return Monitor(env)


class TrainingProgressCallback(BaseCallback):
    """Report occasional progress updates to a callback such as the website UI."""

    def __init__(self, total_timesteps: int, update_callback: Callable | None = None):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.update_callback = update_callback

    def _on_step(self) -> bool:
        # Avoid updating the UI at every single simulation frame.
        if self.n_calls % 2000 == 0 or self.num_timesteps >= self.total_timesteps:
            mean_reward = None
            if self.model.ep_info_buffer:
                recent = [episode["r"] for episode in self.model.ep_info_buffer]
                if recent:
                    mean_reward = float(np.mean(recent))
            if self.update_callback is not None:
                self.update_callback(self.num_timesteps, self.total_timesteps, mean_reward)
        return True


def train_agent(
    total_timesteps: int = 200_000,
    reward_mode: str = "standard",
    update_callback: Callable | None = None,
) -> Path:
    """Train from fresh weights and save a separate model for each reward mode."""
    if total_timesteps < 1_000:
        raise ValueError("Please use at least 1,000 training steps.")
    if reward_mode not in VALID_MODES:
        raise ValueError(f"reward_mode must be one of {VALID_MODES}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_training_env(reward_mode)

    try:
        # MlpPolicy creates a small neural network (two hidden layers by default).
        # PPO starts with new, randomly initialized parameters for this run.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=42,
            n_steps=1024,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.99,
            tensorboard_log=str(LOGS_DIR),
            device="auto",
        )
        callback = TrainingProgressCallback(total_timesteps, update_callback)
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
        path_without_extension = MODELS_DIR / f"lander_{reward_mode}"
        model.save(str(path_without_extension))
        return model_path_for(reward_mode)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train your own LunarLander AI.")
    parser.add_argument("--mode", choices=VALID_MODES, default="standard", help="Which reward rule to train with")
    parser.add_argument("--steps", type=int, default=200_000, help="Number of training frames, e.g. 200000")
    args = parser.parse_args()

    print(f"Training from fresh weights: mode={args.mode}, steps={args.steps:,}")
    print("This may take a while. The time depends on your computer.")
    path = train_agent(total_timesteps=args.steps, reward_mode=args.mode)
    print(f"Training complete. Model saved to: {path}")
    print(f"Now run: python evaluate.py --mode {args.mode} --episodes 10")


if __name__ == "__main__":
    main()
