"""Stable-Baselines3 logic for the CartPole learning workbench."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Optional

# The dummy driver is useful only on headless Linux. Forcing it on macOS can
# crash Pygame/Cocoa while Gymnasium renders the initial RGB frame.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium
import numpy as np
import torch
from torch.nn import functional as torch_functional
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from cartpole_render import CartPoleRenderer


ACTIVATIONS: dict[str, type[torch.nn.Module]] = {
    "ReLU": torch.nn.ReLU,
    "Tanh": torch.nn.Tanh,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU": torch.nn.ELU,
}
OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RMSprop": torch.optim.RMSprop,
    "SGD": torch.optim.SGD,
}
ALGORITHMS = ("DQN", "DDQN")


def double_dqn_next_values(
    online_q_values: torch.Tensor, target_q_values: torch.Tensor
) -> torch.Tensor:
    """Select actions online and evaluate exactly those actions with the target net."""
    next_actions = online_q_values.argmax(dim=1, keepdim=True)
    return target_q_values.gather(1, next_actions)


class DDQN(DQN):
    """Double DQN using SB3's DQN infrastructure and replay mechanics."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            with torch.no_grad():
                # Double DQN: online network selects, target network evaluates.
                next_q_values = double_dqn_next_values(
                    self.q_net(replay_data.next_observations),
                    self.q_net_target(replay_data.next_observations),
                )
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
            current_q_values = self.q_net(replay_data.observations).gather(
                1, replay_data.actions.long()
            )
            loss = torch_functional.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(float(loss.item()))
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def make_cartpole_env() -> gymnasium.Env:
    """Create one independent, officially rendered CartPole instance."""
    return gymnasium.make("CartPole-v1", render_mode="rgb_array")


@dataclass(frozen=True)
class DQNConfig:
    algorithm: str = "DQN"
    total_timesteps: int = 100_000
    learning_rate: float = 0.0005
    buffer_size: int = 200_000
    learning_starts: int = 1_000
    batch_size: int = 128
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.20
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0
    seed: Optional[int] = 42
    net_arch: tuple[int, ...] = (64, 64)
    activation: str = "ReLU"
    optimizer: str = "Adam"
    optimizer_eps: float = 1e-5
    optimizer_weight_decay: float = 0.0

    def validate(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Algorithmus muss einer von {ALGORITHMS} sein.")
        positive_ints = {
            "Trainingsschritte": self.total_timesteps,
            "Replay-Buffer-Größe": self.buffer_size,
            "Batch-Größe": self.batch_size,
            "Trainingsfrequenz": self.train_freq,
            "Target-Update-Intervall": self.target_update_interval,
        }
        for label, value in positive_ints.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} muss eine positive Ganzzahl sein.")
        if not isinstance(self.learning_starts, int) or self.learning_starts < 0:
            raise ValueError("Learning Starts muss eine nichtnegative Ganzzahl sein.")
        if not isinstance(self.gradient_steps, int) or self.gradient_steps == 0 or self.gradient_steps < -1:
            raise ValueError("Gradient Steps muss -1 oder eine positive Ganzzahl sein.")
        if self.batch_size > self.buffer_size:
            raise ValueError("Batch-Größe darf den Replay Buffer nicht überschreiten.")
        if self.learning_starts > self.total_timesteps:
            raise ValueError("Learning Starts darf die Trainingsschritte nicht überschreiten.")
        if self.learning_rate <= 0:
            raise ValueError("Lernrate muss größer als 0 sein.")
        if not 0 < self.gamma <= 1:
            raise ValueError("Gamma muss größer als 0 und höchstens 1 sein.")
        if not 0 < self.tau <= 1:
            raise ValueError("Tau muss größer als 0 und höchstens 1 sein.")
        for label, value in (
            ("Exploration Fraction", self.exploration_fraction),
            ("Initial Epsilon", self.exploration_initial_eps),
            ("Final Epsilon", self.exploration_final_eps),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{label} muss zwischen 0 und 1 liegen.")
        if self.exploration_final_eps > self.exploration_initial_eps:
            raise ValueError("Final Epsilon darf Initial Epsilon nicht überschreiten.")
        if self.max_grad_norm <= 0 or self.optimizer_eps <= 0 or self.optimizer_weight_decay < 0:
            raise ValueError("Gradient Norm und Optimizer Epsilon müssen positiv, Weight Decay nichtnegativ sein.")
        if not self.net_arch or any(not isinstance(size, int) or size <= 0 for size in self.net_arch):
            raise ValueError("Netzarchitektur benötigt positive Layer-Größen, z. B. 64,64.")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"Aktivierungsfunktion muss eine von {tuple(ACTIVATIONS)} sein.")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"Optimizer muss einer von {tuple(OPTIMIZERS)} sein.")

    def model_kwargs(self) -> dict[str, Any]:
        self.validate()
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_update_interval": self.target_update_interval,
            "exploration_fraction": self.exploration_fraction,
            "exploration_initial_eps": self.exploration_initial_eps,
            "exploration_final_eps": self.exploration_final_eps,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "policy_kwargs": {
                "net_arch": list(self.net_arch),
                "activation_fn": ACTIVATIONS[self.activation],
                "optimizer_class": OPTIMIZERS[self.optimizer],
                "optimizer_kwargs": {
                    "eps": self.optimizer_eps,
                    "weight_decay": self.optimizer_weight_decay,
                },
            },
            "verbose": 0,
            "device": "auto",
        }

    def model_signature(self) -> tuple[tuple[str, Any], ...]:
        """Return parameters that define or change the learned DQN itself."""
        return tuple(
            (name, value)
            for name, value in asdict(self).items()
            if name != "total_timesteps"
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DQNConfig":
        data = dict(values)
        data["net_arch"] = tuple(int(value) for value in data["net_arch"])
        config = cls(**data)
        config.validate()
        return config


@dataclass(frozen=True)
class EpisodeMetric:
    episode: int
    reward: float
    length: int
    success: bool
    timesteps: int
    exploration_rate: float


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    reward_std: float
    mean_length: float
    success_rate: float
    rewards: tuple[float, ...]
    lengths: tuple[int, ...]


def evaluate_policy(
    model: DQN, episodes: int, seed: Optional[int] = None
) -> EvaluationResult:
    """Evaluate deterministically in a separate headless environment."""
    env = gymnasium.make("CartPole-v1")
    rewards: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(episodes):
            episode_seed = None if seed is None else seed + episode
            observation, _ = env.reset(seed=episode_seed)
            total_reward = 0.0
            steps = 0
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(int(action))
                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)
            rewards.append(total_reward)
            lengths.append(steps)
    finally:
        env.close()
    return EvaluationResult(
        episodes=episodes,
        mean_reward=mean(rewards),
        reward_std=pstdev(rewards),
        mean_length=mean(lengths),
        success_rate=sum(length >= 500 for length in lengths) / episodes,
        rewards=tuple(rewards),
        lengths=tuple(lengths),
    )


@dataclass(frozen=True)
class AnimationStep:
    frame: np.ndarray
    observation: tuple[float, float, float, float]
    action: int
    reward: float
    step: int
    terminated: bool
    truncated: bool


class WorkbenchCallback(BaseCallback):
    """Collect episode metrics and support cooperative cancellation."""

    def __init__(
        self,
        stop_event: threading.Event,
        output_queue: Optional[queue.Queue] = None,
        episode_offset: int = 0,
        series_name: Optional[tuple[str, int]] = None,
        progress_offset: int = 0,
        comparison_total: Optional[int] = None,
        evaluation_interval: int = 0,
        evaluation_episodes: int = 10,
        evaluation_seed: Optional[int] = None,
        best_model_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None,
    ) -> None:
        super().__init__(verbose=0)
        self.stop_event = stop_event
        self.output_queue = output_queue
        self.episode_offset = episode_offset
        self.series_name = series_name
        self.progress_offset = progress_offset
        self.comparison_total = comparison_total
        self.evaluation_interval = evaluation_interval
        self.evaluation_episodes = evaluation_episodes
        self.evaluation_seed = evaluation_seed
        self.best_model_callback = best_model_callback
        self.episode_metrics: list[EpisodeMetric] = []
        self._reward = 0.0
        self._length = 0
        self._training_start = 0
        self._next_evaluation = evaluation_interval

    def _on_training_start(self) -> None:
        self._training_start = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", [0.0]))
        dones = np.asarray(self.locals.get("dones", [False]))
        infos = self.locals.get("infos", [{}])
        self._reward += float(rewards[0])
        self._length += 1
        if bool(dones[0]):
            info = infos[0] if infos else {}
            time_limit = bool(info.get("TimeLimit.truncated", False)) or self._length >= 500
            metric = EpisodeMetric(
                episode=self.episode_offset + len(self.episode_metrics) + 1,
                reward=self._reward,
                length=self._length,
                success=time_limit,
                timesteps=int(self.num_timesteps),
                exploration_rate=float(getattr(self.model, "exploration_rate", 0.0)),
            )
            self.episode_metrics.append(metric)
            if self.output_queue is not None:
                if self.series_name is None:
                    self.output_queue.put(("episode", metric))
                else:
                    self.output_queue.put(("comparison_episode", (self.series_name, metric)))
            self._reward = 0.0
            self._length = 0
        elapsed = int(self.num_timesteps) - self._training_start
        if self.evaluation_interval > 0 and elapsed >= self._next_evaluation:
            result = evaluate_policy(self.model, self.evaluation_episodes, self.evaluation_seed)
            episode = self.episode_offset + len(self.episode_metrics)
            timesteps = int(self.num_timesteps)
            if self.best_model_callback is not None:
                self.best_model_callback(episode, timesteps, result)
            if self.output_queue is not None:
                if self.series_name is None:
                    self.output_queue.put(("automatic_evaluation", (episode, timesteps, result)))
                else:
                    self.output_queue.put(
                        ("comparison_evaluation", (self.series_name, episode, timesteps, result))
                    )
            self._next_evaluation += self.evaluation_interval
        if self.output_queue is not None and elapsed % 100 == 0:
            if self.series_name is None:
                self.output_queue.put(("progress", elapsed))
            else:
                self.output_queue.put(
                    (
                        "comparison_progress",
                        (self.series_name, self.progress_offset + elapsed, self.comparison_total),
                    )
                )
        return not self.stop_event.is_set()


class CartPoleWorkbench:
    """Own the SB3 model and isolate training, evaluation, and animation envs."""

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig()
        self.config.validate()
        self.training_env: Optional[gymnasium.Env] = None
        self.model: Optional[DQN] = None
        self.training_history: list[EpisodeMetric] = []
        self.total_trained_timesteps = 0

    def create_model(self) -> DQN:
        self.close_training_env()
        self.training_env = make_cartpole_env()
        model_class = DQN if self.config.algorithm == "DQN" else DDQN
        self.model = model_class("MlpPolicy", self.training_env, **self.config.model_kwargs())
        self.training_history.clear()
        self.total_trained_timesteps = 0
        return self.model

    def train(
        self,
        stop_event: Optional[threading.Event] = None,
        output_queue: Optional[queue.Queue] = None,
        reset_num_timesteps: bool = False,
        series_name: Optional[tuple[str, int]] = None,
        progress_offset: int = 0,
        comparison_total: Optional[int] = None,
        evaluation_interval: int = 0,
        evaluation_episodes: int = 10,
        evaluation_seed: Optional[int] = None,
        best_model_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None,
    ) -> list[EpisodeMetric]:
        if self.model is None:
            self.create_model()
            reset_num_timesteps = True
        callback = WorkbenchCallback(
            stop_event or threading.Event(), output_queue, len(self.training_history),
            series_name, progress_offset, comparison_total, evaluation_interval,
            evaluation_episodes, evaluation_seed, best_model_callback,
        )
        start = int(self.model.num_timesteps)
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        self.total_trained_timesteps += int(self.model.num_timesteps) - (0 if reset_num_timesteps else start)
        self.training_history.extend(callback.episode_metrics)
        return callback.episode_metrics

    def evaluate(self, episodes: int, seed: Optional[int] = None) -> EvaluationResult:
        if self.model is None:
            raise RuntimeError("Vor der Evaluation muss ein Modell trainiert oder geladen werden.")
        if not isinstance(episodes, int) or episodes <= 0:
            raise ValueError("Evaluations-Episoden muss eine positive Ganzzahl sein.")
        return evaluate_policy(self.model, episodes, seed)

    def animation_episode(self, seed: Optional[int] = None) -> list[AnimationStep]:
        if self.model is None:
            raise RuntimeError("Vor der Animation muss ein Modell trainiert oder geladen werden.")
        renderer = CartPoleRenderer()
        steps: list[AnimationStep] = []
        try:
            observation, _info, frame = renderer.reset(seed)
            done = False
            index = 0
            while not done:
                action, _ = self.model.predict(observation, deterministic=True)
                next_observation, reward, terminated, truncated, _info, next_frame = renderer.step(int(action))
                index += 1
                steps.append(AnimationStep(
                    frame=np.asarray(frame).copy(),
                    observation=tuple(float(value) for value in observation),
                    action=int(action), reward=float(reward), step=index,
                    terminated=bool(terminated), truncated=bool(truncated),
                ))
                observation = next_observation
                frame = next_frame
                done = bool(terminated or truncated)
        finally:
            renderer.close()
        return steps

    def save(self, path: str | Path) -> tuple[Path, Path, Path]:
        if self.model is None:
            raise RuntimeError("Es gibt noch kein Modell zum Speichern.")
        base = Path(path)
        if base.suffix == ".zip":
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        model_path = base.with_suffix(".zip")
        replay_path = base.with_name(base.name + "_replay.pkl")
        metadata_path = base.with_name(base.name + "_metadata.json")
        self.model.save(base)
        self.model.save_replay_buffer(replay_path)
        metadata = {
            "format_version": 1,
            "environment": "CartPole-v1",
            "algorithm": self.config.algorithm,
            "config": asdict(self.config),
            "total_trained_timesteps": self.total_trained_timesteps,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return model_path, replay_path, metadata_path

    @classmethod
    def load(cls, path: str | Path) -> "CartPoleWorkbench":
        base = Path(path)
        if base.suffix == ".zip":
            base = base.with_suffix("")
        model_path = base.with_suffix(".zip")
        replay_path = base.with_name(base.name + "_replay.pkl")
        metadata_path = base.with_name(base.name + "_metadata.json")
        if not all(item.exists() for item in (model_path, replay_path, metadata_path)):
            raise ValueError("Modell, Replay Buffer oder Metadaten fehlen.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        algorithm = metadata.get("algorithm")
        if metadata.get("format_version") != 1 or metadata.get("environment") != "CartPole-v1" or algorithm not in ALGORITHMS:
            raise ValueError("Die Modelldateien sind nicht mit CartPole DQN kompatibel.")
        config = DQNConfig.from_dict(metadata["config"])
        workbench = cls(config)
        workbench.training_env = make_cartpole_env()
        model_class = DQN if algorithm == "DQN" else DDQN
        workbench.model = model_class.load(model_path, env=workbench.training_env, device="auto", force_reset=True)
        workbench.model.load_replay_buffer(replay_path)
        workbench.total_trained_timesteps = int(metadata.get("total_trained_timesteps", workbench.model.num_timesteps))
        return workbench

    def close_training_env(self) -> None:
        if self.training_env is not None:
            self.training_env.close()
            self.training_env = None

    def close(self) -> None:
        self.close_training_env()
