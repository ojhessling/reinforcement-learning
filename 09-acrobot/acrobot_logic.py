"""PyTorch/SB3 learning logic for the Acrobot workbench."""

from __future__ import annotations

import json
import math
import queue
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterator, Optional

import gymnasium
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn
from torch.nn import functional as functional


ALGORITHMS = (
    "DDQN", "Noisy DDQN", "PER DDQN", "Dueling DDQN",
    "Multi-Step DDQN", "C51 DDQN", "Rainbow DDQN",
)
NOISY_ALGORITHMS = frozenset({"Noisy DDQN", "Rainbow DDQN"})
PER_ALGORITHMS = frozenset({"PER DDQN", "Rainbow DDQN"})
DUELING_ALGORITHMS = frozenset({"Dueling DDQN", "Rainbow DDQN"})
MULTISTEP_ALGORITHMS = frozenset({"Multi-Step DDQN", "Rainbow DDQN"})
DISTRIBUTIONAL_ALGORITHMS = frozenset({"C51 DDQN", "Rainbow DDQN"})
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
}
OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RMSprop": torch.optim.RMSprop,
}


def make_acrobot_env(render_mode: Optional[str] = "rgb_array") -> gymnasium.Env:
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    return gymnasium.make("Acrobot-v1", **kwargs)


def double_dqn_next_values(
    online_q_values: torch.Tensor, target_q_values: torch.Tensor
) -> torch.Tensor:
    actions = online_q_values.argmax(dim=1, keepdim=True)
    return target_q_values.gather(1, actions)


def project_categorical(
    next_probs: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
    discounts: torch.Tensor | float, atoms: torch.Tensor,
) -> torch.Tensor:
    """Categorical projection (Bellemare et al. 2017) of the Bellman-updated
    return distribution onto the fixed atom support `atoms`."""
    n_atoms = atoms.numel()
    v_min, v_max = float(atoms[0]), float(atoms[-1])
    delta_z = (v_max - v_min) / (n_atoms - 1)
    if not torch.is_tensor(discounts):
        discounts = torch.full_like(rewards, float(discounts))
    projected = (rewards + (1 - dones) * discounts * atoms.view(1, -1)).clamp(v_min, v_max)
    b = (projected - v_min) / delta_z
    lower = b.floor().long().clamp(0, n_atoms - 1)
    upper = b.ceil().long().clamp(0, n_atoms - 1)
    equal = lower == upper
    # Wenn b exakt auf ein Atom fällt, ist lower == upper: die Standardformel
    # würde dann beiden Gewichten 0 zuweisen und Wahrscheinlichkeitsmasse
    # verlieren. In diesem Fall geht die volle Masse an genau dieses Atom.
    weight_lower = torch.where(equal, torch.ones_like(b), upper.float() - b)
    weight_upper = torch.where(equal, torch.zeros_like(b), b - lower.float())
    target = torch.zeros_like(next_probs)
    target.scatter_add_(1, lower, next_probs * weight_lower)
    target.scatter_add_(1, upper, next_probs * weight_upper)
    return target


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer from Fortunato et al."""

    def __init__(self, in_features: int, out_features: int, sigma_0: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_0 = sigma_0
        self.noise_enabled = True
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1 / np.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_0 / np.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.sigma_0 / np.sqrt(self.out_features))

    @staticmethod
    def _scaled_noise(size: int, device: torch.device) -> torch.Tensor:
        values = torch.randn(size, device=device)
        return values.sign() * values.abs().sqrt()

    def reset_noise(self) -> None:
        epsilon_in = self._scaled_noise(self.in_features, self.weight_mu.device)
        epsilon_out = self._scaled_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.noise_enabled:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return functional.linear(values, weight, bias)


def _make_linear(in_features: int, out_features: int, noisy: bool, sigma_0: float) -> nn.Module:
    if noisy:
        return NoisyLinear(in_features, out_features, sigma_0)
    return nn.Linear(in_features, out_features)


class RainbowCapableQNetwork(QNetwork):
    """Q-Netzwerk, dessen Bausteine unabhängig voneinander aktivierbar sind.

    `noisy` ersetzt alle linearen Schichten durch NoisyLinear-Schichten,
    `dueling` ersetzt den gemeinsamen Kopf durch getrennte Value-/
    Advantage-Streams, `distributional` ersetzt den skalaren Q-Wert durch
    eine kategoriale Rückgabeverteilung über `n_atoms` feste Atome. Alle drei
    Flags sind unabhängig kombinierbar, sodass Rainbow DDQN keinen Baustein
    dupliziert.
    """

    def __init__(
        self, *args: Any, noisy: bool = False, dueling: bool = False, distributional: bool = False,
        sigma_0: float = 0.5, value_arch: list[int] | None = None, advantage_arch: list[int] | None = None,
        n_atoms: int = 51, v_min: float = -500.0, v_max: float = 0.0, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.noisy, self.dueling, self.distributional = noisy, dueling, distributional
        self.sigma_0 = sigma_0
        self.n_atoms = n_atoms if distributional else 1
        self.action_dim = int(self.action_space.n)
        del self.q_net
        input_dim = self.features_dim
        feature_layers: list[nn.Module] = []
        for size in self.net_arch:
            feature_layers.extend((_make_linear(input_dim, size, noisy, sigma_0), self.activation_fn()))
            input_dim = size
        self.feature_net = nn.Sequential(*feature_layers)
        value_out, advantage_out = self.n_atoms, self.action_dim * self.n_atoms
        if dueling:
            self.value_stream = self._stream(input_dim, value_arch or [64], value_out)
            self.advantage_stream = self._stream(input_dim, advantage_arch or [64], advantage_out)
        else:
            self.output_layer = _make_linear(input_dim, advantage_out, noisy, sigma_0)
        if distributional:
            self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))

    def _stream(self, input_dim: int, architecture: list[int], output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for size in architecture:
            layers.extend((_make_linear(input_dim, size, self.noisy, self.sigma_0), self.activation_fn()))
            input_dim = size
        layers.append(_make_linear(input_dim, output_dim, self.noisy, self.sigma_0))
        return nn.Sequential(*layers)

    def _logits(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(observation, self.features_extractor)
        shared = self.feature_net(features)
        if self.dueling:
            value = self.value_stream(shared).view(-1, 1, self.n_atoms)
            advantage = self.advantage_stream(shared).view(-1, self.action_dim, self.n_atoms)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.output_layer(shared).view(-1, self.action_dim, self.n_atoms)

    def distribution(self, observation: torch.Tensor) -> torch.Tensor:
        """Wahrscheinlichkeit je Action und Atom, Shape (batch, actions, n_atoms)."""
        return functional.softmax(self._logits(observation), dim=-1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self.distributional:
            probs = self.distribution(observation)
            return torch.sum(probs * self.atoms.view(1, 1, -1), dim=-1)
        return self._logits(observation).squeeze(-1)

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def set_noise_enabled(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.noise_enabled = enabled


class AcrobotDQNPolicy(DQNPolicy):
    """Policy, die Noisy-, Dueling- und C51-Baustein unabhängig kombiniert."""

    def __init__(
        self, *args: Any, noisy: bool = False, dueling: bool = False, distributional: bool = False,
        sigma_0: float = 0.5, value_arch: list[int] | None = None, advantage_arch: list[int] | None = None,
        n_atoms: int = 51, v_min: float = -500.0, v_max: float = 0.0, **kwargs: Any,
    ) -> None:
        self.noisy_flag, self.dueling_flag, self.distributional_flag = noisy, dueling, distributional
        self.sigma_0 = sigma_0
        self.value_arch = value_arch or [64]
        self.advantage_arch = advantage_arch or [64]
        self.n_atoms, self.v_min, self.v_max = n_atoms, v_min, v_max
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> RainbowCapableQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return RainbowCapableQNetwork(
            **net_args, noisy=self.noisy_flag, dueling=self.dueling_flag, distributional=self.distributional_flag,
            sigma_0=self.sigma_0, value_arch=self.value_arch, advantage_arch=self.advantage_arch,
            n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max,
        ).to(self.device)

    def reset_noise(self) -> None:
        self.q_net.reset_noise()
        self.q_net_target.reset_noise()

    def set_noise_enabled(self, enabled: bool) -> None:
        self.q_net.set_noise_enabled(enabled)
        self.q_net_target.set_noise_enabled(enabled)


class PrioritizedReplayBuffer:
    """Small proportional PER wrapper around SB3's replay buffer API."""

    def __init__(self, buffer: Any, alpha: float, epsilon: float) -> None:
        self.buffer = buffer
        self.alpha = alpha
        self.epsilon = epsilon
        self.priorities = np.zeros(buffer.buffer_size, dtype=np.float32)

    def add(self, *args: Any, **kwargs: Any) -> None:
        index = self.buffer.pos
        self.buffer.add(*args, **kwargs)
        maximum = float(self.priorities.max()) if self.priorities.any() else 1.0
        self.priorities[index] = maximum

    def sample(self, batch_size: int, beta: float, env: Any = None) -> tuple[Any, np.ndarray, torch.Tensor]:
        size = self.size()
        scaled = np.power(self.priorities[:size], self.alpha)
        probabilities = scaled / scaled.sum()
        indices = np.random.choice(size, batch_size, p=probabilities)
        samples = self.buffer._get_samples(indices, env=env)
        weights = np.power(size * probabilities[indices], -beta)
        weights /= weights.max()
        return samples, indices, torch.as_tensor(weights, device=samples.rewards.device).reshape(-1, 1)

    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor) -> None:
        values = td_errors.detach().abs().cpu().numpy().reshape(-1) + self.epsilon
        # Ein Index kann mehrfach im Batch vorkommen. Dann verwenden wir den
        # größten neuen TD-Fehler, erlauben aber bewusst auch sinkende
        # Prioritäten gegenüber einem früheren Update.
        for index in np.unique(indices):
            self.priorities[index] = values[indices == index].max()

    def size(self) -> int:
        return self.buffer.size()

    def __getattr__(self, name: str) -> Any:
        # Schützt vor Endlosrekursion beim Unpickling: bevor `self.buffer`
        # existiert, würde `getattr(self.buffer, name)` `__getattr__("buffer")`
        # erneut auslösen.
        if name == "buffer":
            raise AttributeError(name)
        return getattr(self.buffer, name)


class NStepReplayBuffer:
    """n-Schritt-Return-Wrapper um SB3s Replay-Buffer-API.

    Kann sowohl den rohen SB3-Buffer als auch einen bereits vorhandenen
    `PrioritizedReplayBuffer` umschließen (NStep außen, PER innen). Dadurch
    fügt PER weiterhin bei jedem tatsächlichen Insert genau eine Priorität an
    exakt dem Buffer-Slot ein, den dieser Insert belegt hat – unabhängig
    davon, dass ein `add()`-Aufruf hier je nach Fensterstand zu keinem, einem
    oder bei Episodenende zu mehreren tatsächlichen Inserts führt.
    """

    def __init__(self, buffer: Any, n_step: int, gamma: float) -> None:
        self.buffer = buffer
        self.n_step = n_step
        self.gamma = gamma
        self.window: deque = deque()
        self.discounts = np.full(buffer.buffer_size, gamma, dtype=np.float32)

    def add(self, obs: Any, next_obs: Any, action: Any, reward: Any, done: Any, infos: Any) -> None:
        self.window.append((obs, next_obs, action, reward, done, infos))
        done_now = bool(np.asarray(done).reshape(-1)[0])
        if len(self.window) >= self.n_step or done_now:
            self._emit()
        if done_now:
            # Am Episodenende werden auch die verkürzten Restfenster sofort
            # aufgelöst, statt auf nie eintreffende weitere Schritte zu warten.
            while self.window:
                self._emit()

    def _emit(self) -> None:
        first_obs, _, first_action, first_reward, _, _ = self.window[0]
        discount = 1.0
        cumulative = np.zeros_like(np.asarray(first_reward, dtype=np.float64))
        last_next_obs = last_done = last_infos = None
        for _, next_obs, _, reward, done, infos in self.window:
            cumulative = cumulative + discount * np.asarray(reward, dtype=np.float64)
            discount *= self.gamma
            last_next_obs, last_done, last_infos = next_obs, done, infos
            if bool(np.asarray(done).reshape(-1)[0]):
                break
        index = self.buffer.pos
        self.buffer.add(first_obs, last_next_obs, first_action, cumulative.astype(np.float32), last_done, last_infos)
        self.discounts[index] = discount
        self.window.popleft()

    def sample(
        self, batch_size: int, beta: Optional[float] = None, env: Any = None
    ) -> tuple[Any, np.ndarray, torch.Tensor | float]:
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            samples, indices, weights = self.buffer.sample(batch_size, beta, env=env)
        else:
            upper_bound = self.buffer.buffer_size if self.buffer.full else self.buffer.pos
            indices = np.random.randint(0, upper_bound, size=batch_size)
            samples = self.buffer._get_samples(indices, env=env)
            weights = 1.0
        discounts = torch.as_tensor(self.discounts[indices], device=samples.rewards.device).reshape(-1, 1)
        return samples._replace(discounts=discounts), indices, weights

    def size(self) -> int:
        return self.buffer.size()

    def __getattr__(self, name: str) -> Any:
        if name == "buffer":
            raise AttributeError(name)
        return getattr(self.buffer, name)


@dataclass(frozen=True)
class AcrobotConfig:
    algorithm: str = "DDQN"
    total_timesteps: int = 100_000
    learning_rate: float = 6.3e-4
    buffer_size: int = 50_000
    learning_starts: int = 0
    batch_size: int = 128
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 4
    target_update_interval: int = 250
    exploration_fraction: float = 0.12
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.1
    max_grad_norm: float = 10.0
    seed: Optional[int] = 42
    net_arch: tuple[int, ...] = (256, 256)
    activation: str = "ReLU"
    optimizer: str = "Adam"
    optimizer_eps: float = 1e-5
    optimizer_weight_decay: float = 0.0
    noisy_sigma_0: float = 0.5
    per_alpha: float = 0.6
    per_beta_0: float = 0.4
    per_beta_steps: int = 100_000
    per_epsilon: float = 1e-6
    value_arch: tuple[int, ...] = (64,)
    advantage_arch: tuple[int, ...] = (64,)
    multistep_n: int = 3
    c51_atoms: int = 51
    c51_v_min: float = -500.0
    c51_v_max: float = 0.0

    def validate(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Algorithmus muss einer von {ALGORITHMS} sein.")
        for label, value in (
            ("Trainingsschritte N", self.total_timesteps), ("Replay Buffer", self.buffer_size),
            ("Batch-Größe", self.batch_size), ("Trainingsfrequenz", self.train_freq),
            ("Gradientenschritte", self.gradient_steps), ("Target-Intervall", self.target_update_interval),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} muss eine positive Ganzzahl sein.")
        if self.learning_starts < 0 or self.learning_starts > self.total_timesteps:
            raise ValueError("Lernstart muss zwischen 0 und N liegen.")
        if self.batch_size > self.buffer_size:
            raise ValueError("Batch-Größe darf den Replay Buffer nicht überschreiten.")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Lernrate, γ und τ liegen außerhalb ihres gültigen Bereichs.")
        if not 0 <= self.exploration_final_eps <= self.exploration_initial_eps <= 1:
            raise ValueError("Es muss 0 ≤ ε_min ≤ ε₀ ≤ 1 gelten.")
        if not 0 <= self.exploration_fraction <= 1:
            raise ValueError("f_ε muss zwischen 0 und 1 liegen.")
        if self.noisy_sigma_0 <= 0:
            raise ValueError("σ₀ muss positiv sein.")
        if not 0 <= self.per_alpha <= 1 or not 0 <= self.per_beta_0 <= 1:
            raise ValueError("PER α und β₀ müssen zwischen 0 und 1 liegen.")
        if self.per_beta_steps <= 0 or self.per_epsilon <= 0:
            raise ValueError("PER-Annealing und ε_PER müssen positiv sein.")
        if not self.net_arch or not self.value_arch or not self.advantage_arch:
            raise ValueError("Netzarchitekturen dürfen nicht leer sein.")
        if not isinstance(self.multistep_n, int) or self.multistep_n <= 0:
            raise ValueError("Schrittanzahl n_step muss eine positive Ganzzahl sein.")
        if not isinstance(self.c51_atoms, int) or self.c51_atoms < 2:
            raise ValueError("Anzahl Atome n_atoms muss eine Ganzzahl ≥ 2 sein.")
        if self.c51_v_min >= self.c51_v_max:
            raise ValueError("V_min muss kleiner als V_max sein.")

    def model_kwargs(self) -> dict[str, Any]:
        noisy = self.algorithm in NOISY_ALGORITHMS
        dueling = self.algorithm in DUELING_ALGORITHMS
        distributional = self.algorithm in DISTRIBUTIONAL_ALGORITHMS
        policy_kwargs: dict[str, Any] = {
            "net_arch": list(self.net_arch),
            "activation_fn": ACTIVATIONS[self.activation],
            "optimizer_class": OPTIMIZERS[self.optimizer],
            "optimizer_kwargs": {"eps": self.optimizer_eps, "weight_decay": self.optimizer_weight_decay},
            "noisy": noisy, "dueling": dueling, "distributional": distributional, "sigma_0": self.noisy_sigma_0,
            "value_arch": list(self.value_arch), "advantage_arch": list(self.advantage_arch),
            "n_atoms": self.c51_atoms, "v_min": self.c51_v_min, "v_max": self.c51_v_max,
        }
        return {
            "learning_rate": self.learning_rate, "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts, "batch_size": self.batch_size,
            "tau": self.tau, "gamma": self.gamma, "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps, "target_update_interval": self.target_update_interval,
            "exploration_fraction": 0.0 if noisy else self.exploration_fraction,
            "exploration_initial_eps": 0.0 if noisy else self.exploration_initial_eps,
            "exploration_final_eps": 0.0 if noisy else self.exploration_final_eps,
            "max_grad_norm": self.max_grad_norm, "seed": self.seed,
            "policy_kwargs": policy_kwargs, "verbose": 0,
        }

    def signature(self) -> tuple[Any, ...]:
        values = asdict(self)
        values.pop("total_timesteps")
        return tuple(values.items())

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AcrobotConfig":
        data = dict(values)
        for key in ("net_arch", "value_arch", "advantage_arch"):
            data[key] = tuple(data[key])
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


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    reward_std: float
    mean_length: float
    success_rate: float
    mean_steps_to_goal: Optional[float]
    best_steps_to_goal: Optional[int]


def angles_degrees(observation: np.ndarray) -> tuple[float, float]:
    """cos/sin-Observation in die beiden Gelenkwinkel (Grad) zurückrechnen."""
    theta1 = math.degrees(math.atan2(observation[1], observation[0]))
    theta2 = math.degrees(math.atan2(observation[3], observation[2]))
    return theta1, theta2


class AcrobotCallback(BaseCallback):
    def __init__(
        self, stop_event: threading.Event, output: Optional[queue.Queue] = None,
        episode_offset: int = 0, series: Optional[str] = None,
        evaluation_interval: int = 0, evaluation_episodes: int = 10,
        evaluation_seed: Optional[int] = None,
        evaluator: Optional[Callable[[int, int], EvaluationResult]] = None,
        best_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None,
    ) -> None:
        super().__init__(verbose=0)
        self.stop_event, self.output = stop_event, output
        self.episode_offset, self.series = episode_offset, series
        self.evaluation_interval, self.evaluation_episodes = evaluation_interval, evaluation_episodes
        self.evaluation_seed, self.evaluator, self.best_callback = evaluation_seed, evaluator, best_callback
        self.metrics: list[EpisodeMetric] = []
        self.reward = 0.0
        self.length = 0
        self.start_step = 0
        self.next_evaluation = evaluation_interval

    def _on_training_start(self) -> None:
        self.start_step = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        reward = float(np.asarray(self.locals["rewards"])[0])
        done = bool(np.asarray(self.locals["dones"])[0])
        info = self.locals.get("infos", [{}])[0]
        self.reward += reward
        self.length += 1
        if done:
            success = not info.get("TimeLimit.truncated", False)
            metric = EpisodeMetric(
                self.episode_offset + len(self.metrics) + 1, self.reward, self.length,
                success, int(self.num_timesteps),
            )
            self.metrics.append(metric)
            kind = "episode" if self.series is None else "comparison_episode"
            payload: Any = metric if self.series is None else (self.series, metric)
            if self.output is not None:
                self.output.put((kind, payload))
            self.reward, self.length = 0.0, 0
        elapsed = int(self.num_timesteps) - self.start_step
        if self.evaluator and self.evaluation_interval and elapsed >= self.next_evaluation:
            result = self.evaluator(self.evaluation_episodes, self.evaluation_seed or 0)
            episode = self.episode_offset + len(self.metrics)
            if self.best_callback:
                self.best_callback(episode, int(self.num_timesteps), result)
            kind = "evaluation" if self.series is None else "comparison_evaluation"
            payload = (episode, int(self.num_timesteps), result) if self.series is None else (
                self.series, episode, int(self.num_timesteps), result
            )
            if self.output is not None:
                self.output.put((kind, payload))
            self.next_evaluation += self.evaluation_interval
        if self.output is not None and elapsed % 100 == 0:
            self.output.put(("progress", (self.series, elapsed)))
        return not self.stop_event.is_set()


class AcrobotDDQN(DQN):
    def __init__(
        self, *args: Any, variant: str = "DDQN",
        config: Optional[AcrobotConfig] = None, **kwargs: Any,
    ) -> None:
        self.variant = variant
        self.workbench_config = config or AcrobotConfig(algorithm=variant)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        buffer = self.replay_buffer
        if self.variant in PER_ALGORITHMS and not isinstance(buffer, PrioritizedReplayBuffer):
            buffer = PrioritizedReplayBuffer(
                buffer, self.workbench_config.per_alpha, self.workbench_config.per_epsilon,
            )
        if self.variant in MULTISTEP_ALGORITHMS and not isinstance(buffer, NStepReplayBuffer):
            buffer = NStepReplayBuffer(buffer, self.workbench_config.multistep_n, self.gamma)
        self.replay_buffer = buffer

    def _distributional_target(self, replay_data: Any, discounts: torch.Tensor | float) -> torch.Tensor:
        atoms = self.q_net.atoms
        with torch.no_grad():
            next_q_values = self.q_net(replay_data.next_observations)
            best_actions = next_q_values.argmax(dim=1)
            next_probs_all = self.q_net_target.distribution(replay_data.next_observations)
            batch_indices = torch.arange(next_probs_all.size(0), device=next_probs_all.device)
            next_probs = next_probs_all[batch_indices, best_actions]
            return project_categorical(next_probs, replay_data.rewards, replay_data.dones, discounts, atoms)

    def _distributional_loss(
        self, replay_data: Any, discounts: torch.Tensor | float, weights: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = self._distributional_target(replay_data, discounts)
        current_probs_all = self.q_net.distribution(replay_data.observations)
        actions = replay_data.actions.long().reshape(-1)
        batch_indices = torch.arange(current_probs_all.size(0), device=current_probs_all.device)
        current_probs = current_probs_all[batch_indices, actions]
        log_probs = torch.log(current_probs.clamp_min(1e-8))
        cross_entropy = -(target * log_probs).sum(dim=1, keepdim=True)
        weights_tensor = weights if torch.is_tensor(weights) else torch.full_like(cross_entropy, float(weights))
        loss = (cross_entropy * weights_tensor).mean()
        return loss, cross_entropy.detach()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        uses_noisy = self.variant in NOISY_ALGORITHMS
        uses_per = self.variant in PER_ALGORITHMS
        uses_multistep = self.variant in MULTISTEP_ALGORITHMS
        uses_distributional = self.variant in DISTRIBUTIONAL_ALGORITHMS
        losses: list[float] = []
        for _ in range(gradient_steps):
            if uses_noisy:
                self.policy.reset_noise()
            indices: Optional[np.ndarray] = None
            weights: torch.Tensor | float = 1.0
            if uses_per:
                progress = min(1.0, self.num_timesteps / self.workbench_config.per_beta_steps)
                beta = self.workbench_config.per_beta_0 + progress * (1 - self.workbench_config.per_beta_0)
                replay_data, indices, weights = self.replay_buffer.sample(
                    batch_size, beta=beta, env=self._vec_normalize_env
                )
            elif uses_multistep:
                replay_data, _, _ = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            else:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            if uses_distributional:
                loss, td_errors = self._distributional_loss(replay_data, discounts, weights)
            else:
                with torch.no_grad():
                    next_values = double_dqn_next_values(
                        self.q_net(replay_data.next_observations),
                        self.q_net_target(replay_data.next_observations),
                    )
                    targets = replay_data.rewards + (1 - replay_data.dones) * discounts * next_values
                current = self.q_net(replay_data.observations).gather(1, replay_data.actions.long())
                td_errors = targets - current
                loss = (functional.smooth_l1_loss(current, targets, reduction="none") * weights).mean()
            losses.append(float(loss.item()))
            self.policy.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            if indices is not None:
                self.replay_buffer.update_priorities(indices, td_errors)
        self._n_updates += gradient_steps
        self.logger.record("train/loss", np.mean(losses))


class AcrobotWorkbench:
    def __init__(self, config: Optional[AcrobotConfig] = None) -> None:
        self.config = config or AcrobotConfig()
        self.config.validate()
        self.env: Optional[gymnasium.Env] = None
        self.model: Optional[AcrobotDDQN] = None
        self.history: list[EpisodeMetric] = []

    def create_model(self) -> AcrobotDDQN:
        self.close()
        self.env = make_acrobot_env()
        self.model = AcrobotDDQN(
            AcrobotDQNPolicy, self.env, variant=self.config.algorithm, config=self.config,
            **self.config.model_kwargs(),
        )
        self.history.clear()
        return self.model

    @contextmanager
    def deterministic_policy(self) -> Iterator[None]:
        noisy = self.config.algorithm in NOISY_ALGORITHMS and self.model is not None
        if noisy:
            self.model.policy.set_noise_enabled(False)
        try:
            yield
        finally:
            if noisy:
                self.model.policy.set_noise_enabled(True)

    def evaluate(self, episodes: int, seed: int = 0) -> EvaluationResult:
        if self.model is None:
            raise RuntimeError("Vor der Evaluation muss ein Modell trainiert sein.")
        env = make_acrobot_env(render_mode=None)
        rewards, lengths, steps_to_goal = [], [], []
        successes = 0
        try:
            with self.deterministic_policy():
                for index in range(episodes):
                    observation, _ = env.reset(seed=seed + index)
                    total, length, done, truncated = 0.0, 0, False, False
                    while not done:
                        action, _ = self.model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, _ = env.step(int(action))
                        total += float(reward)
                        length += 1
                        done = bool(terminated or truncated)
                    rewards.append(total); lengths.append(length)
                    if not truncated:
                        successes += 1
                        steps_to_goal.append(length)
        finally:
            env.close()
        return EvaluationResult(
            episodes, mean(rewards), pstdev(rewards), mean(lengths), successes / episodes,
            mean(steps_to_goal) if steps_to_goal else None,
            min(steps_to_goal) if steps_to_goal else None,
        )

    def train(
        self, stop_event: Optional[threading.Event] = None,
        output: Optional[queue.Queue] = None, series: Optional[str] = None,
        evaluation_interval: int = 0, evaluation_episodes: int = 10,
        best_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None,
    ) -> list[EpisodeMetric]:
        reset = self.model is None
        if reset:
            self.create_model()
        callback = AcrobotCallback(
            stop_event or threading.Event(), output, len(self.history), series,
            evaluation_interval, evaluation_episodes, self.config.seed,
            self.evaluate, best_callback,
        )
        self.model.learn(
            self.config.total_timesteps, callback=callback,
            reset_num_timesteps=reset, progress_bar=False,
        )
        self.history.extend(callback.metrics)
        return callback.metrics

    def save(self, path: str | Path) -> tuple[Path, Path, Path]:
        if self.model is None:
            raise RuntimeError("Es gibt kein Modell zum Speichern.")
        base = Path(path).with_suffix("")
        model_path = base.with_suffix(".zip")
        replay_path = base.with_name(base.name + "_replay.pkl")
        metadata_path = base.with_name(base.name + "_metadata.json")
        self.model.save(base)
        # PER-/Multi-Step-Varianten hüllen den SB3-Replay-Buffer in eigene
        # Wrapper-Klassen (keine ReplayBuffer-Unterklassen), die
        # model.save_replay_buffer()/load_replay_buffer() wegen einer
        # internen isinstance(..., ReplayBuffer)-Prüfung ablehnen würde.
        # Der generische Pickle-Helfer von SB3 kennt diese Prüfung nicht.
        save_to_pkl(replay_path, self.model.replay_buffer, verbose=0)
        metadata = {"format": 1, "environment": "Acrobot-v1", "config": asdict(self.config)}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return model_path, replay_path, metadata_path

    @classmethod
    def load(cls, path: str | Path) -> "AcrobotWorkbench":
        base = Path(path).with_suffix("")
        metadata = json.loads(base.with_name(base.name + "_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != 1 or metadata.get("environment") != "Acrobot-v1":
            raise ValueError("Inkompatibler Acrobot-Checkpoint.")
        workbench = cls(AcrobotConfig.from_dict(metadata["config"]))
        workbench.create_model()
        workbench.model = AcrobotDDQN.load(
            base.with_suffix(".zip"), env=workbench.env, config=workbench.config,
            variant=workbench.config.algorithm,
        )
        replay_buffer = load_from_pkl(base.with_name(base.name + "_replay.pkl"), verbose=0)
        target = replay_buffer
        while isinstance(target, (PrioritizedReplayBuffer, NStepReplayBuffer)):
            target = target.buffer
        target.device = workbench.model.device
        workbench.model.replay_buffer = replay_buffer
        return workbench

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
