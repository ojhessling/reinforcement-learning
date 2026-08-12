"""PyTorch/SB3 learning logic for the MountainCar workbench."""

from __future__ import annotations

import json
import queue
import threading
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
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn
from torch.nn import functional as functional


ALGORITHMS = ("DDQN", "Noisy DDQN", "PER DDQN", "Dueling DDQN")
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


def make_mountaincar_env(render_mode: Optional[str] = "rgb_array") -> gymnasium.Env:
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    return gymnasium.make("MountainCar-v0", **kwargs)


def double_dqn_next_values(
    online_q_values: torch.Tensor, target_q_values: torch.Tensor
) -> torch.Tensor:
    actions = online_q_values.argmax(dim=1, keepdim=True)
    return target_q_values.gather(1, actions)


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


class NoisyQNetwork(QNetwork):
    def __init__(self, *args: Any, sigma_0: float = 0.5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        input_dim = self.features_dim
        action_dim = int(self.action_space.n)
        layers: list[nn.Module] = []
        for size in self.net_arch:
            layers.extend((NoisyLinear(input_dim, size, sigma_0), self.activation_fn()))
            input_dim = size
        layers.append(NoisyLinear(input_dim, action_dim, sigma_0))
        self.q_net = nn.Sequential(*layers)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def set_noise_enabled(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.noise_enabled = enabled


class NoisyDQNPolicy(DQNPolicy):
    def __init__(self, *args: Any, sigma_0: float = 0.5, **kwargs: Any) -> None:
        self.sigma_0 = sigma_0
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> NoisyQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return NoisyQNetwork(**net_args, sigma_0=self.sigma_0).to(self.device)

    def reset_noise(self) -> None:
        self.q_net.reset_noise()
        self.q_net_target.reset_noise()

    def set_noise_enabled(self, enabled: bool) -> None:
        self.q_net.set_noise_enabled(enabled)
        self.q_net_target.set_noise_enabled(enabled)


class DuelingQNetwork(QNetwork):
    def __init__(
        self, *args: Any, value_arch: list[int] | None = None,
        advantage_arch: list[int] | None = None, **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        action_dim = int(self.action_space.n)
        feature_layers: list[nn.Module] = []
        input_dim = self.features_dim
        for size in self.net_arch:
            feature_layers.extend((nn.Linear(input_dim, size), self.activation_fn()))
            input_dim = size
        # QNetwork erzeugt standardmäßig bereits einen Kopf. Die Dueling-
        # Variante ersetzt ihn vollständig, damit keine unbenutzten Parameter
        # im Optimizer verbleiben.
        del self.q_net
        self.feature_net = nn.Sequential(*feature_layers)
        self.value_stream = self._stream(input_dim, value_arch or [64], 1)
        self.advantage_stream = self._stream(input_dim, advantage_arch or [64], action_dim)

    def _stream(self, input_dim: int, architecture: list[int], output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for size in architecture:
            layers.extend((nn.Linear(input_dim, size), self.activation_fn()))
            input_dim = size
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(observation, self.features_extractor)
        shared = self.feature_net(features)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    def __init__(
        self, *args: Any, value_arch: list[int] | None = None,
        advantage_arch: list[int] | None = None, **kwargs: Any,
    ) -> None:
        self.value_arch = value_arch or [64]
        self.advantage_arch = advantage_arch or [64]
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(
            **net_args, value_arch=self.value_arch, advantage_arch=self.advantage_arch
        ).to(self.device)


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
        return getattr(self.buffer, name)


@dataclass(frozen=True)
class MountainCarConfig:
    algorithm: str = "DDQN"
    total_timesteps: int = 120_000
    learning_rate: float = 0.004
    buffer_size: int = 10_000
    learning_starts: int = 1_000
    batch_size: int = 128
    tau: float = 1.0
    gamma: float = 0.98
    train_freq: int = 16
    gradient_steps: int = 8
    target_update_interval: int = 600
    exploration_fraction: float = 0.2
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.07
    max_grad_norm: float = 10.0
    seed: Optional[int] = 42
    net_arch: tuple[int, ...] = (64, 64)
    activation: str = "ReLU"
    optimizer: str = "Adam"
    optimizer_eps: float = 1e-5
    optimizer_weight_decay: float = 0.0
    noisy_sigma_0: float = 0.5
    per_alpha: float = 0.6
    per_beta_0: float = 0.4
    per_beta_steps: int = 120_000
    per_epsilon: float = 1e-6
    value_arch: tuple[int, ...] = (64,)
    advantage_arch: tuple[int, ...] = (64,)

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

    def model_kwargs(self) -> dict[str, Any]:
        policy_kwargs: dict[str, Any] = {
            "net_arch": list(self.net_arch),
            "activation_fn": ACTIVATIONS[self.activation],
            "optimizer_class": OPTIMIZERS[self.optimizer],
            "optimizer_kwargs": {"eps": self.optimizer_eps, "weight_decay": self.optimizer_weight_decay},
        }
        if self.algorithm == "Noisy DDQN":
            policy_kwargs["sigma_0"] = self.noisy_sigma_0
        if self.algorithm == "Dueling DDQN":
            policy_kwargs.update(value_arch=list(self.value_arch), advantage_arch=list(self.advantage_arch))
        return {
            "learning_rate": self.learning_rate, "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts, "batch_size": self.batch_size,
            "tau": self.tau, "gamma": self.gamma, "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps, "target_update_interval": self.target_update_interval,
            "exploration_fraction": 0.0 if self.algorithm == "Noisy DDQN" else self.exploration_fraction,
            "exploration_initial_eps": 0.0 if self.algorithm == "Noisy DDQN" else self.exploration_initial_eps,
            "exploration_final_eps": 0.0 if self.algorithm == "Noisy DDQN" else self.exploration_final_eps,
            "max_grad_norm": self.max_grad_norm, "seed": self.seed,
            "policy_kwargs": policy_kwargs, "verbose": 0,
        }

    def signature(self) -> tuple[Any, ...]:
        values = asdict(self)
        values.pop("total_timesteps")
        return tuple(values.items())

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MountainCarConfig":
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
    max_position: float


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    reward_std: float
    mean_length: float
    success_rate: float
    mean_max_position: float
    best_position: float


class MountainCarCallback(BaseCallback):
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
        self.max_position = -1.2
        self.start_step = 0
        self.next_evaluation = evaluation_interval

    def _on_training_start(self) -> None:
        self.start_step = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        reward = float(np.asarray(self.locals["rewards"])[0])
        done = bool(np.asarray(self.locals["dones"])[0])
        info = self.locals.get("infos", [{}])[0]
        observation = np.asarray(self.locals.get("new_obs", [[-1.2, 0.0]]))[0]
        # VecEnv setzt nach done sofort zurück. Der tatsächlich letzte Zustand
        # liegt dann in terminal_observation und ist für die Zielerkennung
        # entscheidend.
        if done and "terminal_observation" in info:
            observation = np.asarray(info["terminal_observation"])
        self.reward += reward
        self.length += 1
        self.max_position = max(self.max_position, float(observation[0]))
        if done:
            metric = EpisodeMetric(
                self.episode_offset + len(self.metrics) + 1, self.reward, self.length,
                bool(self.max_position >= 0.5 and not info.get("TimeLimit.truncated", False)),
                int(self.num_timesteps), self.max_position,
            )
            self.metrics.append(metric)
            kind = "episode" if self.series is None else "comparison_episode"
            payload: Any = metric if self.series is None else (self.series, metric)
            if self.output is not None:
                self.output.put((kind, payload))
            self.reward, self.length, self.max_position = 0.0, 0, -1.2
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


class MountainCarDDQN(DQN):
    def __init__(
        self, *args: Any, variant: str = "DDQN",
        config: Optional[MountainCarConfig] = None, **kwargs: Any,
    ) -> None:
        self.variant = variant
        self.workbench_config = config or MountainCarConfig(algorithm=variant)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.variant == "PER DDQN" and not isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            self.replay_buffer = PrioritizedReplayBuffer(
                self.replay_buffer, self.workbench_config.per_alpha,
                self.workbench_config.per_epsilon,
            )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses: list[float] = []
        for _ in range(gradient_steps):
            if self.variant == "Noisy DDQN":
                self.policy.reset_noise()
            indices: Optional[np.ndarray] = None
            weights: torch.Tensor | float = 1.0
            if self.variant == "PER DDQN":
                progress = min(1.0, self.num_timesteps / self.workbench_config.per_beta_steps)
                beta = self.workbench_config.per_beta_0 + progress * (1 - self.workbench_config.per_beta_0)
                replay_data, indices, weights = self.replay_buffer.sample(
                    batch_size, beta, env=self._vec_normalize_env
                )
            else:
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
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


class MountainCarWorkbench:
    def __init__(self, config: Optional[MountainCarConfig] = None) -> None:
        self.config = config or MountainCarConfig()
        self.config.validate()
        self.env: Optional[gymnasium.Env] = None
        self.model: Optional[MountainCarDDQN] = None
        self.history: list[EpisodeMetric] = []

    def create_model(self) -> MountainCarDDQN:
        self.close()
        self.env = make_mountaincar_env()
        policy: str | type[BasePolicy] = "MlpPolicy"
        if self.config.algorithm == "Noisy DDQN":
            policy = NoisyDQNPolicy
        elif self.config.algorithm == "Dueling DDQN":
            policy = DuelingDQNPolicy
        self.model = MountainCarDDQN(
            policy, self.env, variant=self.config.algorithm, config=self.config,
            **self.config.model_kwargs(),
        )
        self.history.clear()
        return self.model

    @contextmanager
    def deterministic_policy(self) -> Iterator[None]:
        noisy = self.config.algorithm == "Noisy DDQN" and self.model is not None
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
        env = make_mountaincar_env(render_mode=None)
        rewards, lengths, maxima = [], [], []
        successes = 0
        try:
            with self.deterministic_policy():
                for index in range(episodes):
                    observation, _ = env.reset(seed=seed + index)
                    total, length, maximum, done = 0.0, 0, float(observation[0]), False
                    while not done:
                        action, _ = self.model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, _ = env.step(int(action))
                        total += float(reward)
                        length += 1
                        maximum = max(maximum, float(observation[0]))
                        done = bool(terminated or truncated)
                    rewards.append(total); lengths.append(length); maxima.append(maximum)
                    successes += int(maximum >= 0.5)
        finally:
            env.close()
        return EvaluationResult(
            episodes, mean(rewards), pstdev(rewards), mean(lengths), successes / episodes,
            mean(maxima), max(maxima),
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
        callback = MountainCarCallback(
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
        self.model.save_replay_buffer(replay_path)
        metadata = {"format": 1, "environment": "MountainCar-v0", "config": asdict(self.config)}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return model_path, replay_path, metadata_path

    @classmethod
    def load(cls, path: str | Path) -> "MountainCarWorkbench":
        base = Path(path).with_suffix("")
        metadata = json.loads(base.with_name(base.name + "_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format") != 1 or metadata.get("environment") != "MountainCar-v0":
            raise ValueError("Inkompatibler MountainCar-Checkpoint.")
        workbench = cls(MountainCarConfig.from_dict(metadata["config"]))
        workbench.create_model()
        workbench.model = MountainCarDDQN.load(
            base.with_suffix(".zip"), env=workbench.env, config=workbench.config,
            variant=workbench.config.algorithm,
        )
        workbench.model.load_replay_buffer(base.with_name(base.name + "_replay.pkl"))
        return workbench

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
