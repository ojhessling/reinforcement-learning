"""SB3 learning logic for the BipedalWalker workbench."""

from __future__ import annotations

import json
import math
import queue
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Optional

import gymnasium
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn


ALGORITHMS = ("PPO", "TD3", "SAC")
ALGORITHM_CLASSES: dict[str, type[BaseAlgorithm]] = {"PPO": PPO, "TD3": TD3, "SAC": SAC}
#: PPO ist on-policy und führt keinen Replay Buffer; TD3 und SAC tun es.
OFF_POLICY_ALGORITHMS = frozenset({"TD3", "SAC"})
ON_POLICY_ALGORITHMS = frozenset({"PPO"})

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
SCHEDULES = ("konstant", "linear fallend")
ACTION_NOISES = ("keins", "normal", "Ornstein-Uhlenbeck")
ENTROPY_MODES = ("auto", "fest")

ENV_ID = "BipedalWalker-v3"
#: Die Hardcore-Variante mit Stufen, Gruben und Hindernissen gehört nicht zum
#: Projekt; sie bräuchte ein um Größenordnungen höheres Trainingsbudget.
HARDCORE = False
#: Offizieller Gymnasium-`reward_threshold`: ab diesem Return gilt eine Episode
#: als gelöst. Die Shaping-Normierung ist so gewählt, dass die vollständige
#: Strecke rund +300 ergibt.
SOLVED_RETURN = 300.0
#: Ein Sturz setzt den Reward des letzten Schritts auf exakt -100; das Erreichen
#: des Streckenendes beendet die Episode dagegen ohne Bonus. Ein Schwellenwert
#: dazwischen unterscheidet beide Ausgänge zuverlässig.
FALL_REWARD_THRESHOLD = -50.0
#: Box(-1, 1, (4,)): Drehmomente für Hüfte und Knie beider Beine.
ACTION_DIM = 4
JOINT_NAMES = ("Hüfte 1", "Knie 1", "Hüfte 2", "Knie 2")
#: 24 Beobachtungswerte, davon die letzten zehn Lidar-Messungen.
OBSERVATION_DIM = 24
LIDAR_COUNT = 10
#: Maximales Drehmoment je Motor (`MOTORS_TORQUE` im Environment).
MOTORS_TORQUE = 80

#: Zuordnung Parameter -> Verfahren, die ihn besitzen. Die GUI zeigt in einem
#: Verfahrenstab ausschließlich Parameter des dort gewählten Algorithmus; die
#: Vergleichs-Summary bildet daraus die Liste der echten Unterschiede.
_ALL = frozenset(ALGORITHMS)
FIELD_ALGORITHMS: dict[str, frozenset[str]] = {
    "total_timesteps": _ALL, "learning_rate": _ALL, "learning_rate_schedule": _ALL,
    "batch_size": _ALL, "gamma": _ALL, "seed": _ALL, "actor_arch": _ALL, "critic_arch": _ALL,
    "activation": _ALL, "optimizer": _ALL, "optimizer_eps": _ALL, "optimizer_weight_decay": _ALL,
    "n_steps": frozenset({"PPO"}), "n_epochs": frozenset({"PPO"}),
    "gae_lambda": frozenset({"PPO"}), "clip_range": frozenset({"PPO"}),
    "clip_range_vf": frozenset({"PPO"}), "normalize_advantage": frozenset({"PPO"}),
    "ent_coef": frozenset({"PPO"}), "vf_coef": frozenset({"PPO"}),
    "max_grad_norm": frozenset({"PPO"}), "target_kl": frozenset({"PPO"}),
    "log_std_init": frozenset({"PPO", "SAC"}),
    "use_sde": frozenset({"PPO", "SAC"}), "sde_sample_freq": frozenset({"PPO", "SAC"}),
    "buffer_size": OFF_POLICY_ALGORITHMS, "learning_starts": OFF_POLICY_ALGORITHMS,
    "tau": OFF_POLICY_ALGORITHMS, "train_freq": OFF_POLICY_ALGORITHMS,
    "gradient_steps": OFF_POLICY_ALGORITHMS,
    "action_noise": OFF_POLICY_ALGORITHMS, "action_noise_sigma": OFF_POLICY_ALGORITHMS,
    "policy_delay": frozenset({"TD3"}), "target_policy_noise": frozenset({"TD3"}),
    "target_noise_clip": frozenset({"TD3"}),
    "ent_coef_mode": frozenset({"SAC"}), "ent_coef_value": frozenset({"SAC"}),
    "target_entropy": frozenset({"SAC"}), "target_update_interval": frozenset({"SAC"}),
    "normalize_obs": _ALL, "normalize_reward": _ALL,
    "clip_obs": _ALL, "clip_reward": _ALL,
}
FIELD_LABELS: dict[str, str] = {
    "total_timesteps": "Trainingsschritte N", "learning_rate": "Lernrate α",
    "learning_rate_schedule": "LR-Verlauf", "batch_size": "Batch-Größe B",
    "gamma": "Diskontfaktor γ", "seed": "Zufallsstart s", "actor_arch": "Actor-Hidden h_π",
    "critic_arch": "Critic-Hidden h_q", "activation": "Aktivierung φ", "optimizer": "Optimizer",
    "optimizer_eps": "Optimizer ε", "optimizer_weight_decay": "Weight Decay λ",
    "n_steps": "Rollout n_steps", "n_epochs": "Epochen K", "gae_lambda": "GAE λ_GAE",
    "clip_range": "Clip ε_clip", "clip_range_vf": "Clip Value", "normalize_advantage": "Advantage normieren",
    "ent_coef": "Entropie c_ent", "vf_coef": "Value c_vf", "max_grad_norm": "Grad-Norm",
    "target_kl": "Target KL", "log_std_init": "log σ₀", "use_sde": "gSDE nutzen",
    "sde_sample_freq": "gSDE-Frequenz", "buffer_size": "Replay Buffer |D|",
    "learning_starts": "Lernstart t₀", "tau": "Soft-Update τ", "train_freq": "Trainingsfrequenz fₜ",
    "gradient_steps": "Gradientenschritte G", "action_noise": "Action Noise",
    "action_noise_sigma": "Noise σ", "policy_delay": "Policy Delay d",
    "target_policy_noise": "Target-Noise σ_t", "target_noise_clip": "Noise-Clip c",
    "ent_coef_mode": "Entropie α", "ent_coef_value": "Startwert α",
    "target_entropy": "Zielentropie H*", "target_update_interval": "Target-Intervall C",
    "normalize_obs": "Beobachtungen normalisieren", "normalize_reward": "Rewards normalisieren",
    "clip_obs": "Clip Beobachtungen", "clip_reward": "Clip Reward",
}

INTEGER_FIELDS = frozenset({
    "total_timesteps", "batch_size", "n_steps", "n_epochs", "sde_sample_freq", "buffer_size",
    "learning_starts", "train_freq", "gradient_steps", "policy_delay", "target_update_interval",
})
TUPLE_FIELDS = frozenset({"actor_arch", "critic_arch"})
BOOLEAN_FIELDS = frozenset({"normalize_advantage", "use_sde", "normalize_obs", "normalize_reward"})
OPTIONAL_FLOAT_FIELDS = frozenset({"clip_range_vf", "target_kl"})
CHOICE_FIELDS: dict[str, tuple[str, ...]] = {
    "learning_rate_schedule": SCHEDULES, "activation": tuple(ACTIVATIONS),
    "optimizer": tuple(OPTIMIZERS), "action_noise": ACTION_NOISES, "ent_coef_mode": ENTROPY_MODES,
}
#: `target_entropy` akzeptiert bewusst Text: `auto` oder eine Zahl.
TEXT_FIELDS = frozenset({"target_entropy"})

#: Einheitliche Netzgröße aller Verfahren. Die Zoo-Profile nennen für TD3 und
#: SAC `400,300`; solche Netze brauchen deutlich mehr Rechenzeit je Schritt und
#: lohnen sich erst bei entsprechend großem Budget.
DEFAULT_NET_ARCH = (64, 64)

#: Einheitliches Trainingsbudget aller Verfahren. Die Zoo-Profile nennen
#: 5.000.000 (PPO), 1.000.000 (TD3) und 500.000 (SAC) Schritte; das ist
#: interaktiv nicht abwartbar, und unterschiedliche Budgets machten einen
#: Vergleich zweier Slots ohne Zutun unfair.
DEFAULT_TOTAL_TIMESTEPS = 100_000

#: Getunte `BipedalWalker-v3`-Profile des RL Baselines3 Zoo. Werte, die dort
#: nicht auftauchen, bleiben auf den Voreinstellungen von Stable-Baselines3
#: (siehe Dataclass-Defaults).
DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "PPO": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "learning_rate": 3e-4,
        "gamma": 0.999, "n_steps": 2048, "batch_size": 64, "n_epochs": 10,
        "gae_lambda": 0.95, "clip_range": 0.18, "ent_coef": 0.0,
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH,
        "activation": "Tanh", "optimizer_eps": 1e-5,
        # Das Zoo-Profil verlangt `normalize: true`; ohne Normalisierung der 24
        # sehr unterschiedlich skalierten Werte lernt PPO hier kaum.
        "normalize_obs": True, "normalize_reward": True,
    },
    "TD3": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "learning_rate": 1e-3, "gamma": 0.98,
        "batch_size": 256, "buffer_size": 200_000, "learning_starts": 10_000,
        "train_freq": 1, "gradient_steps": 1,
        "action_noise": "normal", "action_noise_sigma": 0.1,
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH, "activation": "ReLU",
    },
    "SAC": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "learning_rate": 7.3e-4,
        "batch_size": 256, "gamma": 0.98, "tau": 0.02,
        "buffer_size": 300_000, "learning_starts": 10_000,
        "train_freq": 64, "gradient_steps": 64, "ent_coef_mode": "auto",
        "use_sde": True, "log_std_init": -3.0,
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH, "activation": "ReLU",
    },
}


def make_bipedalwalker_env(render_mode: Optional[str] = "rgb_array") -> gymnasium.Env:
    """Offizielles Environment. `hardcore=False` wird ausdrücklich mitgegeben,
    obwohl es dem Standard entspricht: Die Hardcore-Variante gehört nicht zum
    Projekt."""
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    return gymnasium.make(ENV_ID, hardcore=HARDCORE, **kwargs)


class LinearSchedule:
    """Lernrate, die linear von `initial` auf 0 fällt (Zoo-Notation `lin_…`).

    Stable-Baselines3 ruft Schedules mit dem verbleibenden Fortschritt auf:
    1.0 am Trainingsstart, 0.0 am Ende.
    """

    def __init__(self, initial: float) -> None:
        self.initial = float(initial)

    def __call__(self, progress_remaining: float) -> float:
        return self.initial * float(progress_remaining)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, LinearSchedule) and other.initial == self.initial

    def __repr__(self) -> str:
        return f"LinearSchedule({self.initial})"


def make_action_noise(kind: str, sigma: float, action_dim: int = ACTION_DIM):
    """Explizites Explorationsrauschen für die Off-Policy-Verfahren.

    TD3 besitzt einen deterministischen Actor und exploriert ohne dieses
    Rauschen überhaupt nicht; bei SAC ist es optional, weil die Policy selbst
    stochastisch ist.
    """
    if kind == "keins" or sigma <= 0:
        return None
    mean_vector, sigma_vector = np.zeros(action_dim), sigma * np.ones(action_dim)
    if kind == "normal":
        return NormalActionNoise(mean_vector, sigma_vector)
    return OrnsteinUhlenbeckActionNoise(mean_vector, sigma_vector)


@dataclass(frozen=True)
class BipedalWalkerConfig:
    """Vollständige Konfiguration eines Verfahrensslots.

    Die Dataclass-Defaults entsprechen den Voreinstellungen von
    Stable-Baselines3. `default_config()` legt darüber das Zoo-Profil des
    jeweiligen Algorithmus.
    """

    algorithm: str = "PPO"
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "konstant"
    batch_size: int = 64
    gamma: float = 0.99
    seed: Optional[int] = 42
    actor_arch: tuple[int, ...] = (64, 64)
    critic_arch: tuple[int, ...] = (64, 64)
    activation: str = "ReLU"
    optimizer: str = "Adam"
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    # PPO
    n_steps: int = 2048
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    log_std_init: float = 0.0
    # PPO und SAC
    use_sde: bool = False
    sde_sample_freq: int = -1
    # TD3 und SAC
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: str = "keins"
    action_noise_sigma: float = 0.1
    # TD3
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    # SAC
    ent_coef_mode: str = "auto"
    ent_coef_value: float = 1.0
    target_entropy: str = "auto"
    target_update_interval: int = 1
    # Normalisierung (VecNormalize) – gilt für alle drei Verfahren
    normalize_obs: bool = False
    normalize_reward: bool = False
    clip_obs: float = 10.0
    clip_reward: float = 10.0

    @property
    def normalizes(self) -> bool:
        return self.normalize_obs or self.normalize_reward

    def uses(self, field_name: str) -> bool:
        """Gehört der Parameter zum gewählten Algorithmus?"""
        return self.algorithm in FIELD_ALGORITHMS.get(field_name, _ALL)

    def validate(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(
                f"Verfahren: '{self.algorithm}' ist unbekannt. Gültig: {', '.join(ALGORITHMS)}."
            )
        positive_integers = [
            ("Trainingsschritte N", self.total_timesteps), ("Batch-Größe B", self.batch_size),
        ]
        if self.algorithm == "PPO":
            positive_integers += [("Rollout n_steps", self.n_steps), ("Epochen K", self.n_epochs)]
        else:
            positive_integers += [
                ("Replay Buffer |D|", self.buffer_size),
                ("Trainingsfrequenz fₜ", self.train_freq),
                ("Gradientenschritte G", self.gradient_steps),
            ]
        if self.algorithm == "TD3":
            positive_integers.append(("Policy Delay d", self.policy_delay))
        if self.algorithm == "SAC":
            positive_integers.append(("Target-Intervall C", self.target_update_interval))
        for label, value in positive_integers:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label}: '{value}' ist ungültig. Gültig: ganze Zahl ≥ 1.")
        if self.learning_rate <= 0:
            raise ValueError(f"Lernrate α: '{self.learning_rate}' ist ungültig. Gültig: > 0.")
        if self.learning_rate_schedule not in SCHEDULES:
            raise ValueError(
                f"LR-Verlauf: '{self.learning_rate_schedule}' ist ungültig. "
                f"Gültig: {', '.join(SCHEDULES)}."
            )
        if not 0 < self.gamma <= 1:
            raise ValueError(f"Diskontfaktor γ: '{self.gamma}' ist ungültig. Gültig: 0 < γ ≤ 1.")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"Aktivierung φ: '{self.activation}' ist unbekannt. Gültig: {', '.join(ACTIVATIONS)}."
            )
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"Optimizer: '{self.optimizer}' ist unbekannt. Gültig: {', '.join(OPTIMIZERS)}."
            )
        if self.optimizer_eps <= 0:
            raise ValueError(f"Optimizer ε: '{self.optimizer_eps}' ist ungültig. Gültig: > 0.")
        if self.optimizer_weight_decay < 0:
            raise ValueError(
                f"Weight Decay λ: '{self.optimizer_weight_decay}' ist ungültig. Gültig: ≥ 0."
            )
        for label, architecture in (
            ("Actor-Hidden h_π", self.actor_arch), ("Critic-Hidden h_q", self.critic_arch),
        ):
            if not architecture or any(size <= 0 for size in architecture):
                raise ValueError(
                    f"{label}: '{architecture}' ist ungültig. "
                    "Gültig: mindestens eine positive Ganzzahl, z. B. '400,300'."
                )
        for label, value in (("Clip Beobachtungen", self.clip_obs),
                             ("Clip Reward", self.clip_reward)):
            if value <= 0:
                raise ValueError(f"{label}: '{value}' ist ungültig. Gültig: > 0.")
        if self.algorithm == "PPO":
            self._validate_ppo()
        elif self.algorithm == "TD3":
            self._validate_off_policy()
            self._validate_td3()
        else:
            self._validate_off_policy()
            self._validate_sac()

    def _validate_ppo(self) -> None:
        if self.batch_size > self.n_steps:
            raise ValueError(
                f"Batch-Größe B: '{self.batch_size}' übersteigt den Rollout n_steps="
                f"{self.n_steps}. Gültig: 1 bis {self.n_steps}."
            )
        if self.n_steps % self.batch_size != 0:
            raise ValueError(
                f"Batch-Größe B: '{self.batch_size}' teilt den Rollout n_steps={self.n_steps} "
                f"nicht. PPO würde sonst Daten verwerfen. Gültig: Teiler von {self.n_steps}."
            )
        if self.total_timesteps < self.n_steps:
            raise ValueError(
                f"Trainingsschritte N: '{self.total_timesteps}' reicht für keinen vollständigen "
                f"Rollout. Gültig: ≥ n_steps = {self.n_steps}."
            )
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError(f"GAE λ_GAE: '{self.gae_lambda}' ist ungültig. Gültig: 0 bis 1.")
        if self.clip_range <= 0:
            raise ValueError(f"Clip ε_clip: '{self.clip_range}' ist ungültig. Gültig: > 0.")
        if self.clip_range_vf is not None and self.clip_range_vf <= 0:
            raise ValueError(
                f"Clip Value: '{self.clip_range_vf}' ist ungültig. Gültig: > 0 oder leer für aus."
            )
        if self.ent_coef < 0:
            raise ValueError(f"Entropie c_ent: '{self.ent_coef}' ist ungültig. Gültig: ≥ 0.")
        if self.vf_coef < 0:
            raise ValueError(f"Value c_vf: '{self.vf_coef}' ist ungültig. Gültig: ≥ 0.")
        if self.max_grad_norm <= 0:
            raise ValueError(f"Grad-Norm: '{self.max_grad_norm}' ist ungültig. Gültig: > 0.")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError(
                f"Target KL: '{self.target_kl}' ist ungültig. Gültig: > 0 oder leer für aus."
            )

    def _validate_off_policy(self) -> None:
        if not 0 <= self.learning_starts <= self.total_timesteps:
            raise ValueError(
                f"Lernstart t₀: '{self.learning_starts}' liegt außerhalb. "
                f"Gültig: 0 bis {self.total_timesteps}."
            )
        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"Batch-Größe B: '{self.batch_size}' übersteigt den Replay Buffer. "
                f"Gültig: 1 bis {self.buffer_size}."
            )
        if not 0 < self.tau <= 1:
            raise ValueError(f"Soft-Update τ: '{self.tau}' ist ungültig. Gültig: 0 < τ ≤ 1.")
        if self.action_noise not in ACTION_NOISES:
            raise ValueError(
                f"Action Noise: '{self.action_noise}' ist unbekannt. Gültig: {', '.join(ACTION_NOISES)}."
            )
        if self.action_noise_sigma < 0:
            raise ValueError(f"Noise σ: '{self.action_noise_sigma}' ist ungültig. Gültig: ≥ 0.")

    def _validate_td3(self) -> None:
        if self.action_noise == "keins":
            raise ValueError(
                "Action Noise: 'keins' ist für TD3 ungültig. TD3 besitzt einen "
                "deterministischen Actor und exploriert ohne Rauschen nicht. "
                f"Gültig: {', '.join(ACTION_NOISES[1:])}."
            )
        if self.target_policy_noise < 0:
            raise ValueError(f"Target-Noise σ_t: '{self.target_policy_noise}' ist ungültig. Gültig: ≥ 0.")
        if self.target_noise_clip < 0:
            raise ValueError(f"Noise-Clip c: '{self.target_noise_clip}' ist ungültig. Gültig: ≥ 0.")

    def _validate_sac(self) -> None:
        if self.ent_coef_mode not in ENTROPY_MODES:
            raise ValueError(
                f"Entropie α: '{self.ent_coef_mode}' ist unbekannt. Gültig: {', '.join(ENTROPY_MODES)}."
            )
        if self.ent_coef_value <= 0:
            raise ValueError(f"Startwert α: '{self.ent_coef_value}' ist ungültig. Gültig: > 0.")
        if self.target_entropy.strip() != "auto":
            try:
                float(self.target_entropy)
            except ValueError as error:
                raise ValueError(
                    f"Zielentropie H*: '{self.target_entropy}' ist ungültig. "
                    "Gültig: 'auto' oder eine Zahl, z. B. -2."
                ) from error
        if self.use_sde and self.sde_sample_freq < -1:
            raise ValueError(
                f"gSDE-Frequenz: '{self.sde_sample_freq}' ist ungültig. Gültig: -1 oder ≥ 0."
            )

    def learning_rate_value(self) -> float | LinearSchedule:
        if self.learning_rate_schedule == "linear fallend":
            return LinearSchedule(self.learning_rate)
        return self.learning_rate

    def sac_ent_coef(self) -> str | float:
        if self.ent_coef_mode == "auto":
            return f"auto_{self.ent_coef_value}"
        return self.ent_coef_value

    def sac_target_entropy(self) -> str | float:
        text = self.target_entropy.strip()
        return "auto" if text == "auto" else float(text)

    def policy_kwargs(self) -> dict[str, Any]:
        value_key = "vf" if self.algorithm == "PPO" else "qf"
        kwargs: dict[str, Any] = {
            "net_arch": {"pi": list(self.actor_arch), value_key: list(self.critic_arch)},
            "activation_fn": ACTIVATIONS[self.activation],
            "optimizer_class": OPTIMIZERS[self.optimizer],
            "optimizer_kwargs": {
                "eps": self.optimizer_eps, "weight_decay": self.optimizer_weight_decay,
            },
        }
        if self.algorithm in ("PPO", "SAC"):
            # Beide Policies kennen die initiale Streuung der Gauß-Policy; das
            # SAC-Zoo-Profil setzt sie auf -3.
            kwargs["log_std_init"] = self.log_std_init
        return kwargs

    def model_kwargs(self) -> dict[str, Any]:
        """Konstruktorargumente – ausschließlich Schlüssel, die der jeweilige
        Stable-Baselines3-Algorithmus tatsächlich kennt."""
        common = {
            "learning_rate": self.learning_rate_value(), "gamma": self.gamma,
            "batch_size": self.batch_size, "seed": self.seed,
            "policy_kwargs": self.policy_kwargs(), "verbose": 0,
        }
        if self.algorithm == "PPO":
            return {
                **common, "n_steps": self.n_steps, "n_epochs": self.n_epochs,
                "gae_lambda": self.gae_lambda, "clip_range": self.clip_range,
                "clip_range_vf": self.clip_range_vf, "normalize_advantage": self.normalize_advantage,
                "ent_coef": self.ent_coef, "vf_coef": self.vf_coef,
                "max_grad_norm": self.max_grad_norm, "target_kl": self.target_kl,
                "use_sde": self.use_sde, "sde_sample_freq": self.sde_sample_freq,
            }
        off_policy = {
            **common, "buffer_size": self.buffer_size, "learning_starts": self.learning_starts,
            "tau": self.tau, "train_freq": self.train_freq, "gradient_steps": self.gradient_steps,
            "action_noise": make_action_noise(self.action_noise, self.action_noise_sigma),
        }
        if self.algorithm == "TD3":
            return {
                **off_policy, "policy_delay": self.policy_delay,
                "target_policy_noise": self.target_policy_noise,
                "target_noise_clip": self.target_noise_clip,
            }
        return {
            **off_policy, "ent_coef": self.sac_ent_coef(),
            "target_entropy": self.sac_target_entropy(),
            "target_update_interval": self.target_update_interval,
            "use_sde": self.use_sde, "sde_sample_freq": self.sde_sample_freq,
        }

    def signature(self) -> tuple[Any, ...]:
        """Kennzeichnet den Modellaufbau. Ein geändertes Trainingsbudget setzt
        den Lernzustand nicht zurück, jede andere Änderung schon."""
        values = asdict(self)
        values.pop("total_timesteps")
        return tuple(sorted((key, str(value)) for key, value in values.items()
                            if key == "algorithm" or self.uses(key)))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BipedalWalkerConfig":
        data = dict(values)
        for key in TUPLE_FIELDS:
            data[key] = tuple(data[key])
        config = cls(**data)
        config.validate()
        return config


def default_config(algorithm: str) -> BipedalWalkerConfig:
    """Konfiguration mit dem Zoo-Profil des Algorithmus."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Verfahren: '{algorithm}' ist unbekannt. Gültig: {', '.join(ALGORITHMS)}.")
    return BipedalWalkerConfig(algorithm=algorithm, **DEFAULT_PROFILES[algorithm])


def format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "ja" if value else "nein"
    return str(value)


def config_differences(
    first: BipedalWalkerConfig, second: BipedalWalkerConfig
) -> list[tuple[str, str, str]]:
    """Parameter, in denen sich zwei Slots unterscheiden.

    Verglichen werden ausschließlich Parameter, die **beide** Verfahren
    besitzen. Ein Parameter, den nur eines der beiden kennt, ist keine
    Konfigurationsentscheidung, sondern eine Folge der Verfahrenswahl; er steht
    vollständig im jeweiligen Verfahrenstab. Bei zweimal demselben Algorithmus
    umfasst das automatisch alle relevanten Parameter.
    """
    differences = []
    for field in fields(first):
        name = field.name
        if name == "algorithm" or not (first.uses(name) and second.uses(name)):
            continue
        left, right = getattr(first, name), getattr(second, name)
        if left != right:
            differences.append((FIELD_LABELS.get(name, name), format_value(left), format_value(right)))
    return differences


@dataclass(frozen=True)
class EpisodeMetric:
    episode: int
    reward: float
    length: int
    finished: bool
    fell: bool
    timeout: bool
    solved: bool
    timesteps: int


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    reward_std: float
    mean_length: float
    goal_rate: float
    solved_rate: float
    fall_rate: float
    timeout_rate: float


def episode_outcome(
    final_reward: float, truncated: bool, episode_return: float
) -> tuple[bool, bool, bool, bool]:
    """Ausgang einer beendeten Episode als (Ziel, Sturz, Zeitlimit, gelöst).

    Ein Sturz setzt den Reward des letzten Schritts auf exakt -100; das
    Erreichen des Streckenendes beendet die Episode ohne Bonus. Ein Zeitlimit
    ist weder das eine noch das andere: Der Roboter steht noch, war aber zu
    langsam. Diese Unterscheidung ist hier wichtiger als bei einem Environment
    mit klarem Terminalbonus, weil ein vorsichtiger Agent das Zeitlimit
    zuverlässig erreicht, ohne je zu stürzen oder voranzukommen.
    """
    fell = not truncated and final_reward <= FALL_REWARD_THRESHOLD
    finished = not truncated and not fell
    solved = episode_return >= SOLVED_RETURN
    return finished, fell, bool(truncated), solved


def observation_readout(observation: np.ndarray) -> dict[str, Any]:
    """Observation in anzeigefreundliche Größen zerlegen.

    Die Geschwindigkeiten sind im Environment bereits skaliert und die
    Kniewinkel um +1 verschoben. Sie bleiben deshalb normierte Größen; eine
    saubere Umrechnung in physikalische Einheiten existiert nicht. Nur der
    Rumpfwinkel ist ein echter Winkel im Bogenmaß und wird zusätzlich in Grad
    ausgewiesen.
    """
    values = np.asarray(observation, dtype=float).reshape(-1)
    legs = []
    for index, offset in enumerate((4, 9)):
        legs.append({
            "hip_angle": float(values[offset]),
            "hip_speed": float(values[offset + 1]),
            "knee_angle": float(values[offset + 2]),
            "knee_speed": float(values[offset + 3]),
            "contact": bool(values[offset + 4] > 0.5),
        })
    return {
        "hull_angle": float(values[0]),
        "hull_angle_degrees": math.degrees(float(values[0])),
        "hull_angular_velocity": float(values[1]),
        "vx": float(values[2]),
        "vy": float(values[3]),
        "legs": legs,
        "lidar": [float(value) for value in values[-LIDAR_COUNT:]],
    }


def action_readout(action: np.ndarray) -> dict[str, Any]:
    """Die vier Drehmomente in ihre Gelenkbedeutung übersetzen.

    Das Vorzeichen bestimmt die Drehrichtung, der Betrag das maximal anliegende
    Drehmoment als Anteil von `MOTORS_TORQUE`. Ein Wert von 0 bedeutet also
    nicht „Gelenk hält die Position", sondern „kein Moment".
    """
    values = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
    joints = []
    for name, value in zip(JOINT_NAMES, values):
        direction = "—" if value == 0 else ("vor" if value > 0 else "zurück")
        joints.append({
            "joint": name,
            "raw": float(value),
            "direction": direction,
            "torque": abs(float(value)),
            "text": "aus" if value == 0 else f"{direction} {abs(value):.0%}",
        })
    return {"raw": [float(value) for value in values], "joints": joints}


class BipedalWalkerCallback(BaseCallback):
    """Schreibt Episodenmetriken und Zwischenevaluationen in die GUI-Queue.

    `series` kennzeichnet den Verfahrensslot und den Lauftyp, damit Einzel- und
    Vergleichsläufe beider Slots dieselbe Queue teilen können.
    """

    def __init__(
        self, stop_event: threading.Event, output: Optional[queue.Queue] = None,
        episode_offset: int = 0, series: Any = None,
        evaluation_interval: int = 0, evaluation_episodes: int = 5,
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

    def _raw_reward(self) -> float:
        """Reward des letzten Schritts in Originaleinheiten.

        Bei aktiver Reward-Normalisierung liefert `locals["rewards"]` skalierte
        Werte; der Terminalreward -100 wäre daran nicht mehr erkennbar.
        """
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None and vec_normalize.norm_reward:
            return float(np.asarray(vec_normalize.get_original_reward()).reshape(-1)[0])
        return float(np.asarray(self.locals["rewards"])[0])

    def _on_step(self) -> bool:
        reward = self._raw_reward()
        done = bool(np.asarray(self.locals["dones"])[0])
        info = self.locals.get("infos", [{}])[0]
        self.reward += reward
        self.length += 1
        if done:
            truncated = bool(info.get("TimeLimit.truncated", False))
            # Der Monitor-Wrapper sitzt innerhalb der Normalisierung und führt
            # den Episoden-Return in Originaleinheiten; er hat Vorrang.
            episode_info = info.get("episode")
            total = float(episode_info["r"]) if episode_info else self.reward
            length = int(episode_info["l"]) if episode_info else self.length
            finished, fell, timeout, solved = episode_outcome(reward, truncated, total)
            metric = EpisodeMetric(
                self.episode_offset + len(self.metrics) + 1, total, length,
                finished, fell, timeout, solved, int(self.num_timesteps),
            )
            self.metrics.append(metric)
            if self.output is not None:
                self.output.put(("episode", (self.series, metric)))
            self.reward, self.length = 0.0, 0
        elapsed = int(self.num_timesteps) - self.start_step
        if self.evaluator and self.evaluation_interval and elapsed >= self.next_evaluation:
            result = self.evaluator(self.evaluation_episodes, self.evaluation_seed or 0)
            episode = self.episode_offset + len(self.metrics)
            if self.best_callback:
                self.best_callback(episode, int(self.num_timesteps), result)
            if self.output is not None:
                self.output.put(("evaluation", (self.series, episode, int(self.num_timesteps), result)))
            self.next_evaluation += self.evaluation_interval
        if self.output is not None and elapsed % 100 == 0:
            self.output.put(("progress", (self.series, elapsed)))
        return not self.stop_event.is_set()


class BipedalWalkerWorkbench:
    """Ein Verfahrensslot: Environment, Modell, Historie, Checkpoints."""

    def __init__(self, config: Optional[BipedalWalkerConfig] = None) -> None:
        self.config = config or default_config("PPO")
        self.config.validate()
        self.env: Optional[Any] = None
        self.model: Optional[BaseAlgorithm] = None
        self.history: list[EpisodeMetric] = []

    @property
    def uses_replay_buffer(self) -> bool:
        return self.config.algorithm in OFF_POLICY_ALGORITHMS

    @property
    def vec_normalize(self) -> Optional[VecNormalize]:
        return self.env if isinstance(self.env, VecNormalize) else None

    def _make_training_env(self) -> Any:
        """Vektor-Environment mit Monitor und optionaler Normalisierung.

        Der `Monitor` sitzt bewusst **innerhalb** der Normalisierung: Nur so
        führt er den Episoden-Return in Originaleinheiten, an dem sich die
        Schwelle von +300 überhaupt ablesen lässt.
        """
        env: Any = DummyVecEnv([lambda: Monitor(make_bipedalwalker_env(render_mode=None))])
        if self.config.normalizes:
            env = VecNormalize(
                env, norm_obs=self.config.normalize_obs, norm_reward=self.config.normalize_reward,
                clip_obs=self.config.clip_obs, clip_reward=self.config.clip_reward,
                gamma=self.config.gamma,
            )
        return env

    def create_model(self) -> BaseAlgorithm:
        self.close()
        self.env = self._make_training_env()
        algorithm_class = ALGORITHM_CLASSES[self.config.algorithm]
        self.model = algorithm_class("MlpPolicy", self.env, **self.config.model_kwargs())
        self.history.clear()
        return self.model

    def policy_observation(self, observation: np.ndarray) -> np.ndarray:
        """Beobachtung so aufbereiten, wie das Modell sie im Training sieht.

        Die laufenden Statistiken werden dabei **nicht** fortgeschrieben:
        `normalize_obs` liest sie nur. Evaluation und Animation arbeiten
        deshalb mit eingefrorenen Statistiken.
        """
        vec_normalize = self.vec_normalize
        if vec_normalize is not None and vec_normalize.norm_obs:
            return vec_normalize.normalize_obs(np.asarray(observation, dtype=np.float32))
        return observation

    def evaluate(self, episodes: int, seed: int = 0) -> EvaluationResult:
        """Deterministische Evaluation ohne Exploration und ohne Lernupdates.

        Sie läuft auf einer eigenen, unnormalisierten Environment: Die Returns
        bleiben damit in Originaleinheiten und sind mit der Schwelle von +300
        vergleichbar.
        """
        if self.model is None:
            raise RuntimeError("Vor der Evaluation muss ein Modell trainiert sein.")
        env = make_bipedalwalker_env(render_mode=None)
        rewards, lengths = [], []
        goals = falls = timeouts = solutions = 0
        try:
            for index in range(episodes):
                observation, _ = env.reset(seed=seed + index)
                total, length, done, truncated, reward = 0.0, 0, False, False, 0.0
                while not done:
                    action, _ = self.model.predict(
                        self.policy_observation(observation), deterministic=True)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    length += 1
                    done = bool(terminated or truncated)
                finished, fell, timeout, solved = episode_outcome(
                    float(reward), bool(truncated), total)
                rewards.append(total); lengths.append(length)
                goals += int(finished); falls += int(fell)
                timeouts += int(timeout); solutions += int(solved)
        finally:
            env.close()
        return EvaluationResult(
            episodes, mean(rewards), pstdev(rewards), mean(lengths),
            goals / episodes, solutions / episodes, falls / episodes, timeouts / episodes,
        )

    def train(
        self, stop_event: Optional[threading.Event] = None,
        output: Optional[queue.Queue] = None, series: Any = None,
        evaluation_interval: int = 0, evaluation_episodes: int = 5,
        best_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None,
    ) -> list[EpisodeMetric]:
        reset = self.model is None
        if reset:
            self.create_model()
        callback = BipedalWalkerCallback(
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

    def save(self, path: str | Path) -> tuple[Path, Optional[Path], Path]:
        """Speichert genau die Bestandteile, die das Verfahren besitzt.

        PPO ist on-policy: Policy, Value-Netz und Optimizer stecken in der
        SB3-Datei, einen Replay Buffer gibt es nicht. TD3 und SAC sichern ihn
        zusätzlich; bei SAC gehört der gelernte Temperaturparameter zum Modell.
        Bei aktiver Normalisierung kommen die laufenden Statistiken dazu – ohne
        sie verhielte sich ein geladenes Modell anders als das gespeicherte.
        """
        if self.model is None:
            raise RuntimeError("Es gibt kein Modell zum Speichern.")
        base = Path(path).with_suffix("")
        model_path = base.with_suffix(".zip")
        metadata_path = base.with_name(base.name + "_metadata.json")
        replay_path: Optional[Path] = None
        self.model.save(base)
        if self.uses_replay_buffer:
            replay_path = base.with_name(base.name + "_replay.pkl")
            self.model.save_replay_buffer(replay_path)
        if self.vec_normalize is not None:
            self.vec_normalize.save(str(base.with_name(base.name + "_vecnormalize.pkl")))
        metadata = {
            "format": 1, "environment": ENV_ID, "hardcore": HARDCORE,
            "algorithm": self.config.algorithm, "replay_buffer": self.uses_replay_buffer,
            "normalization": self.config.normalizes, "config": asdict(self.config),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return model_path, replay_path, metadata_path

    @classmethod
    def load(
        cls, path: str | Path, expected_algorithm: Optional[str] = None
    ) -> "BipedalWalkerWorkbench":
        base = Path(path).with_suffix("")
        metadata_path = base.with_name(base.name + "_metadata.json")
        if not metadata_path.is_file():
            raise ValueError(
                f"Zum Modell fehlt die Metadatendatei '{metadata_path.name}'. "
                "Ein Speicherstand besteht aus Modell, Metadaten und – je nach "
                "Verfahren – Replay Buffer und Normalisierungsstatistiken."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != 1 or metadata.get("environment") != ENV_ID \
                or metadata.get("hardcore") != HARDCORE:
            raise ValueError(
                f"Inkompatibler Checkpoint: erwartet Format 1 für {ENV_ID} "
                f"(hardcore={HARDCORE}), gefunden Format {metadata.get('format')} für "
                f"{metadata.get('environment')} (hardcore={metadata.get('hardcore')})."
            )
        algorithm = metadata.get("algorithm")
        if expected_algorithm is not None and algorithm != expected_algorithm:
            raise ValueError(
                f"Verfahren passt nicht: Der Stand enthält '{algorithm}', der aktive Slot "
                f"verwendet '{expected_algorithm}'."
            )
        workbench = cls(BipedalWalkerConfig.from_dict(metadata["config"]))
        workbench.env = workbench._make_training_env()
        if workbench.config.normalizes:
            statistics = base.with_name(base.name + "_vecnormalize.pkl")
            if not statistics.is_file():
                workbench.close()
                raise ValueError(
                    f"Zum Stand fehlen die Normalisierungsstatistiken "
                    f"'{statistics.name}'. Ohne sie verhielte sich das Modell anders "
                    "als beim Speichern."
                )
            # Die geladenen Statistiken ersetzen die frischen; das Training darf
            # sie danach weiter fortschreiben.
            workbench.env = VecNormalize.load(str(statistics), workbench.env.venv)
            workbench.env.training = True
        workbench.model = ALGORITHM_CLASSES[workbench.config.algorithm].load(
            base.with_suffix(".zip"), env=workbench.env,
        )
        if workbench.uses_replay_buffer:
            replay_path = base.with_name(base.name + "_replay.pkl")
            if not replay_path.is_file():
                workbench.close()
                raise ValueError(
                    f"Zum {workbench.config.algorithm}-Modell fehlt der Replay Buffer "
                    f"'{replay_path.name}'. Ohne ihn ließe sich das Training nicht "
                    "unverändert fortsetzen."
                )
            workbench.model.load_replay_buffer(replay_path)
        return workbench

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
