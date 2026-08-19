"""SB3 learning logic for the Hopper workbench."""

from __future__ import annotations

import json
import math
import queue
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Optional, Sequence

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


ALGORITHMS = ("PPO", "SAC", "TD3")
ALGORITHM_CLASSES: dict[str, type[BaseAlgorithm]] = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
#: PPO ist on-policy und führt keinen Replay Buffer; SAC und TD3 tun es.
OFF_POLICY_ALGORITHMS = frozenset({"SAC", "TD3"})
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

ENV_ID = "Hopper-v5"
#: Bildgröße des offiziellen Renderers. Die Werte entsprechen den
#: Gymnasium-Voreinstellungen und werden trotzdem ausdrücklich mitgegeben: Sie
#: betreffen nur die Darstellung, machen die Framegröße aber eindeutig.
FRAME_WIDTH, FRAME_HEIGHT = 480, 480
#: Offizieller Gymnasium-`reward_threshold` von `Hopper-v5`.
SOLVED_RETURN = 3800.0
#: Zeitlimit des `TimeLimit`-Wrappers aus `gymnasium.make`.
MAX_EPISODE_STEPS = 1000
#: Box(-1, 1, (3,)): Drehmomente für Hüfte, Knie und Sprunggelenk.
ACTION_DIM = 3
JOINT_NAMES = ("Hüfte", "Knie", "Sprunggelenk")
#: Übersetzung aller drei Motoren in `hopper.xml`; das anliegende Moment ist
#: `aᵢ · GEAR` in Newtonmetern.
GEAR = 200.0
#: 11 Beobachtungswerte; die x-Position ist bewusst nicht enthalten.
OBSERVATION_DIM = 11
#: `Hopper-v5` clippt alle sechs Geschwindigkeiten der Observation auf ±10.
VELOCITY_CLIP = 10.0
#: Grenzen des gesunden Zustands (`healthy_z_range`, `healthy_angle_range`).
HEALTHY_MIN_HEIGHT = 0.7
HEALTHY_ANGLE = 0.2

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
    "ortho_init": frozenset({"PPO"}),
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
    "clip_range": "Clip ε_clip", "clip_range_vf": "Clip Value",
    "normalize_advantage": "Advantage normieren", "ortho_init": "Orthogonale Initialisierung",
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
BOOLEAN_FIELDS = frozenset({
    "normalize_advantage", "ortho_init", "use_sde", "normalize_obs", "normalize_reward",
})
OPTIONAL_FLOAT_FIELDS = frozenset({"clip_range_vf", "target_kl"})
CHOICE_FIELDS: dict[str, tuple[str, ...]] = {
    "learning_rate_schedule": SCHEDULES, "activation": tuple(ACTIVATIONS),
    "optimizer": tuple(OPTIMIZERS), "action_noise": ACTION_NOISES, "ent_coef_mode": ENTROPY_MODES,
}
#: `target_entropy` akzeptiert bewusst Text: `auto` oder eine Zahl.
TEXT_FIELDS = frozenset({"target_entropy"})

#: Einheitliche Netzgröße aller Verfahren. Das Zoo-Profil nennt für PPO und –
#: über den SB3-Default – für SAC bereits `256,256`; nur TD3 weicht mit
#: `400,300` ab. Die Vereinheitlichung macht den Vergleich fair.
DEFAULT_NET_ARCH = (256, 256)

#: Einheitliches Trainingsbudget aller Verfahren. Es ist zugleich das Budget,
#: das die Zoo-Profile für alle drei Verfahren nennen – der Standardwert weicht
#: also nicht vom Profil ab und erfüllt die Workbench-Regel, dass alle
#: Verfahren mit demselben Budget starten. Ein solcher Lauf dauert allerdings
#: Stunden; für einen schnellen Eindruck ist der Wert in der UI zu verkleinern.
DEFAULT_TOTAL_TIMESTEPS = 1_000_000
#: Ein Zehntel des Schrittbudgets: ein Standardlauf liefert zehn Stützstellen.
DEFAULT_EVALUATION_INTERVAL = DEFAULT_TOTAL_TIMESTEPS // 10
DEFAULT_EVALUATION_EPISODES = 5
#: Voreinstellung von Stable-Baselines3; die Zoo-Profile übergehen sie nicht.
#: Beim vollen Budget fasst der Buffer damit genau alle gesammelten Übergänge.
#: Das kostet je Off-Policy-Slot rund 210 MB – wer mehrere Slots gleichzeitig
#: laufen lässt und wenig Speicher hat, verkleinert ihn in der UI.
DEFAULT_BUFFER_SIZE = 1_000_000

#: Getunte `Hopper-v4`-Profile des RL Baselines3 Zoo (Stand geprüft am
#: 19.08.2026). Werte, die dort nicht auftauchen, bleiben auf den
#: Voreinstellungen von Stable-Baselines3 (siehe Dataclass-Defaults).
DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    # Der Zoo führt für Hopper einen vollständig getunten PPO-Block – und zwar
    # mit `n_envs: 1`. Der Rollout von 512 Schritten je Update entspricht hier
    # also exakt dem Profil.
    "PPO": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "learning_rate": 9.80828e-5,
        "gamma": 0.999, "n_steps": 512, "batch_size": 32, "n_epochs": 5,
        "gae_lambda": 0.99, "clip_range": 0.2, "ent_coef": 0.00229519,
        "vf_coef": 0.835671, "max_grad_norm": 0.7,
        "log_std_init": -2.0, "ortho_init": False,
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH, "activation": "ReLU",
        # Das Zoo-Profil verlangt `normalize: true`; die elf Werte sind sehr
        # unterschiedlich skaliert und der Return wächst auf mehrere Tausend.
        "normalize_obs": True, "normalize_reward": True,
    },
    # Der Zoo setzt für SAC nur `learning_starts`; alles Weitere ist SB3-Default.
    "SAC": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "learning_rate": 3e-4,
        "gamma": 0.99, "tau": 0.005, "batch_size": 256,
        "buffer_size": DEFAULT_BUFFER_SIZE, "learning_starts": 10_000,
        "train_freq": 1, "gradient_steps": 1, "ent_coef_mode": "auto",
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH, "activation": "ReLU",
    },
    "TD3": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "learning_rate": 1e-3,
        "gamma": 0.99, "tau": 0.005, "batch_size": 256,
        "buffer_size": DEFAULT_BUFFER_SIZE, "learning_starts": 10_000,
        "train_freq": 1, "gradient_steps": 1,
        "action_noise": "normal", "action_noise_sigma": 0.1,
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH, "activation": "ReLU",
    },
}


def make_hopper_env(render_mode: Optional[str] = "rgb_array") -> gymnasium.Env:
    """Offizielles Environment ohne jede Änderung an Physik, Reward und Abbruch.

    Sämtliche Parameter von `HopperEnv` – `forward_reward_weight`,
    `ctrl_cost_weight`, `healthy_reward`, `terminate_when_unhealthy`, die drei
    `healthy_*_range`, `reset_noise_scale` und
    `exclude_current_positions_from_observation` – bleiben bewusst
    unangetastet: Sie zu verstellen wäre Reward Shaping beziehungsweise eine
    Änderung der Abbruchregeln.

    `width` und `height` betreffen ausschließlich die Bildgröße des Renderers.
    """
    kwargs: dict[str, Any] = {"width": FRAME_WIDTH, "height": FRAME_HEIGHT}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gymnasium.make(ENV_ID, **kwargs)


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
class HopperConfig:
    """Vollständige Konfiguration eines Verfahrensslots.

    Die Dataclass-Defaults entsprechen den Voreinstellungen von
    Stable-Baselines3. `default_config()` legt darüber das Zoo-Profil des
    jeweiligen Algorithmus.
    """

    algorithm: str = "PPO"
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "konstant"
    batch_size: int = 64
    gamma: float = 0.99
    seed: Optional[int] = 42
    actor_arch: tuple[int, ...] = DEFAULT_NET_ARCH
    critic_arch: tuple[int, ...] = DEFAULT_NET_ARCH
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
    ortho_init: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    log_std_init: float = 0.0
    # PPO und SAC
    use_sde: bool = False
    sde_sample_freq: int = -1
    # SAC und TD3
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
                    "Gültig: mindestens eine positive Ganzzahl, z. B. '256,256'."
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
                f"Action Noise: '{self.action_noise}' ist unbekannt. "
                f"Gültig: {', '.join(ACTION_NOISES)}."
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
            raise ValueError(
                f"Target-Noise σ_t: '{self.target_policy_noise}' ist ungültig. Gültig: ≥ 0.")
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
                    f"Gültig: 'auto' (entspricht {-ACTION_DIM}) oder eine Zahl."
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
        """`auto` heißt bei SAC `-dim(A)`; bei drei Actions also -3."""
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
            # PPO-Zoo-Profil setzt sie für Hopper auf -2.
            kwargs["log_std_init"] = self.log_std_init
        if self.algorithm == "PPO":
            # Initialisierung gehört laut Workbench in die UI; das Zoo-Profil
            # schaltet die orthogonale Initialisierung für Hopper ab.
            kwargs["ortho_init"] = self.ortho_init
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
                "clip_range_vf": self.clip_range_vf,
                "normalize_advantage": self.normalize_advantage,
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
    def from_dict(cls, values: dict[str, Any]) -> "HopperConfig":
        data = dict(values)
        for key in TUPLE_FIELDS:
            data[key] = tuple(data[key])
        config = cls(**data)
        config.validate()
        return config


def default_config(algorithm: str) -> HopperConfig:
    """Konfiguration mit dem Zoo-Profil des Algorithmus."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Verfahren: '{algorithm}' ist unbekannt. Gültig: {', '.join(ALGORITHMS)}.")
    return HopperConfig(algorithm=algorithm, **DEFAULT_PROFILES[algorithm])


def format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "ja" if value else "nein"
    return str(value)


def config_differences(configs: Sequence[HopperConfig]) -> list[tuple[str, list[str]]]:
    """Parameter, in denen sich die Slots unterscheiden – je Zeile alle Werte.

    Verglichen werden ausschließlich Parameter, die **alle** beteiligten
    Verfahren besitzen. Ein Parameter, den nur ein Teil von ihnen kennt, ist
    keine Konfigurationsentscheidung, sondern eine Folge der Verfahrenswahl; er
    steht vollständig im jeweiligen Verfahrenstab. Bei mehrfach demselben
    Algorithmus umfasst das automatisch alle relevanten Parameter.
    """
    if len(configs) < 2:
        return []
    differences: list[tuple[str, list[str]]] = []
    for field in fields(configs[0]):
        name = field.name
        if name == "algorithm" or not all(config.uses(name) for config in configs):
            continue
        values = [getattr(config, name) for config in configs]
        if any(value != values[0] for value in values[1:]):
            differences.append((FIELD_LABELS.get(name, name), [format_value(v) for v in values]))
    return differences


@dataclass(frozen=True)
class EpisodeMetric:
    episode: int
    reward: float
    length: int
    survived: bool
    fell: bool
    solved: bool
    timesteps: int


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    reward_std: float
    mean_length: float
    survive_rate: float
    fall_rate: float
    solved_rate: float
    mean_speed: float
    mean_distance: float


def episode_outcome(
    terminated: bool, truncated: bool, episode_return: float
) -> tuple[bool, bool, bool]:
    """Ausgang einer beendeten Episode als (durchgehalten, Sturz, gelöst).

    `Hopper-v5` kennt weder Terminalbonus noch Terminalstrafe: Ein Sturz kostet
    nur die Rewards der Schritte, die nicht mehr stattfinden. `terminated`
    bedeutet deshalb immer „ungesund geworden", also Sturz; `truncated` heißt,
    dass der Roboter die vollen 1000 Schritte durchgehalten hat. Anders als bei
    BipedalWalker ist das Zeitlimit hier der **gute** Ausgang.
    """
    fell = bool(terminated) and not bool(truncated)
    survived = bool(truncated) and not bool(terminated)
    solved = episode_return >= SOLVED_RETURN
    return survived, fell, solved


def observation_readout(observation: np.ndarray) -> dict[str, Any]:
    """Observation in anzeigefreundliche Größen zerlegen.

    Anders als bei BipedalWalker sind das echte physikalische Größen in Meter
    und Radiant. Die sechs Geschwindigkeiten sind im Environment allerdings auf
    ±10 geclippt: Ein Wert von genau ±10 bedeutet „mindestens so schnell".
    Die x-Position steht nicht in der Observation.
    """
    values = np.asarray(observation, dtype=float).reshape(-1)
    velocities = [float(values[index]) for index in range(5, 11)]
    return {
        "height": float(values[0]),
        "torso_angle": float(values[1]),
        "torso_angle_degrees": math.degrees(float(values[1])),
        "joint_angles": [float(values[2]), float(values[3]), float(values[4])],
        "vx": float(values[5]),
        "vz": float(values[6]),
        "torso_angular_velocity": float(values[7]),
        "joint_velocities": [float(values[8]), float(values[9]), float(values[10])],
        "velocities": velocities,
        "velocity_clipped": any(abs(value) >= VELOCITY_CLIP for value in velocities),
        "healthy_height": float(values[0]) > HEALTHY_MIN_HEIGHT,
        "healthy_angle": abs(float(values[1])) < HEALTHY_ANGLE,
    }


def action_readout(action: np.ndarray) -> dict[str, Any]:
    """Die drei Drehmomente in ihre Gelenkbedeutung übersetzen.

    Das Vorzeichen bestimmt die Drehrichtung, der Betrag mal `GEAR` das
    anliegende Moment in Newtonmetern. Ein Wert von 0 bedeutet also nicht
    „Gelenk hält die Position", sondern „kein Moment". Werte außerhalb [-1, 1]
    clippt das Environment (`ctrlrange="-1 1"`).
    """
    values = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
    joints = []
    for name, value in zip(JOINT_NAMES, values):
        direction = "—" if value == 0 else ("+" if value > 0 else "−")
        torque = abs(float(value)) * GEAR
        joints.append({
            "joint": name,
            "raw": float(value),
            "direction": direction,
            "torque": torque,
            "text": "kein Moment" if value == 0 else f"{direction} {torque:5.1f} N·m",
        })
    return {"raw": [float(value) for value in values], "joints": joints}


def reward_readout(info: dict[str, Any]) -> dict[str, float]:
    """Die drei Reward-Anteile direkt aus `info` übernehmen.

    `Hopper-v5` liefert sie als `reward_survive`, `reward_forward` und
    `reward_ctrl` (bereits negativ). Sie werden übernommen und nicht
    nachgerechnet.
    """
    return {
        "survive": float(info.get("reward_survive", 0.0)),
        "forward": float(info.get("reward_forward", 0.0)),
        "ctrl": float(info.get("reward_ctrl", 0.0)),
        "x_position": float(info.get("x_position", 0.0)),
        "x_velocity": float(info.get("x_velocity", 0.0)),
    }


class HopperCallback(BaseCallback):
    """Schreibt Episodenmetriken und Zwischenevaluationen in die GUI-Queue.

    `series` kennzeichnet den Verfahrensslot und den Lauftyp, damit Einzel- und
    Vergleichsläufe aller Slots dieselbe Queue teilen können.
    """

    def __init__(
        self, stop_event: threading.Event, output: Optional[queue.Queue] = None,
        episode_offset: int = 0, series: Any = None,
        evaluation_interval: int = 0, evaluation_episodes: int = DEFAULT_EVALUATION_EPISODES,
        evaluation_seed: Optional[int] = None,
        evaluator: Optional[Callable[[int, int], EvaluationResult]] = None,
        best_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None,
        episode_callback: Optional[Callable[[EpisodeMetric], None]] = None,
    ) -> None:
        super().__init__(verbose=0)
        self.stop_event, self.output = stop_event, output
        self.episode_offset, self.series = episode_offset, series
        self.evaluation_interval, self.evaluation_episodes = evaluation_interval, evaluation_episodes
        self.evaluation_seed = evaluation_seed
        self.evaluator, self.best_callback = evaluator, best_callback
        self.episode_callback = episode_callback
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
        Werte; ein Return ließe sich daran nicht mehr mit der Schwelle von 3800
        vergleichen.
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
            survived, fell, solved = episode_outcome(not truncated, truncated, total)
            metric = EpisodeMetric(
                self.episode_offset + len(self.metrics) + 1, total, length,
                survived, fell, solved, int(self.num_timesteps),
            )
            self.metrics.append(metric)
            # Noch im Worker-Thread und damit synchron zum Lernstand: Nur hier
            # gehört die Policy wirklich zu dieser Episode.
            if self.episode_callback is not None:
                self.episode_callback(metric)
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


class HopperWorkbench:
    """Ein Verfahrensslot: Environment, Modell, Historie, Checkpoints."""

    def __init__(self, config: Optional[HopperConfig] = None) -> None:
        self.config = config or default_config("PPO")
        self.config.validate()
        self.env: Optional[Any] = None
        self.model: Optional[BaseAlgorithm] = None
        self.history: list[EpisodeMetric] = []
        # Lernstand der bisher besten Episode. Gespeichert wird ausschließlich
        # dieser eine Stand je Slot – ein Verlauf über alle Episoden kostete bei
        # großen Budgets Gigabytes.
        self.best_episode: Optional[EpisodeMetric] = None
        self._best_policy_state: Optional[dict[str, Any]] = None
        self._best_lock = threading.Lock()

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
        Schwelle von 3800 überhaupt ablesen lässt.
        """
        env: Any = DummyVecEnv([lambda: Monitor(make_hopper_env(render_mode=None))])
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
        with self._best_lock:
            self.best_episode = None
            self._best_policy_state = None
        return self.model

    def remember_episode(self, metric: EpisodeMetric) -> None:
        """Sichert den Lernstand, sobald eine Episode alle bisherigen schlägt.

        Läuft im Worker-Thread, direkt nach dem Episodenende: Nur dort gehört
        die Policy tatsächlich zu dieser Episode. Kopiert wird eine losgelöste
        Fassung, damit der weiterlaufende Optimizer sie nicht mehr verändert.
        """
        if self.model is None:
            return
        with self._best_lock:
            if self.best_episode is not None and metric.reward <= self.best_episode.reward:
                return
            self.best_episode = metric
            self._best_policy_state = {
                name: value.detach().clone()
                for name, value in self.model.policy.state_dict().items()
            }

    def best_snapshot(self) -> Optional[tuple[EpisodeMetric, dict[str, Any]]]:
        """Beste Episode und die zugehörige Policy-Kopie, falls vorhanden."""
        with self._best_lock:
            if self.best_episode is None or self._best_policy_state is None:
                return None
            return self.best_episode, self._best_policy_state

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
        bleiben damit in Originaleinheiten und sind mit der Schwelle von 3800
        vergleichbar.
        """
        if self.model is None:
            raise RuntimeError("Vor der Evaluation muss ein Modell trainiert sein.")
        env = make_hopper_env(render_mode=None)
        rewards, lengths, speeds, distances = [], [], [], []
        survivals = falls = solutions = 0
        try:
            for index in range(episodes):
                observation, _ = env.reset(seed=seed + index)
                total, length = 0.0, 0
                terminated = truncated = False
                step_speeds: list[float] = []
                distance = 0.0
                while not (terminated or truncated):
                    action, _ = self.model.predict(
                        self.policy_observation(observation), deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total += float(reward)
                    length += 1
                    # x-Position und x-Geschwindigkeit stehen offiziell in
                    # `info`; auf Interna des Environments wird nicht zugegriffen.
                    step_speeds.append(float(info.get("x_velocity", 0.0)))
                    distance = float(info.get("x_position", distance))
                survived, fell, solved = episode_outcome(
                    bool(terminated), bool(truncated), total)
                rewards.append(total)
                lengths.append(length)
                speeds.append(mean(step_speeds) if step_speeds else 0.0)
                distances.append(distance)
                survivals += int(survived)
                falls += int(fell)
                solutions += int(solved)
        finally:
            env.close()
        return EvaluationResult(
            episodes, mean(rewards), pstdev(rewards), mean(lengths),
            survivals / episodes, falls / episodes, solutions / episodes,
            mean(speeds), mean(distances),
        )

    def train(
        self, stop_event: Optional[threading.Event] = None,
        output: Optional[queue.Queue] = None, series: Any = None,
        evaluation_interval: int = 0,
        evaluation_episodes: int = DEFAULT_EVALUATION_EPISODES,
        best_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None,
    ) -> list[EpisodeMetric]:
        reset = self.model is None
        if reset:
            self.create_model()
        callback = HopperCallback(
            stop_event or threading.Event(), output, len(self.history), series,
            evaluation_interval, evaluation_episodes, self.config.seed,
            self.evaluate, best_callback, self.remember_episode,
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
        SB3-Datei, einen Replay Buffer gibt es nicht. SAC und TD3 sichern ihn
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
            "format": 1, "environment": ENV_ID,
            "algorithm": self.config.algorithm, "replay_buffer": self.uses_replay_buffer,
            "normalization": self.config.normalizes, "config": asdict(self.config),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return model_path, replay_path, metadata_path

    @classmethod
    def load(
        cls, path: str | Path, expected_algorithm: Optional[str] = None
    ) -> "HopperWorkbench":
        base = Path(path).with_suffix("")
        metadata_path = base.with_name(base.name + "_metadata.json")
        if not metadata_path.is_file():
            raise ValueError(
                f"Zum Modell fehlt die Metadatendatei '{metadata_path.name}'. "
                "Ein Speicherstand besteht aus Modell, Metadaten und – je nach "
                "Verfahren – Replay Buffer und Normalisierungsstatistiken."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != 1 or metadata.get("environment") != ENV_ID:
            raise ValueError(
                f"Inkompatibler Checkpoint: erwartet Format 1 für {ENV_ID}, gefunden "
                f"Format {metadata.get('format')} für {metadata.get('environment')}."
            )
        algorithm = metadata.get("algorithm")
        if expected_algorithm is not None and algorithm != expected_algorithm:
            raise ValueError(
                f"Verfahren passt nicht: Der Stand enthält '{algorithm}', der aktive Slot "
                f"verwendet '{expected_algorithm}'."
            )
        workbench = cls(HopperConfig.from_dict(metadata["config"]))
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
