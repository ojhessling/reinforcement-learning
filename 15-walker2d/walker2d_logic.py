"""SB3 learning logic for the Walker2d workbench."""

from __future__ import annotations

import json
import math
import pickle
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


#: `PPO` gehört bewusst nicht dazu: Ohne On-Policy-Verfahren steht in diesem
#: Projekt genau eine Asymmetrie zur Debatte – gradientenbasiert gegen
#: gradientenfrei – statt zweier gleichzeitig.
ALGORITHMS = ("TD3", "SAC", "CMA-ES")
#: Verfahren, die Stable-Baselines3 stellt. `CMA-ES` kennt SB3 nicht; dafür
#: tritt `pycma` als Referenzbibliothek an seine Stelle.
SB3_ALGORITHMS = frozenset({"TD3", "SAC"})
ALGORITHM_CLASSES: dict[str, type[BaseAlgorithm]] = {"SAC": SAC, "TD3": TD3}
#: Beide SB3-Verfahren dieses Projekts sind off-policy und führen einen
#: Replay Buffer; `CMA-ES` führt keinen.
OFF_POLICY_ALGORITHMS = frozenset({"SAC", "TD3"})
EVOLUTIONARY_ALGORITHMS = frozenset({"CMA-ES"})

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

ENV_ID = "Walker2d-v5"
#: Bildgröße des offiziellen Renderers. Die Werte entsprechen den
#: Gymnasium-Voreinstellungen und werden trotzdem ausdrücklich mitgegeben: Sie
#: betreffen nur die Darstellung, machen die Framegröße aber eindeutig.
FRAME_WIDTH, FRAME_HEIGHT = 480, 480
#: `Walker2d-v5` führt in der Gymnasium-Registry **keinen** `reward_threshold`.
#: Eine offizielle Gelöst-Schwelle gibt es also nicht. Dies ist eine
#: projektinterne Zielmarke, begründet durch die Benchmarkwerte des RL
#: Baselines3 Zoo (SAC 3863, TD3 4718). Sie wird nirgends „gelöst" genannt.
TARGET_RETURN = 4000.0
#: Zeitlimit des `TimeLimit`-Wrappers aus `gymnasium.make`.
MAX_EPISODE_STEPS = 1000
#: Box(-1, 1, (6,)): Drehmomente für sechs Gelenke.
ACTION_DIM = 6
JOINT_NAMES = ("Hüfte rechts", "Knie rechts", "Fuß rechts",
               "Hüfte links", "Knie links", "Fuß links")
#: Übersetzung aller sechs Motoren aus `walker2d_v5.xml` – hier einheitlich,
#: anders als bei HalfCheetah. Ein Test vergleicht sie mit `actuator_gear`.
GEAR = 100.0
#: 17 Beobachtungswerte; die x-Position ist bewusst nicht enthalten.
OBSERVATION_DIM = 17
#: `Walker2d-v5` clippt alle neun Geschwindigkeiten der Observation auf ±10.
VELOCITY_CLIP = 10.0
#: Gewichte des Rewards.
FORWARD_REWARD_WEIGHT = 1.0
CTRL_COST_WEIGHT = 1e-3
HEALTHY_REWARD = 1.0
#: Grenzen des gesunden Zustands. Die Höhe ist nach **oben und unten** begrenzt:
#: Ein Sprung über 2,0 m beendet die Episode ebenso wie ein Sturz.
HEALTHY_Z_RANGE = (0.8, 2.0)
HEALTHY_ANGLE_RANGE = (-1.0, 1.0)
#: Simulationsschritt: `frame_skip = 4` mal `timestep = 0.002`.
STEP_DURATION = 0.008

#: Zuordnung Parameter -> Verfahren, die ihn besitzen. Die GUI zeigt in einem
#: Verfahrenstab ausschließlich Parameter des dort gewählten Algorithmus; die
#: Vergleichs-Summary bildet daraus die Liste der echten Unterschiede.
_ALL = frozenset(ALGORITHMS)
_SB3 = SB3_ALGORITHMS
_CMA = EVOLUTIONARY_ALGORITHMS
#: Ausnahmslos in jedem Tab stehen nur `total_timesteps` und `seed`; alles
#: andere nur, wo das Verfahren es kennt. `CMA-ES` besitzt weder `learning_rate`
#: noch `batch_size` noch `gamma` – die Felder werden deshalb weggelassen und
#: nicht deaktiviert mitgeschleppt.
FIELD_ALGORITHMS: dict[str, frozenset[str]] = {
    "total_timesteps": _ALL, "seed": _ALL, "actor_arch": _ALL, "activation": _ALL,
    "learning_rate": _SB3, "learning_rate_schedule": _SB3, "batch_size": _SB3,
    "gamma": _SB3, "critic_arch": _SB3,
    "optimizer": _SB3, "optimizer_eps": _SB3, "optimizer_weight_decay": _SB3,
    "log_std_init": frozenset({"SAC"}),
    "use_sde": frozenset({"SAC"}), "sde_sample_freq": frozenset({"SAC"}),
    "buffer_size": OFF_POLICY_ALGORITHMS, "learning_starts": OFF_POLICY_ALGORITHMS,
    "tau": OFF_POLICY_ALGORITHMS, "train_freq": OFF_POLICY_ALGORITHMS,
    "gradient_steps": OFF_POLICY_ALGORITHMS,
    "action_noise": OFF_POLICY_ALGORITHMS, "action_noise_sigma": OFF_POLICY_ALGORITHMS,
    "policy_delay": frozenset({"TD3"}), "target_policy_noise": frozenset({"TD3"}),
    "target_noise_clip": frozenset({"TD3"}),
    "ent_coef_mode": frozenset({"SAC"}), "ent_coef_value": frozenset({"SAC"}),
    "target_entropy": frozenset({"SAC"}), "target_update_interval": frozenset({"SAC"}),
    # CMA-ES
    "sigma0": _CMA, "popsize": _CMA, "episodes_per_candidate": _CMA, "diagonal": _CMA,
    "normalize_obs": _ALL, "clip_obs": _ALL,
    # Rangbasierte Verfahren werten nur die Reihenfolge der Renditen aus; jede
    # monotone Umskalierung ist wirkungslos. Für CMA-ES wäre das Feld eine Attrappe.
    "normalize_reward": _SB3, "clip_reward": _SB3,
}
FIELD_LABELS: dict[str, str] = {
    "total_timesteps": "Trainingsschritte N", "learning_rate": "Lernrate α",
    "learning_rate_schedule": "LR-Verlauf", "batch_size": "Batch-Größe B",
    "gamma": "Diskontfaktor γ", "seed": "Zufallsstart s", "actor_arch": "Actor-Hidden h_π",
    "critic_arch": "Critic-Hidden h_q", "activation": "Aktivierung φ", "optimizer": "Optimizer",
    "optimizer_eps": "Optimizer ε", "optimizer_weight_decay": "Weight Decay λ",
    "log_std_init": "log σ₀", "use_sde": "gSDE nutzen",
    "sde_sample_freq": "gSDE-Frequenz", "buffer_size": "Replay Buffer |D|",
    "learning_starts": "Lernstart t₀", "tau": "Soft-Update τ", "train_freq": "Trainingsfrequenz fₜ",
    "gradient_steps": "Gradientenschritte G", "action_noise": "Action Noise",
    "action_noise_sigma": "Noise σ", "policy_delay": "Policy Delay d",
    "target_policy_noise": "Target-Noise σ_t", "target_noise_clip": "Noise-Clip c",
    "ent_coef_mode": "Entropie α", "ent_coef_value": "Startwert α",
    "target_entropy": "Zielentropie H*", "target_update_interval": "Target-Intervall C",
    "sigma0": "Schrittweite σ₀", "popsize": "Population λ",
    "episodes_per_candidate": "Episoden je Kandidat", "diagonal": "Diagonalvariante",
    "normalize_obs": "Beobachtungen normalisieren", "normalize_reward": "Rewards normalisieren",
    "clip_obs": "Clip Beobachtungen", "clip_reward": "Clip Reward",
}

INTEGER_FIELDS = frozenset({
    "total_timesteps", "batch_size", "sde_sample_freq", "buffer_size",
    "learning_starts", "train_freq", "gradient_steps", "policy_delay",
    "target_update_interval", "episodes_per_candidate",
})
#: `actor_arch` trägt bei CMA-ES die Policy selbst; leer bedeutet dort linear.
TUPLE_FIELDS = frozenset({"actor_arch", "critic_arch"})
BOOLEAN_FIELDS = frozenset({
    "use_sde", "normalize_obs", "normalize_reward", "diagonal",
})
OPTIONAL_FLOAT_FIELDS = frozenset()
CHOICE_FIELDS: dict[str, tuple[str, ...]] = {
    "learning_rate_schedule": SCHEDULES, "activation": tuple(ACTIVATIONS),
    "optimizer": tuple(OPTIMIZERS), "action_noise": ACTION_NOISES, "ent_coef_mode": ENTROPY_MODES,
}
#: `target_entropy` und `popsize` akzeptieren bewusst Text: `auto` oder Zahl.
TEXT_FIELDS = frozenset({"target_entropy", "popsize"})

#: Einheitliche Netzgröße aller Verfahren. Das Zoo-Profil nennt für PPO und –
#: über den SB3-Default – für SAC bereits `256,256`; nur TD3 weicht mit
#: `400,300` ab. Die Vereinheitlichung macht den Vergleich fair und ist die
#: einzige Abweichung von den Profilen.
DEFAULT_NET_ARCH = (256, 256)
#: Netz von TD3 laut Zoo-Profil. Die Netzgrößen werden hier bewusst **nicht**
#: vereinheitlicht: Mit CMA-ES ist eine gemeinsame Architektur ohnehin
#: ausgeschlossen, also bringt es nichts, ein getuntes Profil zu überschreiben.
TD3_NET_ARCH = (400, 300)
#: Policy von CMA-ES: leer heißt linear, `a = tanh(W·s + b)`. Bei 17 Werten und
#: 6 Actions sind das 6·17+6 = 108 Parameter. Größere Netze scheitern an der
#: n×n-Kovarianzmatrix von CMA-ES – bei `256,256` wären es rund 72.000
#: Parameter und über 40 GB Matrix.
CMA_NET_ARCH: tuple[int, ...] = ()

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
DEFAULT_BUFFER_SIZE = 1_000_000

#: `Walker2d-v4`-Profile des RL Baselines3 Zoo (Stand geprüft am 21.08.2026).
#: Werte, die dort nicht auftauchen, bleiben auf den Voreinstellungen von
#: Stable-Baselines3 (siehe Dataclass-Defaults). Für CMA-ES gibt es kein
#: Zoo-Profil; die Werte folgen `pycma` und der Literatur.
#:
#: Kein Profil dieses Projekts verlangt `normalize: true` – das wäre das
#: PPO-Profil gewesen, und PPO ist nicht dabei. `CMA-ES` normalisiert die
#: Beobachtungen trotzdem, aus eigenem Bedarf.
DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "TD3": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "learning_rate": 1e-3,
        "gamma": 0.99, "tau": 0.005, "batch_size": 256,
        "buffer_size": DEFAULT_BUFFER_SIZE, "learning_starts": 10_000,
        "train_freq": 1, "gradient_steps": 1,
        "action_noise": "normal", "action_noise_sigma": 0.1,
        "actor_arch": TD3_NET_ARCH, "critic_arch": TD3_NET_ARCH, "activation": "ReLU",
    },
    # Der Zoo setzt für SAC nur `learning_starts`; alles Weitere ist SB3-Default.
    "SAC": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "learning_rate": 3e-4,
        "gamma": 0.99, "tau": 0.005, "batch_size": 256,
        "buffer_size": DEFAULT_BUFFER_SIZE, "learning_starts": 10_000,
        "train_freq": 1, "gradient_steps": 1, "ent_coef_mode": "auto",
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH, "activation": "ReLU",
    },
    "CMA-ES": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS,
        "actor_arch": CMA_NET_ARCH, "activation": "Tanh",
        "sigma0": 0.5, "popsize": "auto", "episodes_per_candidate": 1,
        "diagonal": False,
        # Ohne laufende Beobachtungsstatistik arbeitet eine lineare Policy auf
        # den sehr unterschiedlich skalierten 17 Werten kaum.
        "normalize_obs": True, "clip_obs": 10.0,
    },
}


def make_walker2d_env(render_mode: Optional[str] = "rgb_array") -> gymnasium.Env:
    """Offizielles Environment ohne jede Änderung an Physik und Reward.

    Sämtliche Parameter von `Walker2dEnv` – `forward_reward_weight`,
    `ctrl_cost_weight`, `reset_noise_scale` und
    `exclude_current_positions_from_observation` – bleiben bewusst
    unangetastet: Sie zu verstellen wäre Reward Shaping.

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
class Walker2dConfig:
    """Vollständige Konfiguration eines Verfahrensslots.

    Die Dataclass-Defaults entsprechen den Voreinstellungen von
    Stable-Baselines3. `default_config()` legt darüber das Zoo-Profil des
    jeweiligen Algorithmus.
    """

    algorithm: str = "TD3"
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "konstant"
    batch_size: int = 64
    gamma: float = 0.99
    seed: Optional[int] = 42
    #: Bei CMA-ES ist das die Policy selbst; leer bedeutet dort linear.
    actor_arch: tuple[int, ...] = DEFAULT_NET_ARCH
    critic_arch: tuple[int, ...] = DEFAULT_NET_ARCH
    activation: str = "ReLU"
    optimizer: str = "Adam"
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    # SAC
    log_std_init: float = 0.0
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
    # CMA-ES
    sigma0: float = 0.5
    #: `auto` entspricht der pycma-Formel `4 + ⌊3·ln n⌋`.
    popsize: str = "auto"
    episodes_per_candidate: int = 1
    #: Diagonalvariante (sep-CMA-ES); nötig erst bei großen Netzen.
    diagonal: bool = False
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
        positive_integers = [("Trainingsschritte N", self.total_timesteps)]
        if self.algorithm in EVOLUTIONARY_ALGORITHMS:
            positive_integers.append(("Episoden je Kandidat", self.episodes_per_candidate))
        else:
            positive_integers += [
                ("Batch-Größe B", self.batch_size),
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
        if self.uses("learning_rate"):
            if self.learning_rate <= 0:
                raise ValueError(f"Lernrate α: '{self.learning_rate}' ist ungültig. Gültig: > 0.")
            if self.learning_rate_schedule not in SCHEDULES:
                raise ValueError(
                    f"LR-Verlauf: '{self.learning_rate_schedule}' ist ungültig. "
                    f"Gültig: {', '.join(SCHEDULES)}."
                )
        if self.uses("gamma") and not 0 < self.gamma <= 1:
            raise ValueError(f"Diskontfaktor γ: '{self.gamma}' ist ungültig. Gültig: 0 < γ ≤ 1.")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"Aktivierung φ: '{self.activation}' ist unbekannt. Gültig: {', '.join(ACTIVATIONS)}."
            )
        if self.uses("optimizer"):
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
        architectures = [("Actor-Hidden h_π", self.actor_arch)]
        if self.uses("critic_arch"):
            architectures.append(("Critic-Hidden h_q", self.critic_arch))
        for label, architecture in architectures:
            # Bei CMA-ES bedeutet eine leere Liste bewusst „lineare Policy".
            if any(size <= 0 for size in architecture):
                raise ValueError(
                    f"{label}: '{architecture}' ist ungültig. "
                    "Gültig: positive Ganzzahlen, z. B. '256,256'."
                )
            if not architecture and self.algorithm not in EVOLUTIONARY_ALGORITHMS:
                raise ValueError(
                    f"{label}: darf für {self.algorithm} nicht leer sein. "
                    "Gültig: mindestens eine positive Ganzzahl, z. B. '256,256'."
                )
        checks = [("Clip Beobachtungen", self.clip_obs)]
        if self.uses("clip_reward"):
            checks.append(("Clip Reward", self.clip_reward))
        for label, value in checks:
            if value <= 0:
                raise ValueError(f"{label}: '{value}' ist ungültig. Gültig: > 0.")
        if self.algorithm == "CMA-ES":
            self._validate_cma()
        elif self.algorithm == "TD3":
            self._validate_off_policy()
            self._validate_td3()
        else:
            self._validate_off_policy()
            self._validate_sac()

    def _validate_cma(self) -> None:
        if self.sigma0 <= 0:
            raise ValueError(f"Schrittweite σ₀: '{self.sigma0}' ist ungültig. Gültig: > 0.")
        text = self.popsize.strip()
        if text != "auto":
            try:
                value = int(text)
            except ValueError as error:
                raise ValueError(
                    f"Population λ: '{self.popsize}' ist ungültig. "
                    "Gültig: 'auto' oder eine ganze Zahl ≥ 2."
                ) from error
            if value < 2:
                raise ValueError(
                    f"Population λ: '{value}' ist zu klein. Aus einem einzigen Kandidaten "
                    "lässt sich keine Rangfolge bilden. Gültig: 'auto' oder ≥ 2."
                )
        # Ohne eine vollständige Generation gibt es keinen einzigen
        # Update-Schritt; das Budget muss mindestens eine zulassen.
        if self.total_timesteps < self.episodes_per_candidate * self.resolved_popsize():
            raise ValueError(
                f"Trainingsschritte N: '{self.total_timesteps}' reicht für keine vollständige "
                f"Generation. Gültig: mindestens λ · Episoden je Kandidat = "
                f"{self.resolved_popsize() * self.episodes_per_candidate}."
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
        if self.algorithm == "SAC":
            # Beide Policies kennen die initiale Streuung der Gauß-Policy; das
            # PPO-Zoo-Profil setzt sie für Walker2d auf -2.
            kwargs["log_std_init"] = self.log_std_init
        return kwargs

    def resolved_popsize(self) -> int:
        """Populationsgröße λ; `auto` entspricht der pycma-Formel `4 + ⌊3·ln n⌋`."""
        text = self.popsize.strip()
        if text != "auto":
            return int(text)
        return 4 + int(3 * math.log(max(2, self.parameter_count())))

    def parameter_count(self) -> int:
        """Zahl der Policy-Gewichte, die CMA-ES optimiert.

        Der Wert entscheidet über die Rechenbarkeit: CMA-ES führt eine
        `n × n`-Kovarianzmatrix.
        """
        sizes = [OBSERVATION_DIM, *self.actor_arch, ACTION_DIM]
        return sum(a * b + b for a, b in zip(sizes, sizes[1:]))

    def model_kwargs(self) -> dict[str, Any]:
        """Konstruktorargumente – ausschließlich Schlüssel, die der jeweilige
        Stable-Baselines3-Algorithmus tatsächlich kennt."""
        if self.algorithm in EVOLUTIONARY_ALGORITHMS:
            raise ValueError(f"{self.algorithm} ist kein Stable-Baselines3-Verfahren.")
        common = {
            "learning_rate": self.learning_rate_value(), "gamma": self.gamma,
            "batch_size": self.batch_size, "seed": self.seed,
            "policy_kwargs": self.policy_kwargs(), "verbose": 0,
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
    def from_dict(cls, values: dict[str, Any]) -> "Walker2dConfig":
        data = dict(values)
        for key in TUPLE_FIELDS:
            data[key] = tuple(data[key])
        config = cls(**data)
        config.validate()
        return config


def default_config(algorithm: str) -> Walker2dConfig:
    """Konfiguration mit dem Zoo-Profil des Algorithmus."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Verfahren: '{algorithm}' ist unbekannt. Gültig: {', '.join(ALGORITHMS)}.")
    return Walker2dConfig(algorithm=algorithm, **DEFAULT_PROFILES[algorithm])


def format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "ja" if value else "nein"
    return str(value)


def config_differences(configs: Sequence[Walker2dConfig]) -> list[tuple[str, list[str]]]:
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
    #: Return ≥ `TARGET_RETURN`. Bewusst nicht `solved` genannt: Es gibt für
    #: `Walker2d-v5` keine offizielle Gelöst-Schwelle.
    reached_target: bool
    timesteps: int
    #: Mittleres `vₓ` der Episode in m/s aus `info["x_velocity"]`.
    mean_speed: float = 0.0
    #: Zurückgelegte Strecke am Episodenende aus `info["x_position"]`.
    distance: float = 0.0


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    reward_std: float
    mean_length: float
    survive_rate: float
    fall_rate: float
    target_rate: float
    mean_speed: float
    mean_distance: float


def episode_outcome(
    terminated: bool, truncated: bool, episode_return: float
) -> tuple[bool, bool, bool]:
    """Ausgang einer beendeten Episode als (durchgehalten, Sturz, Ziel erreicht).

    `Walker2d-v5` terminiert, sobald der Roboter ungesund wird: Höhe außerhalb
    `(0,8; 2,0) m` oder Rumpfwinkel außerhalb `(−1,0; 1,0) rad`. Es gibt weder
    Terminalbonus noch Terminalstrafe – ein Sturz kostet nur die Rewards der
    Schritte, die nicht mehr stattfinden. `truncated` nach 1000 Schritten heißt
    durchgehalten.

    Der dritte Wert heißt bewusst nicht `solved`: `Walker2d-v5` führt keinen
    `reward_threshold`, `TARGET_RETURN` ist eine Projektkonstante.
    """
    fell = bool(terminated) and not bool(truncated)
    survived = bool(truncated) and not bool(terminated)
    reached_target = episode_return >= TARGET_RETURN
    return survived, fell, reached_target


def observation_readout(observation: np.ndarray) -> dict[str, Any]:
    """Observation in anzeigefreundliche Größen zerlegen.

    Echte physikalische Größen in Meter, Radiant und pro Sekunde. Die neun
    Geschwindigkeiten sind – wie beim Hopper und anders als bei HalfCheetah –
    auf ±10 **geclippt**: Ein Wert von genau ±10 heißt „mindestens so schnell".
    Die x-Position steht nicht in der Observation.
    """
    values = np.asarray(observation, dtype=float).reshape(-1)
    velocities = [float(values[index]) for index in range(8, 17)]
    return {
        "height": float(values[0]),
        "torso_angle": float(values[1]),
        "torso_angle_degrees": math.degrees(float(values[1])),
        "joint_angles": [float(values[index]) for index in range(2, 8)],
        "vx": float(values[8]),
        "vz": float(values[9]),
        "torso_angular_velocity": float(values[10]),
        "joint_velocities": [float(values[index]) for index in range(11, 17)],
        "velocities": velocities,
        "velocity_clipped": any(abs(value) >= VELOCITY_CLIP for value in velocities),
        "healthy_height": HEALTHY_Z_RANGE[0] < float(values[0]) < HEALTHY_Z_RANGE[1],
        "healthy_angle": HEALTHY_ANGLE_RANGE[0] < float(values[1]) < HEALTHY_ANGLE_RANGE[1],
    }


def action_readout(action: np.ndarray) -> dict[str, Any]:
    """Die sechs Drehmomente in ihre Gelenkbedeutung übersetzen.

    Das Vorzeichen bestimmt die Drehrichtung, der Betrag mal `GEAR` das
    anliegende Moment in Newtonmetern. Alle sechs Motoren haben hier dieselbe
    Übersetzung – anders als bei HalfCheetah. Ein Wert von 0 bedeutet „kein
    Moment", nicht „Gelenk hält die Position". Werte außerhalb [-1, 1] clippt
    das Environment (`ctrlrange="-1 1"`).
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


def control_cost(action: np.ndarray) -> float:
    """`ctrl_cost = 0,001 · Σ aᵢ²` – höchstens 0,006 je Schritt."""
    values = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
    return float(CTRL_COST_WEIGHT * np.sum(np.square(values)))


def reward_readout(info: dict[str, Any]) -> dict[str, float]:
    """Die drei Reward-Anteile direkt aus `info` übernehmen.

    `Walker2d-v5` liefert `reward_survive`, `reward_forward` und `reward_ctrl`
    (bereits negativ). Die Werte werden übernommen und nicht nachgerechnet.
    """
    return {
        "survive": float(info.get("reward_survive", 0.0)),
        "forward": float(info.get("reward_forward", 0.0)),
        "ctrl": float(info.get("reward_ctrl", 0.0)),
        "x_position": float(info.get("x_position", 0.0)),
        "x_velocity": float(info.get("x_velocity", 0.0)),
    }


def _read_metadata(base: Path, expected_algorithm: Optional[str] = None) -> dict[str, Any]:
    """Metadaten eines Checkpoints lesen und auf Kompatibilität prüfen.

    Gemeinsam für beide Slot-Typen: Format, Environment-Kennung und Algorithmus
    müssen passen, sonst bleibt der aktive Zustand unverändert.
    """
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
    return metadata


def evaluate_policy(
    act: Callable[[Any, np.ndarray], np.ndarray], snapshot: Any,
    episodes: int, seed: int = 0,
) -> EvaluationResult:
    """Deterministische Evaluation eines Lernstands auf eigenen Episoden.

    Gemeinsam für alle Verfahren: `act` bildet einen Lernstand und eine
    Beobachtung auf eine Action ab. Für die SB3-Verfahren ist das die Policy
    ohne Exploration, für CMA-ES der Verteilungsmittelwert. Die Environment ist
    unnormalisiert, die Returns bleiben damit in Originaleinheiten und sind mit
    `TARGET_RETURN` vergleichbar.
    """
    env = make_walker2d_env(render_mode=None)
    rewards, lengths, speeds, distances = [], [], [], []
    survivals = falls = targets = 0
    try:
        for index in range(episodes):
            observation, _ = env.reset(seed=seed + index)
            total, length = 0.0, 0
            terminated = truncated = False
            speed_sum = distance = 0.0
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step(
                    act(snapshot, observation))
                total += float(reward)
                length += 1
                # Tempo und Strecke stehen offiziell in `info`; auf Interna des
                # Environments wird nicht zugegriffen.
                speed_sum += float(info.get("x_velocity", 0.0))
                distance = float(info.get("x_position", distance))
            survived, fell, reached = episode_outcome(
                bool(terminated), bool(truncated), total)
            rewards.append(total)
            lengths.append(length)
            speeds.append(speed_sum / max(1, length))
            distances.append(distance)
            survivals += int(survived)
            falls += int(fell)
            targets += int(reached)
    finally:
        env.close()
    return EvaluationResult(
        episodes, mean(rewards), pstdev(rewards), mean(lengths),
        survivals / episodes, falls / episodes, targets / episodes,
        mean(speeds), mean(distances),
    )


class Walker2dCallback(BaseCallback):
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
        # Laufende Episodensummen für Tempo, Strecke und Steuerkosten. Sie
        # stammen aus `info` und werden nicht nachgerechnet.
        self.speed_sum = 0.0
        self.distance = 0.0
        self.start_step = 0
        self.next_evaluation = evaluation_interval

    def _on_training_start(self) -> None:
        self.start_step = int(self.model.num_timesteps)

    def _raw_reward(self) -> float:
        """Reward des letzten Schritts in Originaleinheiten.

        Bei aktiver Reward-Normalisierung liefert `locals["rewards"]` skalierte
        Werte; ein Return ließe sich daran nicht mehr mit der Schwelle von 4800
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
        # `x_velocity` und `x_position` stehen offiziell in `info`; die
        # x-Position fehlt in der Observation.
        self.speed_sum += float(info.get("x_velocity", 0.0))
        self.distance = float(info.get("x_position", self.distance))
        if done:
            truncated = bool(info.get("TimeLimit.truncated", False))
            # Der Monitor-Wrapper sitzt innerhalb der Normalisierung und führt
            # den Episoden-Return in Originaleinheiten; er hat Vorrang.
            episode_info = info.get("episode")
            total = float(episode_info["r"]) if episode_info else self.reward
            length = int(episode_info["l"]) if episode_info else self.length
            survived, fell, reached = episode_outcome(not truncated, truncated, total)
            metric = EpisodeMetric(
                self.episode_offset + len(self.metrics) + 1, total, length,
                survived, fell, reached, int(self.num_timesteps),
                self.speed_sum / max(1, self.length), self.distance,
            )
            self.metrics.append(metric)
            # Noch im Worker-Thread und damit synchron zum Lernstand: Nur hier
            # gehört die Policy wirklich zu dieser Episode.
            if self.episode_callback is not None:
                self.episode_callback(metric)
            if self.output is not None:
                self.output.put(("episode", (self.series, metric)))
            self.reward, self.length = 0.0, 0
            self.speed_sum, self.distance = 0.0, 0.0
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


class Walker2dWorkbench:
    """Ein Verfahrensslot: Environment, Modell, Historie, Checkpoints."""

    def __init__(self, config: Optional[Walker2dConfig] = None) -> None:
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
        #: Wiederverwendete Policy-Instanz für `act()`; sie wird nie trainiert.
        self._animation_policy: Any = None

    @property
    def uses_replay_buffer(self) -> bool:
        return self.config.algorithm in OFF_POLICY_ALGORITHMS

    @property
    def generations(self) -> Optional[int]:
        """Nur CMA-ES zählt Generationen; hier gibt es keine."""
        return None

    @property
    def num_timesteps(self) -> int:
        """Ausgeführte Environment-Schritte – gleiche Schnittstelle wie CMA-ES."""
        return 0 if self.model is None else int(self.model.num_timesteps)

    @property
    def vec_normalize(self) -> Optional[VecNormalize]:
        return self.env if isinstance(self.env, VecNormalize) else None

    def _make_training_env(self) -> Any:
        """Vektor-Environment mit Monitor und optionaler Normalisierung.

        Der `Monitor` sitzt bewusst **innerhalb** der Normalisierung: Nur so
        führt er den Episoden-Return in Originaleinheiten, an dem sich die
        Schwelle von 4800 überhaupt ablesen lässt.
        """
        env: Any = DummyVecEnv([lambda: Monitor(make_walker2d_env(render_mode=None))])
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

    # --- Policy-Abstraktion: Die GUI kennt nur diese drei Methoden und muss
    # --- deshalb nicht wissen, ob hinter einem Slot SB3 oder pycma steckt.

    def current_snapshot(self) -> Any:
        """Losgelöste Kopie des aktuellen Lernstands für Anzeige und Evaluation."""
        if self.model is None:
            raise RuntimeError("Es gibt noch keinen Lernstand.")
        return {name: value.detach().clone()
                for name, value in self.model.policy.state_dict().items()}

    def act(self, snapshot: Any, observation: np.ndarray) -> np.ndarray:
        """Deterministische Action zu einem Lernstand – ohne Exploration."""
        if self._animation_policy is None:
            self._animation_policy = type(self.model.policy)(
                **self.model.policy._get_constructor_parameters())
        self._animation_policy.load_state_dict(snapshot)
        self._animation_policy.set_training_mode(False)
        action, _ = self._animation_policy.predict(
            self.policy_observation(observation), deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def evaluate(self, episodes: int, seed: int = 0) -> EvaluationResult:
        """Deterministische Evaluation ohne Exploration und ohne Lernupdates."""
        if self.model is None:
            raise RuntimeError("Vor der Evaluation muss ein Modell trainiert sein.")
        return evaluate_policy(self.act, self.current_snapshot(), episodes, seed)

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
        callback = Walker2dCallback(
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
    ) -> "Walker2dWorkbench":
        base = Path(path).with_suffix("")
        metadata = _read_metadata(base, expected_algorithm)
        workbench = cls(Walker2dConfig.from_dict(metadata["config"]))
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


# --------------------------------------------------------------------- CMA-ES


class RunningNormalizer:
    """Laufende Beobachtungsstatistik für Verfahren außerhalb von SB3.

    Gleichwertig zu `VecNormalize`, aber ohne Vektor-Environment: Sie sammelt
    Beobachtungen und schreibt sie erst auf Anforderung fort. CMA-ES ruft
    `commit()` genau einmal je Generation auf – innerhalb einer Generation
    sehen damit alle Kandidaten dieselbe Normalisierung, sonst wäre ihre
    Rangfolge verfälscht.
    """

    def __init__(self, size: int, clip: float = 10.0) -> None:
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)
        self.count = 1e-4
        self.clip = float(clip)
        self._pending: list[np.ndarray] = []

    def observe(self, observation: np.ndarray) -> None:
        """Sammelt eine Beobachtung, ohne die Statistik zu verändern."""
        self._pending.append(np.asarray(observation, dtype=np.float64))

    def commit(self) -> None:
        """Übernimmt die gesammelten Beobachtungen in die Statistik."""
        if not self._pending:
            return
        batch = np.asarray(self._pending)
        self._pending = []
        batch_mean, batch_var, batch_count = batch.mean(0), batch.var(0), len(batch)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + np.square(delta) * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        values = (np.asarray(observation, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(values, -self.clip, self.clip)

    def state(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(),
                "count": self.count, "clip": self.clip}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RunningNormalizer":
        normalizer = cls(len(state["mean"]), state["clip"])
        normalizer.mean = np.asarray(state["mean"], dtype=np.float64)
        normalizer.var = np.asarray(state["var"], dtype=np.float64)
        normalizer.count = state["count"]
        return normalizer


class LinearTanhPolicy:
    """Deterministische Policy, deren Gewichte CMA-ES direkt optimiert.

    `a = tanh(Wₙ · … · φ(W₁·s + b₁) … + bₙ)`. Ohne Hidden Layer ist das eine
    lineare Policy mit `6 · 17 + 6 = 108` Parametern – klein genug für die
    `n × n`-Kovarianzmatrix von CMA-ES. Das abschließende `tanh` bildet exakt
    auf `Box(-1, 1)` ab, ein Clipping erübrigt sich damit.
    """

    def __init__(self, hidden: tuple[int, ...], activation: str) -> None:
        self.sizes = [OBSERVATION_DIM, *hidden, ACTION_DIM]
        self.activation = activation
        self.size = sum(a * b + b for a, b in zip(self.sizes, self.sizes[1:]))

    def _hidden_activation(self, values: np.ndarray) -> np.ndarray:
        if self.activation == "ReLU":
            return np.maximum(values, 0.0)
        if self.activation == "Tanh":
            return np.tanh(values)
        if self.activation == "ELU":
            return np.where(values > 0, values, np.expm1(np.minimum(values, 0.0)))
        return np.where(values > 0, values, 0.01 * values)  # LeakyReLU

    def act(self, parameters: np.ndarray, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        offset = 0
        for index, (rows, columns) in enumerate(zip(self.sizes, self.sizes[1:])):
            weight = parameters[offset:offset + rows * columns].reshape(columns, rows)
            offset += rows * columns
            bias = parameters[offset:offset + columns]
            offset += columns
            values = weight @ values + bias
            if index < len(self.sizes) - 2:
                values = self._hidden_activation(values)
        # Letzte Schicht immer tanh: Der Action-Space ist Box(-1, 1).
        return np.tanh(values).astype(np.float32)


#: Lernstand von CMA-ES für Anzeige und Evaluation: Gewichte plus die
#: eingefrorene Beobachtungsstatistik, mit der sie trainiert wurden.
CmaSnapshot = tuple[np.ndarray, Optional[RunningNormalizer]]


class CmaEsWorkbench:
    """Ein Verfahrensslot mit CMA-ES statt Stable-Baselines3.

    Bietet dieselbe Schnittstelle wie `Walker2dWorkbench`, sodass GUI und
    Runner nicht wissen müssen, welche Verfahrensklasse in einem Slot steckt.
    """

    def __init__(self, config: Optional[Walker2dConfig] = None) -> None:
        self.config = config or default_config("CMA-ES")
        self.config.validate()
        self.env: Optional[Any] = None
        self.model: Optional[Any] = None
        self.history: list[EpisodeMetric] = []
        self.best_episode: Optional[EpisodeMetric] = None
        self._best_parameters: Optional[np.ndarray] = None
        self._best_lock = threading.Lock()
        self.policy = LinearTanhPolicy(tuple(self.config.actor_arch), self.config.activation)
        self.normalizer: Optional[RunningNormalizer] = None
        self.num_timesteps = 0
        self.completed_generations = 0

    @property
    def uses_replay_buffer(self) -> bool:
        return False

    @property
    def vec_normalize(self) -> None:
        return None

    @property
    def generations(self) -> Optional[int]:
        return self.completed_generations

    def create_model(self) -> Any:
        """Legt Suchverteilung, Policy und Beobachtungsstatistik neu an."""
        import cma

        self.close()
        self.policy = LinearTanhPolicy(tuple(self.config.actor_arch), self.config.activation)
        options: dict[str, Any] = {"verbose": -9, "seed": self.config.seed or 0}
        if self.config.popsize.strip() != "auto":
            options["popsize"] = int(self.config.popsize)
        if self.config.diagonal:
            # sep-CMA-ES: nur die Diagonale der Kovarianz, nötig erst bei
            # Netzen mit vielen Tausend Parametern.
            options["CMA_diagonal"] = True
        self.model = cma.CMAEvolutionStrategy(
            np.zeros(self.policy.size), self.config.sigma0, options)
        self.normalizer = (RunningNormalizer(OBSERVATION_DIM, self.config.clip_obs)
                           if self.config.normalize_obs else None)
        self.num_timesteps = 0
        self.completed_generations = 0
        self.history.clear()
        with self._best_lock:
            self.best_episode = None
            self._best_parameters = None
        return self.model

    # ---------------------------------------------- Policy-Abstraktion

    def policy_observation(self, observation: np.ndarray) -> np.ndarray:
        """Beobachtung wie im Training aufbereitet, mit eingefrorener Statistik."""
        if self.normalizer is None:
            return np.asarray(observation, dtype=np.float64)
        return self.normalizer.normalize(observation)

    def current_snapshot(self) -> CmaSnapshot:
        """Der **Verteilungsmittelwert**, nicht ein gezogener Kandidat.

        Das ist der Stand, den auch die deterministische Evaluation nutzt. Ein
        Kandidat wäre bei jeder Episode ein anderer und spränge in der Animation
        sichtbar hin und her, ohne den Lernfortschritt zu zeigen.
        """
        if self.model is None:
            raise RuntimeError("Es gibt noch keinen Lernstand.")
        return np.asarray(self.model.result.xfavorite, dtype=np.float64), self.normalizer

    def act(self, snapshot: CmaSnapshot, observation: np.ndarray) -> np.ndarray:
        parameters, normalizer = snapshot
        values = observation if normalizer is None else normalizer.normalize(observation)
        return self.policy.act(parameters, values)

    def best_snapshot(self) -> Optional[tuple[EpisodeMetric, CmaSnapshot]]:
        with self._best_lock:
            if self.best_episode is None or self._best_parameters is None:
                return None
            return self.best_episode, (self._best_parameters.copy(), self.normalizer)

    def remember_episode(self, metric: EpisodeMetric, parameters: np.ndarray) -> None:
        with self._best_lock:
            if self.best_episode is not None and metric.reward <= self.best_episode.reward:
                return
            self.best_episode = metric
            self._best_parameters = np.asarray(parameters, dtype=np.float64).copy()

    def evaluate(self, episodes: int, seed: int = 0) -> EvaluationResult:
        if self.model is None:
            raise RuntimeError("Vor der Evaluation muss ein Modell trainiert sein.")
        return evaluate_policy(self.act, self.current_snapshot(), episodes, seed)

    # ---------------------------------------------------------- Training

    def _run_episode(self, env: Any, parameters: np.ndarray,
                     seed: Optional[int]) -> tuple[float, int, bool, bool, float, float]:
        """Eine vollständige Episode mit festen Gewichten."""
        observation, _ = env.reset(seed=seed)
        total, length = 0.0, 0
        terminated = truncated = False
        speed_sum = distance = 0.0
        while not (terminated or truncated):
            if self.normalizer is not None:
                # Sammeln, aber erst nach der Generation fortschreiben.
                self.normalizer.observe(observation)
            action = self.policy.act(parameters, self.policy_observation(observation))
            observation, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            length += 1
            speed_sum += float(info.get("x_velocity", 0.0))
            distance = float(info.get("x_position", distance))
        return total, length, bool(terminated), bool(truncated), speed_sum / max(1, length), distance

    def train(
        self, stop_event: Optional[threading.Event] = None,
        output: Optional[queue.Queue] = None, series: Any = None,
        evaluation_interval: int = 0,
        evaluation_episodes: int = DEFAULT_EVALUATION_EPISODES,
        best_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None,
    ) -> list[EpisodeMetric]:
        """Generationen, bis das Schrittbudget erschöpft ist.

        Gezählt wird in **Environment-Schritten**, nicht in Generationen. Eine
        angefangene Generation wird beim Budgetende nicht mehr an den
        Optimierer gemeldet – ohne vollständige Population gibt es kein Update.
        """
        if self.model is None:
            self.create_model()
        stop_event = stop_event or threading.Event()
        env = make_walker2d_env(render_mode=None)
        start_step = self.num_timesteps
        budget = self.config.total_timesteps
        metrics: list[EpisodeMetric] = []
        next_evaluation = evaluation_interval
        rng = np.random.default_rng(self.config.seed)
        try:
            while self.num_timesteps - start_step < budget and not stop_event.is_set():
                candidates = self.model.ask()
                fitness: list[float] = []
                complete = True
                for parameters in candidates:
                    returns = []
                    for _ in range(self.config.episodes_per_candidate):
                        total, length, terminated, truncated, speed, distance = self._run_episode(
                            env, np.asarray(parameters), int(rng.integers(0, 2**31 - 1)))
                        self.num_timesteps += length
                        survived, fell, reached = episode_outcome(terminated, truncated, total)
                        metric = EpisodeMetric(
                            len(self.history) + len(metrics) + 1, total, length,
                            survived, fell, reached, self.num_timesteps, speed, distance)
                        metrics.append(metric)
                        self.remember_episode(metric, parameters)
                        if output is not None:
                            output.put(("episode", (series, metric)))
                        returns.append(total)
                    # pycma minimiert; die Fitness ist die negative Rendite.
                    fitness.append(-float(mean(returns)))
                    elapsed = self.num_timesteps - start_step
                    if output is not None:
                        output.put(("progress", (series, elapsed)))
                    if evaluation_interval and elapsed >= next_evaluation:
                        result = self.evaluate(evaluation_episodes, self.config.seed or 0)
                        episode = len(self.history) + len(metrics)
                        if best_callback:
                            best_callback(episode, self.num_timesteps, result)
                        if output is not None:
                            output.put(("evaluation",
                                        (series, episode, self.num_timesteps, result)))
                        next_evaluation += evaluation_interval
                    if elapsed >= budget or stop_event.is_set():
                        complete = False
                        break
                if complete:
                    self.model.tell(candidates, fitness)
                    self.completed_generations += 1
                    # Erst jetzt: Innerhalb einer Generation sehen alle
                    # Kandidaten dieselbe Normalisierung.
                    if self.normalizer is not None:
                        self.normalizer.commit()
        finally:
            env.close()
        self.history.extend(metrics)
        return metrics

    # -------------------------------------------------------- Checkpoint

    def save(self, path: str | Path) -> tuple[Path, Optional[Path], Path]:
        """Sichert die vollständige Suchverteilung, nicht nur ein Netz.

        Ohne Mittelwert, Schrittweite und Kovarianz ließe sich die Suche nicht
        fortsetzen.
        """
        if self.model is None:
            raise RuntimeError("Es gibt kein Modell zum Speichern.")
        base = Path(path).with_suffix("")
        model_path = base.with_suffix(".zip")
        metadata_path = base.with_name(base.name + "_metadata.json")
        with open(model_path, "wb") as handle:
            pickle.dump({
                "es": self.model,
                "normalizer": None if self.normalizer is None else self.normalizer.state(),
                "num_timesteps": self.num_timesteps,
                "generations": self.completed_generations,
                "best_parameters": None if self._best_parameters is None
                else self._best_parameters.tolist(),
                "best_episode": None if self.best_episode is None else asdict(self.best_episode),
            }, handle)
        metadata = {
            "format": 1, "environment": ENV_ID,
            "algorithm": self.config.algorithm, "replay_buffer": False,
            "normalization": self.config.normalize_obs, "config": asdict(self.config),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return model_path, None, metadata_path

    @classmethod
    def load(cls, path: str | Path,
             expected_algorithm: Optional[str] = None) -> "CmaEsWorkbench":
        base = Path(path).with_suffix("")
        metadata = _read_metadata(base, expected_algorithm)
        workbench = cls(Walker2dConfig.from_dict(metadata["config"]))
        with open(base.with_suffix(".zip"), "rb") as handle:
            state = pickle.load(handle)
        workbench.model = state["es"]
        workbench.num_timesteps = state["num_timesteps"]
        workbench.completed_generations = state["generations"]
        workbench.normalizer = (None if state["normalizer"] is None
                                else RunningNormalizer.from_state(state["normalizer"]))
        if state["best_parameters"] is not None:
            workbench._best_parameters = np.asarray(state["best_parameters"], dtype=np.float64)
            workbench.best_episode = EpisodeMetric(**state["best_episode"])
        return workbench

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def make_workbench(config: Walker2dConfig) -> Any:
    """Der passende Slot zum Algorithmus – SB3 oder pycma."""
    if config.algorithm in EVOLUTIONARY_ALGORITHMS:
        return CmaEsWorkbench(config)
    return Walker2dWorkbench(config)


def load_workbench(path: str | Path, expected_algorithm: Optional[str] = None) -> Any:
    """Lädt einen Checkpoint in den passenden Slot-Typ."""
    metadata = _read_metadata(Path(path).with_suffix(""), expected_algorithm)
    if metadata["algorithm"] in EVOLUTIONARY_ALGORITHMS:
        return CmaEsWorkbench.load(path, expected_algorithm)
    return Walker2dWorkbench.load(path, expected_algorithm)
