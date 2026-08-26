"""SB3-Lernlogik für die Humanoid-Workbench (Abschlussprojekt)."""

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
from sb3_contrib import CrossQ, TQC
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn


ALGORITHMS = ("PPO", "TD3", "SAC", "CrossQ", "TQC")
ALGORITHM_CLASSES: dict[str, type[BaseAlgorithm]] = {
    "PPO": PPO, "SAC": SAC, "TD3": TD3, "CrossQ": CrossQ, "TQC": TQC,
}
#: PPO ist on-policy und führt keinen Replay Buffer; die übrigen tun es.
OFF_POLICY_ALGORITHMS = frozenset({"SAC", "TD3", "CrossQ", "TQC"})
ON_POLICY_ALGORITHMS = frozenset({"PPO"})
#: Verfahren mit stochastischem Actor und automatisch geregelter Entropie –
#: dieselbe Familie wie SAC, nur mit anderem Critic beziehungsweise ohne
#: Target-Netze.
ENTROPY_ALGORITHMS = frozenset({"SAC", "CrossQ", "TQC"})
#: **CrossQ führt keine Target-Netze.** Genau das ist seine Kernidee: Statt
#: eines langsam nachgezogenen Zielnetzes laufen aktueller und nächster Zustand
#: gemeinsam durch den Critic, und Batch Normalization übernimmt die
#: Stabilisierung. `tau` und `target_update_interval` existieren dort nicht und
#: dürfen weder angezeigt noch übergeben werden.
TARGET_NETWORK_ALGORITHMS = frozenset({"SAC", "TD3", "TQC"})

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
#: Kurzformen mit Absicht: Ein Auswahlfeld, das `linear fallend` oder
#: `Ornstein-Uhlenbeck` fassen muss, sprengt die Spaltenbreite und stünde als
#: einziges Feld über die Zeile – die übrigen Felder endeten dann an einer
#: anderen Kante. Die Langformen stehen in Bedienungsanleitung und README.
SCHEDULES = ("konstant", "linear")
ACTION_NOISES = ("keins", "normal", "OU")
ENTROPY_MODES = ("auto", "fest")

ENV_ID = "Humanoid-v5"
#: Bildgröße des offiziellen Renderers, entspricht den Gymnasium-Vorgaben.
FRAME_WIDTH, FRAME_HEIGHT = 480, 480

#: `Humanoid-v5` führt **keinen** `reward_threshold`; eine offizielle
#: Gelöst-Schwelle existiert nicht. `5000` ist eine **projektinterne**
#: Referenzmarke: genau diesen Return erreicht eine Figur, die eine volle
#: Episode lang nicht stürzt (1000 Schritte × `healthy_reward` 5,0), ohne sich
#: vorwärts zu bewegen. Die Marke trennt damit „bleibt aufrecht" von „stürzt"
#: und heißt überall **Zielmarke**, nie „gelöst".
TARGET_RETURN = 5000.0
#: Zeitlimit des `TimeLimit`-Wrappers.
MAX_EPISODE_STEPS = 1000

#: Box(-0.4, 0.4, (17,)). **Nicht** ±1 wie bei Hopper, HalfCheetah und
#: Walker2d – jede Anzeige und Skalierung liest die Grenze aus dem Space.
ACTION_DIM = 17
ACTION_LIMIT = 0.4

#: Reihenfolge der **Aktuatoren** aus `humanoid.xml`, mit ihrer Übersetzung.
#: Moment in N·m ist `aᵢ · gearᵢ`. Die Beine sind bis zu zwölfmal kräftiger
#: übersetzt als die Arme.
ACTUATOR_NAMES = (
    "abdomen_y", "abdomen_z", "abdomen_x",
    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
    "right_shoulder1", "right_shoulder2", "right_elbow",
    "left_shoulder1", "left_shoulder2", "left_elbow",
)
ACTUATOR_LABELS = (
    "Rumpf nicken", "Rumpf drehen", "Rumpf neigen",
    "Hüfte re. seitlich", "Hüfte re. drehen", "Hüfte re. vor/zurück", "Knie rechts",
    "Hüfte li. seitlich", "Hüfte li. drehen", "Hüfte li. vor/zurück", "Knie links",
    "Schulter re. 1", "Schulter re. 2", "Ellbogen rechts",
    "Schulter li. 1", "Schulter li. 2", "Ellbogen links",
)
GEARS = (100.0, 100.0, 100.0,
         100.0, 100.0, 300.0, 200.0,
         100.0, 100.0, 300.0, 200.0,
         25.0, 25.0, 25.0,
         25.0, 25.0, 25.0)

#: Reihenfolge der 17 Gelenkwinkel **in der Observation** (`qpos[7:24]`).
#: Sie ist **nicht** identisch mit der Aktuatorreihenfolge: `abdomen_y` und
#: `abdomen_z` sind vertauscht. Wer `obs[5 + i]` dem Aktuator `i` zuordnet,
#: beschriftet die ersten beiden Werte falsch. Ein Test prüft beide Listen
#: gegen `model.actuator_trnid` bzw. die Gelenkreihenfolge des Modells.
JOINT_NAMES = (
    "abdomen_z", "abdomen_y", "abdomen_x",
    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
    "right_shoulder1", "right_shoulder2", "right_elbow",
    "left_shoulder1", "left_shoulder2", "left_elbow",
)
#: Aktuatorindex -> Index desselben Gelenks in der Observation.
ACTUATOR_TO_JOINT = tuple(JOINT_NAMES.index(name) for name in ACTUATOR_NAMES)

#: Gruppierung für die Einblendung: 348 Werte lassen sich nicht am Stück lesen.
ACTUATOR_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("Rumpf", (0, 1, 2)),
    ("Bein rechts", (3, 4, 5, 6)),
    ("Bein links", (7, 8, 9, 10)),
    ("Arm rechts", (11, 12, 13)),
    ("Arm links", (14, 15, 16)),
)

#: 348 Beobachtungswerte, `float64`. x- und y-Position sind ausgeschlossen
#: (`skipped_qpos = 2`) und stehen in `info`.
OBSERVATION_DIM = 348
#: Grenzen der acht Blöcke, geprüft gegen `observation_structure`.
OBS_HEIGHT = 0
OBS_QUATERNION = slice(1, 5)
OBS_JOINT_ANGLES = slice(5, 22)
OBS_QVEL = slice(22, 45)
OBS_LINEAR_VELOCITY = slice(22, 25)
OBS_ANGULAR_VELOCITY = slice(25, 28)
OBS_JOINT_VELOCITIES = slice(28, 45)
OBS_CINERT = slice(45, 175)
OBS_CVEL = slice(175, 253)
OBS_QFRC_ACTUATOR = slice(253, 270)
OBS_CFRC_EXT = slice(270, 348)

#: Rewardgewichte. `healthy_reward` ist mit Abstand der größte Anteil: Eine
#: volle Episode gibt allein fürs Aufrechtbleiben +5000.
FORWARD_REWARD_WEIGHT = 1.25
CTRL_COST_WEIGHT = 0.1
HEALTHY_REWARD = 5.0
CONTACT_COST_WEIGHT = 5e-7
#: `contact_cost_range = (-inf, 10)`: Die Kontaktkosten sind nach oben gedeckelt.
CONTACT_COST_MAX = 10.0
#: Gesunder Höhenbereich des Rumpfes. Anders als bei Walker2d gibt es **keine**
#: Winkelbedingung – die Figur darf beliebig verdreht sein.
HEALTHY_Z_RANGE = (1.0, 2.0)
#: Simulationsschritt: `frame_skip = 5` mal `timestep = 0.003`.
STEP_DURATION = 0.015

#: Anders als bei Walker2d sind die Geschwindigkeiten **nicht** geclippt.
VELOCITY_CLIP: Optional[float] = None

_ALL = frozenset(ALGORITHMS)
FIELD_ALGORITHMS: dict[str, frozenset[str]] = {
    "total_timesteps": _ALL, "episodes": _ALL,
    "learning_rate": _ALL, "learning_rate_schedule": _ALL,
    "batch_size": _ALL, "gamma": _ALL, "seed": _ALL, "actor_arch": _ALL, "critic_arch": _ALL,
    "activation": _ALL, "optimizer": _ALL, "optimizer_eps": _ALL, "optimizer_weight_decay": _ALL,
    "n_steps": frozenset({"PPO"}), "n_epochs": frozenset({"PPO"}),
    "gae_lambda": frozenset({"PPO"}), "clip_range": frozenset({"PPO"}),
    "clip_range_vf": frozenset({"PPO"}), "normalize_advantage": frozenset({"PPO"}),
    "ent_coef": frozenset({"PPO"}), "vf_coef": frozenset({"PPO"}),
    "max_grad_norm": frozenset({"PPO"}), "target_kl": frozenset({"PPO"}),
    "ortho_init": frozenset({"PPO"}),
    "log_std_init": frozenset({"PPO"}) | ENTROPY_ALGORITHMS,
    "use_sde": frozenset({"PPO"}) | ENTROPY_ALGORITHMS,
    "sde_sample_freq": frozenset({"PPO"}) | ENTROPY_ALGORITHMS,
    "buffer_size": OFF_POLICY_ALGORITHMS, "learning_starts": OFF_POLICY_ALGORITHMS,
    "tau": TARGET_NETWORK_ALGORITHMS, "train_freq": OFF_POLICY_ALGORITHMS,
    "gradient_steps": OFF_POLICY_ALGORITHMS,
    "action_noise": OFF_POLICY_ALGORITHMS, "action_noise_sigma": OFF_POLICY_ALGORITHMS,
    "policy_delay": frozenset({"TD3", "CrossQ"}), "target_policy_noise": frozenset({"TD3"}),
    "target_noise_clip": frozenset({"TD3"}),
    "ent_coef_mode": ENTROPY_ALGORITHMS, "ent_coef_value": ENTROPY_ALGORITHMS,
    "target_entropy": ENTROPY_ALGORITHMS,
    "target_update_interval": frozenset({"SAC", "TQC"}),
    "batch_norm_momentum": frozenset({"CrossQ"}),
    "renorm_warmup_steps": frozenset({"CrossQ"}),
    "n_quantiles": frozenset({"TQC"}), "top_quantiles_to_drop": frozenset({"TQC"}),
    "normalize_obs": _ALL, "normalize_reward": _ALL,
    "clip_obs": _ALL, "clip_reward": _ALL,
}
FIELD_LABELS: dict[str, str] = {
    "total_timesteps": "Trainingsschritte N", "episodes": "Episoden E",
    "learning_rate": "Lernrate α",
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
    "batch_norm_momentum": "BatchNorm-Momentum", "renorm_warmup_steps": "Renorm-Warmlauf",
    "n_quantiles": "Quantile N_q", "top_quantiles_to_drop": "Verworfene Quantile k",
    "normalize_obs": "Beobachtungen normalisieren", "normalize_reward": "Rewards normalisieren",
    "clip_obs": "Clip Beobachtungen", "clip_reward": "Clip Reward",
}

INTEGER_FIELDS = frozenset({
    "total_timesteps", "episodes", "batch_size", "n_steps", "n_epochs", "sde_sample_freq",
    "buffer_size", "learning_starts", "train_freq", "gradient_steps", "policy_delay",
    "target_update_interval", "renorm_warmup_steps", "n_quantiles", "top_quantiles_to_drop",
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
TEXT_FIELDS = frozenset({"target_entropy"})

#: Einheitliche Netzgröße aller Verfahren. Das PPO-Profil nennt `256,256`, der
#: SB3-Default für SAC ebenfalls; nur das TD3-Profil weicht mit `400,300` ab.
#: Die Vereinheitlichung macht den Vergleich fair.
DEFAULT_NET_ARCH = (256, 256)

#: Critic-Breite von CrossQ. Die Referenzimplementierung setzt `1024,1024`; die
#: Batch Normalization erlaubt das größere Netz, ohne dass es instabil wird.
#: Wird der Critic auf `256,256` gestutzt, ist es nicht mehr CrossQ.
CROSSQ_CRITIC_ARCH = (1024, 1024)

#: Obergrenze der **Messläufe für den Bericht**: Sie bemisst den Replay Buffer,
#: der damit in keinem Lauf auch nur einen Übergang verdrängt. Tatsächlich
#: gefahren wurden rund 300.000 Schritte je Lauf – der Verfahrensvergleich mit
#: genau diesem Budget, die Parameterstudie bis zum Handstopp bei ~300.400.
#: Beides liegt bewusst unter den Profilwerten (PPO 1e7, TD3/SAC je 2e6): Ein
#: voller Profillauf dauerte auf der Zielmaschine 6 bis 11 Stunden je
#: Verfahren. Die Figur wird damit **nicht** laufen; was das kostet, steht in
#: `prompt.md` und im README. Diese Läufe werden mit `episodes = 0` gefahren,
#: damit alle Verfahren exakt dieselbe Datenmenge bekommen.
REPORT_TIMESTEPS = 500_000
#: Schrittbudget in der **Voreinstellung** der Oberfläche. Es liegt bewusst in
#: derselben Größenordnung wie die vorgegebenen 1000 Episoden: Gemessen
#: entspricht diese Episodenzahl bei SAC rund 80.000 Schritten. Stünde hier das
#: Budget der Messläufe, griffe immer die Episodengrenze und die Schrittzahl
#: wäre reine Dekoration.
DEFAULT_TOTAL_TIMESTEPS = 100_000
#: Zweite Budgetgrenze, von der Aufgabenstellung vorgegeben. Der Lauf endet,
#: was zuerst eintritt; `0` bedeutet unbegrenzt. Bei Humanoid greift die
#: Episodengrenze früh, weil Episoden mit dem Lernfortschritt länger werden.
DEFAULT_EPISODES = 1000
#: Standardzahl der Episoden für `HumanoidWorkbench.evaluate()`. Die
#: Oberfläche ruft die Evaluation nicht mehr auf – die Summary mittelt
#: stattdessen über die letzten Episoden des Trainings. Die Methode bleibt für
#: eine explorationsfreie Messung von Hand erhalten.
DEFAULT_EVALUATION_EPISODES = 5
#: Verkleinert auf das Budget der Messläufe statt der SB3-Vorgabe `1e6`. Bei
#: 348 Werten belegte ein Buffer über `1e6` Übergänge allein für Beobachtungen
#: 5,19 GB – auf einer 8-GB-Maschine nicht tragbar. Verhaltensneutral, weil
#: auch der längste vorgesehene Lauf 500.000 Schritte hat: Es wird nie ein
#: Übergang verdrängt. Bewusst **nicht** an `DEFAULT_TOTAL_TIMESTEPS`
#: gekoppelt – wer das Budget auf den Berichtswert hochsetzt, soll nicht
#: unbemerkt in eine Verdrängung laufen.
DEFAULT_BUFFER_SIZE = REPORT_TIMESTEPS
#: PyTorch startet sonst so viele Threads wie Kerne und blockiert sich auf
#: Effizienzkernen selbst. Gemessen auf der Zielmaschine (2 Performance-Kerne):
#: 16 Schritte/s im Standard gegen 49,5 mit vier Threads.
DEFAULT_TORCH_THREADS = 4

#: Gemessene mittlere Episodenlänge einer Zufallspolicy (200 Episoden):
#: Median 22, Mittel 24,5, min 16, max 56. Nur für Hilfetexte.
RANDOM_POLICY_EPISODE_LENGTH = 24.5
#: Gemessen: 1000 Episoden entsprechen bei SAC rund so vielen Schritten.
#: Begründet den Standardwert von `DEFAULT_TOTAL_TIMESTEPS`.
STEPS_PER_THOUSAND_EPISODES = 80_000

#: `Humanoid-v4`-Profile des RL Baselines3 Zoo, Stand geprüft am 24.08.2026.
#: Für `v5` sind keine Profile hinterlegt. Werte, die dort fehlen, bleiben auf
#: den SB3-Voreinstellungen (siehe Dataclass-Defaults).
DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    # Vollständig getuntes Profil, nicht geerbt. `gamma = 0.95` ist auffällig
    # niedrig und `normalize: true` verpflichtend – beides aus dem Profil
    # übernommen und nicht „korrigiert".
    "PPO": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "episodes": DEFAULT_EPISODES,
        "learning_rate": 3.56987e-05, "gamma": 0.95,
        "n_steps": 512, "batch_size": 256, "n_epochs": 5,
        "gae_lambda": 0.9, "clip_range": 0.3, "ent_coef": 0.00238306,
        "vf_coef": 0.431892, "max_grad_norm": 2.0,
        "log_std_init": -2.0, "ortho_init": False,
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH, "activation": "ReLU",
        "normalize_obs": True, "normalize_reward": True,
    },
    # Erbt aus `mujoco-defaults`; für TD3 setzt der Block sieben Schlüssel.
    "TD3": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "episodes": DEFAULT_EPISODES,
        "learning_rate": 1e-3, "gamma": 0.99, "tau": 0.005, "batch_size": 256,
        "buffer_size": DEFAULT_BUFFER_SIZE, "learning_starts": 10_000,
        "train_freq": 1, "gradient_steps": 1,
        "action_noise": "normal", "action_noise_sigma": 0.1,
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH, "activation": "ReLU",
    },
    # Erbt aus `mujoco-defaults`; für SAC setzt der Block nur `learning_starts`.
    "SAC": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "episodes": DEFAULT_EPISODES,
        "learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005, "batch_size": 256,
        "buffer_size": DEFAULT_BUFFER_SIZE, "learning_starts": 10_000,
        "train_freq": 1, "gradient_steps": 1, "ent_coef_mode": "auto",
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH, "activation": "ReLU",
    },
    # Voreinstellungen der Referenzimplementierung in `sb3-contrib`. Der breite
    # Critic mit `1024,1024` und die Batch Normalization sind **das Verfahren**
    # und werden deshalb – anders als bei TD3 im Projekt – nicht auf die
    # Netzgröße der übrigen Verfahren vereinheitlicht.
    "CrossQ": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "episodes": DEFAULT_EPISODES,
        "learning_rate": 1e-3, "gamma": 0.99, "batch_size": 256,
        "buffer_size": DEFAULT_BUFFER_SIZE, "learning_starts": 10_000,
        "train_freq": 1, "gradient_steps": 1, "ent_coef_mode": "auto",
        "policy_delay": 3, "batch_norm_momentum": 0.01, "renorm_warmup_steps": 100_000,
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": CROSSQ_CRITIC_ARCH,
        "activation": "ReLU", "log_std_init": -3.0,
    },
    # Voreinstellungen der Referenzimplementierung in `sb3-contrib`; Netz und
    # Lernrate entsprechen SAC, der Unterschied steckt allein im Critic.
    "TQC": {
        "total_timesteps": DEFAULT_TOTAL_TIMESTEPS, "episodes": DEFAULT_EPISODES,
        "learning_rate": 3e-4, "gamma": 0.99, "tau": 0.005, "batch_size": 256,
        "buffer_size": DEFAULT_BUFFER_SIZE, "learning_starts": 10_000,
        "train_freq": 1, "gradient_steps": 1, "ent_coef_mode": "auto",
        "n_quantiles": 25, "top_quantiles_to_drop": 2,
        "actor_arch": DEFAULT_NET_ARCH, "critic_arch": DEFAULT_NET_ARCH,
        "activation": "ReLU", "log_std_init": -3.0,
    },
}

#: Benchmark des Zoo bei **2.000.000** Schritten. Für PPO führt der Zoo keinen
#: Humanoid-Eintrag. Beide Werte liegen über der Zielmarke.
ZOO_BENCHMARK: dict[str, Optional[tuple[float, float]]] = {
    "PPO": None, "TD3": (5566.687, 14.544), "SAC": (6232.287, 279.885),
}
#: Profilbudget je Verfahren – für den Bericht wichtig, weil 500.000 Schritte
#: davon sehr unterschiedliche Anteile sind (PPO 5 %, TD3/SAC je 25 %).
#: Nur für die drei Verfahren des Zoo hinterlegt. Für `CrossQ` und `TQC` nennt
#: dieses Projekt bewusst kein Profilbudget: Ihre Werte stammen aus den
#: Voreinstellungen von `sb3-contrib`, nicht aus einem auf `Humanoid`
#: abgestimmten Zoo-Profil. Ein erfundenes Budget wäre eine Behauptung.
PROFILE_BUDGET: dict[str, int] = {"PPO": 10_000_000, "TD3": 2_000_000, "SAC": 2_000_000}


def set_torch_threads(threads: int = DEFAULT_TORCH_THREADS) -> int:
    """Thread-Zahl von PyTorch setzen und den gesetzten Wert zurückgeben."""
    count = max(1, int(threads))
    torch.set_num_threads(count)
    return count


def make_humanoid_env(render_mode: Optional[str] = "rgb_array") -> gymnasium.Env:
    """Offizielles Environment ohne jede Änderung an Physik und Reward.

    Übergeben werden ausschließlich `render_mode`, `width` und `height`. Alle
    Gewichte, Abbruchregeln und Observationsschalter bleiben unberührt – jede
    Änderung wäre Reward Shaping bzw. eine Änderung des Beobachtungsraums.
    """
    kwargs: dict[str, Any] = {"width": FRAME_WIDTH, "height": FRAME_HEIGHT}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gymnasium.make(ENV_ID, **kwargs)


class Float32Observation(gymnasium.ObservationWrapper):
    """Beobachtungen als `float32` ausgeben, ohne die Werte zu verändern.

    `Humanoid-v5` liefert `float64`. Stable-Baselines3 legt den Replay Buffer
    in der Dtype des Observation-Space an und speichert bei
    `optimize_memory_usage=False` sowohl `observations` als auch
    `next_observations` – bei 348 Werten sind das 5,19 GB für `1e6` Übergänge.
    Die Halbierung ist folgenlos, weil SB3 beim Sampeln ohnehin nach `float32`
    wandelt, bevor irgendetwas das Netz erreicht.

    `optimize_memory_usage=True` wäre die naheliegende Alternative, ist aber
    unzulässig: Sie verträgt sich nicht mit `handle_timeout_termination`, und
    Humanoid trunkiert nach 1000 Schritten.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        space = env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=np.asarray(space.low, dtype=np.float32),
            high=np.asarray(space.high, dtype=np.float32),
            shape=space.shape, dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32)


class LinearSchedule:
    """Linear auf null fallende Lernrate; SB3 ruft sie je Update auf."""

    def __init__(self, initial: float) -> None:
        self.initial = float(initial)

    def __call__(self, progress_remaining: float) -> float:
        return self.initial * float(progress_remaining)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, LinearSchedule) and other.initial == self.initial

    def __repr__(self) -> str:
        return f"LinearSchedule({self.initial!r})"


def make_action_noise(kind: str, sigma: float, action_dim: int = ACTION_DIM):
    """Action Noise im auf ±1 normierten Raum von SB3. `OU` = Ornstein-Uhlenbeck.

    SB3 skaliert Actions intern auf [-1, 1], addiert dort das Rauschen und
    rechnet zurück. `sigma = 0.1` entspricht bei Humanoid deshalb `0,04` in
    Action-Einheiten, weil der Space nur bis ±0,4 reicht.
    """
    if kind == "normal":
        return NormalActionNoise(np.zeros(action_dim), sigma * np.ones(action_dim))
    if kind == "OU":
        return OrnsteinUhlenbeckActionNoise(np.zeros(action_dim), sigma * np.ones(action_dim))
    return None


@dataclass
class HumanoidConfig:
    """Vollständige Konfiguration eines Verfahrensslots.

    Die Dataclass-Defaults entsprechen den Voreinstellungen von
    Stable-Baselines3. `default_config()` legt darüber das Zoo-Profil.
    """

    algorithm: str = "PPO"
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    #: Zweite Budgetgrenze. `0` bedeutet unbegrenzt; der Lauf endet dann allein
    #: am Schrittbudget. Von der Aufgabenstellung vorgegeben (Default 1000).
    episodes: int = DEFAULT_EPISODES
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "konstant"
    batch_size: int = 64
    gamma: float = 0.99
    seed: Optional[int] = 0
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
    buffer_size: int = DEFAULT_BUFFER_SIZE
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
    # SAC, CrossQ und TQC
    ent_coef_mode: str = "auto"
    ent_coef_value: float = 1.0
    target_entropy: str = "auto"
    target_update_interval: int = 1
    # CrossQ
    batch_norm_momentum: float = 0.01
    renorm_warmup_steps: int = 100_000
    # TQC
    n_quantiles: int = 25
    top_quantiles_to_drop: int = 2
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
        if self.algorithm in ("TD3", "CrossQ"):
            positive_integers.append(("Policy Delay d", self.policy_delay))
        if self.algorithm in ("SAC", "TQC"):
            positive_integers.append(("Target-Intervall C", self.target_update_interval))
        if self.algorithm == "CrossQ":
            positive_integers.append(("Renorm-Warmlauf", self.renorm_warmup_steps))
        if self.algorithm == "TQC":
            positive_integers.append(("Quantile N_q", self.n_quantiles))
        for label, value in positive_integers:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label}: '{value}' ist ungültig. Gültig: ganze Zahl ≥ 1.")
        # Die Episodengrenze darf null sein: Das heißt „unbegrenzt".
        if not isinstance(self.episodes, int) or isinstance(self.episodes, bool) \
                or self.episodes < 0:
            raise ValueError(
                f"Episoden E: '{self.episodes}' ist ungültig. "
                "Gültig: ganze Zahl ≥ 0, wobei 0 „unbegrenzt\" bedeutet."
            )
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
                f"Aktivierung φ: '{self.activation}' ist unbekannt. "
                f"Gültig: {', '.join(ACTIVATIONS)}."
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
            if self.algorithm == "CrossQ":
                self._validate_crossq()
            elif self.algorithm == "TQC":
                self._validate_tqc()

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
        if self.uses("tau") and not 0 < self.tau <= 1:
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
                f"Entropie α: '{self.ent_coef_mode}' ist unbekannt. "
                f"Gültig: {', '.join(ENTROPY_MODES)}."
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

    def _validate_crossq(self) -> None:
        """CrossQ ersetzt das Target-Netz durch Batch Normalization – deren
        Momentum entscheidet damit über die Stabilität des ganzen Verfahrens."""
        if not 0 < self.batch_norm_momentum < 1:
            raise ValueError(
                f"BatchNorm-Momentum: '{self.batch_norm_momentum}' ist ungültig. "
                "Gültig: 0 < Momentum < 1."
            )

    def _validate_tqc(self) -> None:
        """Es dürfen nie alle Quantile verworfen werden – der Critic hätte dann
        keinen Wert mehr zu schätzen."""
        if not 0 <= self.top_quantiles_to_drop < self.n_quantiles:
            raise ValueError(
                f"Verworfene Quantile k: '{self.top_quantiles_to_drop}' ist ungültig. "
                f"Gültig: 0 bis {self.n_quantiles - 1}, also weniger als die "
                f"{self.n_quantiles} vorhandenen Quantile."
            )

    def learning_rate_value(self) -> float | LinearSchedule:
        if self.learning_rate_schedule == "linear":
            return LinearSchedule(self.learning_rate)
        return self.learning_rate

    def sac_ent_coef(self) -> str | float:
        if self.ent_coef_mode == "auto":
            return f"auto_{self.ent_coef_value}"
        return self.ent_coef_value

    def sac_target_entropy(self) -> str | float:
        """`auto` heißt bei SAC `-dim(A)`; bei 17 Actions also -17."""
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
        if self.algorithm in frozenset({"PPO"}) | ENTROPY_ALGORITHMS:
            kwargs["log_std_init"] = self.log_std_init
        if self.algorithm == "PPO":
            kwargs["ortho_init"] = self.ortho_init
        if self.algorithm == "CrossQ":
            # `batch_norm` bleibt eingeschaltet: Ohne sie fehlt CrossQ der
            # Ersatz für das Target-Netz, das es nicht hat.
            kwargs["batch_norm"] = True
            kwargs["batch_norm_momentum"] = self.batch_norm_momentum
            kwargs["renorm_warmup_steps"] = self.renorm_warmup_steps
        if self.algorithm == "TQC":
            kwargs["n_quantiles"] = self.n_quantiles
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
        entropie = {
            "ent_coef": self.sac_ent_coef(),
            "target_entropy": self.sac_target_entropy(),
            "use_sde": self.use_sde, "sde_sample_freq": self.sde_sample_freq,
        }
        if self.algorithm == "CrossQ":
            # **Ohne `tau` und ohne `target_update_interval`**: CrossQ besitzt
            # keine Target-Netze, sein Konstruktor kennt beide Argumente nicht.
            return {
                **{schluessel: wert for schluessel, wert in off_policy.items()
                   if schluessel != "tau"},
                **entropie, "policy_delay": self.policy_delay,
            }
        if self.algorithm == "TQC":
            return {
                **off_policy, **entropie,
                "target_update_interval": self.target_update_interval,
                "top_quantiles_to_drop_per_net": self.top_quantiles_to_drop,
            }
        return {
            **off_policy, **entropie,
            "target_update_interval": self.target_update_interval,
        }

    def signature(self) -> tuple[Any, ...]:
        """Kennzeichnet den Modellaufbau. Ein geändertes Budget – Schritte wie
        Episoden – setzt den Lernzustand nicht zurück, jede andere Änderung
        schon."""
        values = asdict(self)
        values.pop("total_timesteps")
        values.pop("episodes")
        return tuple(sorted((key, str(value)) for key, value in values.items()
                            if key == "algorithm" or self.uses(key)))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "HumanoidConfig":
        data = dict(values)
        for key in TUPLE_FIELDS:
            data[key] = tuple(data[key])
        config = cls(**data)
        config.validate()
        return config


def default_config(algorithm: str) -> HumanoidConfig:
    """Slot mit dem Zoo-Profil des Algorithmus vorbelegen."""
    config = HumanoidConfig(algorithm=algorithm, **DEFAULT_PROFILES[algorithm])
    config.validate()
    return config


def format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "ja" if value else "nein"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def config_differences(configs: Sequence[HumanoidConfig]) -> list[tuple[str, list[str]]]:
    """Parameter, in denen sich die Slots unterscheiden – je Zeile alle Werte.

    Verglichen werden ausschließlich Parameter, die **alle** beteiligten
    Verfahren besitzen. Ein Parameter, den nur ein Teil von ihnen kennt, ist
    keine Konfigurationsentscheidung, sondern eine Folge der Verfahrenswahl; er
    steht vollständig im jeweiligen Verfahrenstab. Der Algorithmus selbst
    entfällt, weil ihn die Kopfzeile der Summary bereits nennt.
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
    #: Volle 1000 Schritte erreicht (`truncated`), ohne zu stürzen.
    survived: bool
    #: Ungesund geworden (`terminated`) – die Rumpfhöhe hat den Bereich
    #: (1,0; 2,0) m verlassen.
    fell: bool
    #: Return ≥ Zielmarke 5000. Ausdrücklich **nicht** „gelöst".
    reached_target: bool
    timesteps: int
    #: Mittleres `vₓ` der Episode in m/s.
    mean_speed: float = 0.0
    #: Endstand aus `info["x_position"]` – die Strecke in Laufrichtung.
    distance_x: float = 0.0
    #: Betrag von `info["y_position"]`: seitliche Abweichung. Humanoid ist das
    #: erste 3D-Environment dieser Reihe; der `forward_reward` bewertet nur
    #: `vₓ`, seitliches Abdriften bliebe sonst unsichtbar.
    lateral: float = 0.0
    #: `info["distance_from_origin"]` – Gesamtstrecke in der Ebene.
    distance_origin: float = 0.0
    #: Summen der Episode, negativ wie in `info`.
    ctrl_cost: float = 0.0
    contact_cost: float = 0.0


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    reward_std: float
    mean_length: float
    survival_rate: float
    fall_rate: float
    target_rate: float
    mean_speed: float
    mean_distance_x: float
    mean_lateral: float
    mean_ctrl_cost: float
    mean_contact_cost: float


def episode_outcome(
    episode_return: float, terminated: bool, truncated: bool
) -> tuple[bool, bool, bool]:
    """Ausgang einer beendeten Episode als (durchgehalten, gestürzt, Ziel).

    `Humanoid-v5` besitzt – anders als HalfCheetah – einen terminalen Zustand:
    Verlässt die Rumpfhöhe den Bereich (1,0; 2,0) m, endet die Episode mit
    `terminated`. Eine Winkelbedingung gibt es **nicht**, die Figur darf
    beliebig verdreht sein. Terminalbonus oder -strafe existieren nicht; ein
    Sturz kostet nur die Rewards der Schritte, die nicht mehr stattfinden –
    bei `healthy_reward = 5,0` allerdings ein scharfes Signal.
    """
    survived = bool(truncated and not terminated)
    fell = bool(terminated)
    reached = episode_return >= TARGET_RETURN
    return survived, fell, reached


def torso_tilt_degrees(quaternion: Sequence[float]) -> float:
    """Neigung des Rumpfes gegen die Senkrechte in Grad.

    Das Quaternion ist für sich nicht lesbar. Die z-Achse des Rumpfes zeigt im
    Weltsystem auf `R[:, 2]`; der Winkel zur Senkrechten folgt aus `R[2, 2]`,
    das sich für `(w, x, y, z)` zu `1 − 2(x² + y²)` vereinfacht.
    """
    values = np.asarray(quaternion, dtype=float).reshape(-1)
    w, x, y, z = (float(values[index]) for index in range(4))
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return 0.0
    x, y = x / norm, y / norm
    cosine = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.degrees(math.acos(cosine))


def observation_readout(observation: np.ndarray) -> dict[str, Any]:
    """Die 348 Werte auf eine anzeigbare Auswahl verdichten.

    Die drei großen Blöcke `cinert` (130), `cvel` (78) und `cfrc_ext` (78)
    machen zusammen 286 der 348 Werte aus und erscheinen **nicht** einzeln:
    Einzelne Trägheits- und Kraftkomponenten je Körper erklären kein Verhalten.
    `cfrc_ext` wird als Quadratsumme geführt, weil genau die die Kontaktkosten
    treibt.

    Die Geschwindigkeiten sind – anders als bei Walker2d – **nicht** geclippt.
    """
    values = np.asarray(observation, dtype=float).reshape(-1)
    quaternion = [float(value) for value in values[OBS_QUATERNION]]
    contact_sum = float(np.sum(np.square(values[OBS_CFRC_EXT])))
    return {
        "height": float(values[OBS_HEIGHT]),
        "healthy_z_range": HEALTHY_Z_RANGE,
        "healthy": HEALTHY_Z_RANGE[0] <= float(values[OBS_HEIGHT]) <= HEALTHY_Z_RANGE[1],
        "quaternion": quaternion,
        "tilt_degrees": torso_tilt_degrees(quaternion),
        "joint_angles": [float(value) for value in values[OBS_JOINT_ANGLES]],
        "vx": float(values[OBS_LINEAR_VELOCITY][0]),
        "vy": float(values[OBS_LINEAR_VELOCITY][1]),
        "vz": float(values[OBS_LINEAR_VELOCITY][2]),
        "angular_velocity": [float(value) for value in values[OBS_ANGULAR_VELOCITY]],
        "joint_velocities": [float(value) for value in values[OBS_JOINT_VELOCITIES]],
        "contact_force_squared": contact_sum,
        "contact_cost": min(CONTACT_COST_WEIGHT * contact_sum, CONTACT_COST_MAX),
    }


def action_readout(action: np.ndarray) -> dict[str, Any]:
    """Die 17 Actions in ihre Gelenkbedeutung übersetzen.

    Der Wertebereich ist `±0,4`, **nicht** `±1` wie bei den Vorgängerprojekten –
    er wird aus `ACTION_LIMIT` gelesen und nicht vorausgesetzt. Das Vorzeichen
    bestimmt die Drehrichtung, der Betrag mal der gelenkeigenen Übersetzung das
    Moment in Newtonmetern. Die Übersetzungen reichen von 25 (Arme) bis 300
    (Hüfte vor/zurück): Derselbe Actionwert bedeutet je Gelenk ein anderes
    Moment.
    """
    values = np.clip(np.asarray(action, dtype=float).reshape(-1), -ACTION_LIMIT, ACTION_LIMIT)
    joints = []
    for index, (label, gear, value) in enumerate(zip(ACTUATOR_LABELS, GEARS, values)):
        direction = "—" if value == 0 else ("+" if value > 0 else "−")
        torque = abs(float(value)) * gear
        joints.append({
            "index": index,
            "joint": label,
            "actuator": ACTUATOR_NAMES[index],
            "raw": float(value),
            "gear": gear,
            "direction": direction,
            "torque": torque,
            "text": "kein Moment" if value == 0 else f"{direction} {torque:5.1f} N·m",
        })
    groups = [
        {"name": name, "joints": [joints[index] for index in indices]}
        for name, indices in ACTUATOR_GROUPS
    ]
    return {
        "raw": [float(value) for value in values],
        "joints": joints,
        "groups": groups,
        "limit": ACTION_LIMIT,
    }


def control_cost(action: np.ndarray) -> float:
    """`ctrl_cost = 0,1 · Σ aᵢ²` – höchstens `0,1 · 17 · 0,4² = 0,272`."""
    values = np.clip(np.asarray(action, dtype=float).reshape(-1), -ACTION_LIMIT, ACTION_LIMIT)
    return float(CTRL_COST_WEIGHT * np.sum(np.square(values)))


def contact_cost(external_forces: np.ndarray) -> float:
    """`contact_cost = 5e−7 · Σ cfrc_ext²`, nach oben auf 10 gedeckelt."""
    values = np.asarray(external_forces, dtype=float).reshape(-1)
    return float(min(CONTACT_COST_WEIGHT * np.sum(np.square(values)), CONTACT_COST_MAX))


def reward_readout(info: dict[str, Any]) -> dict[str, float]:
    """Die **vier** Reward-Anteile direkt aus `info` übernehmen.

    `Humanoid-v5` liefert `reward_survive`, `reward_forward`, `reward_ctrl` und
    `reward_contact` (die beiden Kosten bereits negativ). Der Überlebensbonus
    ist mit 5,0 je Schritt der mit Abstand größte Anteil. Nichts wird
    nachgerechnet.
    """
    return {
        "survive": float(info.get("reward_survive", 0.0)),
        "forward": float(info.get("reward_forward", 0.0)),
        "ctrl": float(info.get("reward_ctrl", 0.0)),
        "contact": float(info.get("reward_contact", 0.0)),
        "x_position": float(info.get("x_position", 0.0)),
        "y_position": float(info.get("y_position", 0.0)),
        "x_velocity": float(info.get("x_velocity", 0.0)),
        "y_velocity": float(info.get("y_velocity", 0.0)),
        "distance_from_origin": float(info.get("distance_from_origin", 0.0)),
    }


#: Gründe, aus denen ein Lauf endet. Die Summary weist aus, welcher gegriffen
#: hat – bei zwei Budgetgrenzen wäre das sonst nicht nachvollziehbar.
STOP_REASONS = {
    "steps": "Schrittbudget",
    "episodes": "Episodenbudget",
    "stopped": "Benutzerstopp",
}


class HumanoidCallback(BaseCallback):
    """Schreibt Episodenmetriken und Zwischenevaluationen in die GUI-Queue.

    Beendet den Lauf zusätzlich, sobald die **Episodengrenze** erreicht ist:
    Stable-Baselines3 kennt nur ein Schrittbudget, die Aufgabenstellung
    verlangt aber eine Episodeneingabe.
    """

    def __init__(
        self, stop_event: threading.Event, output: Optional[queue.Queue] = None,
        episode_offset: int = 0, series: Any = None,
        episode_callback: Optional[Callable[[EpisodeMetric], None]] = None,
        episode_limit: int = 0,
        episode_start_callback: Optional[Callable[[], None]] = None,
        episode_action_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        super().__init__(verbose=0)
        self.stop_event, self.output = stop_event, output
        self.episode_offset, self.series = episode_offset, series
        self.episode_callback = episode_callback
        self.episode_start_callback = episode_start_callback
        self.episode_action_callback = episode_action_callback
        #: `0` heißt unbegrenzt; sonst die Gesamtzahl der Episoden des Slots.
        self.episode_limit = max(0, int(episode_limit))
        self.metrics: list[EpisodeMetric] = []
        self.reward = 0.0
        self.length = 0
        self.speed_sum = 0.0
        self.ctrl_sum = 0.0
        self.contact_sum = 0.0
        self.distance_x = 0.0
        self.lateral = 0.0
        self.distance_origin = 0.0
        self.start_step = 0
        #: Wird auf `episodes` gesetzt, wenn die Episodengrenze den Lauf beendet.
        self.stop_reason = "steps"

    def _on_training_start(self) -> None:
        self.start_step = int(self.model.num_timesteps)
        # Der Lauf beginnt mit einer frisch zurückgesetzten Environment: Genau
        # jetzt ist der Zustand der Anfang der ersten Episode.
        if self.episode_start_callback is not None:
            self.episode_start_callback()

    def _raw_reward(self) -> float:
        """Reward des letzten Schritts in Originaleinheiten.

        Bei aktiver Reward-Normalisierung – das PPO-Profil verlangt sie –
        liefert `locals["rewards"]` skalierte Werte; ein Return ließe sich
        daran nicht mehr mit der Zielmarke 5000 vergleichen.
        """
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is not None and vec_normalize.norm_reward:
            return float(np.asarray(vec_normalize.get_original_reward()).reshape(-1)[0])
        return float(np.asarray(self.locals["rewards"])[0])

    def _on_step(self) -> bool:
        if self.episode_action_callback is not None:
            # Genau die Action, die an das Environment ging – nicht die aus der
            # Policy nachgerechnete. Nur so lässt sich die Episode später Schritt
            # für Schritt identisch nachspielen.
            self.episode_action_callback(np.asarray(self.locals["actions"]).reshape(-1))
        reward = self._raw_reward()
        done = bool(np.asarray(self.locals["dones"])[0])
        info = self.locals.get("infos", [{}])[0]
        self.reward += reward
        self.length += 1
        self.speed_sum += float(info.get("x_velocity", 0.0))
        self.ctrl_sum += float(info.get("reward_ctrl", 0.0))
        self.contact_sum += float(info.get("reward_contact", 0.0))
        self.distance_x = float(info.get("x_position", self.distance_x))
        self.lateral = abs(float(info.get("y_position", self.lateral)))
        self.distance_origin = float(info.get("distance_from_origin", self.distance_origin))
        if done:
            # Der Monitor sitzt innerhalb der Normalisierung und führt den
            # Return in Originaleinheiten; er hat Vorrang.
            episode_info = info.get("episode")
            total = float(episode_info["r"]) if episode_info else self.reward
            length = int(episode_info["l"]) if episode_info else self.length
            # SB3 markiert eine abgeschnittene Episode in `info`; alles andere
            # ist ein echter terminaler Zustand, also ein Sturz.
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = not truncated
            mean_speed = self.speed_sum / max(1, self.length)
            survived, fell, reached = episode_outcome(total, terminated, truncated)
            metric = EpisodeMetric(
                self.episode_offset + len(self.metrics) + 1, total, length,
                survived, fell, reached, int(self.num_timesteps),
                mean_speed, self.distance_x, self.lateral, self.distance_origin,
                self.ctrl_sum, self.contact_sum,
            )
            self.metrics.append(metric)
            # Noch im Worker-Thread und damit synchron zum Lernstand.
            if self.episode_callback is not None:
                self.episode_callback(metric)
            if self.output is not None:
                self.output.put(("episode", (self.series, metric)))
            self.reward, self.length = 0.0, 0
            self.speed_sum = self.ctrl_sum = self.contact_sum = 0.0
            self.distance_x = self.lateral = self.distance_origin = 0.0
            # Die Vektor-Environment hat bereits zurückgesetzt: Der Zustand
            # gehört jetzt zur NÄCHSTEN Episode. Genau hier wird deshalb ihr
            # Anfang gesichert – Lernstand, Simulatorzustand, Actionzähler.
            # Off-Policy-Verfahren aktualisieren bei `train_freq = 1` nach jedem
            # Schritt; am Episodenende ist die Policy längst eine andere als
            # die, mit der die Episode gelaufen ist.
            if self.episode_start_callback is not None:
                self.episode_start_callback()
            if self.episode_limit and \
                    self.episode_offset + len(self.metrics) >= self.episode_limit:
                self.stop_reason = "episodes"
                return False
        elapsed = int(self.num_timesteps) - self.start_step
        if self.output is not None and elapsed % 100 == 0:
            self.output.put(("progress", (self.series, elapsed)))
        if self.stop_event.is_set():
            self.stop_reason = "stopped"
            return False
        return True


class HumanoidWorkbench:
    """Ein Verfahrensslot: Environment, Modell, Historie, Checkpoints."""

    def __init__(self, config: Optional[HumanoidConfig] = None) -> None:
        self.config = config or default_config("PPO")
        self.config.validate()
        self.env: Optional[Any] = None
        self.model: Optional[BaseAlgorithm] = None
        self.history: list[EpisodeMetric] = []
        #: Welche der beiden Budgetgrenzen den letzten Lauf beendet hat.
        self.stop_reason: Optional[str] = None
        # Lernstand der bisher besten Episode. Gespeichert wird ausschließlich
        # dieser eine Stand je Slot – ein Verlauf über alle Episoden kostete
        # bei 348 Werten und großen Budgets Gigabytes.
        self.best_episode: Optional[EpisodeMetric] = None
        self._best_policy_state: Optional[dict[str, Any]] = None
        self._best_statistics: Optional[tuple[np.ndarray, np.ndarray]] = None
        # Lernstand zu Beginn der laufenden Episode. Beide Puffer werden
        # einmal angelegt und danach nur noch überschrieben – ein Neuanlegen je
        # Episode kostete bei 348 Eingängen Megabytes an Allokationen.
        self._episode_start_state: Optional[dict[str, Any]] = None
        self._episode_start_statistics: Optional[tuple[np.ndarray, np.ndarray]] = None
        # Aufzeichnung der laufenden Episode: Simulatorzustand am Anfang und
        # jede ausgeführte Action. Damit lässt sich die beste Episode später
        # **exakt** nachspielen statt sie mit der Policy nachzurechnen – eine
        # Wiederholung erreicht sonst nur 80 bis 90 % ihres Returns, weil die
        # Episode explorativ und aus einem anderen Startzustand lief.
        self._episode_actions = np.zeros((MAX_EPISODE_STEPS, ACTION_DIM), dtype=np.float32)
        self._episode_action_count = 0
        self._episode_start_simulator: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._best_actions = np.zeros((MAX_EPISODE_STEPS, ACTION_DIM), dtype=np.float32)
        self._best_action_count = 0
        self._best_simulator: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._best_lock = threading.Lock()

    @property
    def uses_replay_buffer(self) -> bool:
        return self.config.algorithm in OFF_POLICY_ALGORITHMS

    @property
    def vec_normalize(self) -> Optional[VecNormalize]:
        return self.env if isinstance(self.env, VecNormalize) else None

    @property
    def num_timesteps(self) -> int:
        """Ausgeführte Environment-Schritte des Slots."""
        return int(self.model.num_timesteps) if self.model is not None else 0

    def _make_training_env(self) -> Any:
        """Vektor-Environment mit Monitor und optionaler Normalisierung.

        Der `Monitor` sitzt bewusst **innerhalb** der Normalisierung: Nur so
        führt er den Episoden-Return in Originaleinheiten, an dem sich die
        Zielmarke 5000 überhaupt ablesen lässt. Der `Float32Observation`-
        Wrapper liegt darunter und halbiert den Replay Buffer.
        """
        env: Any = DummyVecEnv([
            lambda: Monitor(Float32Observation(make_humanoid_env(render_mode=None)))
        ])
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
        self.stop_reason = None
        with self._best_lock:
            self.best_episode = None
            self._best_policy_state = None
            self._best_statistics = None
            self._best_action_count = 0
            self._best_simulator = None
        self._episode_start_state = None
        self._episode_start_statistics = None
        self._episode_start_simulator = None
        self._episode_action_count = 0
        return self.model

    def _observation_statistics(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Mittelwert und Varianz der Beobachtungsnormalisierung, falls aktiv."""
        vec_normalize = self.vec_normalize
        if vec_normalize is None or not vec_normalize.norm_obs:
            return None
        return (np.array(vec_normalize.obs_rms.mean, copy=True),
                np.array(vec_normalize.obs_rms.var, copy=True))

    def _simulator_state(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Positionen und Geschwindigkeiten des Simulators.

        Für die exakte Wiedergabe unverzichtbar: Ohne denselben Startzustand
        läuft dieselbe Policy in eine andere Episode. Der Zugriff auf
        `unwrapped` ist hier bewusst – für **Messwerte** wäre er unzulässig,
        fürs Nachspielen gibt es keinen anderen Weg.
        """
        env = self.env
        try:
            inner = env.venv.envs[0] if isinstance(env, VecNormalize) else env.envs[0]
            data = inner.unwrapped.data
        except (AttributeError, IndexError):
            return None
        return np.array(data.qpos, copy=True), np.array(data.qvel, copy=True)

    def record_action(self, action: np.ndarray) -> None:
        """Eine ausgeführte Action der laufenden Episode mitschreiben."""
        if self._episode_action_count < MAX_EPISODE_STEPS:
            self._episode_actions[self._episode_action_count] = action
            self._episode_action_count += 1

    def begin_episode(self) -> None:
        """Sichert den Lernstand, mit dem die beginnende Episode läuft.

        Läuft im Worker-Thread zu Beginn jeder Episode. Der Aufwand ist ein
        Speicher-zu-Speicher-Kopieren in bereits angelegte Puffer; das lohnt
        sich, weil erst am Episodenende feststeht, ob dieser Stand der beste
        ist – und weil die Policy bis dahin längst eine andere ist.

        Bei aktiver Normalisierung werden die Statistiken mitgesichert: Eine
        alte Policy mit neuen Statistiken sähe die Beobachtungen anders als im
        Training.
        """
        if self.model is None:
            return
        state = self.model.policy.state_dict()
        if self._episode_start_state is None:
            self._episode_start_state = {name: value.detach().clone()
                                         for name, value in state.items()}
        else:
            for name, value in state.items():
                self._episode_start_state[name].copy_(value.detach())
        self._episode_start_statistics = self._observation_statistics()
        self._episode_start_simulator = self._simulator_state()
        self._episode_action_count = 0

    def remember_episode(self, metric: EpisodeMetric) -> None:
        """Übernimmt den Stand vom Episodenbeginn, sobald eine Episode alle
        bisherigen schlägt.

        Ausdrücklich **nicht** der Stand von jetzt: Off-Policy-Verfahren
        aktualisieren die Policy bei `train_freq = 1` nach jedem Schritt. Am
        Episodenende ist sie eine andere als die, die diese Episode erzeugt
        hat – gemessen rund 2 % Gewichtsänderung je Episode, was den
        deterministischen Return spürbar verschiebt.
        """
        if self.model is None or self._episode_start_state is None:
            return
        with self._best_lock:
            if self.best_episode is not None and metric.reward <= self.best_episode.reward:
                return
            self.best_episode = metric
            if self._best_policy_state is None:
                self._best_policy_state = {name: value.clone()
                                           for name, value in self._episode_start_state.items()}
            else:
                for name, value in self._episode_start_state.items():
                    self._best_policy_state[name].copy_(value)
            self._best_statistics = self._episode_start_statistics
            self._best_actions[:self._episode_action_count] = \
                self._episode_actions[:self._episode_action_count]
            self._best_action_count = self._episode_action_count
            self._best_simulator = self._episode_start_simulator

    def best_snapshot(self) -> Optional[tuple[EpisodeMetric, dict[str, Any],
                                              Optional[tuple[np.ndarray, np.ndarray]]]]:
        """Beste Episode, die zugehörige Policy-Kopie und ihre
        Beobachtungsstatistiken."""
        with self._best_lock:
            if self.best_episode is None or self._best_policy_state is None:
                return None
            return self.best_episode, self._best_policy_state, self._best_statistics

    def best_recording(self) -> Optional[tuple[EpisodeMetric, tuple[np.ndarray, np.ndarray],
                                               np.ndarray]]:
        """Beste Episode als **Aufzeichnung**: Startzustand und alle Actions.

        Damit lässt sie sich Schritt für Schritt identisch nachspielen – gleiche
        Bewegung, gleicher Return, jedes Mal. Die Policy-Wiederholung
        (`best_snapshot`) beantwortet dagegen die andere Frage „wie gut ist
        diese Policy" und erreicht den Wert der Episode systematisch nicht:
        Sie ist das Maximum über Hunderte Episoden und verdankt es zum Teil
        glücklichen Explorationszügen.
        """
        with self._best_lock:
            if self.best_episode is None or self._best_simulator is None \
                    or not self._best_action_count:
                return None
            return (self.best_episode, self._best_simulator,
                    np.array(self._best_actions[:self._best_action_count], copy=True))

    def current_snapshot(self) -> Optional[dict[str, Any]]:
        """Losgelöste Kopie des aktuellen Lernstands für die Animation."""
        if self.model is None:
            return None
        return {name: value.detach().clone()
                for name, value in self.model.policy.state_dict().items()}

    def policy_observation(
        self, observation: np.ndarray,
        statistics: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Beobachtung so aufbereiten, wie das Modell sie im Training sieht.

        Die laufenden Statistiken werden dabei **nicht** fortgeschrieben:
        `normalize_obs` liest sie nur. Wird `statistics` übergeben, gelten
        stattdessen diese – so sieht eine wiederholte Episode dieselben Werte
        wie damals, statt die alte Policy mit heutigen Statistiken zu füttern.
        """
        values = np.asarray(observation, dtype=np.float32)
        vec_normalize = self.vec_normalize
        if vec_normalize is None or not vec_normalize.norm_obs:
            return values
        if statistics is None:
            return vec_normalize.normalize_obs(values)
        mean, var = statistics
        clip = vec_normalize.clip_obs
        return np.clip((values - mean) / np.sqrt(var + vec_normalize.epsilon),
                       -clip, clip).astype(np.float32)

    def act(self, snapshot: Optional[dict[str, Any]], observation: np.ndarray) -> np.ndarray:
        """Deterministische Action eines gegebenen Lernstands."""
        if self.model is None:
            raise RuntimeError("Ohne Modell gibt es keine Action.")
        if snapshot is not None:
            self.model.policy.load_state_dict(snapshot)
        action, _ = self.model.predict(
            self.policy_observation(observation), deterministic=True)
        return action

    def evaluate(self, episodes: int, seed: int = 0) -> EvaluationResult:
        """Deterministische Evaluation ohne Exploration und ohne Lernupdates.

        Sie läuft auf einer eigenen, unnormalisierten Environment: Die Returns
        bleiben damit in Originaleinheiten und sind mit der Zielmarke 5000
        vergleichbar.
        """
        if self.model is None:
            raise RuntimeError("Vor der Evaluation muss ein Modell trainiert sein.")
        env = Float32Observation(make_humanoid_env(render_mode=None))
        rewards, lengths, speeds, distances, laterals = [], [], [], [], []
        ctrl_costs, contact_costs = [], []
        survivals = falls = targets = 0
        try:
            for index in range(episodes):
                observation, _ = env.reset(seed=seed + index)
                total, length = 0.0, 0
                terminated = truncated = False
                speed_sum = ctrl_sum = contact_sum = 0.0
                distance_x = lateral = 0.0
                while not (terminated or truncated):
                    action, _ = self.model.predict(
                        self.policy_observation(observation), deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total += float(reward)
                    length += 1
                    # Positionen, Tempo und Kosten stehen offiziell in `info`;
                    # auf Interna des Environments wird nicht zugegriffen.
                    speed_sum += float(info.get("x_velocity", 0.0))
                    ctrl_sum += float(info.get("reward_ctrl", 0.0))
                    contact_sum += float(info.get("reward_contact", 0.0))
                    distance_x = float(info.get("x_position", distance_x))
                    lateral = abs(float(info.get("y_position", lateral)))
                mean_speed = speed_sum / max(1, length)
                survived, fell, reached = episode_outcome(total, terminated, truncated)
                rewards.append(total)
                lengths.append(length)
                speeds.append(mean_speed)
                distances.append(distance_x)
                laterals.append(lateral)
                ctrl_costs.append(ctrl_sum)
                contact_costs.append(contact_sum)
                survivals += int(survived)
                falls += int(fell)
                targets += int(reached)
        finally:
            env.close()
        return EvaluationResult(
            episodes, mean(rewards), pstdev(rewards), mean(lengths),
            survivals / episodes, falls / episodes, targets / episodes,
            mean(speeds), mean(distances), mean(laterals),
            mean(ctrl_costs), mean(contact_costs),
        )

    def train(
        self, stop_event: Optional[threading.Event] = None,
        output: Optional[queue.Queue] = None, series: Any = None,
    ) -> list[EpisodeMetric]:
        """Trainiert bis zur zuerst erreichten Budgetgrenze.

        **Fortsetzen hängt an, statt zurückzusetzen.** Beide Grenzen gelten
        relativ zum bereits Gelaufenen: Ein zweiter Aufruf liefert erneut das
        volle Budget an Schritten *und* Episoden. Bei den Schritten erledigt
        das Stable-Baselines3 selbst (`reset_num_timesteps=False` rechnet
        `total_timesteps += num_timesteps`); bei den Episoden muss die Grenze
        ausdrücklich um die bereits gelaufenen verschoben werden – sonst wäre
        sie beim zweiten Druck sofort überschritten.

        Welche Grenze gegriffen hat, steht danach in `stop_reason` und geht in
        die Summary ein: Bei zwei Grenzen wäre das sonst nicht nachvollziehbar.
        """
        reset = self.model is None
        if reset:
            self.create_model()
        # Der Ausgang des **vorigen** Laufs darf nicht stehen bleiben: Bei einem
        # Fortsetzen wird kein Modell neu angelegt, und die Summary zeigte sonst
        # stundenlang „Benutzerstopp", während längst wieder trainiert wird.
        self.stop_reason = None
        done = len(self.history)
        callback = HumanoidCallback(
            stop_event or threading.Event(), output, done, series,
            self.remember_episode,
            episode_limit=done + self.config.episodes if self.config.episodes else 0,
            episode_start_callback=self.begin_episode,
            episode_action_callback=self.record_action,
        )
        self.model.learn(
            self.config.total_timesteps, callback=callback,
            reset_num_timesteps=reset, progress_bar=False,
        )
        self.history.extend(callback.metrics)
        self.stop_reason = callback.stop_reason
        return callback.metrics

    def save(self, path: str | Path) -> tuple[Path, Optional[Path], Path]:
        """Speichert genau die Bestandteile, die das Verfahren besitzt.

        PPO ist on-policy: Policy, Value-Netz und Optimizer stecken in der
        SB3-Datei, einen Replay Buffer gibt es nicht. SAC und TD3 sichern ihn
        zusätzlich; bei SAC gehört der gelernte Temperaturparameter zum Modell.
        Bei aktiver Normalisierung – das PPO-Profil verlangt sie – kommen die
        laufenden Statistiken dazu.
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
    ) -> "HumanoidWorkbench":
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
        workbench = cls(HumanoidConfig.from_dict(metadata["config"]))
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
