"""Dark Tkinter GUI for the Humanoid PPO/SAC/TD3 workbench."""

from __future__ import annotations

import math
import queue
import re
import tempfile
import threading
import time
import tkinter as tk
import dataclasses
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from humanoid_logic import (
    ACTION_LIMIT, ACTUATOR_GROUPS, ACTUATOR_TO_JOINT, ALGORITHMS, BOOLEAN_FIELDS,
    CHOICE_FIELDS, CONTACT_COST_MAX, DEFAULT_TORCH_THREADS, DEFAULT_TOTAL_TIMESTEPS,
    FRAME_HEIGHT, FRAME_WIDTH,
    INTEGER_FIELDS, MAX_EPISODE_STEPS, OPTIONAL_FLOAT_FIELDS, REPORT_TIMESTEPS,
    STOP_REASONS,
    TARGET_RETURN, TUPLE_FIELDS, EpisodeMetric, HumanoidConfig,
    HumanoidWorkbench, action_readout, config_differences, default_config,
    observation_readout, reward_readout, set_torch_threads,
)
from humanoid_render import HumanoidRenderer, RendererUnavailable


#: Standardbreite des gleitenden Durchschnitts in Episoden. Bei 1 fällt die
#: geglättete Kurve mit den Rohwerten zusammen.
DEFAULT_SMOOTHING, MIN_SMOOTHING, MAX_SMOOTHING = 20, 1, 500


def rolling_average(values: list[float], window: int = DEFAULT_SMOOTHING) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.asarray([array[max(0, index - window + 1):index + 1].mean()
                       for index in range(len(array))])


def downsample_minmax(x: list[int], y: list[float], limit: int = 2_000) -> tuple[np.ndarray, np.ndarray]:
    x_array, y_array = np.asarray(x), np.asarray(y)
    if len(x_array) <= limit:
        return x_array, y_array
    edges = np.linspace(0, len(x_array), limit // 2 + 1, dtype=int)
    indices = {0, len(x_array) - 1}
    for start, end in zip(edges[:-1], edges[1:]):
        if end > start:
            segment = y_array[start:end]
            indices.update((start + int(segment.argmin()), start + int(segment.argmax())))
    selected = np.asarray(sorted(indices))[:limit]
    return x_array[selected], y_array[selected]


def decimate(x: list[int], y: np.ndarray, limit: int = 2_000) -> tuple[np.ndarray, np.ndarray]:
    """Gleichmäßig ausdünnen, ohne die Form zu verändern.

    Für die **geglättete** Kurve. Min/Max-Verdichtung wie bei den Rohwerten
    wäre hier falsch: Sie machte eine bewusst glatte Linie wieder zackig. Weil
    der gleitende Durchschnitt zwischen benachbarten Episoden ohnehin kaum
    springt, verliert einfaches Auslassen hier nichts – der letzte Punkt bleibt
    in jedem Fall erhalten.
    """
    values = np.asarray(y, dtype=float)
    count = len(values)
    if count <= limit:
        return np.asarray(x), values
    index = np.unique(np.append(np.linspace(0, count - 1, limit).astype(int), count - 1))
    return np.asarray(x)[index], values[index]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def german(value: float, digits: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma und Tausenderpunkt."""
    return f"{value:,.{digits}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def best_columns(count: int, width: int, height: int, chrome: int,
                 frame_aspect: float = 1.0) -> int:
    """Spaltenzahl, die das größte Bild ergibt.

    Eine feste Aufteilung – etwa immer zwei mal zwei – verschenkt Platz, sobald
    der Bereich nicht zufällig dasselbe Seitenverhältnis hat wie das Raster.
    Bei quadratischen Frames in einem breiten, flachen Bereich sind die Bilder
    durch die **Höhe** begrenzt: Mehr Breite bringt dann gar nichts, mehr Zeilen
    kosten dagegen unmittelbar. Drei Anzeigen nebeneinander sind dort deutlich
    größer als zwei über zwei.

    `chrome` ist die Höhe, die je Zelle neben dem Bild gebraucht wird
    (Beschriftung, Auswahl, Rahmen); sie fällt **je Zeile** an und macht Zeilen
    teurer als Spalten.
    """
    if count <= 1:
        return 1
    best, best_scale = 1, -1.0
    for columns in range(1, count + 1):
        rows = math.ceil(count / columns)
        cell_width = width / columns - 16
        cell_height = height / rows - chrome
        if cell_width <= 0 or cell_height <= 0:
            continue
        # Größte Kantenlänge, die in die Zelle passt, ohne das Seitenverhältnis
        # zu verletzen.
        scale = min(cell_width / frame_aspect, cell_height)
        if scale > best_scale:
            best, best_scale = columns, scale
    return best


def panel_grid_positions(count: int, columns: int = 2) -> list[tuple[int, int, int]]:
    """Rasterplätze der Animationsanzeigen als (Zeile, Spalte, Spaltenbreite).

    **Alle belegten Zellen sind gleich groß.** Die letzte Anzeige über eine
    freie Nachbarzelle zu spannen wäre naheliegend, brächte bei quadratischen
    Frames aber nichts: Sie sind in einem breiten Raster durch die Höhe
    begrenzt, eine doppelt breite Zelle liefert dasselbe Bild. Der Gewinn wäre
    null, der Verlust an Gleichmäßigkeit sichtbar.
    """
    columns = max(1, min(columns, max(1, count)))
    return [(index // columns, index % columns, 1) for index in range(max(1, count))]


#: Environment-eigene Bildrate (`env.metadata["render_fps"]`, also 1/dt mit
#: dt = 0,015 s). Bei ihr dauert eine volle Episode 15 Sekunden.
RENDER_FPS = 67
#: Standard der Animation, bewusst **unter** der environment-eigenen Rate:
#: 20 FPS ergeben rund dreifache Zeitlupe. Ein stürzender Humanoid ist bei
#: 67 FPS kaum zu verfolgen, und die langsamere Anzeige kostet zugleich weniger
#: Rechenzeit, die dem Training zugutekommt.
DEFAULT_ANIMATION_FPS = 20
MIN_ANIMATION_FPS, MAX_ANIMATION_FPS = 1, 250

#: Obergrenze der je Kurve gezeichneten Punkte – für Roh- **und** geglättete
#: Kurve. Ohne sie wächst die Zeichenzeit von matplotlib linear mit der
#: Episodenzahl; bei mehreren Slots und zehntausenden Episoden wird die
#: Oberfläche sonst unbenutzbar. Die Messdaten bleiben vollständig, nur die
#: Darstellung wird verdichtet.
MAX_PLOT_POINTS = 2_000

#: Bis zu vier gleichrangige Verfahrensslots.
MAX_SLOTS = 4
SLOT_COUNTS = (2, 3, 4)
#: Startbelegung: die drei zugeteilten Algorithmen nebeneinander, jeder genau
#: einmal. Ein vierter Slot ist wählbar; dann kommt zwangsläufig ein
#: Algorithmus doppelt vor und die Workbench-Regel zum abweichenden Startwert
#: greift.
DEFAULT_SLOT_COUNT = 3
DEFAULT_SLOT_ALGORITHMS = ("PPO", "TD3", "SAC", "SAC")
#: Nur für den optionalen vierten Slot: Ohne Abweichung wäre er mit Slot 3
#: identisch und der Vergleich zeigte keinen Unterschied.
DEFAULT_SLOT_OVERRIDES: dict[int, dict[str, Any]] = {3: {"learning_rate": 1e-4}}
SLOT_LABELS = ("Verfahren 1", "Verfahren 2", "Verfahren 3", "Verfahren 4")
SLOT_SHORT = ("V1", "V2", "V3", "V4")
#: Feste Farbzuordnung der Slots: blau, rot, gelb, grün. Helle, kräftige Töne,
#: damit sie sich auf dem dunklen Hintergrund und voneinander abheben.
SLOT_COLORS = ("#60a5fa", "#f87171", "#facc15", "#4ade80")
#: Referenz- und Schwellenlinien sind weiß und gestrichelt. Weiß ist keiner
#: Slotfarbe zugeordnet, durchgezogen bleibt den Slotkurven vorbehalten.
REFERENCE_COLOR = "#ffffff"
REFERENCE_STYLE = "--"
#: Slotkurven sind ausnahmslos durchgezogen.
SLOT_LINESTYLE = "-"

#: Welche Episode eine Animation zeigt. Gespeichert wird je Slot genau ein
#: zusätzlicher Lernstand – der der besten Episode; ein Verlauf über alle
#: Episoden kostete bei großen Budgets Gigabytes.
CURRENT_EPISODE, BEST_EPISODE = "akt. Ep.", "beste Ep."
#: Der Lernstand der besten Episode, deterministisch und mit festem Seed – im
#: Unterschied zu `BEST_EPISODE`, das die Episode selbst nachspielt. Beide
#: beantworten verschiedene Fragen: die Aufzeichnung „was ist damals passiert",
#: die Policy „wie gut ist dieser Stand ohne das Glück explorativer Züge".
#: Gemessen liegt die Policy-Wiedergabe bei 80 bis 91 % des Episodenwerts; der
#: Abstand ist selbst eine Aussage darüber, wie viel des Spitzenwerts Zufall war.
BEST_POLICY = "beste Pol."
#: Dritte Wahl: Die Animation dieses Slots bleibt aus. Bei drei oder vier
#: gleichzeitig laufenden Verfahren kostet jede sichtbare Anzeige Rechenzeit
#: und einen eigenen Renderprozess – wer nur eines beobachten will, schaltet
#: die übrigen einzeln ab, ohne den globalen Schalter zu benutzen.
INACTIVE_EPISODE = "inaktiv"
EPISODE_CHOICES = (CURRENT_EPISODE, BEST_EPISODE, BEST_POLICY, INACTIVE_EPISODE)

#: Kurze Anzeigenamen; die ausführlichen Bezeichnungen stehen in
#: `FIELD_LABELS` der Logik und erscheinen in Fehlermeldungen und im
#: Unterschiedszeilen der Summary.
SHORT_LABELS = {
    "total_timesteps": "Schritte N", "episodes": "Episoden E",
    "batch_size": "Batch B", "learning_rate": "Lernrate α",
    "learning_rate_schedule": "LR-Verlauf", "gamma": "Diskont γ", "seed": "Seed s",
    "actor_arch": "Actor h_π", "critic_arch": "Critic h_q", "activation": "Aktiv. φ",
    "optimizer": "Optim.", "optimizer_eps": "ε_opt", "optimizer_weight_decay": "Zerfall λ",
    "n_steps": "Rollout n", "n_epochs": "Epochen K", "gae_lambda": "GAE λ", "clip_range": "Clip ε",
    "clip_range_vf": "Clip VF", "target_kl": "KL-Limit", "ent_coef": "Entropie c",
    "vf_coef": "Value c_v", "max_grad_norm": "Grad-Norm",
    "normalize_advantage": "Advantage normieren", "ortho_init": "Ortho-Init",
    "log_std_init": "log σ₀", "use_sde": "gSDE nutzen", "sde_sample_freq": "gSDE f",
    "buffer_size": "Buffer |D|", "learning_starts": "Start t₀", "tau": "Soft τ",
    "train_freq": "Freq. fₜ", "gradient_steps": "Grad. G", "action_noise": "Noise-Typ",
    "action_noise_sigma": "Noise σ", "policy_delay": "Delay d",
    "target_policy_noise": "Ziel σ_t", "target_noise_clip": "Clip c",
    "ent_coef_mode": "Entropie α", "ent_coef_value": "Start α", "target_entropy": "Ziel H*",
    "target_update_interval": "Target C",
    "normalize_obs": "Obs normieren", "normalize_reward": "Reward normieren",
    "clip_obs": "Clip Obs", "clip_reward": "Clip Rew",
}
#: Zeichenbreite der Wertefelder und die Mindestbreite ihrer Rasterspalte in
#: Pixeln. Die Mindestbreite liegt bewusst **über** der Eigenbreite des
#: breitesten Widgettyps: Eine `ttk.Combobox` mit sieben Zeichen misst von sich
#: aus 84 px (Aufklapp-Pfeil inbegriffen), ein `ttk.Entry` nur 69. Läge die
#: Mindestbreite darunter, würde jede Spalte mit einer Combobox darin über sie
#: hinauswachsen, Spalten mit reinen Eingabefeldern aber nicht – untereinander
#: stehende Felder wären dann unterschiedlich breit und gegeneinander
#: verschoben. Mit 90 px (84 Eigenbreite + 6 Abstand) bestimmt allein das
#: Raster die Breite.
VALUE_CHARS, VALUE_WIDTH = 7, 90
#: Mindestbreite der Beschriftungsspalten. Die breiteste reguläre Beschriftung
#: misst 73 px; 80 lässt Luft für den Abstand zum Feld. Zusammen mit
#: `VALUE_WIDTH` ergibt das mit Abständen und Gruppenrahmen 350 px und bleibt
#: damit unter den 352 px, die einer Parameterspalte zur Verfügung stehen. Die
#: breiteste reguläre Beschriftung misst 73 px und passt. Die natürliche Breite
#: der Felder (Entry 69 px, Combobox 84 px) liegt darunter, sodass keines die
#: Spalte über ihre Mindestbreite hinaus aufzieht.
LABEL_WIDTH = 78
#: Rasterspalten eines Parameterblocks: Beschriftung, Wert, Beschriftung, Wert
#: – und eine Füllspalte. Ohne sie bekämen die Wertespalten das übrige Gewicht
#: und dehnten sich je nach Gruppenbreite unterschiedlich weit.
GROUP_COLUMNS = ((0, LABEL_WIDTH), (1, VALUE_WIDTH), (2, LABEL_WIDTH), (3, VALUE_WIDTH))
SPACER_COLUMN = 4
#: Die Verfahrensauswahl braucht drei Spalten: Beschriftung, Wert und – bei den
#: Slotzeilen – die Wahl der Animation daneben.
SELECTION_COLUMNS = ((0, LABEL_WIDTH), (1, VALUE_WIDTH), (2, VALUE_WIDTH))
SELECTION_SPACER = 3
#: Ab dieser Zeile stehen die Slots. Darüber die beiden globalen Einstellungen,
#: die für alle gelten – erst die Regel, dann die Belegung.
SELECTION_ROW_OFFSET = 2


def apply_field_grid(grid: tk.Widget) -> None:
    """Allen Parameterblöcken dieselbe Spaltengeometrie geben.

    Jeder Block ist ein eigenes Raster. Ohne feste Breiten richtet sich seine
    Beschriftungsspalte nach der zufällig längsten Beschriftung darin – die
    Wertefelder verschiedener Blöcke stünden dann untereinander weder an
    derselben Stelle noch gleich breit. Die Füllspalte nimmt den Rest auf,
    damit sich die Wertespalten nicht dehnen.
    """
    for column, width in GROUP_COLUMNS:
        grid.columnconfigure(column, minsize=width, weight=0, uniform="")
    grid.columnconfigure(SPACER_COLUMN, weight=1)

#: `episodes` steht direkt neben `total_timesteps`: Beide sind Budgetgrenzen,
#: der Lauf endet an der zuerst erreichten. `0` heißt unbegrenzt.
_TRAINING = ("total_timesteps", "episodes", "batch_size", "gamma", "seed", "learning_rate",
             "learning_rate_schedule")
_NETWORK = ("actor_arch", "critic_arch", "activation", "optimizer_eps", "optimizer",
            "optimizer_weight_decay")
_REPLAY = ("buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps")
_NETWORK_GROUP = ("right", "Neuronales Netz und Optimizer", _NETWORK)
_TRAINING_GROUP = ("left", "Training", _TRAINING)
#: Normalisierung gilt für alle drei Verfahren; das PPO-Zoo-Profil verlangt sie
#: für dieses Environment ausdrücklich.
_NORMALIZATION = ("clip_obs", "clip_reward", "normalize_obs", "normalize_reward")

#: Aufbau eines Verfahrenstabs: (Spalte, Gruppentitel, Parameter). Ein Tab
#: zeigt ausschließlich Parameter des dort gewählten Algorithmus.
PARAMETER_GROUPS: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "PPO": (
        _TRAINING_GROUP,
        ("left", "Rollout und geclipptes Ziel",
         # `ortho_init` gehört zur Initialisierung und kennt nur PPO; das
         # Zoo-Profil schaltet sie für Humanoid ab. Sie steht hier statt in der
         # Netzgruppe, damit beide Spalten des Tabs etwa gleich hoch bleiben.
         ("n_steps", "n_epochs", "gae_lambda", "clip_range", "clip_range_vf", "target_kl",
          "ent_coef", "vf_coef", "max_grad_norm", "normalize_advantage", "ortho_init")),
        _NETWORK_GROUP,
        ("right", "Exploration (stochastische Policy)",
         ("log_std_init", "sde_sample_freq", "use_sde")),
        ("right", "Normalisierung", _NORMALIZATION),
    ),
    "SAC": (
        _TRAINING_GROUP,
        ("left", "Replay Buffer und Updates", _REPLAY),
        ("left", "Normalisierung", _NORMALIZATION),
        _NETWORK_GROUP,
        ("right", "Entropieregularisierung",
         ("ent_coef_value", "target_entropy", "target_update_interval", "ent_coef_mode")),
        ("right", "Exploration (optional)",
         ("log_std_init", "sde_sample_freq", "action_noise_sigma", "use_sde", "action_noise")),
    ),
    "TD3": (
        _TRAINING_GROUP,
        ("left", "Replay Buffer und Updates", _REPLAY),
        ("left", "Normalisierung", _NORMALIZATION),
        _NETWORK_GROUP,
        ("right", "Twin Critics und Policy Delay",
         ("policy_delay", "target_policy_noise", "target_noise_clip")),
        ("right", "Exploration (Action Noise)", ("action_noise_sigma", "action_noise")),
    ),
}

EXPORT_DIR = Path(__file__).parent / "exports"
#: Breite der beiden Parameterspalten. Sie tragen gemeinsam die Verfahrenstabs
#: und bestimmen damit die Breite des ganzen Bedienpanels. Eine eigene dritte
#: Spalte für die Steuerung gibt es nicht mehr: Sie stand seit der Reduktion
#: auf vier Schaltflächen zu zwei Dritteln leer, und ihre Breite fehlt der
#: Animation.
CONTROL_COLUMNS = (360, 360)
CONTROL_WIDTH = sum(CONTROL_COLUMNS) + 16
PARAMETER_WIDTH = CONTROL_COLUMNS[0] + CONTROL_COLUMNS[1]
#: Breite der Steuerungsspalte und der Statusspalte in der Kopfzeile des
#: Bedienpanels, neben der Verfahrensauswahl.
ACTION_WIDTH, STATUS_WIDTH = 176, 200
#: Mindesthöhe, die dem unteren Bereich aus Diagrammen und Summary bleibt.
MIN_CHART_HEIGHT = 220
#: Mindestbreite der Environment-Anzeige neben dem Bedienpanel und die Breite,
#: die ihr beim Start zusteht, sofern der Bildschirm sie hergibt. Bei bis zu
#: vier Anzeigen im 2x2-Raster braucht sie mehr Platz als bei zweien.
MIN_ENVIRONMENT_WIDTH, WANTED_ENVIRONMENT_WIDTH = 340, 640
#: Platz, den Menüleiste, Fensterrahmen und Dock vom Bildschirm beanspruchen.
SCREEN_MARGIN = 130


class SlotRuntime:
    """Lernzustand eines Verfahrensslots: Einzeltraining und Vergleich getrennt.

    Der Vergleich benutzt bewusst eigene Modelle, damit er das sichtbare
    Experiment des Einzeltrainings nicht verändert.
    """

    def __init__(self, config: HumanoidConfig) -> None:
        self.workbench = HumanoidWorkbench(config)
        self.comparison: Optional[HumanoidWorkbench] = None
        self.history: list[EpisodeMetric] = []
        self.comparison_history: list[EpisodeMetric] = []
        self.lock = threading.Lock()

    @property
    def algorithm(self) -> str:
        return self.workbench.config.algorithm

    def has_data(self) -> bool:
        return bool(self.workbench.model or self.history or self.comparison_history)

    def reset_comparison(self) -> None:
        if self.comparison is not None:
            self.comparison.close()
        self.comparison = None
        self.comparison_history.clear()

    def reset_all(self, config: HumanoidConfig) -> None:
        self.workbench.close()
        self.workbench = HumanoidWorkbench(config)
        self.history.clear()
        self.reset_comparison()

    def close(self) -> None:
        self.workbench.close()
        if self.comparison is not None:
            self.comparison.close()


class HumanoidGUI:
    BG, PANEL, FIELD = "#111827", "#1f2937", "#0f172a"
    FG, MUTED, ACCENT = "#f3f4f6", "#cbd5e1", "#60a5fa"
    #: Strichstärke der hervorgehobenen Slotkurven.
    LINE_WIDTH = 1.2
    #: Deckkraft der dezenten Rohkurve aus den einzelnen Episodenergebnissen.
    #: Kräftig genug, um Ausreißer zu erkennen, blass genug, um den gleitenden
    #: Durchschnitt nicht zu überdecken.
    RAW_ALPHA = 0.10
    PLOT_INTERVAL = 2.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.busy = False
        self.running_slot: Optional[int] = None
        self.comparison_running = False
        self.slot_count = DEFAULT_SLOT_COUNT
        # Animationszustand je Verfahrensslot: Im Vergleich laufen alle aktiven
        # Verfahren gleichzeitig und werden auch gleichzeitig gezeigt.
        self.renderers: list[Optional[HumanoidRenderer]] = [None] * MAX_SLOTS
        self.renderer_failed = False
        self.photos: list[Optional[ImageTk.PhotoImage]] = [None] * MAX_SLOTS
        self.photo_sizes: list[tuple[int, int]] = [(0, 0)] * MAX_SLOTS
        self.last_frames: list[Optional[np.ndarray]] = [None] * MAX_SLOTS
        self.animation_after: list[Optional[str]] = [None] * MAX_SLOTS
        self.animation_observation: list[Optional[np.ndarray]] = [None] * MAX_SLOTS
        self.animation_info: list[dict[str, Any]] = [{} for _ in range(MAX_SLOTS)]
        self.animation_action: list[Optional[np.ndarray]] = [None] * MAX_SLOTS
        self.animation_policy: list[Any] = [None] * MAX_SLOTS
        #: Eingefrorene Beobachtungsstatistiken der wiederholten Episode.
        self.animation_statistics: list[Any] = [None] * MAX_SLOTS
        #: Aufgezeichnete Actions der besten Episode und die Stelle darin. Sind
        #: sie gesetzt, spielt die Animation die Episode Schritt für Schritt
        #: nach, statt sie mit der Policy nachzurechnen.
        self.animation_actions: list[Any] = [None] * MAX_SLOTS
        self.animation_action_index = [0] * MAX_SLOTS
        self.animation_live = [False] * MAX_SLOTS
        self.animation_step = [0] * MAX_SLOTS
        self.animation_episode = [0] * MAX_SLOTS
        self.animation_reward = [0.0] * MAX_SLOTS
        self.animation_from_best = [False] * MAX_SLOTS
        self.animation_interval = max(1, round(1000 / RENDER_FPS))
        self.hover_slot: Optional[int] = None
        # Zuletzt gültige Fensterbreite des gleitenden Durchschnitts: Eine
        # ungültige Eingabe lässt sie stehen, statt die Kurven zu verwerfen.
        self.smoothing_window = DEFAULT_SMOOTHING
        self.last_plot = 0.0
        self.progress_steps: dict[Any, int] = {}
        self.progress_start: dict[Any, int] = {}
        self.progress_total = 1
        #: "episodes" oder "steps" – siehe `_begin_progress`.
        self.progress_mode = "episodes"
        self.entries: list[dict[str, tk.Widget]] = [{} for _ in range(MAX_SLOTS)]
        self.slots = [SlotRuntime(self.initial_config(index)) for index in range(MAX_SLOTS)]
        self._export_snapshot_key: Optional[tuple[Any, ...]] = None
        self._export_stamp: Optional[str] = None
        self._variables()
        self._window()
        self._layout()
        self._sync_slot_widgets()
        self._update_active_label()
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._refresh_single_plot()
        self._training_summary()
        self.root.after_idle(self._initialize_layout)
        self.root.after(100, lambda: self._show_initial_frame(0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # --------------------------------------------------------------- Variablen

    @staticmethod
    def initial_config(slot: int) -> HumanoidConfig:
        """Startkonfiguration eines Slots: Zoo-Profil plus Startabweichungen."""
        config = default_config(DEFAULT_SLOT_ALGORITHMS[slot])
        overrides = DEFAULT_SLOT_OVERRIDES.get(slot)
        return dataclasses.replace(config, **overrides) if overrides else config

    def _variables(self) -> None:
        self.algorithm_vars = [tk.StringVar(value=name) for name in DEFAULT_SLOT_ALGORITHMS]
        self.values: list[dict[str, tk.Variable]] = []
        for slot in range(MAX_SLOTS):
            variables: dict[str, tk.Variable] = {}
            for field in fields(HumanoidConfig):
                if field.name == "algorithm":
                    continue
                variables[field.name] = (tk.BooleanVar() if field.name in BOOLEAN_FIELDS
                                         else tk.StringVar())
            self.values.append(variables)
            self._fill_values(slot, self.initial_config(slot))
        self.slot_count_var = tk.StringVar(value=str(DEFAULT_SLOT_COUNT))
        self.smoothing = tk.StringVar(value=str(DEFAULT_SMOOTHING))
        self.fps = tk.StringVar(value=str(DEFAULT_ANIMATION_FPS))
        self.torch_threads = tk.StringVar(value=str(DEFAULT_TORCH_THREADS))
        #: Welches Verfahren der Einzelgraph zeigt.
        self.single_slot_var = tk.StringVar(value=SLOT_LABELS[0])
        self.status = tk.StringVar(value="Bereit")
        self.active_label = tk.StringVar(value="")
        self.progress = tk.DoubleVar(value=0)
        # Unter dem Bild steht ausschließlich diese eine Zeile.
        self.captions = [tk.StringVar(value="Noch keine Episode gestartet.")
                         for _ in range(MAX_SLOTS)]
        # Je Animation unabhängig wählbar: der aktuelle Lernstand oder der der
        # besten bisherigen Episode.
        self.episode_choice = [tk.StringVar(value=CURRENT_EPISODE) for _ in range(MAX_SLOTS)]

    def _fill_values(self, slot: int, config: HumanoidConfig) -> None:
        """Übernimmt eine Konfiguration in die Eingabefelder eines Slots."""
        for name, variable in self.values[slot].items():
            value = getattr(config, name)
            if name in BOOLEAN_FIELDS:
                variable.set(bool(value))
            elif isinstance(value, tuple):
                variable.set(",".join(map(str, value)))
            else:
                variable.set("" if value is None else str(value))

    # ------------------------------------------------------------------ Layout

    def _window(self) -> None:
        # Start- und Mindestgröße bleiben immer innerhalb des nutzbaren
        # Bildschirms: Eine Wunschbreite, die ihn überschreitet, würde das
        # Fenster unter Menüleiste oder Dock schieben, und eine ebenso große
        # Mindestbreite ließe es sich nicht mehr verkleinern.
        available_width = self.root.winfo_screenwidth() - 40
        available_height = self.root.winfo_screenheight() - SCREEN_MARGIN
        width = min(1640, CONTROL_WIDTH + WANTED_ENVIRONMENT_WIDTH, available_width)
        height = min(980, available_height)
        self.root.title("Humanoid Workbench")
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(CONTROL_WIDTH + MIN_ENVIRONMENT_WIDTH, width), min(660, height))
        self.root.configure(bg=self.BG)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=self.BG, foreground=self.FG)
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.FG)
        style.configure("TLabelframe", background=self.BG, foreground=self.FG, bordercolor="#475569")
        style.configure("TLabelframe.Label", background=self.BG, foreground="#fff",
                        font=("TkDefaultFont", 10, "bold"))
        style.configure("TButton", background="#334155", foreground=self.FG, padding=4)
        style.map("TButton", foreground=[("disabled", "#94a3b8")])
        style.configure("Compact.TButton", background="#334155", foreground=self.FG,
                        padding=(6, 0), font=("TkDefaultFont", 9))
        style.configure("TEntry", fieldbackground=self.FIELD, foreground=self.FG, insertcolor=self.FG)
        # Ohne eigene Konfiguration bleibt die Spinbox im hellen clam-Standard
        # stehen und fällt als einziges Bedienelement aus dem Farbschema.
        style.configure("TSpinbox", fieldbackground=self.FIELD, foreground=self.FG,
                        background="#334155", arrowcolor=self.FG, insertcolor=self.FG,
                        bordercolor="#475569", lightcolor=self.FIELD, darkcolor=self.FIELD)
        style.map("TSpinbox",
                  fieldbackground=[("readonly", self.FIELD), ("disabled", self.FIELD)],
                  foreground=[("disabled", "#94a3b8")],
                  arrowcolor=[("disabled", "#94a3b8")])
        style.configure("TCombobox", fieldbackground=self.FIELD, foreground=self.FG, background="#334155")
        style.map("TCombobox", fieldbackground=[("readonly", self.FIELD)],
                  foreground=[("readonly", self.FG)])
        style.configure("TCheckbutton", background=self.BG, foreground=self.FG)
        style.map("TCheckbutton", foreground=[("disabled", "#94a3b8")])
        style.configure("TProgressbar", troughcolor=self.FIELD, background=self.ACCENT)
        style.configure("TNotebook", background=self.BG, bordercolor="#475569")
        style.configure("TNotebook.Tab", background="#1f2937", foreground=self.MUTED, padding=(12, 4))
        style.map("TNotebook.Tab", background=[("selected", "#334155")],
                  foreground=[("selected", self.FG)])
        self.root.option_add("*TCombobox*Listbox.background", self.FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", self.FG)

    def _figure(self, master: ttk.Frame) -> tuple[Figure, Any, FigureCanvasTkAgg]:
        figure = Figure(figsize=(9, 3), dpi=100, facecolor=self.BG)
        # Kein Titel über den Achsen: Der Tab-Name sagt bereits, was zu sehen
        # ist, und der freie Platz gehört den Kurven.
        figure.subplots_adjust(left=.11, right=.74, bottom=.20, top=.96)
        axes = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=master)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return figure, axes, canvas

    def _layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="Humanoid Workbench",
                  font=("TkDefaultFont", 18, "bold")).pack(side="left")
        ttk.Label(header, text="PPO, SAC und TD3 – sechs Gelenke, ein Ziel: so schnell "
                               "wie möglich nach rechts").pack(side="left", padx=16)
        ttk.Button(header, text="Bedienungsanleitung", command=self.instructions).pack(side="right")
        self.splitter = ttk.Panedwindow(outer, orient="vertical")
        self.splitter.pack(fill="both", expand=True)
        upper, lower = ttk.Frame(self.splitter), ttk.Frame(self.splitter)
        self.splitter.add(upper, weight=1)
        self.splitter.add(lower, weight=1)
        upper.columnconfigure(0, minsize=CONTROL_WIDTH)
        upper.columnconfigure(1, weight=1)
        upper.rowconfigure(0, weight=1)
        self.controls = ttk.Frame(upper, width=CONTROL_WIDTH, padding=(0, 0, 8, 0))
        self.controls.grid(row=0, column=0, sticky="nsew")
        self.controls.grid_propagate(False)
        self._controls(self.controls)
        self._environment(upper)

        lower.columnconfigure(0, weight=3)
        lower.columnconfigure(1, weight=1, minsize=600)
        lower.rowconfigure(0, weight=1)
        chart = ttk.LabelFrame(lower, text="Diagramme", padding=5)
        chart.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.charts = ttk.Notebook(chart)
        self.charts.pack(fill="both", expand=True)
        training_tab, comparison_tab = ttk.Frame(self.charts), ttk.Frame(self.charts)
        single_tab = ttk.Frame(self.charts)
        self.charts.add(training_tab, text="Training")
        self.charts.add(comparison_tab, text="Vergleich")
        # Dritter Tab: genau ein Verfahren. Ein Vergleichsgraph beantwortet
        # „welches Verfahren ist besser", ein Einzelgraph „wie verlief dieses
        # eine Training". Berichte brauchen beides, und eine Parameterstudie
        # über drei Ausprägungen verlangt drei getrennte Plots.
        self.charts.add(single_tab, text="Einzelverfahren")
        self.charts.bind("<<NotebookTabChanged>>", lambda _event: self._chart_tab_changed())
        # Beide Exportschaltflächen sitzen nebeneinander rechts in der
        # Tableiste: Dort ist die Fläche ohnehin frei, sie kosten keine eigene
        # Zeile und verdecken weder Kurven noch Legende. Ein gemeinsamer Rahmen
        # verteilt die Breite, damit sie sich nicht überlappen. `PNG` wirkt auf
        # das gerade sichtbare Diagramm, `TXT` auf die Summary daneben.
        chart_bar = ttk.Frame(self.charts)
        chart_bar.place(relx=1.0, y=1, anchor="ne", x=-2)
        # Die Glättung gilt global für alle Slots und beide Graphen: Kurven mit
        # unterschiedlich breitem Fenster wären nicht vergleichbar.
        ttk.Label(chart_bar, text="Glättung").pack(side="left", padx=(0, 3))
        self.smoothing_box = ttk.Spinbox(
            chart_bar, textvariable=self.smoothing, from_=MIN_SMOOTHING, to=MAX_SMOOTHING,
            increment=1, width=4, justify="right", command=self._smoothing_changed)
        self.smoothing_box.pack(side="left", padx=(0, 8))
        self.smoothing_box.bind("<KeyRelease>", lambda _event: self._smoothing_changed())
        self.export_button = ttk.Button(chart_bar, text="PNG exportieren",
                                        command=self.export_chart, style="Compact.TButton")
        self.export_button.pack(side="left", padx=(0, 4))
        self.summary_button = ttk.Button(chart_bar, text="TXT exportieren",
                                         command=self.export_summary, style="Compact.TButton")
        self.summary_button.pack(side="left", padx=(0, 4))
        # Eine Aktion, je aktivem Slot eine eigene Datei – für Berichte, die
        # einen Plot **je Verfahren** verlangen.
        self.each_button = ttk.Button(chart_bar, text="Je Verfahren PNG",
                                      command=self.export_each_slot, style="Compact.TButton")
        self.each_button.pack(side="left")
        self.figure, self.axes, self.canvas = self._figure(training_tab)
        self.comparison_figure, self.comparison_axes, self.comparison_canvas = self._figure(comparison_tab)
        single_head = ttk.Frame(single_tab)
        single_head.pack(fill="x")
        ttk.Label(single_head, text="Verfahren").pack(side="left", padx=(2, 4))
        self.single_combo = ttk.Combobox(
            single_head, textvariable=self.single_slot_var, state="readonly",
            width=22, justify="left")
        self.single_combo.pack(side="left")
        self.single_combo.bind("<<ComboboxSelected>>",
                               lambda _event: self._refresh_single_plot())
        self.single_figure, self.single_axes, self.single_canvas = self._figure(single_tab)

        info = ttk.LabelFrame(lower, text="Summary", padding=5)
        info.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        table = ttk.Frame(info)
        table.pack(fill="both", expand=True)
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(
            table, wrap="none", height=8, background=self.FIELD, foreground=self.FG,
            insertbackground=self.FG, relief="flat", font="TkFixedFont",
        )
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        summary_y = ttk.Scrollbar(table, orient="vertical", command=self.summary_text.yview)
        # Bei drei oder vier Ergebnisspalten wird die Tabelle breit; sie
        # scrollt dann, statt Werte abzuschneiden.
        summary_x = ttk.Scrollbar(table, orient="horizontal", command=self.summary_text.xview)
        # Fette Variante derselben Schrift für die Zwischenüberschrift. Die
        # Referenz muss erhalten bleiben, sonst räumt Tk das Font-Objekt ab.
        self.summary_bold = tkfont.Font(font=self.summary_text.cget("font"))
        self.summary_bold.configure(weight="bold")
        self.summary_text.tag_configure("bold", font=self.summary_bold)
        self.summary_text.configure(yscrollcommand=summary_y.set, xscrollcommand=summary_x.set,
                                    state="disabled")
        summary_y.grid(row=0, column=1, sticky="ns")
        summary_x.grid(row=1, column=0, sticky="ew")

    def _controls(self, parent: ttk.Frame) -> None:
        """Kopfzeile aus Verfahrensauswahl, Steuerung und Status; darunter über
        die volle Breite die Verfahrenstabs.

        Die Steuerung steht neben der Auswahl statt in einer eigenen Spalte:
        Vier Schaltflächen füllen keine Spalte über die ganze Panelhöhe, und
        die eingesparte Breite kommt der Animation zugute.
        """
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        head = ttk.Frame(parent)
        head.grid(row=0, column=0, sticky="new")
        head.columnconfigure(0, weight=1)

        selection = ttk.LabelFrame(head, text="Verfahren und Rechenleistung", padding=5)
        selection.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        actions = ttk.LabelFrame(head, text="Steuerung", padding=6)
        actions.grid(row=0, column=1, sticky="nsew", padx=4)
        actions.grid_propagate(False)
        actions.configure(width=ACTION_WIDTH)
        status = ttk.LabelFrame(head, text="Lauf", padding=6)
        status.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        status.grid_propagate(False)
        status.configure(width=STATUS_WIDTH)

        self.parameter_tabs = ttk.Notebook(parent)
        self.parameter_tabs.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.tab_frames = [ttk.Frame(self.parameter_tabs, padding=(0, 4, 0, 0))
                           for _ in range(MAX_SLOTS)]
        self.parameter_tabs.bind("<<NotebookTabChanged>>", lambda _event: self._tab_changed())

        self._selection_area(selection)
        self._action_area(actions)
        self._status_area(status)

    def _selection_area(self, selection: ttk.LabelFrame) -> None:
        # Erst die beiden globalen Einstellungen, darunter die Belegung der
        # Slots: Die Zahl oben bestimmt, wie viele Zeilen darunter erscheinen.
        ttk.Label(selection, text="Anzahl Verfahren").grid(row=0, column=0, sticky="w", padx=(0, 4))
        # Spinbox statt Dropdown: Zwei bis vier Werte lohnen keine Aufklappliste,
        # und die Pfeile führen direkt zum Nachbarwert – wie bei `Glättung`.
        self.slot_count_combo = ttk.Spinbox(
            selection, textvariable=self.slot_count_var,
            from_=min(SLOT_COUNTS), to=max(SLOT_COUNTS), increment=1,
            width=VALUE_CHARS - 2, justify="right", state="readonly",
            command=self._slot_count_changed)
        self.slot_count_combo.grid(row=0, column=1, sticky="ew", pady=1)
        # Die Thread-Zahl steht bewusst hier und nicht bei der Animation: Sie
        # ist keine Anzeige-, sondern eine Rechenzeiteinstellung, sie gilt
        # prozessweit – ein Vergleich lässt seine Slots als Threads EINES
        # Prozesses laufen –, und ihr sinnvoller Wert hängt direkt daran, wie
        # viele Verfahren sich den Pool teilen. Beides gehört nebeneinander.
        # Überschrift über der Auswahlspalte: Ohne sie ist nicht erkennbar,
        # worauf sich `aktuell`/`beste`/`inaktiv` bezieht.
        ttk.Label(selection, text="Animation", foreground=self.MUTED, anchor="center").grid(
            row=SELECTION_ROW_OFFSET - 1, column=2, sticky="ew", padx=(6, 0))
        ttk.Label(selection, text="Threads (PyTorch)").grid(
            row=1, column=0, sticky="w", padx=(0, 4), pady=1)
        ttk.Entry(selection, textvariable=self.torch_threads, width=VALUE_CHARS,
                  justify="right").grid(row=1, column=1, sticky="ew", pady=1)
        # Der Hinweis füllt die verbleibenden Zeilen der linken Spalte: Der Block
        # behält damit unabhängig von der Slotzahl dieselbe Höhe und schiebt die
        # Verfahrenstabs nicht nach unten.
        # Kein Hinweistext hier: Er kostete eine volle Rasterzeile Höhe, die dem
        # Parameterbereich fehlt. Was Threads und Budget bedeuten, steht in der
        # Bedienungsanleitung und im README.
        self.algorithm_combos: list[ttk.Combobox] = []
        self.episode_combos: list[ttk.Combobox] = []
        self.algorithm_rows: list[tuple[ttk.Widget, ...]] = []
        for slot, label in enumerate(SLOT_LABELS):
            name = ttk.Label(selection, text=label, foreground=SLOT_COLORS[slot])
            # Schmal: Die Werte sind PPO, TD3 und SAC – drei Zeichen.
            combo = ttk.Combobox(selection, textvariable=self.algorithm_vars[slot],
                                 values=ALGORITHMS, state="readonly",
                                 width=VALUE_CHARS, justify="right")
            combo.bind("<<ComboboxSelected>>", lambda _event, index=slot: self._algorithm_changed(index))
            # Die Animationswahl steht in derselben Zeile wie das Verfahren, zu
            # dem sie gehört – nicht mehr über dem Bild. Unter dem Bild bleibt
            # damit nur noch die eine Zeile mit Episode, Schritt und Return,
            # und die Zelle gewinnt Höhe für das Bild.
            choice = ttk.Combobox(selection, textvariable=self.episode_choice[slot],
                                  values=EPISODE_CHOICES, state="readonly",
                                  width=VALUE_CHARS, justify="right")
            choice.bind("<<ComboboxSelected>>",
                        lambda _event, index=slot: self._episode_choice_changed(index))
            self.algorithm_combos.append(combo)
            self.episode_combos.append(choice)
            self.algorithm_rows.append((name, combo, choice))
        # Alle Slotzeilen gleich hoch: Sonst richtet sich jede Zeile nach dem
        # zufällig höchsten Widget darin und die Abstände springen.
        for row in range(SELECTION_ROW_OFFSET + MAX_SLOTS):
            selection.rowconfigure(row, minsize=24, uniform="slotrow")
        for column, width in SELECTION_COLUMNS:
            selection.columnconfigure(column, minsize=width, weight=0)
        selection.columnconfigure(SELECTION_SPACER, weight=1)

    def _action_area(self, actions: ttk.LabelFrame) -> None:
        """Nur die vier Schaltflächen, untereinander."""
        self.buttons = []
        # Vier Schaltflächen, mehr braucht der Ablauf nicht: trainieren,
        # vergleichen, anhalten, verwerfen. Die Animation läuft während der
        # Läufe mit und wird über `Animation zeigen` sowie die Auswahl je
        # Anzeige gesteuert.
        for text, command in (
            ("Training starten / fortsetzen", self.start_training),
            ("Vergleich starten / fortsetzen", self.start_comparison),
            ("Stoppen", self.stop),
            ("Zurücksetzen", self.reset),
        ):
            button = ttk.Button(actions, text=text, command=command)
            button.pack(fill="x", pady=1, ipady=1)
            self.buttons.append(button)
        self.stop_button = self.buttons[2]
        self.stop_button.configure(state="disabled")

    def _status_area(self, status: ttk.LabelFrame) -> None:
        """Was gerade läuft: aktives Verfahren, Animation, Fortschritt, Meldung."""
        wrap = STATUS_WIDTH - 24
        ttk.Label(status, textvariable=self.active_label, foreground=self.ACCENT,
                  font=("TkDefaultFont", 10, "bold"), wraplength=wrap).pack(fill="x")
        # Keine globale "Animation zeigen"-Schaltfläche mehr: Die Wahl je
        # Verfahren kennt `inaktiv`, und alle vier darauf zu stellen ist
        # dasselbe. Ein zweiter Schalter daneben könnte der ersten Einstellung
        # nur widersprechen.
        rate = ttk.Frame(status)
        rate.pack(fill="x", pady=(2, 0))
        ttk.Label(rate, text="Bildrate (FPS)").pack(side="left")
        ttk.Entry(rate, textvariable=self.fps, width=5, justify="right").pack(side="right")
        ttk.Progressbar(status, variable=self.progress, maximum=100).pack(fill="x", pady=(4, 2))
        ttk.Label(status, textvariable=self.status, foreground=self.ACCENT,
                  wraplength=wrap).pack(fill="x", pady=1)

    def _build_tab(self, slot: int) -> None:
        """Baut einen Verfahrenstab neu auf.

        Parameter, die das gewählte Verfahren nicht kennt, werden gar nicht
        erzeugt statt deaktiviert dargestellt.
        """
        frame = self.tab_frames[slot]
        for child in frame.winfo_children():
            child.destroy()
        algorithm = self.algorithm_vars[slot].get()
        columns = {}
        for index, key in enumerate(("left", "right")):
            column = ttk.Frame(frame)
            column.grid(row=0, column=index, sticky="nsew", padx=(0, 4) if index == 0 else (4, 0))
            frame.columnconfigure(index, minsize=CONTROL_COLUMNS[index] - 8, weight=1)
            columns[key] = column
        entries: dict[str, tk.Widget] = {}
        for column_key, title, names in PARAMETER_GROUPS[algorithm]:
            group = ttk.LabelFrame(columns[column_key], text=title, padding=5)
            group.pack(fill="x", pady=(0, 4))
            entries.update(self._group_fields(group, slot, names))
        self.entries[slot] = entries

    def _group_fields(self, group: ttk.LabelFrame, slot: int,
                      names: tuple[str, ...]) -> dict[str, tk.Widget]:
        """Zwei Label/Feld-Paare je Zeile, alle Wertefelder gleich breit.

        Sonderbreite Felder gibt es nicht mehr: Sie endeten an einer anderen
        Kante als alle übrigen und ließen die Spalte unruhig wirken. Die
        Auswahltexte sind stattdessen kurz gehalten (5.6).
        """
        entries: dict[str, tk.Widget] = {}
        row = position = 0
        for name in names:
            variable = self.values[slot][name]
            if name in BOOLEAN_FIELDS:
                # Checkboxen belegen wie die übrigen Felder einen halben Zeilen-
                # platz; eigene volle Zeilen machten die Spalten zu hoch.
                widget = ttk.Checkbutton(group, text=SHORT_LABELS[name], variable=variable)
                widget.grid(row=row, column=position * 2, columnspan=2, sticky="w", pady=1)
                position += 1
                if position == 2:
                    row, position = row + 1, 0
            else:
                column = position * 2
                ttk.Label(group, text=SHORT_LABELS[name]).grid(row=row, column=column, sticky="w", padx=(0, 4))
                if name in CHOICE_FIELDS:
                    widget = ttk.Combobox(group, textvariable=variable, values=CHOICE_FIELDS[name],
                                          state="readonly", width=VALUE_CHARS, justify="right")
                else:
                    widget = ttk.Entry(group, textvariable=variable, width=VALUE_CHARS,
                                       justify="right")
                # `sticky="ew"` statt `"e"`: Beide Widgettypen füllen dieselbe
                # Spalte und werden dadurch exakt gleich breit. Mit `width=`
                # allein bliebe die Combobox breiter – sie rechnet ihren
                # Aufklapp-Pfeil zusätzlich zum Textfeld.
                widget.grid(row=row, column=column + 1, sticky="ew", padx=(0, 6), pady=1)
                position += 1
                if position == 2:
                    row, position = row + 1, 0
            entries[name] = widget
        apply_field_grid(group)
        return entries

    # -------------------------------------------------------------- Animation

    def _environment(self, parent: ttk.Frame) -> None:
        """Je Slot ein Anzeigefeld im Raster mit höchstens zwei Spalten und
        zwei Zeilen. Sichtbar ist normalerweise nur das aktive Verfahren; im
        Vergleich stehen alle aktiven Verfahren gleichzeitig da."""
        self.env_container = ttk.Frame(parent)
        self.env_container.grid(row=0, column=1, sticky="nsew")
        self.env_frames, self.image_labels, self.caption_labels = [], [], []
        for slot, label in enumerate(SLOT_LABELS):
            frame = ttk.LabelFrame(self.env_container, text=label, padding=5)
            # Unter dem Bild steht ausschließlich Episode, Schritt und Return –
            # bei vier Anzeigen bliebe für mehr kein Platz. Die Zeile wird vor
            # dem Bild gepackt: Der Packer vergibt den Platz in der Reihenfolge
            # des Packens, sodass das expandierende Bild sie sonst aus einer
            # knappen Zelle herausdrückt.
            caption = ttk.Label(frame, textvariable=self.captions[slot], anchor="w",
                                font="TkFixedFont", foreground=SLOT_COLORS[slot])
            caption.pack(side="bottom", fill="x", pady=(3, 0))
            image = ttk.Label(frame, anchor="center")
            image.pack(side="top", fill="both", expand=True)
            image.bind("<Enter>", lambda _event, index=slot: self._hover_enter(index))
            image.bind("<Leave>", lambda _event, index=slot: self._hover_leave(index))
            self.env_frames.append(frame)
            self.image_labels.append(image)
            self.caption_labels.append(caption)
        # Die Messwert-Einblendung liegt als Overlay über dem gesamten Fenster.
        # `place()` verändert kein Raster: Ein- und Ausblenden verschiebt die
        # Bilder deshalb nicht und ändert ihre Größe nicht. Als Elternwidget
        # dient bewusst das Fenster und nicht der Anzeigecontainer – dort wäre
        # sie neben einem Bild abgeschnitten. Über den Nachbarbereich darf sie
        # reichen, über ihr eigenes Bild nie.
        self.hover_panel = tk.Frame(self.root, background=self.PANEL,
                                    highlightthickness=1, highlightbackground="#64748b")
        self.hover_title = tk.Label(self.hover_panel, background=self.PANEL, anchor="w",
                                    font=("TkDefaultFont", 9, "bold"))
        self.hover_title.pack(fill="x", padx=8, pady=(6, 0))
        self.hover_body = tk.Label(self.hover_panel, background=self.PANEL, foreground=self.FG,
                                   justify="left", anchor="w", font="TkFixedFont")
        self.hover_body.pack(fill="both", padx=8, pady=(2, 6))

    def _panel_slots(self) -> list[int]:
        """Anzeigen, die tatsächlich gezeigt werden.

        `inaktiv` blendet die Anzeige **aus**, statt sie stehen zu lassen: Die
        Wahl steht jetzt neben dem Verfahren und nicht mehr über dem Bild, also
        wirkt ein verschwindendes Feld nicht wie ein Umbau der Ansicht – und die
        verbleibenden Anzeigen bekommen den Platz.
        """
        live = [slot for slot in range(self.slot_count)
                if self.animation_live[slot] and self._slot_animated(slot)]
        if live:
            return live
        return [self.active_slot] if self._slot_animated(self.active_slot) else []

    def show_panels(self, slots: list[int]) -> None:
        """Blendet genau diese Anzeigen ein. Wird auch vom Layout-Test genutzt."""
        for slot in range(MAX_SLOTS):
            self.animation_live[slot] = slot in slots
        self._sync_panels()

    def _panel_columns(self, count: int) -> int:
        """Spaltenzahl für die aktuelle Fläche."""
        if count <= 1:
            return 1
        chrome = self.caption_labels[0].winfo_reqheight() + 34
        width = max(1, self.env_container.winfo_width())
        height = max(1, self.env_container.winfo_height())
        return best_columns(count, width, height, chrome, FRAME_WIDTH / FRAME_HEIGHT)

    def _sync_panels(self) -> None:
        """Ordnet die sichtbaren Anzeigefelder im Raster an."""
        slots = self._panel_slots()
        columns = self._panel_columns(len(slots))
        positions = panel_grid_positions(len(slots), columns)
        used_rows, used_columns = set(), set()
        for index, slot in enumerate(slots):
            row, column, span = positions[index]
            self.env_frames[slot].grid(row=row, column=column, columnspan=span, sticky="nsew",
                                       padx=2, pady=2)
            used_rows.add(row)
            used_columns.update(range(column, column + span))
        for slot in range(MAX_SLOTS):
            if slot not in slots:
                self.env_frames[slot].grid_remove()
        # Gleiches Gewicht für alle belegten Zeilen und Spalten: keine Zelle
        # darf die anderen verdrängen.
        # Ungenutzte Zeilen und Spalten verlassen die Uniform-Gruppe, sonst
        # behielten sie ihren Anteil und die einzelne Anzeige bekäme nur die
        # halbe Höhe statt des gesamten Bereichs.
        for row in range(MAX_SLOTS):
            used = row in used_rows
            self.env_container.rowconfigure(row, weight=1 if used else 0,
                                            uniform="cell" if used else "")
        for column in range(MAX_SLOTS):
            used = column in used_columns
            self.env_container.columnconfigure(column, weight=1 if used else 0,
                                               uniform="cell" if used else "")
        if self.hover_slot is not None and self.hover_slot not in slots:
            self._hide_hover()
        self._update_panel_titles()
        # Erst nach dem Neuaufbau des Rasters kennt der Container seine neuen
        # Zellengrößen; ohne das würden die Bilder noch auf die alte Aufteilung
        # skaliert und blieben eine Runde lang zu klein.
        self.env_container.update_idletasks()
        for slot in slots:
            if self.last_frames[slot] is not None:
                self._show_frame(slot, self.last_frames[slot])

    def _panel_configs(self) -> list[HumanoidConfig]:
        """Konfigurationen, die Legende und Anzeigetitel beschriften.

        Im Vergleich sind das die Vergleichsmodelle, sonst die Slots selbst –
        dieselbe Quelle wie im Vergleichsgraphen, damit Titel und Legende nie
        auseinanderlaufen.
        """
        return [runtime.comparison.config if runtime.comparison else runtime.workbench.config
                for runtime in self.slots[:self.slot_count]]

    def _update_panel_titles(self) -> None:
        """`Verfahren 3 – SAC (Lernrate α 0.0003)` – wie in der Legende."""
        configs = self._panel_configs()
        for slot in range(MAX_SLOTS):
            if slot < self.slot_count:
                title = self._series_label(slot, configs, prefix=SLOT_LABELS[slot])
            else:
                title = SLOT_LABELS[slot]
            self.env_frames[slot].configure(text=title)

    def _sync_slot_widgets(self) -> None:
        """Zeigt genau die Dropdowns und Tabs der aktiven Slots."""
        self._sync_single_combo()
        for slot, (label, combo, choice) in enumerate(self.algorithm_rows):
            if slot < self.slot_count:
                row = SELECTION_ROW_OFFSET + slot
                label.grid(row=row, column=0, sticky="w", padx=(0, 4), pady=1)
                combo.grid(row=row, column=1, sticky="ew", pady=1)
                choice.grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=1)
            else:
                label.grid_remove()
                combo.grid_remove()
                choice.grid_remove()
        existing = list(self.parameter_tabs.tabs())
        for slot in range(MAX_SLOTS):
            frame = self.tab_frames[slot]
            present = str(frame) in existing
            if slot < self.slot_count and not present:
                self._build_tab(slot)
                self.parameter_tabs.add(frame, text=SLOT_LABELS[slot])
            elif slot >= self.slot_count and present:
                self.parameter_tabs.forget(frame)
        self._sync_panels()

    def _slot_count_changed(self) -> None:
        """Verkleinern verwirft Slots erst nach Rückfrage; Vergrößern startet
        die neuen Slots mit den Standardwerten ihres Algorithmus."""
        if self.busy:
            self.slot_count_var.set(str(self.slot_count))
            return
        try:
            count = int(self.slot_count_var.get())
        except ValueError:
            self.slot_count_var.set(str(self.slot_count))
            return
        if count == self.slot_count:
            return
        if count < self.slot_count:
            dropped = [slot for slot in range(count, self.slot_count) if self.slots[slot].has_data()]
            if dropped and not messagebox.askyesno(
                "Anzahl Verfahren",
                "Der Lernzustand von "
                + " und ".join(SLOT_LABELS[slot] for slot in dropped)
                + " wird verworfen. Fortfahren?",
                parent=self.root,
            ):
                self.slot_count_var.set(str(self.slot_count))
                return
            for slot in range(count, self.slot_count):
                self._stop_animation(slot)
                self.animation_live[slot] = False
                self.slots[slot].reset_all(default_config(self.algorithm_vars[slot].get()))
                self._close_renderer(slot)
        else:
            for slot in range(self.slot_count, count):
                config = default_config(self.algorithm_vars[slot].get())
                self._fill_values(slot, config)
                self.slots[slot].reset_all(config)
        self.slot_count = count
        self._sync_slot_widgets()
        # Ein weggefallener aktiver Slot darf nicht aktiv bleiben.
        if self.parameter_tabs.index(self.parameter_tabs.select()) >= self.slot_count:
            self.parameter_tabs.select(self.tab_frames[0])
        self._update_active_label()
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._refresh_single_plot()
        self._training_summary()
        self.status.set(f"Bereit – {count} Verfahren aktiv")

    def _initialize_layout(self) -> None:
        self.root.update_idletasks()
        available = self.splitter.winfo_height()
        required = max((child.winfo_y() + child.winfo_reqheight()
                        for child in self.controls.winfo_children()), default=400) + 10
        self.splitter.sashpos(0, min(max(required, int(available * .50)), available - MIN_CHART_HEIGHT))
        for slot, frame in enumerate(self.last_frames):
            if frame is not None:
                self._show_frame(slot, frame)

    def layout_visibility_issues(self) -> list[str]:
        self.root.update_idletasks()
        issues = []
        left, top = self.controls.winfo_rootx(), self.controls.winfo_rooty()
        right, bottom = left + self.controls.winfo_width(), top + self.controls.winfo_height()
        selected = self.parameter_tabs.index(self.parameter_tabs.select())
        hidden = {str(frame) for index, frame in enumerate(self.tab_frames) if index != selected}
        # Dropdowns nicht aktiver Slots werden bewusst entfernt, nicht deaktiviert.
        hidden.update(str(widget) for slot in range(self.slot_count, MAX_SLOTS)
                      for widget in self.algorithm_rows[slot])
        stack = list(self.controls.winfo_children())
        while stack:
            widget = stack.pop()
            # Nicht gewählte Verfahrenstabs sind von Tk bewusst nicht gemappt.
            if any(str(widget).startswith(name) for name in hidden):
                continue
            stack.extend(widget.winfo_children())
            if isinstance(widget, (ttk.Entry, ttk.Combobox, ttk.Button, ttk.Checkbutton,
                                   ttk.Progressbar)):
                if not widget.winfo_ismapped() or widget.winfo_rootx() < left \
                        or widget.winfo_rooty() < top \
                        or widget.winfo_rootx() + widget.winfo_width() > right \
                        or widget.winfo_rooty() + widget.winfo_height() > bottom:
                    issues.append(
                        f"{widget}: ({widget.winfo_rootx()},{widget.winfo_rooty()},"
                        f"{widget.winfo_width()}x{widget.winfo_height()}) außerhalb "
                        f"({left},{top},{right - left}x{bottom - top})"
                    )
        # Eine Parametergruppe, die breiter ist als ihre Rasterspalte, würde
        # ihre Felder unbemerkt in die Nachbarspalte drücken.
        frame = self.tab_frames[selected]
        for index, column in enumerate(frame.grid_slaves(row=0)[::-1]):
            allowed = CONTROL_COLUMNS[index] - 8
            for group in column.winfo_children():
                if group.winfo_reqwidth() > allowed:
                    issues.append(
                        f"Spalte {index}: Gruppe benötigt {group.winfo_reqwidth()} px, "
                        f"verfügbar sind {allowed} px"
                    )
        # Geprüft wird das gerade sichtbare Diagramm: das jeweils andere Tab
        # ist von Tk bewusst nicht gemappt und wäre kein echter Befund.
        visible_canvas = (self.comparison_canvas if self.charts.index(self.charts.select()) == 1
                          else self.canvas).get_tk_widget()
        panels: list[tuple[str, tk.Widget]] = []
        for slot in self._panel_slots():
            panels.append((f"Visualisierung {SLOT_SHORT[slot]}", self.image_labels[slot]))
            panels.append((f"Beschriftung {SLOT_SHORT[slot]}", self.caption_labels[slot]))
            panels.append((f"Episodenwahl {SLOT_SHORT[slot]}", self.episode_combos[slot]))
        for label, widget in (*panels, ("Diagramm", visible_canvas),
                              ("Diagramm-Export", self.export_button),
                              ("Summary", self.summary_text),
                              ("Summary-Export", self.summary_button),
                              ("Glättung", self.smoothing_box)):
            if not widget.winfo_ismapped() or widget.winfo_width() <= 1 \
                    or widget.winfo_height() <= 1:
                issues.append(label)
        # Die Beschriftung muss vollständig in ihrer Zelle liegen: Wird sie vom
        # Bild herausgedrückt, ist sie zwar gemappt, aber abgeschnitten.
        for slot in self._panel_slots():
            caption, frame = self.caption_labels[slot], self.env_frames[slot]
            bottom = frame.winfo_rooty() + frame.winfo_height()
            if caption.winfo_rooty() + caption.winfo_reqheight() > bottom:
                issues.append(f"Beschriftung {SLOT_SHORT[slot]} ragt aus ihrer Zelle")
        return issues

    # ------------------------------------------------------------ Slot-Logik

    @property
    def active_slot(self) -> int:
        """Slot, auf den die Einzellauf-Buttons wirken.

        Während eines Laufs bleibt das Ziel fixiert; ein Tabwechsel zeigt dann
        nur andere Parameter an.
        """
        if self.busy and self.running_slot is not None:
            return self.running_slot
        try:
            return min(self.parameter_tabs.index(self.parameter_tabs.select()), self.slot_count - 1)
        except tk.TclError:
            return 0

    def _update_active_label(self) -> None:
        slot = self.active_slot
        self.active_label.set(f"Aktiv: {SLOT_LABELS[slot]} – {self.algorithm_vars[slot].get()}")

    def _tab_changed(self) -> None:
        self._update_active_label()
        self._sync_panels()
        if not self.busy:
            self._refresh_training_plot()
            self._training_summary()
            if self.last_frames[self.active_slot] is None:
                self._show_initial_frame(self.active_slot)

    def _algorithm_changed(self, slot: int) -> None:
        """Dropdown-Wechsel: Profilwerte laden und nur diesen Slot zurücksetzen."""
        algorithm = self.algorithm_vars[slot].get()
        runtime = self.slots[slot]
        if algorithm == runtime.algorithm:
            return
        if runtime.workbench.model is not None and not messagebox.askyesno(
            "Verfahren wechseln",
            f"{SLOT_LABELS[slot]} auf {algorithm} umstellen? Der Lernzustand dieses "
            "Slots wird verworfen, die übrigen Slots bleiben unverändert.",
            parent=self.root,
        ):
            self.algorithm_vars[slot].set(runtime.algorithm)
            return
        config = default_config(algorithm)
        self._fill_values(slot, config)
        self._build_tab(slot)
        runtime.reset_all(config)
        self._update_active_label()
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._refresh_single_plot()
        self._training_summary()
        self.status.set(f"Bereit – {SLOT_LABELS[slot]} auf {algorithm} gesetzt (Zoo-Profil geladen)")

    def _config(self, slot: int) -> HumanoidConfig:
        data: dict[str, Any] = {}
        for name, variable in self.values[slot].items():
            if name in BOOLEAN_FIELDS:
                data[name] = bool(variable.get())
                continue
            text = str(variable.get()).strip()
            label = SHORT_LABELS.get(name, name)
            try:
                if name in TUPLE_FIELDS:
                    data[name] = tuple(int(item.strip()) for item in text.split(",") if item.strip())
                elif name in OPTIONAL_FLOAT_FIELDS:
                    data[name] = float(text) if text else None
                elif name == "seed":
                    data[name] = int(text) if text else None
                elif name in INTEGER_FIELDS:
                    data[name] = int(text)
                elif name in CHOICE_FIELDS or name == "target_entropy":
                    data[name] = text
                else:
                    data[name] = float(text)
            except ValueError as error:
                expected = ("ganze Zahl" if name in INTEGER_FIELDS or name == "seed" else
                            "Liste ganzer Zahlen, z. B. '256,256'" if name in TUPLE_FIELDS else
                            "Zahl oder leer" if name in OPTIONAL_FLOAT_FIELDS else "Zahl")
                raise ValueError(
                    f"{SLOT_LABELS[slot]} – {label}: '{text}' ist keine gültige Eingabe. "
                    f"Erwartet: {expected}."
                ) from error
        config = HumanoidConfig(algorithm=self.algorithm_vars[slot].get(), **data)
        try:
            config.validate()
        except ValueError as error:
            raise ValueError(f"{SLOT_LABELS[slot]}: {error}") from error
        return config

    def _smoothing_changed(self) -> None:
        """Wirkt sofort auf beide Graphen, auch mitten in einem Lauf.

        Geändert wird ausschließlich die Darstellung: Die Messdaten bleiben
        vollständig erhalten, ein laufender Lauf wird nicht berührt. Eine
        ungültige Eingabe lässt die zuletzt gültige Breite stehen und wird in
        der Statuszeile erklärt – ein Dialog erschiene hier bei jedem
        Tastendruck.
        """
        text = self.smoothing.get().strip()
        try:
            window = int(text)
        except ValueError:
            self.status.set(f"Glättung: '{text}' ist keine ganze Zahl. "
                            f"Gültig: {MIN_SMOOTHING} bis {MAX_SMOOTHING} Episoden.")
            return
        if not MIN_SMOOTHING <= window <= MAX_SMOOTHING:
            self.status.set(f"Glättung: '{window}' liegt außerhalb. "
                            f"Gültig: {MIN_SMOOTHING} bis {MAX_SMOOTHING} Episoden.")
            return
        if window == self.smoothing_window:
            return
        self.smoothing_window = window
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._refresh_single_plot()

    def _apply_threads(self) -> int:
        """Thread-Zahl von PyTorch setzen; ungültige Eingabe fällt auf den
        Standard zurück und wird in der Statuszeile erklärt."""
        text = self.torch_threads.get().strip()
        try:
            count = int(text)
            if count < 1:
                raise ValueError
        except ValueError:
            count = DEFAULT_TORCH_THREADS
            self.torch_threads.set(str(count))
            self.status.set(f"Threads: '{text}' ist ungültig – {count} verwendet.")
        return set_torch_threads(count)

    def _set_busy(self, busy: bool, text: str) -> None:
        self.busy = busy
        self.status.set(text)
        for button in self.buttons:
            button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        for combo in self.algorithm_combos:
            combo.configure(state="disabled" if busy else "readonly")
        self.slot_count_combo.configure(state="disabled" if busy else "readonly")
        if not busy:
            self._stop_live_animation()
            self.running_slot = None
            self.comparison_running = False
        self._update_active_label()

    def _begin_progress(self, series_starts: dict, episode_budget: int,
                        step_budget: int) -> None:
        """Fortschritt vorbereiten.

        Gezählt wird in **Episoden** – das ist die Größe, die der Benutzer
        vorgibt und im Graphen wiederfindet. Nur wenn die Episodengrenze auf
        „unbegrenzt" steht, gibt es keinen Nenner; dann zählen ersatzweise die
        Schritte.
        """
        self.progress_mode = "episodes" if episode_budget else "steps"
        self.progress_start = dict(series_starts)
        self.progress_steps = {series: 0 for series in series_starts}
        self.progress_total = episode_budget or step_budget
        self.progress.set(0)

    @staticmethod
    def _budget_text(config: HumanoidConfig) -> str:
        """Beide Grenzen benennen – es endet die zuerst erreichte."""
        steps = f"{thousands(config.total_timesteps)} Schritte"
        if not config.episodes:
            return steps
        return f"{thousands(config.episodes)} Episoden oder {steps}"

    def _history_for(self, series) -> list:
        mode, slot = series
        return (self.slots[slot].history if mode == "single"
                else self.slots[slot].comparison_history)

    def _update_progress(self) -> None:
        if self.progress_mode == "episodes":
            done = sum(max(0, len(self._history_for(series)) - start)
                       for series, start in self.progress_start.items())
        else:
            done = sum(self.progress_steps.values())
        self.progress.set(min(100, 100 * done / max(1, self.progress_total)))

    # ----------------------------------------------------------------- Training

    def start_training(self) -> None:
        self._apply_threads()
        if self.busy:
            return
        slot = self.active_slot
        runtime = self.slots[slot]
        try:
            config = self._config(slot)
            self._animation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root)
            return
        if runtime.workbench.model is not None \
                and config.signature() != runtime.workbench.config.signature():
            if not messagebox.askyesno(
                "Neues Modell",
                f"Geänderte Modellparameter erfordern einen Reset von {SLOT_LABELS[slot]}. Fortfahren?",
                parent=self.root,
            ):
                return
            runtime.workbench.close()
            runtime.workbench = HumanoidWorkbench(config)
            runtime.history.clear()
        else:
            runtime.workbench.config = config
        self.running_slot = slot
        self.stop_event.clear()
        # Ein zweiter Druck setzt nichts zurück: Beide Grenzen gelten relativ
        # zum bereits Gelaufenen, der Fortschritt zählt ab hier neu.
        self._begin_progress({("single", slot): len(runtime.history)},
                             config.episodes, config.total_timesteps)
        self.charts.select(0)
        self._set_busy(True, f"Läuft – {SLOT_LABELS[slot]} ({config.algorithm}) über "
                             f"{self._budget_text(config)}")
        self.worker = threading.Thread(target=self._train_worker, args=(slot,), daemon=True)
        self.worker.start()
        self.root.after(50, self._poll)
        self._start_live_animation([slot])

    def _train_worker(self, slot: int) -> None:
        try:
            self.slots[slot].workbench.train(self.stop_event, self.events, ("single", slot))
            self.events.put(("training_done", self.stop_event.is_set()))
        except Exception as error:
            self.events.put(("error", error))

    # --------------------------------------------------------------- Vergleich

    def start_comparison(self) -> None:
        self._apply_threads()
        if self.busy:
            return
        try:
            configs = [self._config(slot) for slot in range(self.slot_count)]
            self._animation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültiger Vergleich", str(error), parent=self.root)
            return
        budgets = {config.total_timesteps for config in configs}
        if len(budgets) > 1 and not messagebox.askyesno(
            "Unterschiedliche Budgets",
            "Die Slots trainieren unterschiedlich lang: "
            + ", ".join(f"{SLOT_SHORT[slot]} {thousands(config.total_timesteps)}"
                        for slot, config in enumerate(configs))
            + " Schritte. Der Vergleich ist dann nicht budgetgleich. Trotzdem starten?",
            parent=self.root,
        ):
            return
        for slot, config in enumerate(configs):
            runtime = self.slots[slot]
            if runtime.comparison is not None \
                    and runtime.comparison.config.signature() != config.signature():
                runtime.reset_comparison()
            if runtime.comparison is None:
                runtime.comparison = HumanoidWorkbench(config)
            else:
                runtime.comparison.config = config
        self.stop_event.clear()
        self.comparison_running = True
        self._begin_progress(
            {("compare", slot): len(self.slots[slot].comparison_history)
             for slot in range(self.slot_count)},
            sum(config.episodes for config in configs) if all(c.episodes for c in configs) else 0,
            sum(config.total_timesteps for config in configs))
        self.charts.select(1)
        self._set_busy(True, "Läuft – Vergleich: "
                             + " gegen ".join(f"{SLOT_SHORT[slot]} {config.algorithm}"
                                              for slot, config in enumerate(configs))
                             + f" über je {self._budget_text(configs[0])}")
        self._refresh_comparison_plot()
        self._refresh_single_plot()
        self._comparison_summary()
        self.worker = threading.Thread(target=self._comparison_worker, daemon=True)
        self.worker.start()
        self.root.after(50, self._poll)
        self._start_live_animation(list(range(self.slot_count)))

    def _comparison_worker(self) -> None:
        errors: queue.Queue = queue.Queue()
        slots = list(range(self.slot_count))
        barrier = threading.Barrier(len(slots))

        def run(slot: int) -> None:
            try:
                barrier.wait()
                self.slots[slot].comparison.train(
                    self.stop_event, self.events, ("compare", slot))
            except Exception as error:
                errors.put(error)
                self.stop_event.set()

        threads = [threading.Thread(target=run, args=(slot,), daemon=True) for slot in slots]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if not errors.empty():
            self.events.put(("error", errors.get()))
        else:
            self.events.put(("comparison_done", self.stop_event.is_set()))

    def stop(self) -> None:
        if self.busy:
            self.stop_event.set()
            self.status.set("Stoppen angefordert …")

    # -------------------------------------------------------------------- Poll

    def _poll(self) -> None:
        redraw_training = redraw_comparison = False
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "episode":
                (mode, slot), metric = payload
                if mode == "single":
                    self.slots[slot].history.append(metric)
                    redraw_training = True
                else:
                    self.slots[slot].comparison_history.append(metric)
                    redraw_comparison = True
            elif kind == "progress":
                series, elapsed = payload
                self.progress_steps[series] = max(elapsed, self.progress_steps.get(series, 0))
            elif kind == "training_done":
                self._set_busy(False, "Gestoppt – Training" if payload else "Abgeschlossen – Training")
                redraw_training = True
            elif kind == "comparison_done":
                self._set_busy(False, "Gestoppt – Vergleich" if payload else "Abgeschlossen – Vergleich")
                redraw_comparison = True
            elif kind == "error":
                self._set_busy(False, "Fehler")
                messagebox.showerror("Fehler", str(payload), parent=self.root)
        self._update_progress()
        throttled = time.monotonic() - self.last_plot >= self.PLOT_INTERVAL
        if redraw_training:
            self._training_summary()
            if throttled or not self.busy:
                self._refresh_training_plot()
                self.last_plot = time.monotonic()
        if redraw_comparison:
            self._comparison_summary()
            if throttled or not self.busy:
                self._refresh_comparison_plot()
                self._refresh_single_plot()
                self.last_plot = time.monotonic()
        if self.busy:
            self.root.after(50, self._poll)

    # ------------------------------------------------------------------ Plots

    def _style(self, axes: Any) -> None:
        axes.set_facecolor(self.FIELD)
        axes.tick_params(colors=self.MUTED)
        axes.grid(color="#475569", alpha=.4)
        for spine in axes.spines.values():
            spine.set_color("#64748b")
        axes.title.set_color(self.FG)
        axes.xaxis.label.set_color(self.MUTED)
        axes.yaxis.label.set_color(self.MUTED)
        legend = axes.legend(loc="center left", bbox_to_anchor=(1.02, .5),
                             borderaxespad=0, fontsize=8)
        legend.get_frame().set_facecolor(self.PANEL)
        legend.get_frame().set_edgecolor("#64748b")
        for text in legend.get_texts():
            text.set_color(self.FG)
        self._reserve_legend_space(axes, legend)

    @staticmethod
    def _reserve_legend_space(axes: Any, legend: Any) -> None:
        """Passt den Achsenbereich an die tatsächliche Legendenbreite an.

        Lange Labels wie `V3 – SAC (Lernrate α 0.0003)` sind breiter als eine
        fest gewählte Reserve und würden sonst am rechten Figure-Rand
        abgeschnitten.
        """
        figure = axes.get_figure()
        try:
            renderer = figure.canvas.get_renderer()
        except AttributeError:
            return
        width = legend.get_window_extent(renderer).width
        figure_width = figure.get_size_inches()[0] * figure.dpi
        if figure_width <= 0:
            return
        target = 1.0 - width / figure_width - .03
        # Untergrenze, damit die Kurven auch bei sehr langen Labels Platz behalten.
        target = max(.42, min(.80, target))
        if abs(figure.subplotpars.right - target) > .01:
            figure.subplots_adjust(right=target)

    @staticmethod
    def _limit_to_data(axes: Any, values: list[float]) -> None:
        """Y-Achse an den **Daten** ausrichten, nicht an der Zielmarke.

        Die Zielmarke von 5000 liegt bei einem Budget unterhalb des Profilwerts
        weit über allem Erreichten – untrainiert endet eine Episode bei rund
        120. Würde die Achse sie erzwingen, drückte eine einzelne gestrichelte
        Linie alle Kurven in einen Bruchteil der Bildhöhe und machte genau die
        Unterschiede unlesbar, um die es geht. Die Legende nennt den Wert
        stattdessen ausdrücklich.
        """
        if not values:
            return
        low, high = min(values), max(values)
        padding = max(1.0, .08 * (high - low))
        axes.set_ylim(low - padding, high + padding)

    def _reference_line(self, axes: Any) -> None:
        # Ausdrücklich „Zielmarke", nicht „gelöst": `Humanoid-v5` führt keinen
        # offiziellen `reward_threshold`; die Marke ist projektintern gesetzt.
        axes.axhline(TARGET_RETURN, color=REFERENCE_COLOR, linestyle=REFERENCE_STYLE,
                     linewidth=self.LINE_WIDTH, label="Zielmarke")

    def _series_label(self, slot: int, configs: Optional[list[HumanoidConfig]] = None,
                      prefix: Optional[str] = None) -> str:
        """`V1 – PPO`; teilen sich mehrere Slots einen Algorithmus, zusätzlich
        der wichtigste abweichende Parameter.

        Mit `prefix` liefert dieselbe Beschriftung den Titel einer Anzeige,
        etwa `Verfahren 3 – SAC (Lernrate α 0.0003)`.
        """
        algorithm = (configs[slot].algorithm if configs else self.slots[slot].algorithm)
        label = f"{prefix or SLOT_SHORT[slot]} – {algorithm}"
        if configs and sum(1 for config in configs if config.algorithm == algorithm) > 1:
            differences = config_differences(configs)
            if differences:
                name, values = differences[0]
                label += f" ({name} {values[slot]})"
        return label

    def _plot_curve(self, axes: Any, slot: int, history: list[EpisodeMetric],
                    label: str) -> list[float]:
        """Rohkurve und gleitenden Durchschnitt eines Slots zeichnen.

        **Beide** Kurven sind auf `MAX_PLOT_POINTS` gedeckelt. Ohne die
        Deckelung der geglätteten Kurve wüchse die Zeichenzeit linear mit der
        Episodenzahl – bei vier Slots und zehntausenden Episoden wäre die
        Oberfläche nicht mehr bedienbar. Gemessen wird weiter vollständig.
        """
        if not history:
            return []
        episodes = [item.episode for item in history]
        rewards = [item.reward for item in history]
        raw_x, raw_y = downsample_minmax(episodes, rewards, MAX_PLOT_POINTS)
        axes.plot(raw_x, raw_y, color=SLOT_COLORS[slot], alpha=self.RAW_ALPHA, linewidth=.8)
        mean_x, mean_y = decimate(episodes, rolling_average(rewards, self.smoothing_window),
                                  MAX_PLOT_POINTS)
        axes.plot(mean_x, mean_y, color=SLOT_COLORS[slot], linestyle=SLOT_LINESTYLE,
                  linewidth=self.LINE_WIDTH, label=label)
        return rewards

    def _refresh_training_plot(self) -> None:
        self.axes.clear()
        slot = self.active_slot
        runtime = self.slots[slot]
        scale = self._plot_curve(
            self.axes, slot, runtime.history,
            f"Training – {SLOT_LABELS[slot]} ({runtime.algorithm})")
        # Die deterministischen Zwischenevaluationen stehen ausschließlich in
        # der Summary: Im Graphen liefen sie auf einer anderen Stützstellenzahl
        # und lenkten von der eigentlichen Lernkurve ab.
        self._reference_line(self.axes)
        self._limit_to_data(self.axes, scale)
        self.axes.set(xlabel="Episode", ylabel="Return")
        self._style(self.axes)
        self.canvas.draw_idle()

    def _refresh_comparison_plot(self) -> None:
        self.comparison_axes.clear()
        configs = [runtime.comparison.config if runtime.comparison else runtime.workbench.config
                   for runtime in self.slots[:self.slot_count]]
        scale: list[float] = []
        for slot in range(self.slot_count):
            # Rohwerte dezent in derselben Slotfarbe, damit sie die kräftige
            # Kurve des gleitenden Durchschnitts nicht überdecken.
            scale.extend(self._plot_curve(
                self.comparison_axes, slot, self.slots[slot].comparison_history,
                self._series_label(slot, configs)))
        self._reference_line(self.comparison_axes)
        self._limit_to_data(self.comparison_axes, scale)
        self.comparison_axes.set(xlabel="Episode", ylabel="Return")
        self._style(self.comparison_axes)
        self.comparison_canvas.draw_idle()
        # Anzeigetitel und Legende stammen aus derselben Quelle.
        self._update_panel_titles()

    # ---------------------------------------------------------------- Summary

    def _write_summary(self, text: str, bold_lines: tuple[str, ...] = ()) -> None:
        """Summary schreiben; genannte Zeilen werden fett ausgezeichnet.

        Der Zwischentitel trennt die Kennzahlen des ganzen Laufs von denen des
        Glättungsfensters. Fett gesetzt ist er als Einziges, damit die Trennung
        beim Überfliegen sofort sichtbar ist.
        """
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        for needle in bold_lines:
            if not needle:
                continue
            index = self.summary_text.search(needle, "1.0", "end")
            if index:
                self.summary_text.tag_add("bold", index, f"{index} lineend")
        self.summary_text.configure(state="disabled")

    def _summary_text(self) -> str:
        return self.summary_text.get("1.0", "end").rstrip("\n")

    #: Markierung für eine Zwischenüberschrift in der Tabelle. Sie belegt
    #: keine Wertespalte und geht deshalb nicht in die Spaltenbreite ein.
    SECTION = object()

    def _table(self, title: str, rows: tuple[Any, ...],
               statistics: dict[str, tuple[Any, ...]], footer: list[str],
               section_title: str = "") -> None:
        heading = [title, ""] if title else []
        if not statistics:
            lines = [*heading, "Noch keine vollständig abgeschlossene Episode.", *footer]
            self._write_summary("\n".join(lines))
            return
        names = list(statistics)
        labels = [row for row in rows if row is not self.SECTION]
        label_width = max(len(label) for label in labels) + 1
        # Die Breite richtet sich nach Spaltennamen **und** Werten: Ein langer
        # Wert würde die Tabelle sonst zerreißen statt sie nur zu verbreitern.
        widest_value = max((len(str(value)) for row in statistics.values() for value in row),
                           default=0)
        value_width = max(16, max(len(name) for name in names) + 2, widest_value + 2)
        header = "Statistik".ljust(label_width) + "".join(name.rjust(value_width) for name in names)
        lines = [*heading, header, "─" * len(header)]
        index = 0
        for row in rows:
            if row is self.SECTION:
                # Keine Leerzeile davor: Die fette Schrift trennt bereits, und
                # jede Zeile zählt – die Summary soll ohne Scrollen passen.
                lines.append(section_title)
                continue
            lines.append(row.ljust(label_width) + "".join(
                str(statistics[name][index]).rjust(value_width) for name in names))
            index += 1
        if footer:
            lines.extend(["", *footer])
        self._write_summary("\n".join(lines), (section_title,))

    @staticmethod
    def best_episode(history: list[EpisodeMetric]) -> Optional[EpisodeMetric]:
        """Episode mit dem höchsten explorativen Return – über **alle**
        Episoden, nicht nur über das Glättungsfenster: „beste der letzten 20"
        wäre keine sinnvolle Kennzahl."""
        return max(history, key=lambda item: item.reward) if history else None

    def _window_size(self, history: list[EpisodeMetric]) -> int:
        """Wie viele Episoden die Mittelwerte umfassen.

        Es ist dieselbe Zahl wie die Glättung des Graphen: Was die Kurve zeigt
        und was die Summary mittelt, soll dieselbe Aussage sein. Liegen weniger
        Episoden vor, wird über alle vorhandenen gemittelt.
        """
        return min(self.smoothing_window, len(history)) if history else 0

    def section_title(self, history: list[EpisodeMetric]) -> str:
        count = self._window_size(history)
        if not count:
            return "Ø der letzten Episoden"
        return f"Ø der letzten {count} Episode{'n' if count != 1 else ''}"

    #: Kopfzeilen: Umfang und Ausgang des Laufs.
    STAT_HEAD = ("Episoden", "Schritte", "Budget N/E", "Ende durch", "Beste Episode")
    #: Alles unterhalb der Zwischenüberschrift mittelt über das Glättungsfenster.
    #: Die drei Quoten stehen in der Reihenfolge Sturz, Durchhalten, Ziel.
    #: Ohne `Ø`: Die Zwischenüberschrift sagt bereits für alle darunter, dass
    #: gemittelt wird. Das Zeichen an jeder Zeile zu wiederholen kostet nur
    #: Breite in einer ohnehin schmalen Spalte.
    STAT_WINDOW = ("Return", "Länge", "Tempo m/s", "Strecke x m", "seitlich m",
                   "Sturzquote", "Durchhaltequote", "Zielquote")
    STAT_LABELS = (*STAT_HEAD, SECTION, *STAT_WINDOW)

    def _statistics(self, history: list[EpisodeMetric], steps: int, budget: int,
                    episode_budget: int = 0, stop_reason: Optional[str] = None) -> tuple[Any, ...]:
        """Kennzahlen eines Slots.

        Die Mittelwerte umfassen nur die **letzten** Episoden (Fenster wie die
        Glättung). Über den gesamten Lauf gemittelt hinge jede Kennzahl noch am
        untrainierten Anfang und bewegte sich kaum – gerade das, was man sehen
        will, verschwände im Mittel.
        """
        budgets = f"{thousands(budget)} / {thousands(episode_budget) if episode_budget else '∞'}"
        ending = STOP_REASONS.get(stop_reason or "", "läuft noch" if steps else "—")
        if not history:
            return (0, thousands(steps), budgets, ending, "—", *(["—"] * len(self.STAT_WINDOW)))
        best = self.best_episode(history)
        recent = history[-self._window_size(history):]
        return (
            thousands(len(history)), thousands(steps), budgets, ending,
            f"#{best.episode}: {german(best.reward)}",
            german(float(np.mean([item.reward for item in recent]))),
            german(float(np.mean([item.length for item in recent])), 0),
            german(float(np.mean([item.mean_speed for item in recent])), 2),
            german(float(np.mean([item.distance_x for item in recent])), 1),
            german(float(np.mean([item.lateral for item in recent])), 2),
            f"{np.mean([item.fell for item in recent]):.1%}".replace(".", ","),
            f"{np.mean([item.survived for item in recent]):.1%}".replace(".", ","),
            f"{np.mean([item.reached_target for item in recent]):.1%}".replace(".", ","),
        )

    #: Zwei kurze Zeilen. Alles Weitere steht in der Bedienungsanleitung – eine
    #: lange Fußzeile zwänge zum Scrollen und verdeckte die Tabelle.
    FOOTER = [f"Zielmarke {german(TARGET_RETURN, 0)} = {MAX_EPISODE_STEPS} Schritte aufrecht, "
              f"Durchhalten = {MAX_EPISODE_STEPS} Schritte erreicht"]

    def _training_summary(self) -> None:
        """Ohne eigene Titelzeile: Die Spaltenüberschrift nennt den Slot, die
        Zeile `Algorithmus` das Verfahren – und der aktive Tab sagt es ohnehin.
        Zwei gesparte Zeilen entscheiden darüber, ob die Tabelle ohne Scrollen
        ins Feld passt."""
        slot = self.active_slot
        runtime = self.slots[slot]
        model = runtime.workbench.model
        config = runtime.workbench.config
        stats = self._statistics(runtime.history, model.num_timesteps if model else 0,
                                 config.total_timesteps, config.episodes,
                                 runtime.workbench.stop_reason)
        self._table("", ("Algorithmus", *self.STAT_LABELS),
                    {SLOT_LABELS[slot]: (runtime.algorithm, *stats)}, list(self.FOOTER),
                    self.section_title(runtime.history))

    def _duplicated_algorithms(self, configs: list[HumanoidConfig]) -> list[str]:
        """Algorithmen, die mehr als einen aktiven Slot belegen."""
        names = [config.algorithm for config in configs]
        return sorted({name for name in names if names.count(name) > 1})

    def _comparison_summary(self) -> None:
        statistics: dict[str, tuple[Any, ...]] = {}
        configs = []
        longest: list[EpisodeMetric] = []
        for slot in range(self.slot_count):
            runtime = self.slots[slot]
            config = runtime.comparison.config if runtime.comparison else runtime.workbench.config
            configs.append(config)
            model = runtime.comparison.model if runtime.comparison else None
            if len(runtime.comparison_history) > len(longest):
                longest = runtime.comparison_history
            statistics[SLOT_LABELS[slot]] = (config.algorithm, *self._statistics(
                runtime.comparison_history, model.num_timesteps if model else 0,
                config.total_timesteps, config.episodes,
                runtime.comparison.stop_reason if runtime.comparison else None))
        if not any(self.slots[slot].comparison_history for slot in range(self.slot_count)):
            statistics = {}
        # Unterschiedliche Parameter als eigene Tabellenzeilen direkt unter dem
        # Algorithmus: Die Tabelle ist ohnehin „Bezeichnung + ein Wert je Slot" –
        # genau das, was ein gesonderter Block darunter nur wiederholen würde.
        # Als Zeilen kosten sie eine statt fünf Zeilen und stehen in derselben
        # Spalte wie die Kennzahlen, auf die sie sich auswirken.
        differences = self._difference_rows(configs)
        for index, label in enumerate(SLOT_LABELS[:self.slot_count]):
            if label in statistics:
                head = statistics[label][:1]
                rest = statistics[label][1:]
                statistics[label] = (*head, *(values[index] for _, values in differences), *rest)
        rows = ("Algorithmus", *(name for name, _ in differences), *self.STAT_LABELS)
        # Ohne Überschrift: Die Spaltenköpfe sagen bereits, was verglichen wird.
        self._table("", rows, statistics, list(self.FOOTER), self.section_title(longest))

    def _difference_rows(self, configs: list[HumanoidConfig]) -> list[tuple[str, list[str]]]:
        """Abweichende Parameter – **nur** bei mehrfach belegtem Algorithmus.

        Bei einer Parameterstudie tragen mehrere Slots denselben Algorithmus;
        ohne diese Zeilen sagte die Summary nicht, welche Spalte welchen Wert
        hatte. Bei lauter verschiedenen Verfahren erklärt die Kopfzeile bereits
        alles, und die Zeilen würden die Tabelle nur aus dem sichtbaren Bereich
        schieben – deshalb entfallen sie dort.
        """
        if not self._duplicated_algorithms(configs):
            return []
        return config_differences(configs)

    # ------------------------------------------------------ Einzelgraph (8.9)

    def _single_slot(self) -> int:
        """Slot, den der Einzelgraph zeigt."""
        try:
            return SLOT_LABELS.index(self.single_slot_var.get())
        except ValueError:
            return 0

    def _sync_single_combo(self) -> None:
        """Auswahl auf die aktiven Slots begrenzen."""
        labels = [f"{SLOT_LABELS[slot]} ({self.slots[slot].algorithm})"
                  for slot in range(self.slot_count)]
        plain = list(SLOT_LABELS[:self.slot_count])
        self.single_combo.configure(values=plain)
        self.single_combo_labels = labels
        if self.single_slot_var.get() not in plain:
            self.single_slot_var.set(plain[0])

    def _slot_curve_source(self, slot: int) -> tuple[list[EpisodeMetric], str]:
        """Messreihe des Slots samt Herkunft.

        Der Einzelgraph ist auch **nach** einem Vergleichslauf verfügbar, ohne
        dass neu trainiert werden muss: Die Daten liegen bereits vor. Ein
        Vergleichslauf hat Vorrang, weil er der jüngere Lauf ist.
        """
        runtime = self.slots[slot]
        if runtime.comparison_history:
            return runtime.comparison_history, "Vergleich"
        return runtime.history, "Training"

    def _draw_slot_curve(self, axes: Any, slot: int, configs: Optional[list] = None) -> list[float]:
        history, origin = self._slot_curve_source(slot)
        return self._plot_curve(axes, slot, history,
                                f"{origin} – {self._series_label(slot, configs)}")

    def _refresh_single_plot(self) -> None:
        """Einzelgraph: genau ein Verfahren, sonst wie der Vergleichsgraph.

        Dieselbe X-Achse, dieselbe Metrik und dieselbe Glättung – nur so sind
        Einzel- und Vergleichsbild nebeneinander lesbar.
        """
        self.single_axes.clear()
        slot = self._single_slot()
        configs = [runtime.comparison.config if runtime.comparison else runtime.workbench.config
                   for runtime in self.slots[:self.slot_count]]
        scale = self._draw_slot_curve(self.single_axes, slot, configs)
        self._reference_line(self.single_axes)
        self._limit_to_data(self.single_axes, scale)
        self.single_axes.set(xlabel="Episode", ylabel="Return")
        self._style(self.single_axes)
        self.single_canvas.draw_idle()

    def _chart_tab_changed(self) -> None:
        if self._chart_index() == 2:
            self._sync_single_combo()
            self._refresh_single_plot()

    def _chart_index(self) -> int:
        try:
            return self.charts.index(self.charts.select())
        except tk.TclError:
            return 0

    # ----------------------------------------------------------------- Export

    def _comparison_visible(self) -> bool:
        return self._chart_index() == 1

    def _config_snapshot_lines(self) -> list[str]:
        lines = []
        for slot in range(self.slot_count):
            runtime = self.slots[slot]
            config = runtime.comparison.config if (self._comparison_visible() and runtime.comparison) \
                else runtime.workbench.config
            if not self._comparison_visible() and slot != self.active_slot:
                continue
            lines.append(f"[{SLOT_LABELS[slot]}]")
            lines.extend(f"{key}: {value}" for key, value in asdict(config).items()
                         if key == "algorithm" or config.uses(key))
            lines.append("")
        lines.append(f"slot_count: {self.slot_count}")
        lines.append(f"smoothing_window: {self.smoothing_window}")
        return lines

    def _export_snapshot(self) -> tuple[str, tuple[Any, ...]]:
        """Kennzeichnet den aktuell sichtbaren Trainings-/Vergleichsstand, damit
        PNG und TXT denselben Dateinamensstamm teilen."""
        if self._comparison_visible():
            slug = "vergleich-" + "-".join(slugify(self.slots[slot].algorithm)
                                           for slot in range(self.slot_count))
            key = ("comparison", tuple(len(self.slots[slot].comparison_history)
                                       for slot in range(self.slot_count)))
        else:
            slot = self.active_slot
            slug = f"{SLOT_SHORT[slot].lower()}-{slugify(self.slots[slot].algorithm)}"
            key = ("single", slot, len(self.slots[slot].history))
        return slug, key

    def _export_base_name(self) -> str:
        slug, key = self._export_snapshot()
        if key != self._export_snapshot_key:
            self._export_snapshot_key = key
            self._export_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"humanoid_{slug}_{self._export_stamp}"

    def export_chart(self) -> None:
        EXPORT_DIR.mkdir(exist_ok=True)
        index = self._chart_index()
        if index == 2:
            slot = self._single_slot()
            figure = self.single_figure
            suffix = f"{SLOT_SHORT[slot].lower()}-{slugify(self.slots[slot].algorithm)}"
        else:
            figure = self.comparison_figure if index == 1 else self.figure
            suffix = "vergleich" if index == 1 else "training"
        path = filedialog.asksaveasfilename(
            parent=self.root, initialdir=EXPORT_DIR, defaultextension=".png",
            filetypes=(("PNG-Bild", "*.png"),),
            initialfile=f"{self._export_base_name()}_{suffix}.png",
        )
        if not path:
            return
        try:
            figure.savefig(path, facecolor=figure.get_facecolor())
        except OSError as error:
            messagebox.showerror("Export fehlgeschlagen", str(error), parent=self.root)
            return
        self.status.set(f"Abgeschlossen – Diagramm exportiert: {Path(path).name}")

    def export_each_slot(self) -> None:
        """Je aktivem Slot eine eigene PNG-Datei, in **einer** Aktion.

        Vorgaben verlangen einen Reward-Plot **je Verfahren**; eine
        Parameterstudie über drei Ausprägungen verlangt drei **getrennte**
        Plots. Beides von Hand zusammenzuklicken wäre fehleranfällig.
        """
        EXPORT_DIR.mkdir(exist_ok=True)
        ready = [slot for slot in range(self.slot_count) if self._slot_curve_source(slot)[0]]
        if not ready:
            messagebox.showinfo(
                "Nichts zu exportieren",
                "Noch kein Verfahren hat Messwerte. Starte zuerst ein Training "
                "oder einen Vergleich.", parent=self.root)
            return
        directory = filedialog.askdirectory(
            parent=self.root, initialdir=EXPORT_DIR,
            title="Ordner für die Einzelgraphen wählen")
        if not directory:
            return
        base = self._export_base_name()
        configs = [runtime.comparison.config if runtime.comparison else runtime.workbench.config
                   for runtime in self.slots[:self.slot_count]]
        remembered = self.single_slot_var.get()
        written: list[str] = []
        try:
            for slot in ready:
                name = (f"{base}_{SLOT_SHORT[slot].lower()}-"
                        f"{slugify(self.slots[slot].algorithm)}.png")
                target = Path(directory) / name
                if target.exists():
                    if not messagebox.askyesno(
                            "Datei überschreiben?",
                            f"'{name}' existiert bereits. Überschreiben?", parent=self.root):
                        continue
                self.single_slot_var.set(SLOT_LABELS[slot])
                self._refresh_single_plot()
                self.single_figure.savefig(
                    target, facecolor=self.single_figure.get_facecolor())
                written.append(name)
        except OSError as error:
            messagebox.showerror("Export fehlgeschlagen", str(error), parent=self.root)
            return
        finally:
            self.single_slot_var.set(remembered)
            self._refresh_single_plot()
        self.status.set(
            f"Abgeschlossen – {len(written)} Einzelgraph(en) exportiert nach "
            f"{Path(directory).name}")

    def export_summary(self) -> None:
        EXPORT_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.root, initialdir=EXPORT_DIR, defaultextension=".txt",
            filetypes=(("Textdatei", "*.txt"),),
            initialfile=f"{self._export_base_name()}_config.txt",
        )
        if not path:
            return
        text = self._summary_text() + "\n\nKonfiguration:\n" \
            + "\n".join(self._config_snapshot_lines()) + "\n"
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as error:
            messagebox.showerror("Export fehlgeschlagen", str(error), parent=self.root)
            return
        self.status.set(f"Abgeschlossen – Summary exportiert: {Path(path).name}")

    # -------------------------------------------------------------- Animation

    def _renderer(self, slot: int) -> HumanoidRenderer:
        """Renderprozess des Slots.

        Jede sichtbare Anzeige läuft in einem eigenen Prozess mit eigener
        Environment-Instanz; Environment-Instanzen werden nicht geteilt.
        """
        if self.renderers[slot] is None:
            self.renderers[slot] = HumanoidRenderer(seed=self.slots[slot].workbench.config.seed)
        return self.renderers[slot]

    def _close_renderer(self, slot: int) -> None:
        if self.renderers[slot] is not None:
            self.renderers[slot].close()
            self.renderers[slot] = None

    def _renderer_failed(self, error: Exception) -> None:
        """MuJoCo fehlt oder der Grafikkontext scheitert: Animation abschalten,
        Training und Evaluation bleiben nutzbar."""
        self._stop_animation()
        for slot in range(MAX_SLOTS):
            self.animation_live[slot] = False
        self._sync_panels()
        self.status.set("Animation nicht verfügbar – Training läuft weiter")
        if not self.renderer_failed:
            self.renderer_failed = True
            messagebox.showwarning("Animation nicht verfügbar", str(error), parent=self.root)

    def _when_ready(self, slot: int, action: Any) -> None:
        """Führt `action` aus, sobald der Renderprozess bereit ist.

        Der erste Start importiert MuJoCo und baut den OpenGL-Kontext auf. Das
        dauert einige Sekunden; die GUI fragt deshalb nach, statt zu warten.
        """
        try:
            if self._renderer(slot).poll_ready():
                action()
                return
        except RendererUnavailable as error:
            self._renderer_failed(error)
            return
        self.animation_after[slot] = self.root.after(200, lambda: self._when_ready(slot, action))

    def _animation_settings(self) -> int:
        """Bildrate prüfen und für die laufende Animation übernehmen."""
        text = self.fps.get().strip()
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError(
                f"Bildrate (FPS): '{text}' ist keine ganze Zahl. "
                f"Gültig: {MIN_ANIMATION_FPS} bis {MAX_ANIMATION_FPS}."
            ) from error
        if not MIN_ANIMATION_FPS <= value <= MAX_ANIMATION_FPS:
            raise ValueError(
                f"Bildrate (FPS): '{value}' liegt außerhalb. "
                f"Gültig: {MIN_ANIMATION_FPS} bis {MAX_ANIMATION_FPS}."
            )
        self.animation_interval = max(1, round(1000 / value))
        return value

    def _animation_workbench(self, slot: int) -> HumanoidWorkbench:
        """Der Slot, dessen Episode gezeigt wird: im Vergleich das
        Vergleichsmodell, sonst das Einzeltrainingsmodell."""
        runtime = self.slots[slot]
        if self.comparison_running and runtime.comparison is not None:
            return runtime.comparison
        return runtime.workbench

    def _animation_model(self, slot: int) -> Any:
        return self._animation_workbench(slot).model

    def trained_episodes(self, slot: int) -> int:
        """Nummer der zuletzt trainierten beziehungsweise verglichenen Episode.

        Die Animation zeigt immer den Lernstand nach dieser Episode und
        beschriftet sich mit ihrer Nummer – derselben Nummer, die auch auf der
        X-Achse der Graphen steht. Ein eigener Animationszähler wäre gegenüber
        dem Diagramm irreführend.
        """
        runtime = self.slots[slot]
        history = runtime.comparison_history if self.comparison_running else runtime.history
        return len(history)

    def _show_initial_frame(self, slot: int = 0) -> None:
        def show() -> None:
            observation, info, frame = self._renderer(slot).reset(
                self.slots[slot].workbench.config.seed)
            self.animation_observation[slot] = observation
            self.animation_info[slot] = info
            self.animation_action[slot] = None
            self._show_frame(slot, frame)
            self._update_caption(slot)

        self.status.set("MuJoCo-Renderer startet …")
        self._when_ready(slot, show)

    # ---------------------------------------------------- Beschriftung und Hover

    def _update_caption(self, slot: int) -> None:
        """Die eine Zeile unter dem Bild: Episode, Schritt, Return.

        Kurz gehalten, weil in einer schmalen Rasterzelle wenig Platz ist;
        welches Verfahren gezeigt wird, steht im Titel der Anzeige. Fester
        Aufbau und feste Breiten, damit die Bilder beim Weiterzählen nicht
        springen. Das `*` markiert die Wiederholung der besten Episode.
        """
        marker = "*" if self.animation_from_best[slot] else " "
        self.captions[slot].set(
            f"E: {self.animation_episode[slot]:4d}{marker} · "
            f"S: {self.animation_step[slot]:4d}/{MAX_EPISODE_STEPS} · "
            f"R: {german(self.animation_reward[slot]):>9}"
        )

    def _hover_lines(self, slot: int) -> str:
        """Inhalt der Einblendung neben dem Bild: Action, Zustand und Reward.

        Die Zeilen bleiben kurz und die Zahlen erhalten feste Breiten, damit die
        Einblendung schmal genug fürs Fenster bleibt und beim Aktualisieren
        nicht wandert.
        """
        observation = self.animation_observation[slot]
        if observation is None:
            return "Noch keine Episode gestartet."
        values = observation_readout(observation)
        rewards = reward_readout(self.animation_info[slot])
        # 348 Observationswerte lassen sich nicht anzeigen. Gezeigt wird, was
        # das Verhalten erklärt; cinert, cvel und cfrc_ext (286 Werte) nur
        # verdichtet. Der Wertebereich ist ±0,4, nicht ±1.
        lines = [f"Action (Bereich ±{ACTION_LIMIT}, Moment = aᵢ · gear)"]
        action = self.animation_action[slot]
        if action is None:
            lines.append("  noch keine Action gewählt")
        else:
            for group in action_readout(action)["groups"]:
                lines.append(f"  {group['name']}")
                for joint in group["joints"]:
                    lines.append(f"    a{joint['index']:<2} {joint['raw']:+.3f} "
                                 f"{joint['joint']:<21}{joint['gear']:>4.0f} {joint['text']}")
        angles, speeds = values["joint_angles"], values["joint_velocities"]
        low, high = values["healthy_z_range"]
        healthy = "gesund" if values["healthy"] else "AUSSERHALB"
        omega = values["angular_velocity"]

        def block(source: list[float]) -> list[str]:
            """Die 17 Gelenkwerte in denselben fünf Gruppen wie die Actions.

            Die Reihenfolge in der Observation ist **nicht** die der Aktuatoren:
            `abdomen_y` und `abdomen_z` sind vertauscht. Deshalb wird über
            `ACTUATOR_TO_JOINT` gemappt statt direkt indiziert.
            """
            return [f"    {name:<12}" + " ".join(
                f"{source[ACTUATOR_TO_JOINT[index]]:+.2f}" for index in indices)
                for name, indices in ACTUATOR_GROUPS]

        lines += [
            "",
            "Zustand",
            f"  Rumpfhöhe   {values['height']:+.3f} m  ({healthy}, {low}–{high} m)",
            f"  Neigung     {values['tilt_degrees']:5.1f}° gegen die Senkrechte",
            f"  Quaternion  " + " ".join(f"{v:+.2f}" for v in values["quaternion"]),
            "  Gelenkwinkel",
            *block(angles),
            f"  Tempo       vₓ {values['vx']:+.3f}  v_y {values['vy']:+.3f}  "
            f"v_z {values['vz']:+.3f}",
            f"  Drehrate    " + " ".join(f"{value:+.2f}" for value in omega),
            "  Gelenktempo",
            *block(speeds),
            "  Geschwindigkeiten sind unbeschränkt (kein Clipping).",
            f"  Kontaktkraft Σcfrc² {values['contact_force_squared']:.3g}"
            f"  → Kosten {values['contact_cost']:.4f} (max {CONTACT_COST_MAX:g})",
            "",
            "Reward des letzten Schritts (vier Anteile)",
            f"  Überleben {rewards['survive']:+.3f}   vorwärts {rewards['forward']:+.3f}",
            f"  Steuerung {rewards['ctrl']:+.4f}   Kontakt  {rewards['contact']:+.4f}",
            f"  Position  x {rewards['x_position']:+.2f} m  y {rewards['y_position']:+.2f} m"
            f"  (Abstand {rewards['distance_from_origin']:.2f} m)",
        ]
        return "\n".join(lines)

    def _hover_enter(self, slot: int) -> None:
        self.hover_slot = slot
        self._refresh_hover()

    def _hover_leave(self, slot: int) -> None:
        if self.hover_slot == slot:
            self._hide_hover()

    def _hide_hover(self) -> None:
        self.hover_slot = None
        self.hover_panel.place_forget()

    def _refresh_hover(self) -> None:
        slot = self.hover_slot
        if slot is None:
            return
        self.hover_title.configure(text=f"{SLOT_LABELS[slot]} – {self.slots[slot].algorithm}",
                                   foreground=SLOT_COLORS[slot])
        self.hover_body.configure(text=self._hover_lines(slot))
        self._place_hover(slot)

    def _place_hover(self, slot: int) -> None:
        """Legt die Einblendung **neben** das Bild – nie darüber.

        `place()` ist ein Overlay über dem Container: Das Raster und damit
        Größe und Position der Bilder bleiben unverändert. Reicht der Platz
        rechts vom Bild nicht, wandert die Einblendung nach links; sie darf
        dabei über den Nachbarbereich reichen, aber nie über das eigene Bild.
        """
        label = self.image_labels[slot]
        if not label.winfo_ismapped():
            self._hide_hover()
            return
        self.hover_panel.update_idletasks()
        panel_width = self.hover_panel.winfo_reqwidth()
        panel_height = self.hover_panel.winfo_reqheight()
        window_x, window_y = self.root.winfo_rootx(), self.root.winfo_rooty()
        window_width, window_height = self.root.winfo_width(), self.root.winfo_height()
        photo_width, photo_height = self.photo_sizes[slot]
        # Bildrechteck innerhalb des zentrierenden Labels.
        image_left = label.winfo_rootx() + max(0, (label.winfo_width() - photo_width) // 2)
        image_right = image_left + photo_width
        gap = 8
        # Beide Kandidaten liegen konstruktionsbedingt vollständig neben dem
        # Bild. Gewählt wird der, der ins Fenster passt; passt keiner, der mit
        # dem kleineren Überlauf. Zurechtgerückt wird nie – das schöbe die
        # Einblendung über ihr eigenes Bild.
        right_x = image_right + gap - window_x
        left_x = image_left - panel_width - gap - window_x
        if right_x + panel_width <= window_width:
            x = right_x
        elif left_x >= 0:
            x = left_x
        else:
            x = left_x if -left_x < right_x + panel_width - window_width else right_x
        y = label.winfo_rooty() - window_y + max(0, (label.winfo_height() - photo_height) // 2)
        y = max(0, min(y, max(0, window_height - panel_height)))
        self.hover_panel.place(x=x, y=y)
        self.hover_panel.lift()

    def _frame_size(self, slot: int) -> tuple[int, int]:
        """Zielgröße des Bildes, abgeleitet aus dem freien Platz des Containers.

        Die Größe des Bildlabels selbst taugt dafür nicht: Sie folgt dem zuletzt
        gesetzten Bild, sodass ein einmal großes Bild seinen Platz behielte und
        die übrigen Anzeigen verdrängen würde.
        """
        count = max(1, len(self._panel_slots()))
        positions = panel_grid_positions(count, self._panel_columns(count))
        # Belegte Zeilen und Spalten zählen, nicht die Rasterbreite: Die einzelne
        # Anzeige spannt über beide Spalten und darf deshalb den vollen Platz
        # bekommen.
        rows = len({row for row, _, _ in positions})
        columns = len({column for _, column, _ in positions})
        # Alles, was in der Zelle neben dem Bild Platz braucht: die Zeile
        # darüber, die Beschriftung darunter, Rahmen und Innenabstand. Geraten
        # werden darf das nicht – eine zu kleine Schätzung macht das Bild zu
        # hoch und schneidet die Beschriftung ab.
        chrome = self.caption_labels[slot].winfo_reqheight() + 34
        width = max(140, self.env_container.winfo_width() // columns - 16)
        height = max(100, self.env_container.winfo_height() // rows - chrome)
        return width, height

    def _show_frame(self, slot: int, frame: np.ndarray) -> None:
        self.last_frames[slot] = np.asarray(frame)
        label = self.image_labels[slot]
        width, height = self._frame_size(slot)
        image = Image.fromarray(self.last_frames[slot])
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo_sizes[slot] = image.size
        self.photos[slot] = ImageTk.PhotoImage(image)
        label.configure(image=self.photos[slot])

    def animate(self) -> None:
        """Einzelne sichtbare Episode des aktiven Verfahrens."""
        if self.busy:
            return
        slot = self.active_slot
        if self.slots[slot].workbench.model is None:
            messagebox.showinfo("Keine Episode",
                                f"Trainiere zuerst ein Modell für {SLOT_LABELS[slot]}.",
                                parent=self.root)
            return
        try:
            self._animation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültige Animation", str(error), parent=self.root)
            return
        self._stop_animation(slot)

        def run() -> None:
            self._begin_episode(slot, live=False)
            self._animate_step(slot)

        self._when_ready(slot, run)

    def _refresh_snapshot(self, slot: int, model: Any) -> None:
        """Aktualisiert die Policy-Kopie, mit der die Animation spielt.

        `copy.deepcopy` scheitert an Tensoren, die keine Blätter des
        Autograd-Graphen sind. Stattdessen entsteht einmal je Lauf eine leere
        Policy aus den Konstruktorparametern, die danach je Episode mit einer
        losgelösten Kopie der aktuellen Gewichte gefüllt wird. Ein Worker-Thread
        schreibt diese Gewichte gerade; ein dabei gemischt gelesener Stand ist
        für eine Anzeige unkritisch, ein Zugriff auf das lernende Netz selbst
        wäre es nicht.
        """
        self._apply_policy_state(
            slot, model,
            {name: value.detach().clone() for name, value in model.policy.state_dict().items()})

    def _apply_policy_state(self, slot: int, model: Any, state: dict[str, Any]) -> None:
        """Lädt einen Policy-Zustand in die Anzeigekopie des Slots."""
        if self.animation_policy[slot] is None:
            self.animation_policy[slot] = type(model.policy)(
                **model.policy._get_constructor_parameters())
        self.animation_policy[slot].load_state_dict(state)
        self.animation_policy[slot].set_training_mode(False)

    def _begin_episode(self, slot: int, live: bool) -> None:
        """Setzt den Renderer zurück und wählt den Lernstand, den die Animation
        spielt: den aktuellen oder den der besten bisherigen Episode."""
        workbench = self._animation_workbench(slot)
        model = workbench.model
        choice = self.episode_choice[slot].get()
        # `beste Ep.` spielt die Episode nach; fehlt eine Aufzeichnung – etwa
        # bei einem geladenen Modell –, fällt sie auf die Policy zurück.
        recording = workbench.best_recording() if choice == BEST_EPISODE else None
        best = (workbench.best_snapshot()
                if choice in (BEST_EPISODE, BEST_POLICY) else None)
        configured_seed = self.slots[slot].workbench.config.seed
        self.animation_actions[slot] = None
        self.animation_action_index[slot] = 0
        if recording is not None:
            # Exakte Wiedergabe: Simulatorzustand vom Episodenbeginn setzen und
            # die aufgezeichneten Actions abspielen. Die Policy nachzurechnen
            # erreicht den Return der Episode systematisch nicht – sie ist das
            # Maximum über Hunderte Episoden und verdankt ihn zum Teil
            # glücklichen Explorationszügen und ihrem Startzustand.
            best_metric, (qpos, qvel), actions = recording
            self.animation_actions[slot] = actions
            self.animation_from_best[slot] = True
            observation, info, frame = self._renderer(slot).reset_to(qpos, qvel, configured_seed)
            self.animation_observation[slot] = observation
            self.animation_info[slot] = info
            self.animation_action[slot] = None
            self.animation_step[slot] = 0
            self.animation_episode[slot] = best_metric.episode
            self.animation_reward[slot] = 0.0
            self._show_frame(slot, frame)
            self._update_caption(slot)
            if self.hover_slot == slot:
                self._refresh_hover()
            return
        if best is not None:
            # Der Lernstand der besten Episode, mit festem Seed – so sieht die
            # Wiederholung jedes Mal gleich aus. Mitgeführt werden die
            # Beobachtungsstatistiken von damals: Eine alte Policy mit heutigen
            # Statistiken sähe die Beobachtungen anders als im Training.
            self._apply_policy_state(slot, model, best[1])
            self.animation_statistics[slot] = best[2]
            episode, seed = best[0].episode, configured_seed
        elif live:
            self.animation_statistics[slot] = None
            self._refresh_snapshot(slot, model)
            # Laufende Episoden starten ohne festen Seed, damit nacheinander
            # gezeigte Episoden nicht identisch aussehen.
            episode, seed = self.trained_episodes(slot), None
        else:
            self.animation_policy[slot] = None
            self.animation_statistics[slot] = None
            episode, seed = self.trained_episodes(slot), configured_seed
        self.animation_from_best[slot] = best is not None
        observation, info, frame = self._renderer(slot).reset(seed)
        self.animation_observation[slot] = observation
        self.animation_info[slot] = info
        self.animation_action[slot] = None
        self.animation_step[slot] = 0
        self.animation_episode[slot] = episode
        self.animation_reward[slot] = 0.0
        self._show_frame(slot, frame)
        self._update_caption(slot)
        if self.hover_slot == slot:
            self._refresh_hover()

    def _start_live_animation(self, slots: list[int]) -> None:
        """Blendet die Animation für einen laufenden Trainings- oder
        Vergleichslauf ein.

        Gestartet wird nur, was noch nicht läuft und nicht auf `inaktiv` steht.
        """
        slots = [slot for slot in slots
                 if not self.animation_live[slot] and self._slot_animated(slot)]
        if not slots:
            return
        for slot in slots:
            self.animation_live[slot] = True
        self._sync_panels()
        for slot in slots:
            self._live_episode(slot)

    def _animation_note(self, text: str) -> None:
        """Hinweis zur Animation – aber nur, wenn gerade nichts läuft.

        Während eines Laufs gehört die Statuszeile dem Lauf: Sie nennt die
        beteiligten Verfahren und ihr Budget. Eine Animationsmeldung würde das
        überschreiben, und die eigentliche Information wäre weg.
        """
        if not self.busy:
            self.status.set(text)

    def _episode_choice_changed(self, slot: int) -> None:
        """Die Wahl wirkt sofort und betrifft ausschließlich diese Anzeige."""
        choice = self.episode_choice[slot].get()
        if choice == INACTIVE_EPISODE:
            # Feld ausblenden; die übrigen Anzeigen bekommen den Platz.
            self._stop_animation(slot)
            self.animation_live[slot] = False
            self._sync_panels()
            return
        if self.busy and slot in self._live_slots():
            self._start_live_animation([slot])
            return
        if choice in (BEST_EPISODE, BEST_POLICY) \
                and self._animation_workbench(slot).best_snapshot() is None:
            self._animation_note(f"{SLOT_LABELS[slot]}: noch keine beste Episode – "
                                 "es läuft weiter der aktuelle Lernstand")
        if self._live_active(slot):
            self._stop_animation(slot)
            self._live_episode(slot)

    def _slot_animated(self, slot: int) -> bool:
        """Ist die Animation dieses Slots eingeschaltet?

        Neben dem globalen Schalter entscheidet die Auswahl je Anzeige: `inaktiv`
        schaltet genau diesen Slot ab, ohne die übrigen zu berühren.
        """
        return self.episode_choice[slot].get() != INACTIVE_EPISODE

    def _live_active(self, slot: int) -> bool:
        return self.animation_live[slot] and self.busy and self._slot_animated(slot)

    def _live_slots(self) -> list[int]:
        """Slots, deren laufendes Training sichtbar gemacht wird."""
        if not self.busy:
            return []
        if self.comparison_running:
            candidates = list(range(self.slot_count))
        elif self.running_slot is None:
            return []
        else:
            candidates = [self.running_slot]
        return [slot for slot in candidates if self._slot_animated(slot)]

    def _live_episode(self, slot: int) -> None:
        if not self._live_active(slot):
            return
        try:
            # Eine geänderte Bildrate wirkt ab der nächsten sichtbaren Episode.
            self._animation_settings()
        except ValueError:
            pass  # ungültige Eingabe: bisherige Bildrate behalten
        if self._animation_model(slot) is None:
            # Das Modell entsteht erst im Worker-Thread; kurz warten.
            self.animation_after[slot] = self.root.after(250, lambda: self._live_episode(slot))
            return

        def run() -> None:
            if not self._live_active(slot):
                return
            try:
                self._begin_episode(slot, live=True)
            except RendererUnavailable as error:
                self._renderer_failed(error)
                return
            except Exception as error:
                self._stop_live_animation()
                self._animation_note(f"Animation gestoppt – {error}")
                return
            self._animate_step(slot)

        self._when_ready(slot, run)

    def _stop_live_animation(self) -> None:
        if not any(self.animation_live):
            return
        for slot in range(MAX_SLOTS):
            if self.animation_live[slot]:
                self.animation_live[slot] = False
                self.animation_policy[slot] = None
                self._stop_animation(slot)
        self._sync_panels()

    def _policy_action(self, slot: int) -> np.ndarray:
        recorded = self.animation_actions[slot]
        if recorded is not None:
            # Aufgezeichnete Episode: Die Actions stehen fest, es wird nichts
            # nachgerechnet. Sind sie aufgebraucht, endet die Episode ohnehin
            # im selben Schritt wie damals.
            index = min(self.animation_action_index[slot], len(recorded) - 1)
            self.animation_action_index[slot] = index + 1
            return np.asarray(recorded[index], dtype=np.float32).reshape(-1)
        # Bei aktiver Normalisierung muss die Policy die Beobachtung so sehen
        # wie im Training – mit eingefrorenen Statistiken.
        observation = self._animation_workbench(slot).policy_observation(
            self.animation_observation[slot], self.animation_statistics[slot])
        policy = self.animation_policy[slot]
        if policy is not None:
            action, _ = policy.predict(observation, deterministic=True)
        else:
            action, _ = self._animation_model(slot).predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    @staticmethod
    def _episode_outcome_text(terminated: bool) -> str:
        """`Humanoid-v5` besitzt keinen terminalen Zustand: `terminated` ist
        immer False, jede Episode endet nach `MAX_EPISODE_STEPS`."""
        return f"Abgeschlossen – {MAX_EPISODE_STEPS} Schritte gelaufen"

    def _run_episode_without_animation(self, slot: int) -> None:
        """Sichtbare Episode ohne Einzelbildanimation: nur das Endbild zählt."""
        terminated = truncated = False
        frame = self.last_frames[slot]
        while not (terminated or truncated):
            action = self._policy_action(slot)
            observation, reward, terminated, truncated, info, frame = self._renderer(slot).step(action)
            self.animation_observation[slot] = observation
            self.animation_info[slot] = info
            self.animation_action[slot] = action
            self.animation_step[slot] += 1
            self.animation_reward[slot] += reward
        if frame is not None:
            self._show_frame(slot, frame)
        self._update_caption(slot)
        if self.hover_slot == slot:
            self._refresh_hover()
        self.status.set(self._episode_outcome_text(terminated))

    def _animate_step(self, slot: int) -> None:
        live = self.animation_live[slot]
        if live and not self._live_active(slot):
            self._stop_live_animation()
            return
        if live:
            # Ein Fehler in der Anzeige darf den laufenden Trainingslauf weder
            # abbrechen noch stillschweigend die Animation beenden.
            try:
                self._animate_frame(slot, live)
            except RendererUnavailable as error:
                self._renderer_failed(error)
            except Exception as error:
                self._stop_live_animation()
                self.status.set(f"Animation gestoppt – {error}")
            return
        if not self._slot_animated(slot):
            self._stop_animation(slot)
            return
        self._animate_frame(slot, live)

    def _animate_frame(self, slot: int, live: bool) -> None:
        action = self._policy_action(slot)
        observation, reward, terminated, truncated, info, frame = self._renderer(slot).step(action)
        self.animation_observation[slot] = observation
        self.animation_info[slot] = info
        self.animation_action[slot] = action
        self.animation_step[slot] += 1
        self.animation_reward[slot] += reward
        self._show_frame(slot, frame)
        self._update_caption(slot)
        # Die Einblendung zeigt den Stand des gerade dargestellten Frames.
        if self.hover_slot == slot:
            self._refresh_hover()
        if terminated or truncated:
            if live:
                # Nächste Episode zeigt den inzwischen weiter trainierten Stand.
                self.animation_after[slot] = self.root.after(
                    self.animation_interval, lambda: self._live_episode(slot))
            else:
                self.animation_after[slot] = None
                self.status.set(self._episode_outcome_text(terminated))
            return
        self.animation_after[slot] = self.root.after(
            self.animation_interval, lambda: self._animate_step(slot))

    def _stop_animation(self, slot: Optional[int] = None) -> None:
        slots = range(MAX_SLOTS) if slot is None else (slot,)
        for index in slots:
            if self.animation_after[index]:
                self.root.after_cancel(self.animation_after[index])
                self.animation_after[index] = None

    # ------------------------------------------------------------ Modelldatei

    def reset(self) -> None:
        if self.busy:
            return
        slot = self.active_slot
        if not messagebox.askyesno("Zurücksetzen",
                                   f"Lernzustand von {SLOT_LABELS[slot]} verwerfen?",
                                   parent=self.root):
            return
        try:
            config = self._config(slot)
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root)
            return
        self.slots[slot].reset_all(config)
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._refresh_single_plot()
        self._training_summary()
        self.status.set(f"Bereit – {SLOT_LABELS[slot]} zurückgesetzt")

    def _show_manual(self, text: str) -> None:
        """Bedienungsanleitung in einem eigenen, scrollbaren Fenster.

        Ein Meldungsdialog taugt dafür nicht: Er kann nicht scrollen und wächst
        mit dem Text, bis seine Schaltfläche unter den Bildschirmrand rutscht –
        dann lässt er sich nicht mehr schließen. Dieses Fenster passt sich dem
        Bildschirm an, scrollt und reagiert auf Escape.
        """
        window = tk.Toplevel(self.root)
        window.title("Bedienungsanleitung")
        window.configure(background=self.BG)
        window.transient(self.root)

        # Die Schaltfläche bekommt ihren Platz VOR dem Textbereich zugeteilt:
        # Der Packer vergibt in der Reihenfolge des Packens, sodass der
        # expandierende Text sie sonst aus einem knappen Fenster drückt – genau
        # der Grund, aus dem sich ein zu großer Meldungsdialog nicht schließen
        # lässt.
        ttk.Button(window, text="Schließen", command=window.destroy).pack(
            side="bottom", pady=(0, 8))
        body = ttk.Frame(window, padding=8)
        body.pack(side="top", fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        view = tk.Text(body, wrap="none", background=self.FIELD, foreground=self.FG,
                       insertbackground=self.FG, relief="flat", font="TkFixedFont",
                       padx=8, pady=6)
        view.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(body, orient="vertical", command=view.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(body, orient="horizontal", command=view.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        view.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        view.insert("1.0", text)
        # Überschriften fett: Sie stehen als einzelne Zeile ohne Einrückung
        # zwischen zwei Leerzeilen.
        bold = tkfont.Font(font=view.cget("font"))
        bold.configure(weight="bold")
        view.tag_configure("kapitel", font=bold, foreground=self.ACCENT)
        lines = text.split("\n")
        for index, line in enumerate(lines, start=1):
            previous = lines[index - 2] if index >= 2 else ""
            if line and not line.startswith(" ") and not previous.strip() \
                    and not line.endswith(".") and len(line) < 40:
                view.tag_add("kapitel", f"{index}.0", f"{index}.end")
        view.configure(state="disabled")

        # Größe aus dem Inhalt ableiten, aber am Bildschirm begrenzen. Gemessen
        # wird die **längste Zeile selbst**: Zeichenzahl mal Breite einer Ziffer
        # geht daneben, sobald die Darstellung nicht exakt gleich breit ist.
        character = tkfont.Font(font=view.cget("font"))
        widest = max(character.measure(line) for line in lines)
        width = min(widest + 60, int(self.root.winfo_screenwidth() * .75))
        height = min(character.metrics("linespace") * (len(lines) + 2) + 80,
                     int(self.root.winfo_screenheight() * .8))
        left = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - width) // 2)
        top = self.root.winfo_rooty() + 40
        window.geometry(f"{int(width)}x{int(height)}+{int(left)}+{int(top)}")
        window.minsize(420, 240)

        # Escape schließt – ohne das bliebe das Fenster stehen, sobald die
        # Schaltfläche einmal nicht sichtbar ist.
        window.bind("<Escape>", lambda _event: window.destroy())
        view.bind("<Escape>", lambda _event: window.destroy())
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        # `focus_force` statt `focus_set`: Ohne zugeteilten Tastaturfokus kommt
        # kein Tastendruck an, und Escape bliebe wirkungslos, bis jemand ins
        # Fenster klickt.
        window.focus_force()
        view.focus_set()
        window.grab_set()

    def instructions(self) -> None:
        self._show_manual(
            "Ablauf\n"
            "1. 'Anzahl Verfahren' legt fest, wie viele Slots aktiv sind (2 bis 4,\n"
            f"   Standard {DEFAULT_SLOT_COUNT}: PPO, TD3 und SAC je einmal). Gern auch\n"
            "   mehrfach denselben Algorithmus, um Parametrisierungen zu vergleichen.\n"
            "2. Im jeweiligen Tab die Parameter setzen. Der sichtbare Tab ist das aktive\n"
            "   Verfahren; 'Training' und 'Zurücksetzen' wirken darauf.\n"
            "3. 'Training starten / fortsetzen' trainiert das aktive Verfahren,\n"
            "   'Vergleich starten / fortsetzen' alle aktiven Slots parallel.\n\n"
            "Budget: zwei Grenzen\n"
            "Jeder Slot führt 'Trainingsschritte N' UND 'Episoden E'. Der Lauf endet an\n"
            "der zuerst erreichten; die Summary zeigt unter 'Ende durch', welche es war.\n"
            "'Episoden E = 0' heißt unbegrenzt, dann zählt allein das Schrittbudget.\n"
            "Beide Grenzen gelten RELATIV: Ein zweiter Druck auf 'Training fortsetzen'\n"
            "setzt nichts zurück, sondern hängt erneut das volle Budget an – Kurve und\n"
            "Summary wachsen weiter.\n"
            "Warum beides: Eine Episode endet beim Sturz. Untrainiert fällt die Figur\n"
            "nach rund 25 Schritten, mit Lernfortschritt werden Episoden länger. 1000\n"
            "Episoden sind deshalb mal 25.000, mal 1.000.000 Schritte. Für einen FAIREN\n"
            "Vergleich ist das Schrittbudget die richtige Grenze: Das Verfahren, das\n"
            "besser lernt, bekäme bei gleicher Episodenzahl sonst mehr Trainingsdaten.\n\n"
            "Environment\n"
            "Humanoid-v5 (MuJoCo): eine dreidimensionale Figur von 42 kg mit 17 Gelenken\n"
            "und 348 Beobachtungswerten. Der Wertebereich der Actions ist ±0,4 – NICHT\n"
            "±1 wie in den Vorgängerprojekten. Die Übersetzungen unterscheiden sich stark\n"
            "(25 an den Armen bis 300 an der Hüfte), derselbe Actionwert bedeutet je\n"
            "Gelenk ein anderes Moment.\n"
            "Reward = 5,0 (Überleben) + 1,25·vₓ - 0,1·Σaᵢ² - 5e-7·Σcfrc².\n"
            "Der Überlebensbonus DOMINIERT: Eine Figur, die 1000 Schritte nur steht,\n"
            "sammelt bereits 5000 Return.\n"
            "Die Episode endet mit einem STURZ, sobald die Rumpfhöhe den Bereich\n"
            "1,0 bis 2,0 m verlässt – eine Winkelbedingung gibt es nicht, die Figur darf\n"
            f"beliebig verdreht sein. Sonst nach {MAX_EPISODE_STEPS} Schritten\n"
            "('durchgehalten'). Weder Bonus noch Strafe am Ende; ein Sturz kostet nur\n"
            "die Rewards der Schritte, die nicht mehr stattfinden.\n"
            f"Die Zielmarke {german(TARGET_RETURN, 0)} ist PROJEKTINTERN gesetzt und\n"
            f"entspricht {MAX_EPISODE_STEPS} Schritten aufrecht. Humanoid-v5 führt keinen\n"
            "offiziellen reward_threshold – 'gelöst' gibt es hier nicht.\n\n"
            "Methoden\n"
            "PPO ist on-policy: Rollouts, GAE, geclipptes Ziel, kein Replay Buffer.\n"
            "SAC und TD3 sind off-policy mit zwei Critics – TD3 mit deterministischem\n"
            "Actor und Action Noise, SAC mit stochastischem Actor und gelernter Entropie.\n"
            "Bei gleichem Schrittbudget bevorteilt der Vergleich die Off-Policy-Verfahren:\n"
            "Sie lernen aus jedem gespeicherten Übergang mehrfach, PPO verwirft seine\n"
            "Daten nach jedem Update. Hinzu kommt, dass die Profile für verschieden lange\n"
            f"Läufe getunt sind – {thousands(DEFAULT_TOTAL_TIMESTEPS)} Schritte sind bei PPO\n"
            "5 % seines Profilbudgets, bei TD3 und SAC je 25 %.\n\n"
            "Diagramme\n"
            "'Training' zeigt das aktive Verfahren, 'Vergleich' alle Slots gemeinsam,\n"
            "'Einzelverfahren' genau einen wählbaren Slot – auch nach einem Vergleichs-\n"
            "lauf, ohne neu zu trainieren. Die Kurven sind durchgezogen und farbig nach\n"
            "Slot: V1 blau, V2 rot, V3 gelb, V4 grün; die weiße gestrichelte Linie ist\n"
            "die Zielmarke. Die Y-Achse folgt den MESSWERTEN, nicht der Marke – sonst\n"
            "drückte diese alle Kurven in einen Bruchteil der Bildhöhe.\n"
            "'Glättung' rechts in der Tableiste stellt ein, über wie viele Episoden der\n"
            "gleitende Durchschnitt mittelt – global für alle Slots und alle Graphen,\n"
            "sofort wirksam, auch mitten im Lauf. Dieselbe Zahl bestimmt, über wie viele\n"
            "Episoden die Summary mittelt.\n"
            "Export: 'PNG exportieren' sichert das sichtbare Diagramm, 'TXT exportieren'\n"
            "die Summary, 'Je Verfahren PNG' schreibt in einer Aktion je aktivem Slot\n"
            "eine eigene Datei – für Berichte, die einen Plot je Verfahren verlangen.\n\n"
            "Summary\n"
            "Oben Umfang und Ausgang des Laufs, darunter – abgetrennt durch eine fett\n"
            "gesetzte Zwischenüberschrift – alle Mittelwerte über die LETZTEN Episoden\n"
            "(Fenster wie die Glättung). Über den ganzen Lauf gemittelt hinge jede\n"
            "Kennzahl noch am untrainierten Anfang. 'Beste Episode' zählt dagegen über\n"
            "alle Episoden.\n"
            "Der Block 'Unterschiede' erscheint nur, wenn ein Algorithmus mehrere Slots\n"
            "belegt – also bei einer Parameterstudie. Bei lauter verschiedenen Verfahren\n"
            "sagt die Kopfzeile bereits alles.\n\n"
            "Animation\n"
            "Neben jedem Verfahren steht unter der Ueberschrift 'Animation' ein Feld\n"
            "mit vier Moeglichkeiten. 'akt. Ep.' zeigt den laufenden Lernstand.\n"
            "'inaktiv' blendet diese eine Anzeige aus – die uebrigen bekommen ihren\n"
            "Platz und werden groesser. Einen globalen Schalter 'Animation zeigen'\n"
            "gibt es nicht mehr: Alle Felder auf 'inaktiv' zu stellen ist dasselbe.\n"
            "'beste Ep.' spielt die bisher beste Episode EXAKT nach. Aufgezeichnet werden\n"
            "der Simulatorzustand zu ihrem Beginn und jede ausgefuehrte Action; die\n"
            "Wiedergabe setzt den Zustand und spielt die Actions ab. Bewegung und\n"
            "Return sind damit identisch mit der Trainingsepisode. Ein '*' hinter der\n"
            "Nummer macht kenntlich, dass nicht der aktuelle Stand laeuft.\n"
            "'beste Pol.' laesst stattdessen den Lernstand dieser Episode determi-\n"
            "nistisch laufen. Das erreicht nur 80 bis 91 % des Returns, beantwortet\n"
            "aber die andere Frage: nicht 'was ist damals passiert', sondern 'wie gut\n"
            "ist dieser Stand ohne das Glueck explorativer Zuege'. Der Abstand zwischen\n"
            "beiden misst, wie viel des Spitzenwerts Zufall war.\n"
            "Die Anzeigen ordnen sich nach der Form des Bereichs - gewaehlt wird die\n"
            "Aufteilung, die das groesste Bild ergibt.\n"
            "Unter jedem Bild stehen nur Episode, Schritt und Return. Alle weiteren\n"
            "Messwerte – die 17 Actions mit ihrer Übersetzung, Rumpfhöhe und Neigung,\n"
            "Gelenkwinkel, Geschwindigkeiten und die Zerlegung des Rewards in seine vier\n"
            "Anteile – erscheinen erst, wenn der Mauszeiger über dem Bild steht; sie\n"
            "werden dann NEBEN dem Bild eingeblendet, sodass beides sichtbar bleibt.\n"
            "Von den 348 Observationswerten zeigt die Einblendung eine Auswahl; die\n"
            "286 Werte aus cinert, cvel und cfrc_ext erklären kein Verhalten und\n"
            "erscheinen nur verdichtet als Quadratsumme der Kontaktkräfte.\n"
            "'Animation zeigen' wirkt jederzeit, auch im laufenden Lauf, und zeigt alle\n"
            "aktiven Verfahren gleichzeitig – jedes in einem eigenen Renderprozess. Das\n"
            f"kostet Rechenzeit. Die Bildrate (Standard {DEFAULT_ANIMATION_FPS} FPS) ist eine\n"
            f"Obergrenze; die environment-eigene Rate wäre {RENDER_FPS} FPS, bei {DEFAULT_ANIMATION_FPS}\n"
            "läuft die Anzeige also in etwa dreifacher Zeitlupe – ein stürzender\n"
            "Humanoid ist sonst kaum zu verfolgen.\n\n"
            "Laufzeit und Speicher\n"
            f"Die Voreinstellung {thousands(DEFAULT_TOTAL_TIMESTEPS)} Schritte passt zur Vorgabe von 1000\n"
            "Episoden: Gemessen entsprechen diese bei SAC rund 80.000 Schritten, beide\n"
            f"Grenzen liegen also in derselben Größenordnung. Die Messläufe für den Bericht\n"
            f"nutzen {thousands(REPORT_TIMESTEPS)} Schritte bei 'Episoden E = 0'. Auch das liegt UNTER\n"
            "den Zoo-Profilen (PPO 10 Mio., TD3/SAC je 2 Mio.). Die Figur wird damit\n"
            "NICHT laufen; erwartbar ist, dass sie sich zunehmend länger hält.\n"
            "Gemessen: PPO rund 480 Schritte/s, TD3 und SAC rund 50 – ein voller Lauf\n"
            "dauert bei den Off-Policy-Verfahren also Stunden.\n"
            f"'Threads (PyTorch)' (Standard {DEFAULT_TORCH_THREADS}) steht neben 'Anzahl\n"
            "Verfahren', weil beide zusammenhängen: Es bestimmt, über wie viele Kerne\n"
            "PyTorch EINE Matrixmultiplikation verteilt – nicht, wie viele Verfahren\n"
            "gleichzeitig laufen. Ein Vergleich führt seine Slots als Threads EINES\n"
            "Prozesses aus; alle teilen sich denselben Pool. Die Einstellung gilt also\n"
            "für die ganze Anwendung, nicht je Lauf, und wird beim Start eines Laufs\n"
            "übernommen. Von den 6 Kernen sind nur 2 Performance-Kerne; nimmt PyTorch\n"
            "alle sechs, warten die schnellen Threads auf die langsamen – Faktor 3.\n"
            "Jeder Off-Policy-Slot belegt mit dem Standardbuffer rund 1,3 GB.\n\n"
            "Kein Lernerfolg?\n"
            "Zu kurzes Budget, zu großer Lernstart t₀, zu kleines Action Noise bei TD3,\n"
            "eine zu hohe Lernrate oder bei PPO fehlende Normalisierung. Zur Einordnung:\n"
            "Der Zoo-Benchmark erreicht bei 2 Mio. Schritten SAC rund 6.232 und TD3 rund\n"
            "5.567; für PPO führt er keinen Humanoid-Eintrag."
        )

    def close(self) -> None:
        self.stop_event.set()
        self._stop_animation()
        for slot in range(MAX_SLOTS):
            self._close_renderer(slot)
        for runtime in self.slots:
            runtime.close()
        self.root.destroy()
