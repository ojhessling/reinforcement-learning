"""Dark Tkinter GUI for the Hopper PPO/SAC/TD3 workbench."""

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
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from hopper_logic import (
    ALGORITHMS, BOOLEAN_FIELDS, CHOICE_FIELDS, DEFAULT_EVALUATION_EPISODES,
    DEFAULT_EVALUATION_INTERVAL, DEFAULT_TOTAL_TIMESTEPS, INTEGER_FIELDS, MAX_EPISODE_STEPS,
    OPTIONAL_FLOAT_FIELDS,
    SOLVED_RETURN, TUPLE_FIELDS, EpisodeMetric, EvaluationResult, HopperConfig,
    HopperWorkbench, action_readout, config_differences, default_config,
    observation_readout, reward_readout,
)
from hopper_render import HopperRenderer, RendererUnavailable


def rolling_average(values: list[float], window: int = 20) -> np.ndarray:
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


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def german(value: float, digits: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma und Tausenderpunkt."""
    return f"{value:,.{digits}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def panel_grid_positions(count: int) -> list[tuple[int, int, int]]:
    """Rasterplätze der Animationsanzeigen als (Zeile, Spalte, Spaltenbreite).

    Höchstens zwei Spalten und höchstens zwei Zeilen: eine Anzeige füllt den
    gesamten Bereich, zwei stehen nebeneinander, drei und vier verteilen sich
    auf zwei je Zeile und zwei je Spalte. Alle belegten Zellen sind gleich groß;
    nur die einzelne Anzeige spannt bewusst über beide Spalten.
    """
    if count <= 1:
        return [(0, 0, 2)]
    return [(index // 2, index % 2, 1) for index in range(min(count, 4))]


#: Standardbildrate der Animation: die environment-eigene Rate
#: (`env.metadata["render_fps"] == 125`, also 1/dt mit dt = 0,008 s).
RENDER_FPS = 125
#: Die Obergrenze der Workbench von 120 läge unter dem Standardwert und machte
#: ihn selbst ungültig. 250 ist das Doppelte der environment-eigenen Rate.
MIN_ANIMATION_FPS, MAX_ANIMATION_FPS = 1, 250

#: Bis zu vier gleichrangige Verfahrensslots.
MAX_SLOTS = 4
SLOT_COUNTS = (2, 3, 4)
#: Startbelegung: die drei Algorithmen des Projekts nebeneinander, dazu ein
#: zweiter SAC-Slot. So steht ab Werk sowohl der Vergleich verschiedener
#: Verfahren als auch der zweier Parametrisierungen desselben Verfahrens da.
DEFAULT_SLOT_COUNT = 4
DEFAULT_SLOT_ALGORITHMS = ("PPO", "TD3", "SAC", "SAC")
#: Abweichungen vom Zoo-Profil, die **nur** die Startbelegung setzt: Ohne sie
#: wären die beiden SAC-Slots identisch und der Vergleich zeigte keinen
#: Unterschied. Später hinzugefügte Slots starten gemäß Workbench mit den
#: unveränderten Standardwerten ihres Algorithmus.
DEFAULT_SLOT_OVERRIDES: dict[int, dict[str, Any]] = {3: {"learning_rate": 6e-4}}
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
CURRENT_EPISODE, BEST_EPISODE = "aktuell", "beste"
EPISODE_CHOICES = (CURRENT_EPISODE, BEST_EPISODE)

#: Kurze Anzeigenamen; die ausführlichen Bezeichnungen stehen in
#: `FIELD_LABELS` der Logik und erscheinen in Fehlermeldungen und im
#: Unterschiedsblock der Summary.
SHORT_LABELS = {
    "total_timesteps": "Schritte N", "batch_size": "Batch B", "learning_rate": "Lernrate α",
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
#: Felder mit langen Auswahlwerten belegen eine eigene, volle Zeile.
WIDE_FIELDS = frozenset({"learning_rate_schedule", "action_noise"})

_TRAINING = ("total_timesteps", "batch_size", "gamma", "seed", "learning_rate",
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
         # Zoo-Profil schaltet sie für Hopper ab. Sie steht hier statt in der
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
#: Breite der drei Bedienspalten: zwei Parameterspalten tragen gemeinsam die
#: Verfahrenstabs, die dritte ist den Steuerungsbuttons vorbehalten.
CONTROL_COLUMNS = (360, 360, 226)
CONTROL_WIDTH = sum(CONTROL_COLUMNS) + 16
PARAMETER_WIDTH = CONTROL_COLUMNS[0] + CONTROL_COLUMNS[1]
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

    def __init__(self, config: HopperConfig, best_base: Path) -> None:
        self.workbench = HopperWorkbench(config)
        self.comparison: Optional[HopperWorkbench] = None
        self.history: list[EpisodeMetric] = []
        self.evaluations: list[tuple[int, EvaluationResult]] = []
        self.comparison_history: list[EpisodeMetric] = []
        self.comparison_evaluations: list[tuple[int, EvaluationResult]] = []
        self.best: Optional[EvaluationResult] = None
        self.best_base = best_base
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
        self.comparison_evaluations.clear()

    def reset_all(self, config: HopperConfig) -> None:
        self.workbench.close()
        self.workbench = HopperWorkbench(config)
        self.history.clear()
        self.evaluations.clear()
        self.best = None
        self.reset_comparison()

    def close(self) -> None:
        self.workbench.close()
        if self.comparison is not None:
            self.comparison.close()


class HopperGUI:
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
        self.renderers: list[Optional[HopperRenderer]] = [None] * MAX_SLOTS
        self.renderer_failed = False
        self.photos: list[Optional[ImageTk.PhotoImage]] = [None] * MAX_SLOTS
        self.photo_sizes: list[tuple[int, int]] = [(0, 0)] * MAX_SLOTS
        self.last_frames: list[Optional[np.ndarray]] = [None] * MAX_SLOTS
        self.animation_after: list[Optional[str]] = [None] * MAX_SLOTS
        self.animation_observation: list[Optional[np.ndarray]] = [None] * MAX_SLOTS
        self.animation_info: list[dict[str, Any]] = [{} for _ in range(MAX_SLOTS)]
        self.animation_action: list[Optional[np.ndarray]] = [None] * MAX_SLOTS
        self.animation_policy: list[Any] = [None] * MAX_SLOTS
        self.animation_live = [False] * MAX_SLOTS
        self.animation_step = [0] * MAX_SLOTS
        self.animation_episode = [0] * MAX_SLOTS
        self.animation_reward = [0.0] * MAX_SLOTS
        self.animation_from_best = [False] * MAX_SLOTS
        self.animation_interval = max(1, round(1000 / RENDER_FPS))
        self.hover_slot: Optional[int] = None
        self.last_plot = 0.0
        self.progress_steps: dict[Any, int] = {}
        self.progress_total = 1
        self.checkpoints = tempfile.TemporaryDirectory(prefix="hopper-")
        self.entries: list[dict[str, tk.Widget]] = [{} for _ in range(MAX_SLOTS)]
        self.slots = [
            SlotRuntime(self.initial_config(index), Path(self.checkpoints.name) / f"best_slot{index}")
            for index in range(MAX_SLOTS)
        ]
        self._export_snapshot_key: Optional[tuple[Any, ...]] = None
        self._export_stamp: Optional[str] = None
        self._variables()
        self._window()
        self._layout()
        self._sync_slot_widgets()
        self._update_active_label()
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._training_summary()
        self.root.after_idle(self._initialize_layout)
        self.root.after(100, lambda: self._show_initial_frame(0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # --------------------------------------------------------------- Variablen

    @staticmethod
    def initial_config(slot: int) -> HopperConfig:
        """Startkonfiguration eines Slots: Zoo-Profil plus Startabweichungen."""
        config = default_config(DEFAULT_SLOT_ALGORITHMS[slot])
        overrides = DEFAULT_SLOT_OVERRIDES.get(slot)
        return dataclasses.replace(config, **overrides) if overrides else config

    def _variables(self) -> None:
        self.algorithm_vars = [tk.StringVar(value=name) for name in DEFAULT_SLOT_ALGORITHMS]
        self.values: list[dict[str, tk.Variable]] = []
        for slot in range(MAX_SLOTS):
            variables: dict[str, tk.Variable] = {}
            for field in fields(HopperConfig):
                if field.name == "algorithm":
                    continue
                variables[field.name] = (tk.BooleanVar() if field.name in BOOLEAN_FIELDS
                                         else tk.StringVar())
            self.values.append(variables)
            self._fill_values(slot, self.initial_config(slot))
        self.slot_count_var = tk.StringVar(value=str(DEFAULT_SLOT_COUNT))
        self.evaluation_episodes = tk.StringVar(value=str(DEFAULT_EVALUATION_EPISODES))
        self.evaluation_interval = tk.StringVar(value=str(DEFAULT_EVALUATION_INTERVAL))
        self.animation_enabled = tk.BooleanVar(value=True)
        self.fps = tk.StringVar(value=str(RENDER_FPS))
        self.status = tk.StringVar(value="Bereit")
        self.active_label = tk.StringVar(value="")
        self.progress = tk.DoubleVar(value=0)
        # Unter dem Bild steht ausschließlich diese eine Zeile.
        self.captions = [tk.StringVar(value="Noch keine Episode gestartet.")
                         for _ in range(MAX_SLOTS)]
        # Je Animation unabhängig wählbar: der aktuelle Lernstand oder der der
        # besten bisherigen Episode.
        self.episode_choice = [tk.StringVar(value=CURRENT_EPISODE) for _ in range(MAX_SLOTS)]

    def _fill_values(self, slot: int, config: HopperConfig) -> None:
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
        self.root.title("Hopper Workbench")
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
        ttk.Label(header, text="Hopper Workbench",
                  font=("TkDefaultFont", 18, "bold")).pack(side="left")
        ttk.Label(header, text="PPO, SAC und TD3 – drei Gelenke, ein Ziel: aufrecht bleiben "
                               "und vorankommen").pack(side="left", padx=16)
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
        self.charts.add(training_tab, text="Training")
        self.charts.add(comparison_tab, text="Vergleich")
        # Beide Exportschaltflächen sitzen nebeneinander rechts in der
        # Tableiste: Dort ist die Fläche ohnehin frei, sie kosten keine eigene
        # Zeile und verdecken weder Kurven noch Legende. Ein gemeinsamer Rahmen
        # verteilt die Breite, damit sie sich nicht überlappen. `PNG` wirkt auf
        # das gerade sichtbare Diagramm, `TXT` auf die Summary daneben.
        export_bar = ttk.Frame(self.charts)
        export_bar.place(relx=1.0, y=1, anchor="ne", x=-2)
        self.export_button = ttk.Button(export_bar, text="PNG exportieren",
                                        command=self.export_chart, style="Compact.TButton")
        self.export_button.pack(side="left", padx=(0, 4))
        self.summary_button = ttk.Button(export_bar, text="TXT exportieren",
                                         command=self.export_summary, style="Compact.TButton")
        self.summary_button.pack(side="left")
        self.figure, self.axes, self.canvas = self._figure(training_tab)
        self.comparison_figure, self.comparison_axes, self.comparison_canvas = self._figure(comparison_tab)

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
        self.summary_text.configure(yscrollcommand=summary_y.set, xscrollcommand=summary_x.set,
                                    state="disabled")
        summary_y.grid(row=0, column=1, sticky="ns")
        summary_x.grid(row=1, column=0, sticky="ew")

    def _controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, minsize=CONTROL_COLUMNS[0])
        parent.columnconfigure(1, minsize=CONTROL_COLUMNS[1])
        parent.columnconfigure(2, minsize=CONTROL_COLUMNS[2], weight=1)
        parameters = ttk.Frame(parent, width=PARAMETER_WIDTH)
        parameters.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=(0, 4))
        actions = ttk.LabelFrame(parent, text="Steuerung", padding=6)
        actions.grid(row=0, column=2, sticky="new", padx=(4, 0))
        self._parameter_area(parameters)
        self._action_area(actions)

    def _parameter_area(self, parent: ttk.Frame) -> None:
        selection = ttk.LabelFrame(parent, text="Verfahren und Evaluation", padding=5)
        selection.pack(fill="x", pady=(0, 4))
        # Global, weil für einen fairen Vergleich in allen Läufen identisch.
        ttk.Label(selection, text="Anzahl Verfahren").grid(row=0, column=2, sticky="w", padx=(0, 4))
        self.slot_count_combo = ttk.Combobox(
            selection, textvariable=self.slot_count_var,
            values=[str(count) for count in SLOT_COUNTS], state="readonly", width=8, justify="right")
        self.slot_count_combo.grid(row=0, column=3, sticky="e")
        self.slot_count_combo.bind("<<ComboboxSelected>>", lambda _event: self._slot_count_changed())
        ttk.Label(selection, text="Eval-Episoden M").grid(row=1, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(selection, textvariable=self.evaluation_episodes, width=8,
                  justify="right").grid(row=1, column=3, sticky="e")
        ttk.Label(selection, text="Eval-Intervall").grid(row=2, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(selection, textvariable=self.evaluation_interval, width=8,
                  justify="right").grid(row=2, column=3, sticky="e")
        # Der Hinweis sitzt in der vierten Zeile neben dem Dropdown von
        # Verfahren 4: Der Block behält damit unabhängig von der Slotzahl
        # dieselbe Höhe und schiebt die Verfahrenstabs nicht nach unten.
        ttk.Label(selection, text="Intervall 0 = keine Auto-Evaluation.",
                  foreground=self.MUTED).grid(row=3, column=2, columnspan=2, sticky="w",
                                              pady=(2, 0))
        self.algorithm_combos: list[ttk.Combobox] = []
        self.algorithm_rows: list[tuple[ttk.Label, ttk.Combobox]] = []
        for slot, label in enumerate(SLOT_LABELS):
            name = ttk.Label(selection, text=label, foreground=SLOT_COLORS[slot])
            combo = ttk.Combobox(selection, textvariable=self.algorithm_vars[slot],
                                 values=ALGORITHMS, state="readonly", width=10, justify="right")
            combo.bind("<<ComboboxSelected>>", lambda _event, index=slot: self._algorithm_changed(index))
            self.algorithm_combos.append(combo)
            self.algorithm_rows.append((name, combo))
        selection.rowconfigure(MAX_SLOTS - 1, minsize=24)
        selection.columnconfigure(1, weight=1)
        selection.columnconfigure(3, weight=1)

        self.parameter_tabs = ttk.Notebook(parent)
        self.parameter_tabs.pack(fill="both", expand=True)
        self.tab_frames = [ttk.Frame(self.parameter_tabs, padding=(0, 4, 0, 0))
                           for _ in range(MAX_SLOTS)]
        self.parameter_tabs.bind("<<NotebookTabChanged>>", lambda _event: self._tab_changed())

    def _action_area(self, actions: ttk.LabelFrame) -> None:
        ttk.Label(actions, textvariable=self.active_label, foreground=self.ACCENT,
                  font=("TkDefaultFont", 10, "bold"), wraplength=CONTROL_COLUMNS[2] - 24).pack(
            fill="x", pady=(0, 4))
        self.buttons = []
        for text, command in (
            ("Training starten / fortsetzen", self.start_training),
            ("Stoppen", self.stop),
            ("Deterministisch evaluieren", self.start_evaluation),
            ("Sichtbare Episode abspielen", self.animate),
            ("Vergleich starten / fortsetzen", self.start_comparison),
            ("Bestes Modell wiederherstellen", self.restore_best),
            ("Neues Modell", self.reset),
        ):
            button = ttk.Button(actions, text=text, command=command)
            button.pack(fill="x", pady=1, ipady=1)
            self.buttons.append(button)
        self.stop_button, self.best_button = self.buttons[1], self.buttons[5]
        self.stop_button.configure(state="disabled")
        self.best_button.configure(state="disabled")
        ttk.Checkbutton(actions, text="Animation zeigen", variable=self.animation_enabled,
                        command=self._animation_toggled).pack(fill="x", pady=(4, 0))
        rate = ttk.Frame(actions)
        rate.pack(fill="x", pady=(2, 0))
        ttk.Label(rate, text="Bildrate (FPS)").pack(side="left")
        ttk.Entry(rate, textvariable=self.fps, width=6, justify="right").pack(side="right")
        ttk.Progressbar(actions, variable=self.progress, maximum=100).pack(fill="x", pady=(4, 2))
        ttk.Label(actions, textvariable=self.status, foreground=self.ACCENT,
                  wraplength=CONTROL_COLUMNS[2] - 24).pack(fill="x", pady=1)

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
        """Zwei Label/Feld-Paare je Zeile; breite Auswahlfelder eine volle Zeile."""
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
            elif name in WIDE_FIELDS:
                if position:
                    row, position = row + 1, 0
                ttk.Label(group, text=SHORT_LABELS[name]).grid(row=row, column=0, sticky="w", padx=(0, 4))
                widget = ttk.Combobox(group, textvariable=variable, values=CHOICE_FIELDS[name],
                                      state="readonly", width=16, justify="right")
                widget.grid(row=row, column=1, columnspan=3, sticky="ew", pady=1)
                row += 1
            else:
                column = position * 2
                ttk.Label(group, text=SHORT_LABELS[name]).grid(row=row, column=column, sticky="w", padx=(0, 4))
                if name in CHOICE_FIELDS:
                    widget = ttk.Combobox(group, textvariable=variable, values=CHOICE_FIELDS[name],
                                          state="readonly", width=9, justify="right")
                else:
                    widget = ttk.Entry(group, textvariable=variable, width=8, justify="right")
                widget.grid(row=row, column=column + 1, sticky="e", padx=(0, 6), pady=1)
                position += 1
                if position == 2:
                    row, position = row + 1, 0
            entries[name] = widget
        group.columnconfigure((1, 3), weight=1)
        return entries

    # -------------------------------------------------------------- Animation

    def _environment(self, parent: ttk.Frame) -> None:
        """Je Slot ein Anzeigefeld im Raster mit höchstens zwei Spalten und
        zwei Zeilen. Sichtbar ist normalerweise nur das aktive Verfahren; im
        Vergleich stehen alle aktiven Verfahren gleichzeitig da."""
        self.env_container = ttk.Frame(parent)
        self.env_container.grid(row=0, column=1, sticky="nsew")
        self.env_frames, self.image_labels, self.caption_labels = [], [], []
        self.episode_combos: list[ttk.Combobox] = []
        for slot, label in enumerate(SLOT_LABELS):
            frame = ttk.LabelFrame(self.env_container, text=label, padding=5)
            # Die Wahl steht über dem Bild: Unter dem Bild bleibt ausschließlich
            # die eine Zeile mit Episode, Schritt, Verfahren und Return.
            chooser = ttk.Frame(frame)
            chooser.pack(fill="x")
            ttk.Label(chooser, text="Episode").pack(side="left")
            combo = ttk.Combobox(chooser, textvariable=self.episode_choice[slot],
                                 values=EPISODE_CHOICES, state="readonly", width=8,
                                 justify="right")
            combo.pack(side="right")
            combo.bind("<<ComboboxSelected>>",
                       lambda _event, index=slot: self._episode_choice_changed(index))
            self.episode_combos.append(combo)
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
        live = [slot for slot in range(self.slot_count) if self.animation_live[slot]]
        return live if live else [self.active_slot]

    def show_panels(self, slots: list[int]) -> None:
        """Blendet genau diese Anzeigen ein. Wird auch vom Layout-Test genutzt."""
        for slot in range(MAX_SLOTS):
            self.animation_live[slot] = slot in slots
        self._sync_panels()

    def _sync_panels(self) -> None:
        """Ordnet die sichtbaren Anzeigefelder im Raster an."""
        slots = self._panel_slots()
        positions = panel_grid_positions(len(slots))
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
        for row in range(2):
            used = row in used_rows
            self.env_container.rowconfigure(row, weight=1 if used else 0,
                                            uniform="cell" if used else "")
        for column in range(2):
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

    def _panel_configs(self) -> list[HopperConfig]:
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
        for slot, (label, combo) in enumerate(self.algorithm_rows):
            if slot < self.slot_count:
                label.grid(row=slot, column=0, sticky="w", padx=(0, 4), pady=1)
                combo.grid(row=slot, column=1, sticky="ew", padx=(0, 10), pady=1)
            else:
                label.grid_remove()
                combo.grid_remove()
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
                              ("Summary-Export", self.summary_button)):
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
            self.best_button.configure(
                state="normal" if self.slots[self.active_slot].best is not None else "disabled")
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
        if slot == self.active_slot:
            self.best_button.configure(state="disabled")
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._training_summary()
        self.status.set(f"Bereit – {SLOT_LABELS[slot]} auf {algorithm} gesetzt (Zoo-Profil geladen)")

    def _config(self, slot: int) -> HopperConfig:
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
        config = HopperConfig(algorithm=self.algorithm_vars[slot].get(), **data)
        try:
            config.validate()
        except ValueError as error:
            raise ValueError(f"{SLOT_LABELS[slot]}: {error}") from error
        return config

    def _evaluation_settings(self) -> tuple[int, int]:
        try:
            episodes = int(self.evaluation_episodes.get().strip())
        except ValueError as error:
            raise ValueError(
                f"Eval-Episoden M: '{self.evaluation_episodes.get()}' ist keine ganze Zahl. "
                "Gültig: ≥ 1."
            ) from error
        if episodes <= 0:
            raise ValueError(f"Eval-Episoden M: '{episodes}' ist ungültig. Gültig: ganze Zahl ≥ 1.")
        try:
            interval = int(self.evaluation_interval.get().strip())
        except ValueError as error:
            raise ValueError(
                f"Eval-Intervall: '{self.evaluation_interval.get()}' ist keine ganze Zahl. "
                "Gültig: ≥ 0."
            ) from error
        if interval < 0:
            raise ValueError(f"Eval-Intervall: '{interval}' ist ungültig. Gültig: ganze Zahl ≥ 0.")
        return episodes, interval

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
            if self.slots[self.active_slot].best is None:
                self.best_button.configure(state="disabled")
        self._update_active_label()

    # ----------------------------------------------------------------- Training

    def start_training(self) -> None:
        if self.busy:
            return
        slot = self.active_slot
        runtime = self.slots[slot]
        try:
            config = self._config(slot)
            episodes, interval = self._evaluation_settings()
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
            runtime.workbench = HopperWorkbench(config)
            runtime.history.clear()
            runtime.evaluations.clear()
            runtime.best = None
        else:
            runtime.workbench.config = config
        self.running_slot = slot
        self.stop_event.clear()
        self.progress.set(0)
        self.progress_steps = {("single", slot): 0}
        self.progress_total = config.total_timesteps
        self.charts.select(0)
        self._set_busy(True, f"Läuft – {SLOT_LABELS[slot]} ({config.algorithm}) über "
                             f"{thousands(config.total_timesteps)} Schritte")
        self.worker = threading.Thread(target=self._train_worker, args=(slot, episodes, interval),
                                       daemon=True)
        self.worker.start()
        self.root.after(50, self._poll)
        self._start_live_animation([slot])

    def _train_worker(self, slot: int, episodes: int, interval: int) -> None:
        try:
            self.slots[slot].workbench.train(
                self.stop_event, self.events, ("single", slot), evaluation_interval=interval,
                evaluation_episodes=episodes,
                best_callback=lambda _episode, _steps, result: self._save_best(slot, result),
            )
            self.events.put(("training_done", self.stop_event.is_set()))
        except Exception as error:
            self.events.put(("error", error))

    # --------------------------------------------------------------- Vergleich

    def start_comparison(self) -> None:
        if self.busy:
            return
        try:
            configs = [self._config(slot) for slot in range(self.slot_count)]
            episodes, interval = self._evaluation_settings()
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
                runtime.comparison = HopperWorkbench(config)
            else:
                runtime.comparison.config = config
        self.stop_event.clear()
        self.progress.set(0)
        self.comparison_running = True
        self.progress_steps = {("compare", slot): 0 for slot in range(self.slot_count)}
        self.progress_total = sum(config.total_timesteps for config in configs)
        self.charts.select(1)
        self._set_busy(True, "Läuft – Vergleich: "
                             + " gegen ".join(f"{SLOT_SHORT[slot]} {config.algorithm}"
                                              for slot, config in enumerate(configs))
                             + f" über {thousands(self.progress_total)} Schritte")
        self._refresh_comparison_plot()
        self._comparison_summary()
        self.worker = threading.Thread(target=self._comparison_worker, args=(episodes, interval),
                                       daemon=True)
        self.worker.start()
        self.root.after(50, self._poll)
        self._start_live_animation(list(range(self.slot_count)))

    def _comparison_worker(self, episodes: int, interval: int) -> None:
        errors: queue.Queue = queue.Queue()
        slots = list(range(self.slot_count))
        barrier = threading.Barrier(len(slots))

        def run(slot: int) -> None:
            try:
                barrier.wait()
                self.slots[slot].comparison.train(
                    self.stop_event, self.events, ("compare", slot),
                    evaluation_interval=interval, evaluation_episodes=episodes,
                )
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

    # -------------------------------------------------------------- Evaluation

    def start_evaluation(self) -> None:
        if self.busy:
            return
        slot = self.active_slot
        if self.slots[slot].workbench.model is None:
            messagebox.showinfo("Keine Evaluation",
                                f"Trainiere zuerst ein Modell für {SLOT_LABELS[slot]}.",
                                parent=self.root)
            return
        try:
            episodes, _ = self._evaluation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültige Evaluation", str(error), parent=self.root)
            return
        self.running_slot = slot
        self._set_busy(True, f"Läuft – deterministische Evaluation ({SLOT_LABELS[slot]})")
        self.worker = threading.Thread(target=self._evaluation_worker, args=(slot, episodes),
                                       daemon=True)
        self.worker.start()
        self.root.after(50, self._poll)

    def _evaluation_worker(self, slot: int, episodes: int) -> None:
        runtime = self.slots[slot]
        try:
            result = runtime.workbench.evaluate(episodes, runtime.workbench.config.seed or 0)
            self._save_best(slot, result)
            self.events.put(("manual_evaluation", (slot, len(runtime.history), result)))
        except Exception as error:
            self.events.put(("error", error))

    def _save_best(self, slot: int, result: EvaluationResult) -> None:
        runtime = self.slots[slot]
        with runtime.lock:
            if runtime.best is None or result.mean_reward > runtime.best.mean_reward:
                runtime.workbench.save(runtime.best_base)
                runtime.best = result

    def restore_best(self) -> None:
        if self.busy:
            return
        slot = self.active_slot
        runtime = self.slots[slot]
        if runtime.best is None:
            return
        try:
            restored = HopperWorkbench.load(runtime.best_base, runtime.algorithm)
        except Exception as error:
            messagebox.showerror("Wiederherstellen fehlgeschlagen", str(error), parent=self.root)
            return
        runtime.workbench.close()
        runtime.workbench = restored
        self.status.set(f"Abgeschlossen – bestes Modell von {SLOT_LABELS[slot]} wiederhergestellt")
        self._training_summary()

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
            elif kind == "evaluation":
                (mode, slot), episode, _steps, result = payload
                if mode == "single":
                    self.slots[slot].evaluations.append((episode, result))
                    redraw_training = True
                else:
                    self.slots[slot].comparison_evaluations.append((episode, result))
                    redraw_comparison = True
            elif kind == "progress":
                series, elapsed = payload
                self.progress_steps[series] = max(elapsed, self.progress_steps.get(series, 0))
                done = sum(self.progress_steps.values())
                self.progress.set(min(100, 100 * done / max(1, self.progress_total)))
            elif kind == "training_done":
                self._set_busy(False, "Gestoppt – Training" if payload else "Abgeschlossen – Training")
                redraw_training = True
            elif kind == "comparison_done":
                self._set_busy(False, "Gestoppt – Vergleich" if payload else "Abgeschlossen – Vergleich")
                redraw_comparison = True
            elif kind == "manual_evaluation":
                slot, episode, result = payload
                self.slots[slot].evaluations.append((episode, result))
                self._set_busy(False, "Abgeschlossen – Evaluation")
                redraw_training = True
            elif kind == "error":
                self._set_busy(False, "Fehler")
                messagebox.showerror("Fehler", str(payload), parent=self.root)
        if self.slots[self.active_slot].best is not None:
            self.best_button.configure(state="disabled" if self.busy else "normal")
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
        """Y-Achse an den Daten ausrichten.

        Die Gelöst-Marke von 3800 liegt bei einem Standardlauf weit über den
        Messwerten. Würde die Achse sie erzwingen, wären alle Kurven unten
        zusammengedrückt; die Legende nennt den Wert stattdessen ausdrücklich.
        """
        if not values:
            return
        low, high = min(values), max(values)
        padding = max(1.0, .08 * (high - low))
        axes.set_ylim(low - padding, high + padding)

    def _reference_line(self, axes: Any) -> None:
        axes.axhline(SOLVED_RETURN, color=REFERENCE_COLOR, linestyle=REFERENCE_STYLE,
                     linewidth=self.LINE_WIDTH, label=f"Gelöst ab {german(SOLVED_RETURN, 0)}")

    def _series_label(self, slot: int, configs: Optional[list[HopperConfig]] = None,
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

    def _refresh_training_plot(self) -> None:
        self.axes.clear()
        slot = self.active_slot
        runtime = self.slots[slot]
        scale: list[float] = []
        if runtime.history:
            episodes = [item.episode for item in runtime.history]
            rewards = [item.reward for item in runtime.history]
            x, y = downsample_minmax(episodes, rewards)
            self.axes.plot(x, y, color=SLOT_COLORS[slot], alpha=self.RAW_ALPHA, linewidth=.8)
            self.axes.plot(episodes, rolling_average(rewards), color=SLOT_COLORS[slot],
                           linestyle=SLOT_LINESTYLE, linewidth=self.LINE_WIDTH,
                           label=f"Training – {SLOT_LABELS[slot]} ({runtime.algorithm})")
            scale.extend(rewards)
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
            runtime = self.slots[slot]
            if not runtime.comparison_history:
                continue
            episodes = [item.episode for item in runtime.comparison_history]
            rewards = [item.reward for item in runtime.comparison_history]
            x, y = downsample_minmax(episodes, rewards)
            # Rohwerte dezent in derselben Slotfarbe, damit sie die kräftige
            # Kurve des gleitenden Durchschnitts nicht überdecken.
            self.comparison_axes.plot(x, y, color=SLOT_COLORS[slot], alpha=self.RAW_ALPHA,
                                      linewidth=.8)
            self.comparison_axes.plot(episodes, rolling_average(rewards), color=SLOT_COLORS[slot],
                                      linestyle=SLOT_LINESTYLE, linewidth=self.LINE_WIDTH,
                                      label=self._series_label(slot, configs))
            scale.extend(rewards)
        self._reference_line(self.comparison_axes)
        self._limit_to_data(self.comparison_axes, scale)
        self.comparison_axes.set(xlabel="Episode", ylabel="Return")
        self._style(self.comparison_axes)
        self.comparison_canvas.draw_idle()
        # Anzeigetitel und Legende stammen aus derselben Quelle.
        self._update_panel_titles()

    # ---------------------------------------------------------------- Summary

    def _write_summary(self, text: str) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")

    def _summary_text(self) -> str:
        return self.summary_text.get("1.0", "end").rstrip("\n")

    def _table(self, title: str, labels: tuple[str, ...], statistics: dict[str, tuple[Any, ...]],
               footer: list[str]) -> None:
        heading = [title, ""] if title else []
        if not statistics:
            # Der Unterschiedsblock bleibt auch ohne Messwerte sichtbar: er
            # erklärt, was verglichen wird, bevor der erste Lauf startet.
            lines = [*heading, "Noch keine vollständig abgeschlossene Episode.", *footer]
            self._write_summary("\n".join(lines))
            return
        names = list(statistics)
        label_width = max(len(label) for label in labels) + 1
        value_width = max(16, max(len(name) for name in names) + 2)
        header = "Statistik".ljust(label_width) + "".join(name.rjust(value_width) for name in names)
        lines = [*heading, header, "─" * len(header)]
        for index, label in enumerate(labels):
            lines.append(label.ljust(label_width) + "".join(
                str(statistics[name][index]).rjust(value_width) for name in names))
        if footer:
            lines.extend(["", *footer])
        self._write_summary("\n".join(lines))

    @staticmethod
    def best_episode(history: list[EpisodeMetric]) -> Optional[EpisodeMetric]:
        """Episode mit dem höchsten explorativen Return."""
        return max(history, key=lambda item: item.reward) if history else None

    @classmethod
    def _statistics(cls, history: list[EpisodeMetric], steps: int, budget: int) -> tuple[Any, ...]:
        if not history:
            return (0, thousands(steps), thousands(budget), "—", "—", "—", "—", "—", "—")
        best = cls.best_episode(history)
        return (
            len(history), thousands(steps), thousands(budget),
            german(float(np.mean([item.reward for item in history]))),
            f"#{best.episode}: {german(best.reward)}",
            german(float(np.mean([item.length for item in history])), 0),
            f"{np.mean([item.survived for item in history]):.1%}".replace(".", ","),
            f"{np.mean([item.fell for item in history]):.1%}".replace(".", ","),
            f"{np.mean([item.solved for item in history]):.1%}".replace(".", ","),
        )

    STAT_LABELS = ("Episoden", "Schritte", "Budget", "Ø Return", "Beste Episode", "Ø Länge",
                   "Durchhaltequote", "Sturzquote", "Gelöst-Quote")

    def _evaluation_footer(self, prefix: str, result: EvaluationResult) -> str:
        return (f"{prefix}: Ø {german(result.mean_reward)} ± {german(result.reward_std)} | "
                f"Länge {german(result.mean_length, 0)} | "
                f"durchgehalten {result.survive_rate:.0%} | gelöst {result.solved_rate:.0%} | "
                f"v̄ₓ {german(result.mean_speed, 2)} m/s | Strecke {german(result.mean_distance, 2)} m")

    def _training_summary(self) -> None:
        slot = self.active_slot
        runtime = self.slots[slot]
        model = runtime.workbench.model
        stats = self._statistics(runtime.history, model.num_timesteps if model else 0,
                                 runtime.workbench.config.total_timesteps)
        footer = ["Höhere Returns sind besser; gelöst ab "
                  f"{german(SOLVED_RETURN, 0)}. Kurvenpunkte entstehen nur für "
                  "vollständig abgeschlossene Episoden.",
                  f"Durchhalten = volle {MAX_EPISODE_STEPS} Schritte (truncated), "
                  "Sturz = ungesund geworden (terminated)."]
        if runtime.evaluations:
            footer.append(self._evaluation_footer("Letzte Evaluation", runtime.evaluations[-1][1]))
        if runtime.best is not None:
            footer.append(self._evaluation_footer("Beste Evaluation", runtime.best)
                          + "  (als Checkpoint gesichert)")
        self._table(f"Training – {SLOT_LABELS[slot]} ({runtime.algorithm})", self.STAT_LABELS,
                    {SLOT_LABELS[slot]: stats}, footer)

    def _comparison_summary(self) -> None:
        statistics: dict[str, tuple[Any, ...]] = {}
        configs = []
        for slot in range(self.slot_count):
            runtime = self.slots[slot]
            config = runtime.comparison.config if runtime.comparison else runtime.workbench.config
            configs.append(config)
            model = runtime.comparison.model if runtime.comparison else None
            statistics[SLOT_LABELS[slot]] = (config.algorithm, *self._statistics(
                runtime.comparison_history, model.num_timesteps if model else 0,
                config.total_timesteps))
        if not any(self.slots[slot].comparison_history for slot in range(self.slot_count)):
            statistics = {}
        footer = [f"Höhere Returns sind besser; gelöst ab {german(SOLVED_RETURN, 0)}."]
        for slot in range(self.slot_count):
            points = self.slots[slot].comparison_evaluations
            if points:
                footer.append(self._evaluation_footer(
                    f"{SLOT_SHORT[slot]} letzte Evaluation", points[-1][1]))
        footer.extend(self._difference_lines(configs))
        # Ohne Überschrift: Die Spaltenköpfe sagen bereits, was verglichen wird.
        self._table("", ("Algorithmus", *self.STAT_LABELS), statistics, footer)

    def _difference_lines(self, configs: list[HopperConfig]) -> list[str]:
        """Ohne diesen Block wäre ein Vergleich mehrerer Parametrisierungen
        desselben Algorithmus nicht interpretierbar."""
        differences = config_differences(configs)
        note = ([] if len({config.algorithm for config in configs}) == 1 else
                ["  (nur gemeinsame Parameter; verfahrenseigene stehen im jeweiligen Tab)"])
        if not differences:
            return ["", "Unterschiede: keine – alle Slots sind gleich konfiguriert.", *note]
        shorts = SLOT_SHORT[:len(configs)]
        width = max(len(name) for name, _ in differences)
        column = max(10, max(len(value) for _, values in differences for value in values) + 2)
        lines = ["", "Unterschiede:", *note,
                 "  " + "".ljust(width) + "".join(short.rjust(column) for short in shorts)]
        lines.extend("  " + name.ljust(width) + "".join(value.rjust(column) for value in values)
                     for name, values in differences)
        return lines

    # ----------------------------------------------------------------- Export

    def _comparison_visible(self) -> bool:
        return self.charts.index(self.charts.select()) == 1

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
        lines.append(f"evaluation_episodes: {self.evaluation_episodes.get()}")
        lines.append(f"evaluation_interval: {self.evaluation_interval.get()}")
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
        return f"hopper_{slug}_{self._export_stamp}"

    def export_chart(self) -> None:
        EXPORT_DIR.mkdir(exist_ok=True)
        comparison = self._comparison_visible()
        figure = self.comparison_figure if comparison else self.figure
        suffix = "vergleich" if comparison else "training"
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

    def _renderer(self, slot: int) -> HopperRenderer:
        """Renderprozess des Slots.

        Jede sichtbare Anzeige läuft in einem eigenen Prozess mit eigener
        Environment-Instanz; Environment-Instanzen werden nicht geteilt.
        """
        if self.renderers[slot] is None:
            self.renderers[slot] = HopperRenderer(seed=self.slots[slot].workbench.config.seed)
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
        self.animation_enabled.set(False)
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

    def _animation_workbench(self, slot: int) -> HopperWorkbench:
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
        lines = ["Action (Moment = aᵢ · gear 200)"]
        action = self.animation_action[slot]
        if action is None:
            lines.append("  noch keine Action gewählt")
        else:
            for index, joint in enumerate(action_readout(action)["joints"]):
                lines.append(f"  a{index} {joint['raw']:+.3f}  {joint['joint']:<12} {joint['text']}")
        angles, speeds = values["joint_angles"], values["joint_velocities"]
        lines += [
            "",
            "Zustand (gesund: Höhe > 0,70 · Winkel ±0,20)",
            f"  Höhe        {values['height']:+.3f} m"
            f"{'' if values['healthy_height'] else '   ← ungesund'}",
            f"  Rumpfwinkel {values['torso_angle']:+.3f} rad ({values['torso_angle_degrees']:+.1f}°)"
            f"{'' if values['healthy_angle'] else '  ← ungesund'}",
            f"  Winkel      Hüfte {angles[0]:+.3f} Knie {angles[1]:+.3f} Fuß {angles[2]:+.3f}",
            f"  Tempo       vₓ {values['vx']:+.3f}  v_z {values['vz']:+.3f}  "
            f"ω {values['torso_angular_velocity']:+.3f}",
            f"  Gelenktempo {speeds[0]:+.3f} {speeds[1]:+.3f} {speeds[2]:+.3f}",
            "  Alle sechs Tempi sind auf ±10 geclippt"
            f"{' – Grenze erreicht!' if values['velocity_clipped'] else '.'}",
            "",
            "Reward des letzten Schritts",
            f"  Überleben {rewards['survive']:+.3f}  vorwärts {rewards['forward']:+.3f}",
            f"  Steuerkosten {rewards['ctrl']:+.4f}",
            f"  Strecke {rewards['x_position']:+.3f} m (aus info)",
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
        positions = panel_grid_positions(max(1, len(self._panel_slots())))
        # Belegte Zeilen und Spalten zählen, nicht die Rasterbreite: Die einzelne
        # Anzeige spannt über beide Spalten und darf deshalb den vollen Platz
        # bekommen.
        rows = len({row for row, _, _ in positions})
        columns = len({column for _, column, _ in positions})
        # Alles, was in der Zelle neben dem Bild Platz braucht: die Zeile
        # darüber, die Beschriftung darunter, Rahmen und Innenabstand. Geraten
        # werden darf das nicht – eine zu kleine Schätzung macht das Bild zu
        # hoch und schneidet die Beschriftung ab.
        chrome = (self.caption_labels[slot].winfo_reqheight()
                  + self.episode_combos[slot].winfo_reqheight() + 34)
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
            if self.animation_enabled.get():
                self._animate_step(slot)
            else:
                self._run_episode_without_animation(slot)

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
        best = workbench.best_snapshot() if self.episode_choice[slot].get() == BEST_EPISODE \
            else None
        configured_seed = self.slots[slot].workbench.config.seed
        if best is not None:
            # Der Lernstand der besten Episode, mit festem Seed – so sieht die
            # Wiederholung jedes Mal gleich aus.
            self._apply_policy_state(slot, model, best[1])
            episode, seed = best[0].episode, configured_seed
        elif live:
            self._refresh_snapshot(slot, model)
            # Laufende Episoden starten ohne festen Seed, damit nacheinander
            # gezeigte Episoden nicht identisch aussehen.
            episode, seed = self.trained_episodes(slot), None
        else:
            self.animation_policy[slot] = None
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
        Vergleichslauf ein."""
        if not self.animation_enabled.get():
            return
        slots = [slot for slot in slots if not self.animation_live[slot]]
        if not slots:
            return
        for slot in slots:
            self.animation_live[slot] = True
        self._sync_panels()
        for slot in slots:
            self._live_episode(slot)

    def _episode_choice_changed(self, slot: int) -> None:
        """Die Wahl wirkt sofort: Die sichtbare Episode beginnt neu."""
        if self.episode_choice[slot].get() == BEST_EPISODE \
                and self._animation_workbench(slot).best_snapshot() is None:
            self.status.set(f"{SLOT_LABELS[slot]}: noch keine beste Episode – "
                            "es läuft weiter der aktuelle Lernstand")
        if self._live_active(slot):
            self._stop_animation(slot)
            self._live_episode(slot)

    def _live_active(self, slot: int) -> bool:
        return self.animation_live[slot] and self.busy and self.animation_enabled.get()

    def _live_slots(self) -> list[int]:
        """Slots, deren laufendes Training gerade sichtbar gemacht werden kann."""
        if not self.busy:
            return []
        if self.comparison_running:
            return list(range(self.slot_count))
        return [] if self.running_slot is None else [self.running_slot]

    def _animation_toggled(self) -> None:
        """Der Schalter wirkt auch mitten in einem laufenden Lauf."""
        if not self.busy:
            return
        if self.animation_enabled.get():
            self._start_live_animation(self._live_slots())
        else:
            self._stop_live_animation()

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
                self.status.set(f"Animation gestoppt – {error}")
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
        # Bei aktiver Normalisierung muss die Policy die Beobachtung so sehen
        # wie im Training – mit eingefrorenen Statistiken.
        observation = self._animation_workbench(slot).policy_observation(
            self.animation_observation[slot])
        policy = self.animation_policy[slot]
        if policy is not None:
            action, _ = policy.predict(observation, deterministic=True)
        else:
            action, _ = self._animation_model(slot).predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    @staticmethod
    def _episode_outcome_text(terminated: bool) -> str:
        """`Hopper-v5` kennt keine Terminalstrafe: `terminated` heißt immer
        „ungesund geworden", `truncated` heißt „volle 1000 Schritte geschafft"."""
        if terminated:
            return "Abgeschlossen – gestürzt (ungesund geworden)"
        return f"Abgeschlossen – {MAX_EPISODE_STEPS} Schritte durchgehalten"

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
        if not self.animation_enabled.get():
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
        if not messagebox.askyesno("Neues Modell",
                                   f"Lernzustand von {SLOT_LABELS[slot]} verwerfen?",
                                   parent=self.root):
            return
        try:
            config = self._config(slot)
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root)
            return
        self.slots[slot].reset_all(config)
        self.best_button.configure(state="disabled")
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._training_summary()
        self.status.set(f"Bereit – {SLOT_LABELS[slot]} zurückgesetzt")

    def instructions(self) -> None:
        messagebox.showinfo(
            "Bedienungsanleitung",
            "Ablauf\n"
            "1. 'Anzahl Verfahren' legt fest, wie viele Slots aktiv sind (2 bis 4,\n"
            f"   Standard {DEFAULT_SLOT_COUNT}). Für jeden Slot einen Algorithmus wählen – gern\n"
            "   mehrfach denselben, um Parametrisierungen zu vergleichen.\n"
            "2. Im jeweiligen Tab die Parameter setzen. Der sichtbare Tab ist das aktive\n"
            "   Verfahren; alle Einzellauf-Buttons wirken darauf.\n"
            "3. 'Training starten / fortsetzen' trainiert das aktive Verfahren,\n"
            "   'Vergleich starten / fortsetzen' alle aktiven Slots parallel.\n\n"
            "Environment\n"
            "Hopper-v5 (MuJoCo): a₀ bis a₂ sind Drehmomente für Hüfte, Knie und\n"
            "Sprunggelenk, jeweils -1 bis +1 mal gear = 200 N·m. Der Reward ist\n"
            "1,0 fürs Überleben + 1,0·vₓ - 0,001·Σaᵢ². Es gibt weder Terminalbonus\n"
            "noch Terminalstrafe: Ein Sturz kostet nur die Rewards der Schritte, die\n"
            "nicht mehr stattfinden. Die Episode endet mit 'terminated', sobald der\n"
            "Roboter ungesund wird (Höhe ≤ 0,7 m oder Rumpfwinkel außerhalb ±0,2), und\n"
            f"mit 'truncated' nach {MAX_EPISODE_STEPS} Schritten – Letzteres ist hier der\n"
            f"gute Ausgang. Gelöst ab Return {german(SOLVED_RETURN, 0)}.\n\n"
            "Methoden\n"
            "PPO ist on-policy: Rollouts, GAE, geclipptes Ziel, kein Replay Buffer.\n"
            "SAC und TD3 sind off-policy mit zwei Critics – TD3 mit deterministischem\n"
            "Actor und Action Noise, SAC mit stochastischem Actor und gelernter Entropie.\n"
            "Bei gleichem Schrittbudget bevorteilt der Vergleich die Off-Policy-Verfahren\n"
            "systematisch: Sie lernen aus jedem gespeicherten Übergang mehrfach, PPO\n"
            "verwirft seine Daten nach jedem Update. Training exploriert, Evaluation ist\n"
            "deterministisch und ohne Lernupdates.\n\n"
            "Ansichten und Animation\n"
            "'Training' zeigt das aktive Verfahren, 'Vergleich' alle Slots samt ihrer\n"
            "Unterschiede in der Summary. Die Kurven sind durchgezogen und farbig nach\n"
            "Slot: V1 blau, V2 rot, V3 gelb, V4 grün; die weiße gestrichelte Linie ist\n"
            "die Gelöst-Marke. Sie liegt bei einem Standardlauf weit über den Daten, die\n"
            "Y-Achse folgt deshalb den Messwerten.\n"
"Beide Graphen zeigen nur den explorativen Episoden-Return; die\n"
            "deterministischen Evaluationen stehen in der Summary, zusammen mit der\n"
            "besten Episode je Slot.\n"
            "Über jedem Animationsbild steht ein eigenes Feld 'Episode': 'aktuell' zeigt\n"
            "den laufenden Lernstand und beschriftet sich mit der zuletzt trainierten\n"
            "Episodennummer – derselben wie auf der X-Achse. 'beste' spielt den Stand der\n"
            "bisher besten Episode mit festem Seed erneut ab; ein '*' hinter der Nummer\n"
            "macht das kenntlich. Gesichert wird je Slot nur dieser eine zusätzliche\n"
            "Stand. Jede Animation wählt unabhängig von den anderen.\n"
            "Unter jedem Animationsbild stehen nur Episode, Schritt, Verfahren und\n"
            "Return. Alle weiteren Messwerte – Action, Zustand, Reward-Zerlegung –\n"
            "erscheinen erst, wenn der Mauszeiger über dem Bild steht; sie werden dann\n"
            "neben dem Bild eingeblendet, sodass beides gleichzeitig sichtbar bleibt.\n"
            "'Animation zeigen' wirkt jederzeit, auch im laufenden Lauf, und zeigt dann\n"
            "alle aktiven Verfahren gleichzeitig – jedes in einem eigenen Renderprozess.\n"
            f"Das kostet Rechenzeit; die Bildrate (Standard {RENDER_FPS} FPS) ist eine\n"
            "Obergrenze und wird bei vier Anzeigen selten erreicht.\n\n"
            "Laufzeit\n"
            f"Der Standardwert von {thousands(DEFAULT_TOTAL_TIMESTEPS)} Schritten ist das Budget der\n"
            "Zoo-Profile. Ein solcher Lauf dauert auf einem Laptop Stunden – bei mehreren\n"
            "Slots parallel länger, mit Animation zusätzlich. Für einen ersten Eindruck\n"
            "'Trainingsschritte N' auf etwa 100.000 und 'Eval-Intervall' auf 10.000\n"
            "setzen; wer das Budget verkleinert, muss das Intervall mitverkleinern, sonst\n"
            "gibt es keine Stützstelle. Jeder Off-Policy-Slot belegt mit dem vollen\n"
            "Replay Buffer rund 210 MB.\n\n"
            "Kein Lernerfolg?\n"
            "Zu kurzes Budget, zu großer Lernstart t₀, zu kleines Action Noise bei TD3,\n"
            "eine zu hohe Lernrate oder bei PPO fehlende Normalisierung. Selbst mit dem\n"
            f"vollen Budget erreicht laut Zoo-Benchmark keines der Verfahren im Mittel die\n"
            f"{german(SOLVED_RETURN, 0)}: PPO rund 2.410, SAC rund 2.326, TD3 rund 3.606.",
            parent=self.root,
        )

    def close(self) -> None:
        self.stop_event.set()
        self._stop_animation()
        for slot in range(MAX_SLOTS):
            self._close_renderer(slot)
        for runtime in self.slots:
            runtime.close()
        self.checkpoints.cleanup()
        self.root.destroy()
