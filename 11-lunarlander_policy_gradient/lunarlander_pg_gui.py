"""Dark Tkinter GUI for the LunarLander PPO/TD3/SAC workbench."""

from __future__ import annotations

import queue
import re
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from lunarlander_pg_logic import (
    ALGORITHMS, BOOLEAN_FIELDS, CHOICE_FIELDS, INTEGER_FIELDS, OFF_POLICY_ALGORITHMS,
    OPTIONAL_FLOAT_FIELDS, SOLVED_RETURN, TUPLE_FIELDS,
    EpisodeMetric, EvaluationResult, LunarLanderPGConfig, LunarLanderPGWorkbench,
    action_readout, config_differences, default_config, observation_readout,
)
from lunarlander_pg_render import LunarLanderRenderer


def rolling_average(values: list[float], window: int = 20) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.asarray([array[max(0, index - window + 1):index + 1].mean() for index in range(len(array))])


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


#: Standardbildrate der Animation: die environment-eigene Rate
#: (`env.metadata["render_fps"] == 50`). Sie ist über das Feld `Bildrate (FPS)`
#: einstellbar und steuert ausschließlich die Darstellung, nie das Environment.
RENDER_FPS = 50
MIN_ANIMATION_FPS, MAX_ANIMATION_FPS = 1, 120

SLOT_LABELS = ("Verfahren 1", "Verfahren 2")
SLOT_SHORT = ("V1", "V2")
SLOT_COLORS = ("#60a5fa", "#34d399")
SLOT_STYLES = ("-", "--")
DEFAULT_SLOT_ALGORITHMS = ("PPO", "SAC")

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
    "normalize_advantage": "Advantage normieren", "log_std_init": "log σ₀",
    "use_sde": "gSDE nutzen", "sde_sample_freq": "gSDE f", "buffer_size": "Buffer |D|",
    "learning_starts": "Start t₀", "tau": "Soft τ", "train_freq": "Freq. fₜ",
    "gradient_steps": "Grad. G", "action_noise": "Noise-Typ", "action_noise_sigma": "Noise σ",
    "policy_delay": "Delay d", "target_policy_noise": "Ziel σ_t", "target_noise_clip": "Clip c",
    "ent_coef_mode": "Entropie α", "ent_coef_value": "Start α", "target_entropy": "Ziel H*",
    "target_update_interval": "Target C",
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

#: Aufbau eines Verfahrenstabs: (Spalte, Gruppentitel, Parameter). Ein Tab
#: zeigt ausschließlich Parameter des dort gewählten Algorithmus.
PARAMETER_GROUPS: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "PPO": (
        _TRAINING_GROUP,
        ("left", "Rollout und geclipptes Ziel",
         ("n_steps", "n_epochs", "gae_lambda", "clip_range", "clip_range_vf", "target_kl",
          "ent_coef", "vf_coef", "max_grad_norm", "normalize_advantage")),
        _NETWORK_GROUP,
        ("right", "Exploration (stochastische Policy)",
         ("log_std_init", "sde_sample_freq", "use_sde")),
    ),
    "TD3": (
        _TRAINING_GROUP,
        ("left", "Replay Buffer und Updates", _REPLAY),
        _NETWORK_GROUP,
        ("right", "Twin Critics und Policy Delay",
         ("policy_delay", "target_policy_noise", "target_noise_clip")),
        ("right", "Exploration (Action Noise)", ("action_noise_sigma", "action_noise")),
    ),
    "SAC": (
        _TRAINING_GROUP,
        ("left", "Replay Buffer und Updates", _REPLAY),
        _NETWORK_GROUP,
        ("right", "Entropieregularisierung",
         ("ent_coef_value", "target_entropy", "target_update_interval", "ent_coef_mode")),
        ("right", "Exploration (optional)",
         ("sde_sample_freq", "action_noise_sigma", "use_sde", "action_noise")),
    ),
}

EXPORT_DIR = Path(__file__).parent / "exports"
#: Breite der drei Bedienspalten: zwei Parameterspalten tragen gemeinsam die
#: Verfahrenstabs, die dritte ist den Steuerungsbuttons vorbehalten.
CONTROL_COLUMNS = (360, 360, 226)
CONTROL_WIDTH = sum(CONTROL_COLUMNS) + 16
PARAMETER_WIDTH = CONTROL_COLUMNS[0] + CONTROL_COLUMNS[1]
#: Mindesthöhe, die dem unteren Bereich aus Diagrammen und Summary bleibt.
MIN_CHART_HEIGHT = 240
#: Mindestbreite der Environment-Anzeige neben dem Bedienpanel und die Breite,
#: die ihr beim Start zusteht, sofern der Bildschirm sie hergibt.
MIN_ENVIRONMENT_WIDTH, WANTED_ENVIRONMENT_WIDTH = 300, 560
#: Platz, den Menüleiste, Fensterrahmen und Dock vom Bildschirm beanspruchen.
SCREEN_MARGIN = 130


class SlotRuntime:
    """Lernzustand eines Verfahrensslots: Einzeltraining und Vergleich getrennt.

    Der Vergleich benutzt bewusst eigene Modelle, damit er das sichtbare
    Experiment des Einzeltrainings nicht verändert.
    """

    def __init__(self, config: LunarLanderPGConfig, best_base: Path) -> None:
        self.workbench = LunarLanderPGWorkbench(config)
        self.comparison: Optional[LunarLanderPGWorkbench] = None
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

    def reset_comparison(self) -> None:
        if self.comparison is not None:
            self.comparison.close()
        self.comparison = None
        self.comparison_history.clear()
        self.comparison_evaluations.clear()

    def close(self) -> None:
        self.workbench.close()
        if self.comparison is not None:
            self.comparison.close()


class LunarLanderPGGUI:
    BG, PANEL, FIELD = "#111827", "#1f2937", "#0f172a"
    FG, MUTED, ACCENT = "#f3f4f6", "#cbd5e1", "#60a5fa"
    EVAL_COLOR = "#f59e0b"
    PLOT_INTERVAL = 2.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.busy = False
        self.running_slot: Optional[int] = None
        self.comparison_running = False
        # Animationszustand je Verfahrensslot: Im Vergleich laufen beide
        # Verfahren gleichzeitig und werden auch gleichzeitig gezeigt.
        self.renderers: list[Optional[LunarLanderRenderer]] = [LunarLanderRenderer(), None]
        self.photos: list[Optional[ImageTk.PhotoImage]] = [None, None]
        self.last_frames: list[Optional[np.ndarray]] = [None, None]
        self.animation_after: list[Optional[str]] = [None, None]
        self.animation_observation: list[Optional[np.ndarray]] = [None, None]
        self.animation_policy: list[Any] = [None, None]
        self.animation_live = [False, False]
        self.animation_step = [0, 0]
        self.animation_reward = [0.0, 0.0]
        self.animation_interval = round(1000 / RENDER_FPS)
        self.last_plot = 0.0
        self.progress_steps: dict[Any, int] = {}
        self.progress_total = 1
        self.checkpoints = tempfile.TemporaryDirectory(prefix="lunarlander-pg-")
        self.entries: list[dict[str, tk.Widget]] = [{}, {}]
        self.slots = [
            SlotRuntime(default_config(name), Path(self.checkpoints.name) / f"best_slot{index}")
            for index, name in enumerate(DEFAULT_SLOT_ALGORITHMS)
        ]
        self._export_snapshot_key: Optional[tuple[Any, ...]] = None
        self._export_stamp: Optional[str] = None
        self._variables()
        self._window()
        self._layout()
        for slot in range(len(self.slots)):
            self._build_tab(slot)
        self._update_active_label()
        self._show_initial_frame()
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._training_summary()
        self.root.after_idle(self._initialize_layout)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # --------------------------------------------------------------- Variablen

    def _variables(self) -> None:
        self.algorithm_vars = [tk.StringVar(value=name) for name in DEFAULT_SLOT_ALGORITHMS]
        self.values: list[dict[str, tk.Variable]] = []
        for slot, name in enumerate(DEFAULT_SLOT_ALGORITHMS):
            variables: dict[str, tk.Variable] = {}
            for field in fields(LunarLanderPGConfig):
                if field.name == "algorithm":
                    continue
                variables[field.name] = (tk.BooleanVar() if field.name in BOOLEAN_FIELDS
                                         else tk.StringVar())
            self.values.append(variables)
            self._fill_values(slot, default_config(name))
        self.evaluation_episodes = tk.StringVar(value="5")
        self.evaluation_interval = tk.StringVar(value="10000")
        self.animation_enabled = tk.BooleanVar(value=True)
        self.fps = tk.StringVar(value=str(RENDER_FPS))
        self.status = tk.StringVar(value="Bereit")
        self.active_label = tk.StringVar(value="")
        self.progress = tk.DoubleVar(value=0)
        self.observations = [tk.StringVar(value="Noch keine Episode gestartet.")
                             for _ in SLOT_LABELS]

    def _fill_values(self, slot: int, config: LunarLanderPGConfig) -> None:
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
        width = min(1560, CONTROL_WIDTH + WANTED_ENVIRONMENT_WIDTH, available_width)
        height = min(960, available_height)
        self.root.title("LunarLander Policy-Gradient Workbench")
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(CONTROL_WIDTH + MIN_ENVIRONMENT_WIDTH, width), min(640, height))
        self.root.configure(bg=self.BG)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=self.BG, foreground=self.FG)
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.FG)
        style.configure("TLabelframe", background=self.BG, foreground=self.FG, bordercolor="#475569")
        style.configure("TLabelframe.Label", background=self.BG, foreground="#fff", font=("TkDefaultFont", 10, "bold"))
        style.configure("TButton", background="#334155", foreground=self.FG, padding=4)
        style.map("TButton", foreground=[("disabled", "#94a3b8")])
        style.configure("TEntry", fieldbackground=self.FIELD, foreground=self.FG, insertcolor=self.FG)
        style.configure("TCombobox", fieldbackground=self.FIELD, foreground=self.FG, background="#334155")
        style.map("TCombobox", fieldbackground=[("readonly", self.FIELD)], foreground=[("readonly", self.FG)])
        style.configure("TCheckbutton", background=self.BG, foreground=self.FG)
        style.map("TCheckbutton", foreground=[("disabled", "#94a3b8")])
        style.configure("TProgressbar", troughcolor=self.FIELD, background=self.ACCENT)
        style.configure("TNotebook", background=self.BG, bordercolor="#475569")
        style.configure("TNotebook.Tab", background="#1f2937", foreground=self.MUTED, padding=(12, 4))
        style.map("TNotebook.Tab", background=[("selected", "#334155")], foreground=[("selected", self.FG)])
        self.root.option_add("*TCombobox*Listbox.background", self.FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", self.FG)

    def _figure(self, master: ttk.Frame) -> tuple[Figure, Any, FigureCanvasTkAgg]:
        figure = Figure(figsize=(9, 3), dpi=100, facecolor=self.BG)
        figure.subplots_adjust(left=.11, right=.76, bottom=.30, top=.82)
        axes = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=master)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return figure, axes, canvas

    def _layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="LunarLander Policy-Gradient Workbench",
                  font=("TkDefaultFont", 18, "bold")).pack(side="left")
        ttk.Label(header, text="PPO, TD3 und SAC – kontinuierlich steuern und sicher landen").pack(side="left", padx=16)
        ttk.Button(header, text="Bedienungsanleitung", command=self.instructions).pack(side="right")
        self.splitter = ttk.Panedwindow(outer, orient="vertical")
        self.splitter.pack(fill="both", expand=True)
        upper, lower = ttk.Frame(self.splitter), ttk.Frame(self.splitter)
        self.splitter.add(upper, weight=1); self.splitter.add(lower, weight=1)
        upper.columnconfigure(0, minsize=CONTROL_WIDTH); upper.columnconfigure(1, weight=1); upper.rowconfigure(0, weight=1)
        self.controls = ttk.Frame(upper, width=CONTROL_WIDTH, padding=(0, 0, 8, 0))
        self.controls.grid(row=0, column=0, sticky="nsew"); self.controls.grid_propagate(False)
        self._controls(self.controls)
        self._environment(upper)

        lower.columnconfigure(0, weight=3); lower.columnconfigure(1, weight=1, minsize=560); lower.rowconfigure(0, weight=1)
        chart = ttk.LabelFrame(lower, text="Diagramme", padding=5)
        chart.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        chart_header = ttk.Frame(chart); chart_header.pack(fill="x")
        ttk.Button(chart_header, text="Diagramm exportieren (PNG)", command=self.export_chart).pack(side="right")
        self.charts = ttk.Notebook(chart)
        self.charts.pack(fill="both", expand=True, pady=(4, 0))
        training_tab, comparison_tab = ttk.Frame(self.charts), ttk.Frame(self.charts)
        self.charts.add(training_tab, text="Training")
        self.charts.add(comparison_tab, text="Vergleich")
        self.figure, self.axes, self.canvas = self._figure(training_tab)
        self.comparison_figure, self.comparison_axes, self.comparison_canvas = self._figure(comparison_tab)

        info = ttk.LabelFrame(lower, text="Summary", padding=5)
        info.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        info_header = ttk.Frame(info); info_header.pack(fill="x")
        ttk.Button(info_header, text="Summary exportieren (TXT)", command=self.export_summary).pack(side="right")
        table = ttk.Frame(info); table.pack(fill="both", expand=True, pady=(4, 0))
        table.rowconfigure(0, weight=1); table.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(
            table, wrap="none", height=8, background=self.FIELD, foreground=self.FG,
            insertbackground=self.FG, relief="flat", font="TkFixedFont",
        )
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        summary_y = ttk.Scrollbar(table, orient="vertical", command=self.summary_text.yview)
        summary_x = ttk.Scrollbar(table, orient="horizontal", command=self.summary_text.xview)
        self.summary_text.configure(yscrollcommand=summary_y.set, xscrollcommand=summary_x.set, state="disabled")
        summary_y.grid(row=0, column=1, sticky="ns"); summary_x.grid(row=1, column=0, sticky="ew")

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
        self.algorithm_combos = []
        for slot, label in enumerate(SLOT_LABELS):
            ttk.Label(selection, text=label).grid(row=slot, column=0, sticky="w", padx=(0, 4), pady=1)
            combo = ttk.Combobox(selection, textvariable=self.algorithm_vars[slot],
                                 values=ALGORITHMS, state="readonly", width=10, justify="right")
            combo.grid(row=slot, column=1, sticky="ew", padx=(0, 10), pady=1)
            combo.bind("<<ComboboxSelected>>", lambda _event, index=slot: self._algorithm_changed(index))
            self.algorithm_combos.append(combo)
        # Evaluationseinstellungen gelten global, damit beide Slots dieselben
        # Stützstellen liefern und der Vergleich fair bleibt.
        ttk.Label(selection, text="Eval-Episoden M").grid(row=0, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(selection, textvariable=self.evaluation_episodes, width=8,
                  justify="right").grid(row=0, column=3, sticky="e")
        ttk.Label(selection, text="Eval-Intervall").grid(row=1, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(selection, textvariable=self.evaluation_interval, width=8,
                  justify="right").grid(row=1, column=3, sticky="e")
        ttk.Label(selection, text="Intervall 0 = keine Auto-Evaluation. Beide Slots dürfen "
                                  "denselben Algorithmus verwenden.",
                  foreground=self.MUTED, wraplength=PARAMETER_WIDTH - 30).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(2, 0))
        selection.columnconfigure(1, weight=1); selection.columnconfigure(3, weight=1)

        self.parameter_tabs = ttk.Notebook(parent)
        self.parameter_tabs.pack(fill="both", expand=True)
        self.tab_frames = []
        for slot, label in enumerate(SLOT_LABELS):
            frame = ttk.Frame(self.parameter_tabs, padding=(0, 4, 0, 0))
            self.parameter_tabs.add(frame, text=label)
            self.tab_frames.append(frame)
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
            button.pack(fill="x", pady=1, ipady=1); self.buttons.append(button)
        self.stop_button, self.best_button = self.buttons[1], self.buttons[5]
        self.stop_button.configure(state="disabled"); self.best_button.configure(state="disabled")
        ttk.Checkbutton(actions, text="Animation zeigen", variable=self.animation_enabled,
                        command=self._animation_toggled).pack(fill="x", pady=(4, 0))
        rate = ttk.Frame(actions); rate.pack(fill="x", pady=(2, 0))
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
                if position:
                    row, position = row + 1, 0
                widget = ttk.Checkbutton(group, text=SHORT_LABELS[name], variable=variable)
                widget.grid(row=row, column=0, columnspan=4, sticky="w", pady=1)
                row += 1
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

    def _environment(self, parent: ttk.Frame) -> None:
        """Je Slot ein Anzeigefeld. Sichtbar ist normalerweise nur das aktive
        Verfahren; laufen im Vergleich beide Animationen, stehen beide
        nebeneinander."""
        self.env_container = ttk.Frame(parent)
        self.env_container.grid(row=0, column=1, sticky="nsew")
        self.env_container.columnconfigure(0, weight=1)
        self.env_frames, self.image_labels, self.readout_labels = [], [], []
        for slot, label in enumerate(SLOT_LABELS):
            frame = ttk.LabelFrame(self.env_container,
                                   text=f"{label} – offizieller Gymnasium-RGB-Frame", padding=6)
            image = ttk.Label(frame, anchor="center"); image.pack(fill="both", expand=True)
            readout = ttk.Label(frame, textvariable=self.observations[slot], justify="left",
                                font="TkFixedFont")
            readout.pack(anchor="w", pady=(4, 0))
            self.env_frames.append(frame)
            self.image_labels.append(image)
            self.readout_labels.append(readout)
        self._sync_panels()

    def _panel_slots(self) -> list[int]:
        live = [slot for slot in range(len(self.slots)) if self.animation_live[slot]]
        return live if live else [self.active_slot]

    def _sync_panels(self) -> None:
        """Blendet genau die Anzeigefelder ein, die gerade etwas zeigen.

        Zwei Anzeigen stehen untereinander: Nebeneinander bliebe von einem
        600 × 400-Frame samt Messwertzeile in der Breite zu wenig übrig.
        """
        slots = self._panel_slots()
        for slot, frame in enumerate(self.env_frames):
            if slot in slots:
                frame.grid(row=slots.index(slot), column=0, sticky="nsew",
                           pady=(0, 4) if slots.index(slot) == 0 and len(slots) > 1 else 0)
            else:
                frame.grid_remove()
        for row in range(len(self.env_frames)):
            self.env_container.rowconfigure(row, weight=1 if row < len(slots) else 0)
        for slot in slots:
            if self.last_frames[slot] is not None:
                self._show_frame(slot, self.last_frames[slot])

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
        hidden = {str(frame) for index, frame in enumerate(self.tab_frames)
                  if index != self.parameter_tabs.index(self.parameter_tabs.select())}
        stack = list(self.controls.winfo_children())
        while stack:
            widget = stack.pop()
            # Der nicht gewählte Verfahrenstab ist von Tk bewusst nicht gemappt.
            if any(str(widget).startswith(name) for name in hidden):
                continue
            stack.extend(widget.winfo_children())
            if isinstance(widget, (ttk.Entry, ttk.Combobox, ttk.Button, ttk.Checkbutton, ttk.Progressbar)):
                if not widget.winfo_ismapped() or widget.winfo_rootx() < left or widget.winfo_rooty() < top \
                        or widget.winfo_rootx() + widget.winfo_width() > right \
                        or widget.winfo_rooty() + widget.winfo_height() > bottom:
                    issues.append(
                        f"{widget}: ({widget.winfo_rootx()},{widget.winfo_rooty()},"
                        f"{widget.winfo_width()}x{widget.winfo_height()}) außerhalb "
                        f"({left},{top},{right - left}x{bottom - top})"
                    )
        # Eine Parametergruppe, die breiter ist als ihre Rasterspalte, würde
        # ihre Felder unbemerkt in die Nachbarspalte drücken.
        frame = self.tab_frames[self.parameter_tabs.index(self.parameter_tabs.select())]
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
        panels = [("Visualisierung", self.image_labels[slot]) for slot in self._panel_slots()]
        for label, widget in (*panels,
                              ("Diagramm", visible_canvas),
                              ("Summary", self.summary_text)):
            if not widget.winfo_ismapped() or widget.winfo_width() <= 1 or widget.winfo_height() <= 1:
                issues.append(label)
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
        return self.parameter_tabs.index(self.parameter_tabs.select())

    def _update_active_label(self) -> None:
        slot = self.active_slot
        self.active_label.set(f"Aktiv: {SLOT_LABELS[slot]} – {self.algorithm_vars[slot].get()}")

    def _tab_changed(self) -> None:
        self._update_active_label()
        self._sync_panels()
        if not self.busy:
            self.best_button.configure(
                state="normal" if self.slots[self.active_slot].best is not None else "disabled")
            self._refresh_training_plot(); self._training_summary()

    def _algorithm_changed(self, slot: int) -> None:
        """Dropdown-Wechsel: Profilwerte laden und nur diesen Slot zurücksetzen."""
        algorithm = self.algorithm_vars[slot].get()
        runtime = self.slots[slot]
        if algorithm == runtime.algorithm:
            return
        if runtime.workbench.model is not None and not messagebox.askyesno(
            "Verfahren wechseln",
            f"{SLOT_LABELS[slot]} auf {algorithm} umstellen? Der Lernzustand dieses "
            "Slots wird verworfen, der andere Slot bleibt unverändert.",
            parent=self.root,
        ):
            self.algorithm_vars[slot].set(runtime.algorithm)
            return
        config = default_config(algorithm)
        self._fill_values(slot, config)
        self._build_tab(slot)
        runtime.workbench.close(); runtime.workbench = LunarLanderPGWorkbench(config)
        runtime.history.clear(); runtime.evaluations.clear(); runtime.best = None
        runtime.reset_comparison()
        self._update_active_label()
        if slot == self.active_slot:
            self.best_button.configure(state="disabled")
        self._refresh_training_plot(); self._refresh_comparison_plot(); self._training_summary()
        self.status.set(f"Bereit – {SLOT_LABELS[slot]} auf {algorithm} gesetzt (Zoo-Profil geladen)")

    def _config(self, slot: int) -> LunarLanderPGConfig:
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
                            "Liste ganzer Zahlen, z. B. '400,300'" if name in TUPLE_FIELDS else
                            "Zahl oder leer" if name in OPTIONAL_FLOAT_FIELDS else "Zahl")
                raise ValueError(
                    f"{SLOT_LABELS[slot]} – {label}: '{text}' ist keine gültige Eingabe. "
                    f"Erwartet: {expected}."
                ) from error
        config = LunarLanderPGConfig(algorithm=self.algorithm_vars[slot].get(), **data)
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
                f"Eval-Episoden M: '{self.evaluation_episodes.get()}' ist keine ganze Zahl. Gültig: ≥ 1."
            ) from error
        if episodes <= 0:
            raise ValueError(f"Eval-Episoden M: '{episodes}' ist ungültig. Gültig: ganze Zahl ≥ 1.")
        try:
            interval = int(self.evaluation_interval.get().strip())
        except ValueError as error:
            raise ValueError(
                f"Eval-Intervall: '{self.evaluation_interval.get()}' ist keine ganze Zahl. Gültig: ≥ 0."
            ) from error
        if interval < 0:
            raise ValueError(f"Eval-Intervall: '{interval}' ist ungültig. Gültig: ganze Zahl ≥ 0.")
        return episodes, interval

    def _set_busy(self, busy: bool, text: str) -> None:
        self.busy = busy; self.status.set(text)
        for button in self.buttons:
            button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        for combo in self.algorithm_combos:
            combo.configure(state="disabled" if busy else "readonly")
        if not busy:
            self._stop_live_animation()
            self.running_slot = None
            self.comparison_running = False
            if self.slots[self.active_slot].best is None:
                self.best_button.configure(state="disabled")
        self._update_active_label()

    # ----------------------------------------------------------------- Training

    def start_training(self) -> None:
        if self.busy: return
        slot = self.active_slot
        runtime = self.slots[slot]
        try:
            config = self._config(slot)
            episodes, interval = self._evaluation_settings()
            self._animation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root); return
        if runtime.workbench.model is not None and config.signature() != runtime.workbench.config.signature():
            if not messagebox.askyesno(
                "Neues Modell",
                f"Geänderte Modellparameter erfordern einen Reset von {SLOT_LABELS[slot]}. Fortfahren?",
                parent=self.root,
            ):
                return
            runtime.workbench.close(); runtime.workbench = LunarLanderPGWorkbench(config)
            runtime.history.clear(); runtime.evaluations.clear(); runtime.best = None
        else:
            runtime.workbench.config = config
        self.running_slot = slot
        self.stop_event.clear(); self.progress.set(0)
        self.progress_steps = {("single", slot): 0}
        self.progress_total = config.total_timesteps
        self.charts.select(0)
        self._set_busy(True, f"Läuft – {SLOT_LABELS[slot]} ({config.algorithm}) über "
                             f"{config.total_timesteps:,} Schritte".replace(",", "."))
        self.worker = threading.Thread(target=self._train_worker, args=(slot, episodes, interval), daemon=True)
        self.worker.start(); self.root.after(50, self._poll)
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
        if self.busy: return
        try:
            configs = [self._config(slot) for slot in range(len(self.slots))]
            episodes, interval = self._evaluation_settings()
            self._animation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültiger Vergleich", str(error), parent=self.root); return
        if configs[0].total_timesteps != configs[1].total_timesteps and not messagebox.askyesno(
            "Unterschiedliche Budgets",
            f"{SLOT_LABELS[0]} trainiert {configs[0].total_timesteps:,} Schritte, "
            f"{SLOT_LABELS[1]} dagegen {configs[1].total_timesteps:,} Schritte. "
            "Der Vergleich ist dann nicht budgetgleich. Trotzdem starten?".replace(",", "."),
            parent=self.root,
        ):
            return
        for slot, config in enumerate(configs):
            runtime = self.slots[slot]
            if runtime.comparison is not None and \
                    runtime.comparison.config.signature() != config.signature():
                runtime.reset_comparison()
            if runtime.comparison is None:
                runtime.comparison = LunarLanderPGWorkbench(config)
            else:
                runtime.comparison.config = config
        self.stop_event.clear(); self.progress.set(0)
        self.comparison_running = True
        self.progress_steps = {("compare", slot): 0 for slot in range(len(self.slots))}
        self.progress_total = sum(config.total_timesteps for config in configs)
        self.charts.select(1)
        self._set_busy(True, "Läuft – Vergleich: "
                             + " gegen ".join(f"{SLOT_SHORT[slot]} {config.algorithm}"
                                              for slot, config in enumerate(configs))
                             + f" über {self.progress_total:,} Schritte".replace(",", "."))
        self._refresh_comparison_plot(); self._comparison_summary()
        self.worker = threading.Thread(target=self._comparison_worker, args=(episodes, interval), daemon=True)
        self.worker.start(); self.root.after(50, self._poll)
        self._start_live_animation(list(range(len(self.slots))))

    def _comparison_worker(self, episodes: int, interval: int) -> None:
        errors: queue.Queue = queue.Queue()
        barrier = threading.Barrier(len(self.slots))

        def run(slot: int) -> None:
            try:
                barrier.wait()
                self.slots[slot].comparison.train(
                    self.stop_event, self.events, ("compare", slot),
                    evaluation_interval=interval, evaluation_episodes=episodes,
                )
            except Exception as error:
                errors.put(error); self.stop_event.set()

        threads = [threading.Thread(target=run, args=(slot,), daemon=True)
                   for slot in range(len(self.slots))]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        if not errors.empty():
            self.events.put(("error", errors.get()))
        else:
            self.events.put(("comparison_done", self.stop_event.is_set()))

    # -------------------------------------------------------------- Evaluation

    def start_evaluation(self) -> None:
        if self.busy: return
        slot = self.active_slot
        if self.slots[slot].workbench.model is None:
            messagebox.showinfo("Keine Evaluation",
                                f"Trainiere zuerst ein Modell für {SLOT_LABELS[slot]}.",
                                parent=self.root)
            return
        try:
            episodes, _ = self._evaluation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültige Evaluation", str(error), parent=self.root); return
        self.running_slot = slot
        self._set_busy(True, f"Läuft – deterministische Evaluation ({SLOT_LABELS[slot]})")
        self.worker = threading.Thread(target=self._evaluation_worker, args=(slot, episodes), daemon=True)
        self.worker.start(); self.root.after(50, self._poll)

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
        if self.busy: return
        slot = self.active_slot
        runtime = self.slots[slot]
        if runtime.best is None: return
        try:
            restored = LunarLanderPGWorkbench.load(runtime.best_base, runtime.algorithm)
        except Exception as error:
            messagebox.showerror("Wiederherstellen fehlgeschlagen", str(error), parent=self.root); return
        runtime.workbench.close(); runtime.workbench = restored
        self.status.set(f"Abgeschlossen – bestes Modell von {SLOT_LABELS[slot]} wiederhergestellt")
        self._training_summary()

    def stop(self) -> None:
        if self.busy:
            self.stop_event.set(); self.status.set("Stoppen angefordert …")

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
                    self.slots[slot].history.append(metric); redraw_training = True
                else:
                    self.slots[slot].comparison_history.append(metric); redraw_comparison = True
            elif kind == "evaluation":
                (mode, slot), episode, _steps, result = payload
                if mode == "single":
                    self.slots[slot].evaluations.append((episode, result)); redraw_training = True
                else:
                    self.slots[slot].comparison_evaluations.append((episode, result)); redraw_comparison = True
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
                self._set_busy(False, "Abgeschlossen – Evaluation"); redraw_training = True
            elif kind == "error":
                self._set_busy(False, "Fehler"); messagebox.showerror("Fehler", str(payload), parent=self.root)
        if self.slots[self.active_slot].best is not None:
            self.best_button.configure(state="disabled" if self.busy else "normal")
        throttled = time.monotonic() - self.last_plot >= self.PLOT_INTERVAL
        if redraw_training:
            self._training_summary()
            if throttled or not self.busy:
                self._refresh_training_plot(); self.last_plot = time.monotonic()
        if redraw_comparison:
            self._comparison_summary()
            if throttled or not self.busy:
                self._refresh_comparison_plot(); self.last_plot = time.monotonic()
        if self.busy:
            self.root.after(50, self._poll)

    # ------------------------------------------------------------------ Plots

    def _style(self, axes: Any) -> None:
        axes.set_facecolor(self.FIELD); axes.tick_params(colors=self.MUTED); axes.grid(color="#475569", alpha=.4)
        for spine in axes.spines.values(): spine.set_color("#64748b")
        axes.title.set_color(self.FG); axes.xaxis.label.set_color(self.MUTED); axes.yaxis.label.set_color(self.MUTED)
        legend = axes.legend(loc="center left", bbox_to_anchor=(1.02, .5), borderaxespad=0, fontsize=8)
        legend.get_frame().set_facecolor(self.PANEL); legend.get_frame().set_edgecolor("#64748b")
        for text in legend.get_texts(): text.set_color(self.FG)

    def _series_label(self, slot: int, configs: Optional[list[LunarLanderPGConfig]] = None) -> str:
        """`V1 – PPO`; bei zweimal demselben Algorithmus zusätzlich der
        wichtigste abweichende Parameter."""
        label = f"{SLOT_SHORT[slot]} – {self.slots[slot].algorithm}"
        if configs and configs[0].algorithm == configs[1].algorithm:
            differences = config_differences(configs[0], configs[1])
            if differences:
                name, first, second = differences[0]
                label += f" ({name} {first if slot == 0 else second})"
        return label

    def _refresh_training_plot(self) -> None:
        self.axes.clear()
        slot = self.active_slot
        runtime = self.slots[slot]
        if runtime.history:
            episodes = [item.episode for item in runtime.history]
            rewards = [item.reward for item in runtime.history]
            x, y = downsample_minmax(episodes, rewards)
            self.axes.plot(x, y, color=SLOT_COLORS[slot], alpha=.15, linewidth=.8)
            self.axes.plot(episodes, rolling_average(rewards), color=SLOT_COLORS[slot], linewidth=1.6,
                           label=f"Training – {SLOT_LABELS[slot]} ({runtime.algorithm})")
        if runtime.evaluations:
            # Evaluation optisch klar vom explorativen Training getrennt.
            self.axes.plot([point[0] for point in runtime.evaluations],
                           [point[1].mean_reward for point in runtime.evaluations],
                           color=self.EVAL_COLOR, marker="o", markersize=4, linestyle="--",
                           linewidth=1.3, label="Evaluation (deterministisch)")
        self.axes.axhline(SOLVED_RETURN, color="#a3e635", linestyle=":", linewidth=1.2, label="Gelöst ab +200")
        self.axes.set(title="Episoden-Return", xlabel="Episode", ylabel="Return")
        self._style(self.axes); self.canvas.draw_idle()

    def _refresh_comparison_plot(self) -> None:
        self.comparison_axes.clear()
        configs = [runtime.comparison.config if runtime.comparison else runtime.workbench.config
                   for runtime in self.slots]
        for slot, runtime in enumerate(self.slots):
            if not runtime.comparison_history: continue
            episodes = [item.episode for item in runtime.comparison_history]
            rewards = [item.reward for item in runtime.comparison_history]
            x, y = downsample_minmax(episodes, rewards)
            self.comparison_axes.plot(x, y, color=SLOT_COLORS[slot], alpha=.08, linewidth=.7)
            self.comparison_axes.plot(episodes, rolling_average(rewards), color=SLOT_COLORS[slot],
                                      linestyle=SLOT_STYLES[slot], linewidth=1.6,
                                      label=self._series_label(slot, configs))
        self.comparison_axes.axhline(SOLVED_RETURN, color="#a3e635", linestyle=":", linewidth=1.2,
                                     label="Gelöst ab +200")
        self.comparison_axes.set(title="Live-Vergleich der Verfahrensslots", xlabel="Episode", ylabel="Return")
        self._style(self.comparison_axes); self.comparison_canvas.draw_idle()

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
        if not statistics:
            # Der Unterschiedsblock bleibt auch ohne Messwerte sichtbar: er
            # erklärt, was verglichen wird, bevor der erste Lauf startet.
            lines = [title, "", "Noch keine vollständig abgeschlossene Episode.", *footer]
            self._write_summary("\n".join(lines))
            return
        names = list(statistics)
        label_width = max(len(label) for label in labels) + 1
        value_width = max(13, max(len(name) for name in names) + 2)
        header = "Statistik".ljust(label_width) + "".join(name.rjust(value_width) for name in names)
        lines = [title, "", header, "─" * len(header)]
        for index, label in enumerate(labels):
            lines.append(label.ljust(label_width) + "".join(
                str(statistics[name][index]).rjust(value_width) for name in names))
        if footer:
            lines.extend(["", *footer])
        self._write_summary("\n".join(lines))

    @staticmethod
    def _statistics(history: list[EpisodeMetric], steps: int, budget: int) -> tuple[Any, ...]:
        thousands = lambda value: f"{value:,}".replace(",", ".")
        if not history:
            return (0, thousands(steps), thousands(budget), "—", "—", "—", "—")
        return (
            len(history), thousands(steps), thousands(budget),
            f"{np.mean([item.reward for item in history]):.1f}",
            f"{np.mean([item.landed for item in history]):.1%}",
            f"{np.mean([item.solved for item in history]):.1%}",
            f"{np.mean([item.crashed for item in history]):.1%}",
        )

    def _training_summary(self) -> None:
        slot = self.active_slot
        runtime = self.slots[slot]
        model = runtime.workbench.model
        stats = self._statistics(runtime.history, model.num_timesteps if model else 0,
                                 runtime.workbench.config.total_timesteps)
        labels = ("Episoden", "Schritte", "Budget", "Ø Return", "Landequote", "Gelöst-Quote", "Absturzquote")
        footer = ["Kurvenpunkte entstehen nur für vollständig abgeschlossene Episoden."]
        if runtime.evaluations:
            last = runtime.evaluations[-1][1]
            footer.append(
                f"Letzte Evaluation: Ø {last.mean_reward:.1f} ± {last.reward_std:.1f} | "
                f"Landung {last.landing_rate:.0%} | gelöst {last.solved_rate:.0%} | "
                f"Länge {last.mean_length:.0f}"
            )
        if runtime.best is not None:
            best = runtime.best
            footer.append(
                f"Beste Evaluation: Ø {best.mean_reward:.1f} | Landung {best.landing_rate:.0%} | "
                f"gelöst {best.solved_rate:.0%}  (als Checkpoint gesichert)"
            )
        self._table(f"Training – {SLOT_LABELS[slot]} ({runtime.algorithm})", labels,
                    {SLOT_LABELS[slot]: stats}, footer)

    def _comparison_summary(self) -> None:
        statistics: dict[str, tuple[Any, ...]] = {}
        configs = []
        for slot, runtime in enumerate(self.slots):
            config = runtime.comparison.config if runtime.comparison else runtime.workbench.config
            configs.append(config)
            model = runtime.comparison.model if runtime.comparison else None
            statistics[SLOT_LABELS[slot]] = (config.algorithm, *self._statistics(
                runtime.comparison_history, model.num_timesteps if model else 0,
                config.total_timesteps))
        if not any(runtime.comparison_history for runtime in self.slots):
            statistics = {}
        labels = ("Algorithmus", "Episoden", "Schritte", "Budget", "Ø Return", "Landequote",
                  "Gelöst-Quote", "Absturzquote")
        footer = []
        for slot, runtime in enumerate(self.slots):
            points = runtime.comparison_evaluations
            if points:
                last = points[-1][1]
                footer.append(f"{SLOT_SHORT[slot]} letzte Evaluation: Ø {last.mean_reward:.1f} | "
                              f"gelöst {last.solved_rate:.0%}")
        footer.extend(self._difference_lines(configs))
        self._table("Vergleich der Verfahrensslots", labels, statistics, footer)

    def _difference_lines(self, configs: list[LunarLanderPGConfig]) -> list[str]:
        """Ohne diesen Block wäre ein Vergleich zweier Parametrisierungen
        desselben Algorithmus nicht interpretierbar."""
        differences = config_differences(configs[0], configs[1])
        note = ([] if configs[0].algorithm == configs[1].algorithm else
                ["  (nur gemeinsame Parameter; verfahrenseigene stehen im jeweiligen Tab)"])
        if not differences:
            return ["", "Unterschiede: keine – beide Slots sind gleich konfiguriert.", *note]
        width = max(len(name) for name, _, _ in differences)
        lines = ["", f"Unterschiede ({SLOT_SHORT[0]} | {SLOT_SHORT[1]}):", *note]
        lines.extend(f"  {name.ljust(width)}  {first} | {second}" for name, first, second in differences)
        return lines

    # ----------------------------------------------------------------- Export

    def _comparison_visible(self) -> bool:
        return self.charts.index(self.charts.select()) == 1

    def _config_snapshot_lines(self) -> list[str]:
        lines = []
        for slot, runtime in enumerate(self.slots):
            config = runtime.comparison.config if (self._comparison_visible() and runtime.comparison) \
                else runtime.workbench.config
            if not self._comparison_visible() and slot != self.active_slot:
                continue
            lines.append(f"[{SLOT_LABELS[slot]}]")
            lines.extend(f"{key}: {value}" for key, value in asdict(config).items()
                         if key == "algorithm" or config.uses(key))
            lines.append("")
        lines.append(f"evaluation_episodes: {self.evaluation_episodes.get()}")
        lines.append(f"evaluation_interval: {self.evaluation_interval.get()}")
        return lines

    def _export_snapshot(self) -> tuple[str, tuple[Any, ...]]:
        """Kennzeichnet den aktuell sichtbaren Trainings-/Vergleichsstand, damit
        PNG und TXT denselben Dateinamensstamm teilen."""
        if self._comparison_visible():
            slug = "vergleich-" + "-".join(slugify(runtime.algorithm) for runtime in self.slots)
            key = ("comparison", tuple(len(runtime.comparison_history) for runtime in self.slots))
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
        return f"lunarlander_pg_{slug}_{self._export_stamp}"

    def export_chart(self) -> None:
        EXPORT_DIR.mkdir(exist_ok=True)
        comparison = self._comparison_visible()
        figure = self.comparison_figure if comparison else self.figure
        suffix = "vergleich" if comparison else "training"
        path = filedialog.asksaveasfilename(
            parent=self.root, initialdir=EXPORT_DIR, defaultextension=".png",
            filetypes=(("PNG-Bild", "*.png"),), initialfile=f"{self._export_base_name()}_{suffix}.png",
        )
        if not path: return
        try:
            figure.savefig(path, facecolor=figure.get_facecolor())
        except OSError as error:
            messagebox.showerror("Export fehlgeschlagen", str(error), parent=self.root); return
        self.status.set(f"Abgeschlossen – Diagramm exportiert: {Path(path).name}")

    def export_summary(self) -> None:
        EXPORT_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.root, initialdir=EXPORT_DIR, defaultextension=".txt",
            filetypes=(("Textdatei", "*.txt"),), initialfile=f"{self._export_base_name()}_config.txt",
        )
        if not path: return
        text = self._summary_text() + "\n\nKonfiguration:\n" + "\n".join(self._config_snapshot_lines()) + "\n"
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as error:
            messagebox.showerror("Export fehlgeschlagen", str(error), parent=self.root); return
        self.status.set(f"Abgeschlossen – Summary exportiert: {Path(path).name}")

    # -------------------------------------------------------------- Animation

    def _renderer(self, slot: int) -> LunarLanderRenderer:
        """Renderprozess des Slots.

        Solange nur eine Episode gleichzeitig sichtbar ist, teilen sich beide
        Slots den ersten Prozess; erst der Vergleich mit zwei gleichzeitig
        laufenden Animationen erzeugt den zweiten.
        """
        if self.renderers[slot] is not None:
            return self.renderers[slot]
        if all(self.animation_live):
            self.renderers[slot] = LunarLanderRenderer()
            return self.renderers[slot]
        return self.renderers[0]

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

    def _animation_model(self, slot: int) -> Any:
        """Das Modell, das die sichtbare Episode spielt: im Vergleich das
        Vergleichsmodell des Slots, sonst dessen Einzeltrainingsmodell."""
        runtime = self.slots[slot]
        if self.comparison_running and runtime.comparison is not None:
            return runtime.comparison.model
        return runtime.workbench.model

    def _show_initial_frame(self, slot: int = 0) -> None:
        observation, _info, frame = self._renderer(slot).reset(
            self.slots[slot].workbench.config.seed)
        self.animation_observation[slot] = observation
        self._show_frame(slot, frame)
        self._show_readout(slot, observation)

    def _show_readout(self, slot: int, observation: np.ndarray,
                      action: Optional[np.ndarray] = None) -> None:
        values = observation_readout(observation)
        legs = f"{'ja' if values['left_leg'] else 'nein'} / {'ja' if values['right_leg'] else 'nein'}"
        first = (f"Schritt: {self.animation_step[slot]:4d}   "
                 f"Return: {self.animation_reward[slot]:+8.1f}   "
                 f"{SLOT_LABELS[slot]} ({self.slots[slot].algorithm})")
        if action is None:
            second = "Action: —"
        else:
            readout = action_readout(action)
            second = (f"Action: a₀={readout['main_raw']:+.2f} a₁={readout['lateral_raw']:+.2f}   "
                      f"Haupttriebwerk: {readout['main_text']}   "
                      f"Steuertriebwerk: {readout['lateral_text']}")
        third = (f"x: {values['x']:+.3f}   y: {values['y']:+.3f}   "
                 f"vx: {values['vx']:+.3f}   vy: {values['vy']:+.3f}")
        fourth = (f"Winkel: {values['angle_degrees']:+6.1f}°   "
                  f"ω: {values['angular_velocity']:+.3f} rad/s   Beine: {legs}")
        self.observations[slot].set("\n".join((first, second, third, fourth)))

    def _frame_size(self, slot: int) -> tuple[int, int]:
        """Zielgröße des Bildes, abgeleitet aus dem freien Platz des Containers.

        Die Größe des Bildlabels selbst taugt dafür nicht: Sie folgt dem zuletzt
        gesetzten Bild, sodass ein einmal großes Bild seinen Platz behielte und
        das zweite Anzeigefeld verdrängen würde.
        """
        panels = max(1, len(self._panel_slots()))
        chrome = self.readout_labels[slot].winfo_reqheight() + 44
        width = max(160, self.env_container.winfo_width() - 24)
        height = max(110, self.env_container.winfo_height() // panels - chrome)
        return width, height

    def _show_frame(self, slot: int, frame: np.ndarray) -> None:
        self.last_frames[slot] = np.asarray(frame)
        label = self.image_labels[slot]
        width, height = self._frame_size(slot)
        image = Image.fromarray(self.last_frames[slot])
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photos[slot] = ImageTk.PhotoImage(image)
        label.configure(image=self.photos[slot])

    def animate(self) -> None:
        """Einzelne sichtbare Episode des aktiven Verfahrens."""
        if self.busy: return
        slot = self.active_slot
        if self.slots[slot].workbench.model is None:
            messagebox.showinfo("Keine Episode",
                                f"Trainiere zuerst ein Modell für {SLOT_LABELS[slot]}.",
                                parent=self.root)
            return
        try:
            self._animation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültige Animation", str(error), parent=self.root); return
        self._stop_animation(slot)
        self._begin_episode(slot, live=False)
        if self.animation_enabled.get():
            self._animate_step(slot)
        else:
            self._run_episode_without_animation(slot)

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
        if self.animation_policy[slot] is None:
            self.animation_policy[slot] = type(model.policy)(
                **model.policy._get_constructor_parameters())
        self.animation_policy[slot].load_state_dict(
            {name: value.detach().clone() for name, value in model.policy.state_dict().items()})
        self.animation_policy[slot].set_training_mode(False)

    def _begin_episode(self, slot: int, live: bool) -> None:
        """Setzt den Renderer zurück und aktualisiert bei laufendem Training die
        Policy-Kopie, die die Animation spielt."""
        model = self._animation_model(slot)
        if live:
            self._refresh_snapshot(slot, model)
        else:
            self.animation_policy[slot] = None
        # Laufende Episoden starten ohne festen Seed, damit nacheinander
        # gezeigte Episoden nicht identisch aussehen.
        seed = None if live else self.slots[slot].workbench.config.seed
        observation, _info, frame = self._renderer(slot).reset(seed)
        self.animation_observation[slot] = observation
        self.animation_step[slot] = 0
        self.animation_reward[slot] = 0.0
        self._show_frame(slot, frame)
        self._show_readout(slot, observation)

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
        # Der erste reset() eines frisch gestarteten Renderprozesses wartet, bis
        # dieser seine Importe erledigt hat. Das passiert hier einmalig beim
        # Start und nicht später mitten in der laufenden Animation.
        for slot in slots:
            self._renderer(slot).reset(None)
        for slot in slots:
            self._live_episode(slot)

    def _live_active(self, slot: int) -> bool:
        return self.animation_live[slot] and self.busy and self.animation_enabled.get()

    def _live_slots(self) -> list[int]:
        """Slots, deren laufendes Training gerade sichtbar gemacht werden kann."""
        if not self.busy:
            return []
        if self.comparison_running:
            return list(range(len(self.slots)))
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
        try:
            self._begin_episode(slot, live=True)
        except Exception as error:
            self._stop_live_animation()
            self.status.set(f"Animation gestoppt – {error}")
            return
        self._animate_step(slot)

    def _stop_live_animation(self) -> None:
        if not any(self.animation_live):
            return
        for slot in range(len(self.slots)):
            if self.animation_live[slot]:
                self.animation_live[slot] = False
                self.animation_policy[slot] = None
                self._stop_animation(slot)
        self._sync_panels()

    def _policy_action(self, slot: int) -> np.ndarray:
        policy = self.animation_policy[slot]
        observation = self.animation_observation[slot]
        if policy is not None:
            action, _ = policy.predict(observation, deterministic=True)
        else:
            action, _ = self._animation_model(slot).predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _episode_outcome_text(self, reward: float, truncated: bool) -> str:
        if truncated:
            return "Abgeschlossen – Zeitlimit erreicht"
        if reward > 50:
            return "Abgeschlossen – sicher gelandet"
        if reward < -50:
            return "Abgeschlossen – abgestürzt"
        return "Abgeschlossen – Episode beendet"

    def _run_episode_without_animation(self, slot: int) -> None:
        """Sichtbare Episode ohne Einzelbildanimation: nur das Endbild zählt."""
        terminated = truncated = False
        reward = 0.0
        action = None
        while not (terminated or truncated):
            action = self._policy_action(slot)
            observation, reward, terminated, truncated, _info, frame = self._renderer(slot).step(action)
            self.animation_observation[slot] = observation
            self.animation_step[slot] += 1
            self.animation_reward[slot] += reward
        self._show_frame(slot, frame)
        self._show_readout(slot, self.animation_observation[slot], action)
        self.status.set(self._episode_outcome_text(reward, truncated))

    def _animate_step(self, slot: int) -> None:
        live = self.animation_live[slot]
        if live and not self._live_active(slot):
            self._stop_live_animation(); return
        if live:
            # Ein Fehler in der Anzeige darf den laufenden Trainingslauf weder
            # abbrechen noch stillschweigend die Animation beenden.
            try:
                self._animate_frame(slot, live)
            except Exception as error:
                self._stop_live_animation()
                self.status.set(f"Animation gestoppt – {error}")
            return
        if not self.animation_enabled.get():
            self._stop_animation(slot); return
        self._animate_frame(slot, live)

    def _animate_frame(self, slot: int, live: bool) -> None:
        action = self._policy_action(slot)
        observation, reward, terminated, truncated, _info, frame = self._renderer(slot).step(action)
        self.animation_observation[slot] = observation
        self.animation_step[slot] += 1
        self.animation_reward[slot] += reward
        self._show_frame(slot, frame)
        self._show_readout(slot, observation, action)
        if terminated or truncated:
            if live:
                # Nächste Episode zeigt den inzwischen weiter trainierten Stand.
                self.animation_after[slot] = self.root.after(
                    self.animation_interval, lambda: self._live_episode(slot))
            else:
                self.animation_after[slot] = None
                self.status.set(self._episode_outcome_text(reward, truncated))
            return
        self.animation_after[slot] = self.root.after(
            self.animation_interval, lambda: self._animate_step(slot))

    def _stop_animation(self, slot: Optional[int] = None) -> None:
        slots = range(len(self.slots)) if slot is None else (slot,)
        for index in slots:
            if self.animation_after[index]:
                self.root.after_cancel(self.animation_after[index])
                self.animation_after[index] = None

    # ------------------------------------------------------------ Modelldatei

    def reset(self) -> None:
        if self.busy: return
        slot = self.active_slot
        if not messagebox.askyesno("Neues Modell",
                                   f"Lernzustand von {SLOT_LABELS[slot]} verwerfen?",
                                   parent=self.root):
            return
        try:
            config = self._config(slot)
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root); return
        runtime = self.slots[slot]
        runtime.workbench.close(); runtime.workbench = LunarLanderPGWorkbench(config)
        runtime.history.clear(); runtime.evaluations.clear(); runtime.best = None
        runtime.reset_comparison()
        self.best_button.configure(state="disabled")
        self._refresh_training_plot(); self._refresh_comparison_plot(); self._training_summary()
        self.status.set(f"Bereit – {SLOT_LABELS[slot]} zurückgesetzt")

    def instructions(self) -> None:
        messagebox.showinfo(
            "Bedienungsanleitung",
            "Ablauf\n"
            "1. Für 'Verfahren 1' und 'Verfahren 2' je einen Algorithmus wählen – gern\n"
            "   zweimal denselben, um zwei Parametrisierungen zu vergleichen.\n"
            "2. Im jeweiligen Tab die Parameter setzen. Der sichtbare Tab ist das aktive\n"
            "   Verfahren; alle Einzellauf-Buttons wirken darauf.\n"
            "3. 'Training starten / fortsetzen' trainiert das aktive Verfahren,\n"
            "   'Vergleich starten / fortsetzen' beide Slots parallel.\n\n"
            "Environment\n"
            "LunarLander-v3 kontinuierlich: a₀ regelt das Haupttriebwerk (≤ 0 aus, sonst\n"
            "50–100 % Schub), a₁ die Steuertriebwerke (< -0,5 links, > 0,5 rechts). Der\n"
            "Reward belohnt Annäherung, ruhige Lage und Beinkontakt und bestraft\n"
            "Treibstoff. Landung +100, Absturz -100, gelöst ab +200.\n\n"
            "Methoden\n"
            "PPO ist on-policy: Rollouts, GAE, geclipptes Ziel, kein Replay Buffer.\n"
            "TD3 und SAC sind off-policy mit zwei Critics – TD3 mit deterministischem\n"
            "Actor und Action Noise, SAC mit stochastischem Actor und gelernter Entropie.\n"
            "Training exploriert, Evaluation ist deterministisch und ohne Lernupdates.\n\n"
            "Ansichten und Animation\n"
            "'Training' zeigt das aktive Verfahren, 'Vergleich' beide Slots samt ihrer\n"
            "Unterschiede in der Summary. 'Animation zeigen' wirkt jederzeit, auch im\n"
            "laufenden Lauf, und zeigt dann beide Verfahren – das kostet Rechenzeit. Die\n"
            f"Bildrate ist einstellbar (Standard {RENDER_FPS} FPS).\n\n"
            "Kein Lernerfolg?\n"
            "Zu kurzes Budget (PPO braucht als On-Policy-Verfahren deutlich mehr\n"
            "Schritte), zu großer Lernstart t₀, zu kleines Action Noise bei TD3 oder\n"
            "eine zu hohe Lernrate.",
            parent=self.root,
        )

    def close(self) -> None:
        self.stop_event.set()
        self._stop_animation()
        for renderer in self.renderers:
            if renderer is not None:
                renderer.close()
        for runtime in self.slots:
            runtime.close()
        self.checkpoints.cleanup()
        self.root.destroy()
