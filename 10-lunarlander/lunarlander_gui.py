"""Dark Tkinter GUI for the LunarLander Rainbow-DDQN workbench."""

from __future__ import annotations

import queue
import re
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from lunarlander_logic import (
    ACTIVATIONS, ALGORITHMS, DISTRIBUTIONAL_ALGORITHMS, DUELING_ALGORITHMS, MULTISTEP_ALGORITHMS,
    NOISY_ALGORITHMS, OPTIMIZERS, PER_ALGORITHMS, SOLVED_RETURN,
    EpisodeMetric, EvaluationResult, LunarLanderConfig, LunarLanderWorkbench, observation_readout,
)
from lunarlander_render import LunarLanderRenderer


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


ACTION_LABELS = ("nichts tun", "links", "Haupttriebwerk", "rechts")
#: Feste Abspielgeschwindigkeit aus der Bildrate des Environments
#: (`env.metadata["render_fps"] == 50`). Die Workbench verbietet ein
#: Eingabefeld für die Animationsgeschwindigkeit; die Animation ist nur
#: ein- und ausschaltbar.
RENDER_FPS = 50
ANIMATION_INTERVAL_MS = round(1000 / RENDER_FPS)
FIELD_ALGORITHMS = {
    "noisy_sigma_0": NOISY_ALGORITHMS,
    "per_alpha": PER_ALGORITHMS,
    "per_beta_0": PER_ALGORITHMS,
    "per_beta_steps": PER_ALGORITHMS,
    "per_epsilon": PER_ALGORITHMS,
    "value_arch": DUELING_ALGORITHMS,
    "advantage_arch": DUELING_ALGORITHMS,
    "multistep_n": MULTISTEP_ALGORITHMS,
    "c51_atoms": DISTRIBUTIONAL_ALGORITHMS,
    "c51_v_min": DISTRIBUTIONAL_ALGORITHMS,
    "c51_v_max": DISTRIBUTIONAL_ALGORITHMS,
}
COMPARISON_COLORS = (
    "#60a5fa", "#34d399", "#f87171", "#c084fc", "#fbbf24", "#2dd4bf", "#f472b6",
)
EXPORT_DIR = Path(__file__).parent / "exports"
#: Breite der drei Bedienspalten (zwei Parameterspalten, eine Buttonspalte)
#: und die daraus abgeleitete Gesamtbreite des Bedienpanels.
CONTROL_COLUMNS = (320, 336, 236)
CONTROL_WIDTH = sum(CONTROL_COLUMNS) + 16
#: Mindesthöhe, die dem unteren Bereich aus Diagrammen und Summary bleibt.
MIN_CHART_HEIGHT = 250
#: Platz, den Menüleiste, Fensterrahmen und Dock vom Bildschirm beanspruchen.
#: Ohne diesen Abzug reicht das Fenster beim Start unter das macOS-Dock.
SCREEN_MARGIN = 130


class LunarLanderGUI:
    BG, PANEL, FIELD = "#111827", "#1f2937", "#0f172a"
    FG, MUTED, ACCENT = "#f3f4f6", "#cbd5e1", "#60a5fa"
    EVAL_COLOR = "#f59e0b"
    PLOT_INTERVAL = 2.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.workbench = LunarLanderWorkbench()
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.busy = False
        self.renderer = LunarLanderRenderer()
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.last_frame: Optional[np.ndarray] = None
        self.animation_after: Optional[str] = None
        self.animation_observation: Optional[np.ndarray] = None
        self.animation_step = 0
        self.animation_reward = 0.0
        self.live_history: list[EpisodeMetric] = []
        self.evaluation_points: list[tuple[int, EvaluationResult]] = []
        self.comparison_history: dict[str, list[EpisodeMetric]] = {}
        self.comparison_evaluations: dict[str, list[tuple[int, EvaluationResult]]] = {}
        self.comparison_workbenches: dict[str, LunarLanderWorkbench] = {}
        self.active_algorithms: tuple[str, ...] = ()
        self.best_evaluation: Optional[EvaluationResult] = None
        self.best_checkpoint = tempfile.TemporaryDirectory(prefix="lunarlander-best-")
        self.best_base = Path(self.best_checkpoint.name) / "best"
        self.best_lock = threading.Lock()
        self.last_plot = 0.0
        self.progress_by_algorithm: dict[Optional[str], int] = {}
        self._export_snapshot_key: Optional[tuple[Any, ...]] = None
        self._export_stamp: Optional[str] = None
        self._variables()
        self._window()
        self._layout()
        self._show_initial_frame()
        self._refresh_training_plot()
        self._refresh_comparison_plot()
        self._training_summary()
        self.root.after_idle(self._initialize_layout)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _variables(self) -> None:
        defaults = LunarLanderConfig()
        self.values: dict[str, tk.StringVar] = {}
        for name, value in defaults.__dict__.items():
            if name == "algorithm":
                continue
            if isinstance(value, tuple):
                value = ",".join(map(str, value))
            self.values[name] = tk.StringVar(value="" if value is None else str(value))
        self.algorithm = tk.StringVar(value=defaults.algorithm)
        self.evaluation_episodes = tk.StringVar(value="5")
        self.evaluation_interval = tk.StringVar(value="10000")
        self.compare = {name: tk.BooleanVar(value=name in ("DDQN", "Rainbow DDQN")) for name in ALGORITHMS}
        self.animation_enabled = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Bereit")
        self.progress = tk.DoubleVar(value=0)
        self.observation = tk.StringVar(value="Noch keine Episode gestartet.")

    def _window(self) -> None:
        width = min(1560, max(CONTROL_WIDTH + 420, self.root.winfo_screenwidth() - 40))
        height = min(940, self.root.winfo_screenheight() - SCREEN_MARGIN)
        self.root.title("LunarLander Rainbow-DDQN Workbench")
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(CONTROL_WIDTH + 420, width), min(700, height))
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
        figure.subplots_adjust(left=.11, right=.78, bottom=.30, top=.82)
        axes = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=master)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return figure, axes, canvas

    def _layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="LunarLander Rainbow-DDQN Workbench", font=("TkDefaultFont", 18, "bold")).pack(side="left")
        ttk.Label(header, text="Sieben Double-DQN-Varianten – sicher auf der Plattform landen").pack(side="left", padx=16)
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

    def _entry(self, parent: ttk.Frame, row: int, column: int, label: str, name: str,
               width: int = 8) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4))
        entry = ttk.Entry(parent, textvariable=self.values[name], width=width, justify="right")
        entry.grid(row=row, column=column + 1, sticky="e", padx=(0, 6))
        return entry

    def _entry_pairs(self, parent: ttk.Frame, specifications: tuple[tuple[str, str], ...],
                     start_row: int = 0) -> dict[str, ttk.Entry]:
        """Parameter paarweise anordnen: zwei Label/Feld-Paare je Zeile.

        Das halbiert die Höhe der Parametergruppen, sodass Bedienpanel und
        unterer Diagrammbereich gemeinsam auf typische Laptop-Auflösungen
        passen, ohne dass ein Feld abgeschnitten wird.
        """
        entries: dict[str, ttk.Entry] = {}
        for index, (label, name) in enumerate(specifications):
            row, group = divmod(index, 2)
            entries[name] = self._entry(parent, start_row + row, group * 2, label, name)
        parent.columnconfigure((1, 3), weight=1)
        return entries

    def _controls(self, parent: ttk.Frame) -> None:
        # Bedarfsgerechte Spaltenbreiten statt uniformer Drittel: die
        # Parametergruppen sind unterschiedlich breit, und eine zu schmale
        # Spalte würde ihre Felder über den Spaltenrand hinausdrücken.
        parent.columnconfigure(0, minsize=CONTROL_COLUMNS[0])
        parent.columnconfigure(1, minsize=CONTROL_COLUMNS[1])
        parent.columnconfigure(2, minsize=CONTROL_COLUMNS[2], weight=1)
        left, middle = ttk.Frame(parent), ttk.Frame(parent)
        actions = ttk.LabelFrame(parent, text="Steuerung", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        middle.grid(row=0, column=1, sticky="nsew", padx=4)
        actions.grid(row=0, column=2, sticky="new", padx=(4, 0))

        training = ttk.LabelFrame(left, text="Training und Replay Buffer", padding=5); training.pack(fill="x", pady=(0, 3))
        ttk.Label(training, text="Algorithmus").grid(row=0, column=0, sticky="w", padx=(0, 4))
        combo = ttk.Combobox(training, textvariable=self.algorithm, values=ALGORITHMS, state="readonly")
        combo.grid(row=0, column=1, columnspan=3, sticky="ew", pady=(0, 2))
        combo.bind("<<ComboboxSelected>>", lambda _event: self._algorithm_fields())
        self._entry_pairs(training, (
            ("Schritte N", "total_timesteps"), ("Start t₀", "learning_starts"),
            ("Buffer |D|", "buffer_size"), ("Batch B", "batch_size"),
            ("Freq. fₜ", "train_freq"), ("Grad. G", "gradient_steps"),
        ), start_row=1)

        exploration = ttk.LabelFrame(left, text="Exploration", padding=5); exploration.pack(fill="x", pady=3)
        exploration_fields = self._entry_pairs(exploration, (
            ("Start ε₀", "exploration_initial_eps"), ("Ende εₘᵢₙ", "exploration_final_eps"),
            ("Abkling fₑ", "exploration_fraction"),
        ))
        self.exploration_entries = list(exploration_fields.values())

        evaluation = ttk.LabelFrame(left, text="Evaluation", padding=5); evaluation.pack(fill="x", pady=3)
        ttk.Label(evaluation, text="Episoden M").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(evaluation, textvariable=self.evaluation_episodes, width=8, justify="right").grid(row=0, column=1, sticky="e", padx=(0, 6))
        ttk.Label(evaluation, text="Intervall").grid(row=0, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(evaluation, textvariable=self.evaluation_interval, width=8, justify="right").grid(row=0, column=3, sticky="e")
        ttk.Label(evaluation, text="Intervall 0 = keine Auto-Evaluation",
                  foreground=self.MUTED).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 0))
        evaluation.columnconfigure((1, 3), weight=1)

        comparison = ttk.LabelFrame(left, text="Algorithmen im Vergleich", padding=5); comparison.pack(fill="x", pady=3)
        for index, name in enumerate(ALGORITHMS):
            ttk.Checkbutton(comparison, text=name, variable=self.compare[name]).grid(
                row=index // 2, column=index % 2, sticky="w", padx=(0, 6)
            )

        learning = ttk.LabelFrame(middle, text="Lernen und Target Network", padding=5); learning.pack(fill="x", pady=(0, 3))
        self._entry_pairs(learning, (
            ("Lernrate α", "learning_rate"), ("Diskont γ", "gamma"),
            ("Soft-Upd. τ", "tau"), ("Target C", "target_update_interval"),
            ("Grad-Norm", "max_grad_norm"), ("Seed s", "seed"),
        ))

        network = ttk.LabelFrame(middle, text="Neuronales Netz und Optimizer", padding=5); network.pack(fill="x", pady=3)
        ttk.Label(network, text="Hidden h").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(network, textvariable=self.values["net_arch"], width=8, justify="right").grid(row=0, column=1, sticky="e", padx=(0, 6))
        ttk.Label(network, text="Aktiv. φ").grid(row=0, column=2, sticky="w", padx=(0, 4))
        ttk.Combobox(network, textvariable=self.values["activation"], values=tuple(ACTIVATIONS),
                     state="readonly", width=8, justify="right").grid(row=0, column=3, sticky="e")
        ttk.Label(network, text="Optimizer").grid(row=1, column=0, sticky="w", padx=(0, 4))
        ttk.Combobox(network, textvariable=self.values["optimizer"], values=tuple(OPTIMIZERS),
                     state="readonly", width=8, justify="right").grid(row=1, column=1, sticky="e", padx=(0, 6))
        ttk.Label(network, text="ε_opt").grid(row=1, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(network, textvariable=self.values["optimizer_eps"], width=8, justify="right").grid(row=1, column=3, sticky="e")
        ttk.Label(network, text="Zerfall λ").grid(row=2, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(network, textvariable=self.values["optimizer_weight_decay"], width=8, justify="right").grid(row=2, column=1, sticky="e", padx=(0, 6))
        network.columnconfigure((1, 3), weight=1)

        variants = ttk.LabelFrame(middle, text="Variantenparameter", padding=5); variants.pack(fill="x", pady=3)
        self.variant_entries = self._entry_pairs(variants, (
            ("Noisy σ₀", "noisy_sigma_0"), ("PER α", "per_alpha"),
            ("PER β₀", "per_beta_0"), ("PER ε", "per_epsilon"),
            ("β-Steps", "per_beta_steps"), ("n_step", "multistep_n"),
            ("Value hᵥ", "value_arch"), ("Advant hₐ", "advantage_arch"),
            ("n_atoms", "c51_atoms"), ("V_min", "c51_v_min"),
            ("V_max", "c51_v_max"),
        ))

        self.buttons = []
        for text, command in (("Training starten / fortsetzen", self.start_training), ("Stoppen", self.stop),
                              ("Deterministisch evaluieren", self.start_evaluation),
                              ("Sichtbare Episode abspielen", self.animate),
                              ("Algorithmen vergleichen", self.start_comparison),
                              ("Bestes Modell wiederherstellen", self.restore_best), ("Neues Modell", self.reset),
                              ("Modell und Replay speichern", self.save), ("Modell und Replay laden", self.load)):
            button = ttk.Button(actions, text=text, command=command)
            button.pack(fill="x", pady=1, ipady=1); self.buttons.append(button)
        self.stop_button, self.best_button = self.buttons[1], self.buttons[5]
        self.stop_button.configure(state="disabled"); self.best_button.configure(state="disabled")
        ttk.Checkbutton(actions, text=f"Animation zeigen ({RENDER_FPS} FPS)",
                        variable=self.animation_enabled).pack(fill="x", pady=(6, 0))
        ttk.Progressbar(actions, variable=self.progress, maximum=100).pack(fill="x", pady=(6, 2))
        ttk.Label(actions, textvariable=self.status, foreground=self.ACCENT, wraplength=230).pack(fill="x", pady=2)
        self._algorithm_fields()

    def _algorithm_fields(self) -> None:
        selected = self.algorithm.get()
        for name, entry in self.variant_entries.items():
            entry.configure(state="normal" if selected in FIELD_ALGORITHMS[name] else "disabled")
        state = "disabled" if selected in NOISY_ALGORITHMS else "normal"
        for entry in self.exploration_entries:
            entry.configure(state=state)

    def _environment(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Offizieller Gymnasium-RGB-Frame", padding=6)
        frame.grid(row=0, column=1, sticky="nsew")
        self.image_label = ttk.Label(frame, anchor="center"); self.image_label.pack(fill="both", expand=True)
        ttk.Label(frame, textvariable=self.observation, justify="left", font="TkFixedFont").pack(anchor="w", pady=(4, 0))

    def _initialize_layout(self) -> None:
        self.root.update_idletasks()
        available = self.splitter.winfo_height()
        required = max((child.winfo_y() + child.winfo_reqheight() for child in self.controls.winfo_children()), default=400) + 10
        self.splitter.sashpos(0, min(max(required, int(available * .50)), available - MIN_CHART_HEIGHT))
        if self.last_frame is not None:
            self._show_frame(self.last_frame)

    def layout_visibility_issues(self) -> list[str]:
        self.root.update_idletasks()
        issues = []
        left, top = self.controls.winfo_rootx(), self.controls.winfo_rooty()
        right, bottom = left + self.controls.winfo_width(), top + self.controls.winfo_height()
        stack = list(self.controls.winfo_children())
        while stack:
            widget = stack.pop(); stack.extend(widget.winfo_children())
            if isinstance(widget, (ttk.Entry, ttk.Combobox, ttk.Button, ttk.Checkbutton, ttk.Progressbar)):
                if not widget.winfo_ismapped() or widget.winfo_rootx() < left or widget.winfo_rooty() < top or widget.winfo_rootx() + widget.winfo_width() > right or widget.winfo_rooty() + widget.winfo_height() > bottom:
                    issues.append(
                        f"{widget}: ({widget.winfo_rootx()},{widget.winfo_rooty()},"
                        f"{widget.winfo_width()}x{widget.winfo_height()}) außerhalb "
                        f"({left},{top},{right - left}x{bottom - top})"
                    )
        # Eine Parametergruppe, die breiter ist als ihre Rasterspalte, würde
        # ihre Felder unbemerkt in die Nachbarspalte drücken. Der reine
        # Positionsvergleich oben erkennt das nicht, weil beide Spalten
        # innerhalb desselben Bedienpanels liegen.
        for column, child in enumerate(self.controls.grid_slaves(row=0)[::-1]):
            allowed = CONTROL_COLUMNS[column] if column < len(CONTROL_COLUMNS) else child.winfo_width()
            for group in child.winfo_children():
                if group.winfo_reqwidth() > allowed:
                    issues.append(
                        f"Spalte {column}: Gruppe benötigt {group.winfo_reqwidth()} px, "
                        f"verfügbar sind {allowed} px"
                    )
        # Geprüft wird das gerade sichtbare Diagramm: das jeweils andere Tab
        # ist von Tk bewusst nicht gemappt und wäre kein echter Befund.
        visible_canvas = (self.comparison_canvas if self.charts.index(self.charts.select()) == 1
                          else self.canvas).get_tk_widget()
        for label, widget in (("Visualisierung", self.image_label),
                              ("Diagramm", visible_canvas),
                              ("Summary", self.summary_text)):
            if not widget.winfo_ismapped() or widget.winfo_width() <= 1 or widget.winfo_height() <= 1:
                issues.append(label)
        return issues

    def _config(self) -> LunarLanderConfig:
        integers = {"total_timesteps", "buffer_size", "learning_starts", "batch_size", "train_freq",
                    "gradient_steps", "target_update_interval", "per_beta_steps", "multistep_n", "c51_atoms"}
        tuples = {"net_arch", "value_arch", "advantage_arch"}
        strings = {"activation", "optimizer"}
        labels = {
            "total_timesteps": "Trainingsschritte N", "buffer_size": "Replay Buffer |D|",
            "learning_starts": "Lernstart t₀", "batch_size": "Batch-Größe B",
            "train_freq": "Trainingsfrequenz fₜ", "gradient_steps": "Gradientenschritte G",
            "target_update_interval": "Target-Intervall C", "per_beta_steps": "β-Anneal.-Schritte",
            "multistep_n": "Schritte n_step", "c51_atoms": "Atome n_atoms", "net_arch": "Hidden Layers h",
            "value_arch": "Value-Stream hᵥ", "advantage_arch": "Advantage-Stream hₐ", "seed": "Zufallsstart s",
        }
        data: dict[str, Any] = {}
        for name, variable in self.values.items():
            text = variable.get().strip()
            label = labels.get(name, name)
            try:
                if name in tuples:
                    data[name] = tuple(int(item.strip()) for item in text.split(",") if item.strip())
                elif name in integers:
                    data[name] = int(text)
                elif name == "seed":
                    data[name] = int(text) if text else None
                elif name in strings:
                    data[name] = text
                else:
                    data[name] = float(text)
            except ValueError as error:
                expected = "ganze Zahl" if name in integers or name == "seed" else (
                    "Liste ganzer Zahlen, z. B. '256,256'" if name in tuples else "Zahl")
                raise ValueError(f"{label}: '{text}' ist keine gültige Eingabe. Erwartet: {expected}.") from error
        config = LunarLanderConfig(algorithm=self.algorithm.get(), **data)
        config.validate()
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
        if not busy and self.best_evaluation is None:
            self.best_button.configure(state="disabled")

    # ----------------------------------------------------------------- Training

    def start_training(self) -> None:
        if self.busy: return
        try:
            config = self._config()
            episodes, interval = self._evaluation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root); return
        if self.workbench.model is not None and config.signature() != self.workbench.config.signature():
            if not messagebox.askyesno("Neues Modell", "Geänderte Modellparameter erfordern einen Reset. Fortfahren?", parent=self.root):
                return
            self.workbench.close(); self.workbench = LunarLanderWorkbench(config)
            self.evaluation_points.clear(); self.best_evaluation = None
        else:
            self.workbench.config = config
        self._clear_comparison()
        self.live_history = list(self.workbench.history)
        self.stop_event.clear(); self.progress.set(0)
        self.progress_by_algorithm = {None: 0}
        self.charts.select(0)
        self._set_busy(True, f"Läuft – {config.algorithm} über {config.total_timesteps:,} Schritte".replace(",", "."))
        self.worker = threading.Thread(target=self._train_worker, args=(episodes, interval), daemon=True)
        self.worker.start(); self.root.after(50, self._poll)

    def _train_worker(self, episodes: int, interval: int) -> None:
        try:
            self.workbench.train(
                self.stop_event, self.events, evaluation_interval=interval,
                evaluation_episodes=episodes, best_callback=self._save_best,
            )
            self.events.put(("training_done", self.stop_event.is_set()))
        except Exception as error:
            self.events.put(("error", error))

    # --------------------------------------------------------------- Vergleich

    def start_comparison(self) -> None:
        if self.busy: return
        try:
            config = self._config()
            episodes, interval = self._evaluation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültiger Vergleich", str(error), parent=self.root); return
        algorithms = [name for name in ALGORITHMS if self.compare[name].get()]
        if len(algorithms) < 2:
            messagebox.showerror("Ungültiger Vergleich", "Wähle mindestens zwei Algorithmen.", parent=self.root); return
        if self.comparison_workbenches and next(iter(self.comparison_workbenches.values())).config.signature() != config.signature():
            self._clear_comparison()
        for name in algorithms:
            self.comparison_history.setdefault(name, [])
            self.comparison_evaluations.setdefault(name, [])
            self.comparison_workbenches.setdefault(name, LunarLanderWorkbench(replace(config, algorithm=name)))
        self.active_algorithms = tuple(algorithms)
        self.stop_event.clear(); self.progress.set(0)
        self.progress_by_algorithm = {name: 0 for name in algorithms}
        total = len(algorithms) * config.total_timesteps
        self.charts.select(1)
        self._set_busy(True, f"Läuft – Vergleich: {len(algorithms)} Algorithmen × "
                             f"{config.total_timesteps:,} Schritte = {total:,} Schritte".replace(",", "."))
        self._refresh_comparison_plot(); self._comparison_summary()
        self.worker = threading.Thread(target=self._comparison_worker, args=(algorithms, episodes, interval), daemon=True)
        self.worker.start(); self.root.after(50, self._poll)

    def _comparison_worker(self, algorithms: list[str], episodes: int, interval: int) -> None:
        errors: queue.Queue = queue.Queue(); barrier = threading.Barrier(len(algorithms))
        def run(name: str) -> None:
            try:
                barrier.wait()
                self.comparison_workbenches[name].train(
                    self.stop_event, self.events, name,
                    evaluation_interval=interval, evaluation_episodes=episodes,
                )
            except Exception as error:
                errors.put(error); self.stop_event.set()
        threads = [threading.Thread(target=run, args=(name,), daemon=True) for name in algorithms]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        if not errors.empty():
            self.events.put(("error", errors.get()))
        else:
            self.events.put(("comparison_done", self.stop_event.is_set()))

    # -------------------------------------------------------------- Evaluation

    def start_evaluation(self) -> None:
        if self.busy or self.workbench.model is None:
            if self.workbench.model is None:
                messagebox.showinfo("Keine Evaluation", "Trainiere zuerst ein Modell.", parent=self.root)
            return
        try:
            episodes, _ = self._evaluation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültige Evaluation", str(error), parent=self.root); return
        self._set_busy(True, "Läuft – deterministische Evaluation")
        self.worker = threading.Thread(target=self._evaluation_worker, args=(episodes,), daemon=True)
        self.worker.start(); self.root.after(50, self._poll)

    def _evaluation_worker(self, episodes: int) -> None:
        try:
            result = self.workbench.evaluate(episodes, self.workbench.config.seed or 0)
            episode = len(self.live_history or self.workbench.history)
            self._save_best(episode, self.workbench.model.num_timesteps, result)
            self.events.put(("manual_evaluation", (episode, result)))
        except Exception as error:
            self.events.put(("error", error))

    def _save_best(self, _episode: int, _steps: int, result: EvaluationResult) -> None:
        with self.best_lock:
            if self.best_evaluation is None or result.mean_reward > self.best_evaluation.mean_reward:
                self.workbench.save(self.best_base)
                self.best_evaluation = result

    def restore_best(self) -> None:
        if self.busy or self.best_evaluation is None: return
        try:
            restored = LunarLanderWorkbench.load(self.best_base)
        except Exception as error:
            messagebox.showerror("Wiederherstellen fehlgeschlagen", str(error), parent=self.root); return
        self.workbench.close(); self.workbench = restored
        self.live_history = list(restored.history)
        self.status.set("Abgeschlossen – bestes Modell wiederhergestellt")
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
                self.live_history.append(payload); redraw_training = True
            elif kind == "comparison_episode":
                name, metric = payload; self.comparison_history[name].append(metric); redraw_comparison = True
            elif kind == "evaluation":
                episode, _steps, result = payload
                self.evaluation_points.append((episode, result)); redraw_training = True
            elif kind == "comparison_evaluation":
                name, episode, _steps, result = payload
                self.comparison_evaluations[name].append((episode, result)); redraw_comparison = True
            elif kind == "progress":
                name, elapsed = payload
                self.progress_by_algorithm[name] = max(elapsed, self.progress_by_algorithm.get(name, 0))
                completed = min(self.progress_by_algorithm.values()) if self.active_algorithms else self.progress_by_algorithm[name]
                self.progress.set(min(100, 100 * completed / max(1, self.workbench.config.total_timesteps)))
            elif kind == "training_done":
                self._set_busy(False, "Gestoppt – Training" if payload else "Abgeschlossen – Training")
                redraw_training = True
            elif kind == "comparison_done":
                self._set_busy(False, "Gestoppt – Vergleich" if payload else "Abgeschlossen – Vergleich")
                redraw_comparison = True
            elif kind == "manual_evaluation":
                episode, result = payload
                self.evaluation_points.append((episode, result))
                self._set_busy(False, "Abgeschlossen – Evaluation"); redraw_training = True
            elif kind == "error":
                self._set_busy(False, "Fehler"); messagebox.showerror("Fehler", str(payload), parent=self.root)
        if self.best_evaluation is not None:
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

    def _style(self, axes: Any, figure: Figure) -> None:
        axes.set_facecolor(self.FIELD); axes.tick_params(colors=self.MUTED); axes.grid(color="#475569", alpha=.4)
        for spine in axes.spines.values(): spine.set_color("#64748b")
        axes.title.set_color(self.FG); axes.xaxis.label.set_color(self.MUTED); axes.yaxis.label.set_color(self.MUTED)
        legend = axes.legend(loc="center left", bbox_to_anchor=(1.02, .5), borderaxespad=0, fontsize=8)
        legend.get_frame().set_facecolor(self.PANEL); legend.get_frame().set_edgecolor("#64748b")
        for text in legend.get_texts(): text.set_color(self.FG)

    def _refresh_training_plot(self) -> None:
        self.axes.clear()
        history = self.live_history or self.workbench.history
        if history:
            episodes = [item.episode for item in history]
            rewards = [item.reward for item in history]
            x, y = downsample_minmax(episodes, rewards)
            self.axes.plot(x, y, color=self.ACCENT, alpha=.15, linewidth=.8)
            self.axes.plot(episodes, rolling_average(rewards), color=self.ACCENT, linewidth=1.6,
                           label=f"Training – {self.workbench.config.algorithm}")
        if self.evaluation_points:
            # Evaluation optisch klar vom explorativen Training getrennt.
            self.axes.plot([point[0] for point in self.evaluation_points],
                           [point[1].mean_reward for point in self.evaluation_points],
                           color=self.EVAL_COLOR, marker="o", markersize=4, linestyle="--",
                           linewidth=1.3, label="Evaluation (deterministisch)")
        self.axes.axhline(SOLVED_RETURN, color="#a3e635", linestyle=":", linewidth=1.2, label="Gelöst ab +200")
        self.axes.set(title="Episoden-Return", xlabel="Episode", ylabel="Return")
        self._style(self.axes, self.figure); self.canvas.draw_idle()

    def _refresh_comparison_plot(self) -> None:
        self.comparison_axes.clear()
        colors = dict(zip(ALGORITHMS, COMPARISON_COLORS))
        for name, history in self.comparison_history.items():
            if not history: continue
            episodes = [item.episode for item in history]
            rewards = [item.reward for item in history]
            x, y = downsample_minmax(episodes, rewards)
            self.comparison_axes.plot(x, y, color=colors[name], alpha=.08, linewidth=.7)
            self.comparison_axes.plot(episodes, rolling_average(rewards), color=colors[name],
                                      linewidth=1.6, label=name)
        self.comparison_axes.axhline(SOLVED_RETURN, color="#a3e635", linestyle=":", linewidth=1.2,
                                     label="Gelöst ab +200")
        self.comparison_axes.set(title="Live-Algorithmenvergleich", xlabel="Episode", ylabel="Return")
        self._style(self.comparison_axes, self.comparison_figure); self.comparison_canvas.draw_idle()

    # ---------------------------------------------------------------- Summary

    def _write_summary(self, text: str) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")

    def _summary_text(self) -> str:
        return self.summary_text.get("1.0", "end").rstrip("\n")

    def _table(self, title: str, statistics: dict[str, tuple[Any, ...]], footer: list[str]) -> None:
        labels = ("Episoden", "Schritte", "Ø Return", "Landequote", "Gelöst-Quote", "Absturzquote")
        if not statistics:
            self._write_summary(f"{title}\n\nNoch keine vollständig abgeschlossene Episode.")
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

    def _training_summary(self) -> None:
        history = self.live_history or self.workbench.history
        steps = self.workbench.model.num_timesteps if self.workbench.model else 0
        if history:
            stats = (
                len(history), f"{steps:,}".replace(",", "."),
                f"{np.mean([item.reward for item in history]):.1f}",
                f"{np.mean([item.landed for item in history]):.1%}",
                f"{np.mean([item.solved for item in history]):.1%}",
                f"{np.mean([item.crashed for item in history]):.1%}",
            )
        else:
            stats = (0, f"{steps:,}".replace(",", "."), "—", "—", "—", "—")
        footer = [
            f"Trainingsbudget: {self.workbench.config.total_timesteps:,}".replace(",", ".") + " Schritte",
            "Kurvenpunkte entstehen nur für vollständig abgeschlossene Episoden.",
        ]
        if self.evaluation_points:
            last = self.evaluation_points[-1][1]
            footer.append(
                f"Letzte Evaluation: Ø {last.mean_reward:.1f} ± {last.reward_std:.1f} | "
                f"Landung {last.landing_rate:.0%} | gelöst {last.solved_rate:.0%} | "
                f"Länge {last.mean_length:.0f}"
            )
        if self.best_evaluation is not None:
            best = self.best_evaluation
            footer.append(
                f"Beste Evaluation: Ø {best.mean_reward:.1f} | Landung {best.landing_rate:.0%} | "
                f"gelöst {best.solved_rate:.0%}  (als Checkpoint gesichert)"
            )
        self._table("Training", {self.workbench.config.algorithm: stats}, footer)

    def _comparison_summary(self) -> None:
        stats = {}
        for name, history in self.comparison_history.items():
            if not history: continue
            stats[name] = (
                len(history), f"{history[-1].timesteps:,}".replace(",", "."),
                f"{np.mean([item.reward for item in history]):.1f}",
                f"{np.mean([item.landed for item in history]):.1%}",
                f"{np.mean([item.solved for item in history]):.1%}",
                f"{np.mean([item.crashed for item in history]):.1%}",
            )
        footer = []
        for name in self.active_algorithms:
            points = self.comparison_evaluations.get(name) or []
            if points:
                last = points[-1][1]
                footer.append(f"{name}: letzte Evaluation Ø {last.mean_reward:.1f} | "
                              f"gelöst {last.solved_rate:.0%}")
        self._table("Vergleich", stats, footer)

    # ----------------------------------------------------------------- Export

    def _config_snapshot_lines(self) -> list[str]:
        if self.active_algorithms:
            base = next(iter(self.comparison_workbenches.values())).config
            lines = [f"Verglichene Algorithmen: {', '.join(self.active_algorithms)}"]
            values = {key: value for key, value in asdict(base).items() if key != "algorithm"}
        else:
            lines = []
            values = asdict(self.workbench.config)
        lines.extend(f"{key}: {value}" for key, value in values.items())
        lines.append(f"evaluation_episodes: {self.evaluation_episodes.get()}")
        lines.append(f"evaluation_interval: {self.evaluation_interval.get()}")
        return lines

    def _export_snapshot(self) -> tuple[str, tuple[Any, ...]]:
        """Kennzeichnet den aktuell sichtbaren Trainings-/Vergleichsstand.

        PNG- und TXT-Export teilen sich denselben Dateinamensstamm, solange
        sich zwischen beiden Exporten keine weitere Episode geändert hat – so
        bleiben Diagramm und zugehörige Konfigurationsdatei eindeutig
        zuordenbar, ohne sich auf zufällig gleiche Uhrzeiten zu verlassen.
        """
        if self.active_algorithms:
            slug = "vergleich-" + "-".join(slugify(name) for name in self.active_algorithms)
            key = ("comparison", self.active_algorithms, tuple(
                len(self.comparison_history.get(name, ())) for name in self.active_algorithms))
        else:
            slug = slugify(self.workbench.config.algorithm)
            history = self.live_history or self.workbench.history
            key = ("single", self.workbench.config.algorithm, len(history))
        return slug, key

    def _export_base_name(self) -> str:
        slug, key = self._export_snapshot()
        if key != self._export_snapshot_key:
            self._export_snapshot_key = key
            self._export_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"lunarlander_{slug}_{self._export_stamp}"

    def export_chart(self) -> None:
        EXPORT_DIR.mkdir(exist_ok=True)
        comparison = self.charts.index(self.charts.select()) == 1
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

    def _show_initial_frame(self) -> None:
        observation, _info, frame = self.renderer.reset(self.workbench.config.seed)
        self.animation_observation = observation
        self._show_frame(frame)
        self._show_readout(observation)

    def _show_readout(self, observation: np.ndarray, action: Optional[int] = None) -> None:
        values = observation_readout(observation)
        legs = f"{'ja' if values['left_leg'] else 'nein'} / {'ja' if values['right_leg'] else 'nein'}"
        first = (f"Schritt: {self.animation_step:4d}   Return: {self.animation_reward:+8.1f}   "
                 f"Action: {ACTION_LABELS[action] if action is not None else '—'}")
        second = (f"x: {values['x']:+.3f}   y: {values['y']:+.3f}   "
                  f"vx: {values['vx']:+.3f}   vy: {values['vy']:+.3f}")
        third = (f"Winkel: {values['angle_degrees']:+6.1f}°   "
                 f"\u03c9: {values['angular_velocity']:+.3f} rad/s   Beine: {legs}")
        self.observation.set("\n".join((first, second, third)))

    def _show_frame(self, frame: np.ndarray) -> None:
        self.last_frame = np.asarray(frame)
        width = max(240, self.image_label.winfo_width() - 12)
        height = max(160, self.image_label.winfo_height() - 12)
        image = Image.fromarray(self.last_frame)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

    def animate(self) -> None:
        if self.busy: return
        if self.workbench.model is None:
            messagebox.showinfo("Keine Episode", "Trainiere zuerst ein Modell.", parent=self.root); return
        self._stop_animation()
        observation, _info, frame = self.renderer.reset(self.workbench.config.seed)
        self.animation_observation, self.animation_step, self.animation_reward = observation, 0, 0.0
        self._show_frame(frame)
        if self.animation_enabled.get():
            self._animate_step()
        else:
            self._run_episode_without_animation()

    def _policy_action(self) -> int:
        with self.workbench.deterministic_policy():
            action, _ = self.workbench.model.predict(self.animation_observation, deterministic=True)
        return int(action)

    def _episode_outcome_text(self, reward: float, truncated: bool) -> str:
        if truncated:
            return "Abgeschlossen – Zeitlimit erreicht"
        if reward > 50:
            return "Abgeschlossen – sicher gelandet"
        if reward < -50:
            return "Abgeschlossen – abgestürzt"
        return "Abgeschlossen – Episode beendet"

    def _run_episode_without_animation(self) -> None:
        """Sichtbare Episode ohne Einzelbildanimation: nur das Endbild zählt."""
        terminated = truncated = False
        reward = 0.0
        while not (terminated or truncated):
            action = self._policy_action()
            observation, reward, terminated, truncated, _info, frame = self.renderer.step(action)
            self.animation_observation = observation
            self.animation_step += 1
            self.animation_reward += reward
        self._show_frame(frame)
        self._show_readout(self.animation_observation, action)
        self.status.set(self._episode_outcome_text(reward, truncated))

    def _animate_step(self) -> None:
        action = self._policy_action()
        observation, reward, terminated, truncated, _info, frame = self.renderer.step(action)
        self.animation_observation = observation
        self.animation_step += 1
        self.animation_reward += reward
        self._show_frame(frame)
        self._show_readout(observation, action)
        if terminated or truncated:
            self.animation_after = None
            self.status.set(self._episode_outcome_text(reward, truncated))
            return
        self.animation_after = self.root.after(ANIMATION_INTERVAL_MS, self._animate_step)

    def _stop_animation(self) -> None:
        if self.animation_after:
            self.root.after_cancel(self.animation_after)
            self.animation_after = None

    # ------------------------------------------------------------ Modelldatei

    def reset(self) -> None:
        if self.busy: return
        if not messagebox.askyesno("Neues Modell", "Aktuellen Lernzustand verwerfen?", parent=self.root):
            return
        try:
            config = self._config()
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root); return
        self.workbench.close(); self.workbench = LunarLanderWorkbench(config)
        self.live_history.clear(); self.evaluation_points.clear(); self.best_evaluation = None
        self.best_button.configure(state="disabled")
        self._clear_comparison()
        self._refresh_training_plot(); self._refresh_comparison_plot(); self._training_summary()
        self.status.set("Bereit")

    def save(self) -> None:
        if self.workbench.model is None:
            messagebox.showinfo("Nichts zu speichern", "Trainiere zuerst ein Modell.", parent=self.root); return
        path = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".zip", filetypes=(("SB3-Modell", "*.zip"),),
            initialfile=f"lunarlander_{slugify(self.workbench.config.algorithm)}.zip",
        )
        if not path: return
        try:
            self.workbench.save(path)
        except Exception as error:
            messagebox.showerror("Speichern fehlgeschlagen", str(error), parent=self.root); return
        self.status.set("Abgeschlossen – Modell gespeichert")

    def load(self) -> None:
        if self.busy: return
        path = filedialog.askopenfilename(parent=self.root, filetypes=(("SB3-Modell", "*.zip"),))
        if not path: return
        try:
            loaded = LunarLanderWorkbench.load(path)
        except Exception as error:
            messagebox.showerror("Laden fehlgeschlagen", str(error), parent=self.root); return
        self.workbench.close(); self.workbench = loaded
        self.algorithm.set(loaded.config.algorithm)
        for name, value in loaded.config.__dict__.items():
            if name == "algorithm":
                continue
            if isinstance(value, tuple):
                value = ",".join(map(str, value))
            self.values[name].set("" if value is None else str(value))
        self.live_history = list(loaded.history)
        self.evaluation_points.clear(); self.best_evaluation = None
        self.best_button.configure(state="disabled")
        self._algorithm_fields(); self._refresh_training_plot(); self._training_summary()
        self.status.set("Bereit – Modell geladen")

    def _clear_comparison(self) -> None:
        for workbench in self.comparison_workbenches.values():
            workbench.close()
        self.comparison_workbenches.clear(); self.comparison_history.clear()
        self.comparison_evaluations.clear(); self.active_algorithms = ()

    def instructions(self) -> None:
        messagebox.showinfo(
            "Bedienungsanleitung",
            "Ablauf\n"
            "1. Algorithmus und Parameter wählen, dann 'Training starten / fortsetzen'.\n"
            "2. Während des Trainings wird im eingestellten Schrittintervall automatisch\n"
            "   deterministisch evaluiert; der beste Stand wird als Checkpoint gesichert.\n"
            "3. 'Algorithmen vergleichen' trainiert die ausgewählten Varianten parallel.\n\n"
            "Environment und Reward\n"
            "LunarLander-v3: vier Actions (nichts, links, Haupttriebwerk, rechts).\n"
            "Der Reward belohnt Annäherung an die Plattform, ruhige Lage und Beinkontakt\n"
            "und bestraft Treibstoffverbrauch. Landung gibt +100, Absturz -100.\n"
            "Höhere Werte sind besser; ab +200 gilt eine Episode als gelöst.\n\n"
            "Methoden\n"
            "Alle sieben Varianten teilen dieselbe Double-DQN-Zielberechnung. Noisy, PER,\n"
            "Dueling, Multi-Step und C51 sind einzeln zuschaltbar; Rainbow DDQN kombiniert\n"
            "alle fünf. Variantenfelder sind nur im passenden Modus aktiv.\n\n"
            "Training und Evaluation\n"
            "Training exploriert (ε-greedy beziehungsweise NoisyNet), Evaluation ist immer\n"
            "deterministisch, ohne Rauschen und ohne Lernupdates.\n\n"
            "Ansichten\n"
            "Der Tab 'Training' zeigt Episodenwerte, den gleitenden Mittelwert über 20\n"
            "Episoden und getrennt davon die deterministischen Evaluationen. Der Tab\n"
            "'Vergleich' stellt die Algorithmen gegenüber. Die Animation läuft mit fester\n"
            f"Bildrate von {RENDER_FPS} FPS und lässt sich abschalten.\n\n"
            "Kein Lernerfolg?\n"
            "Typische Ursachen: zu kurzes Trainingsbudget, zu schnell abfallendes Epsilon,\n"
            "zu kleiner Replay Buffer oder ein für C51 zu enger Bereich V_min/V_max.",
            parent=self.root,
        )

    def close(self) -> None:
        self.stop_event.set()
        self._stop_animation()
        self.renderer.close()
        self.workbench.close()
        self._clear_comparison()
        self.best_checkpoint.cleanup()
        self.root.destroy()
