"""Responsive Tkinter interface for the CartPole DQN workbench."""

from __future__ import annotations

import queue
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from cartpole_logic import (
    ACTIVATIONS,
    ALGORITHMS,
    OPTIMIZERS,
    CartPoleWorkbench,
    DQNConfig,
    EpisodeMetric,
    EvaluationResult,
)
from cartpole_render import CartPoleRenderer


def rolling_average(values: np.ndarray, window_size: int = 20) -> np.ndarray:
    """Return a trailing average while retaining the initial partial window."""
    return np.asarray([
        np.mean(values[max(0, index - window_size + 1):index + 1])
        for index in range(len(values))
    ])


def downsample_minmax(
    x_values: list[int] | np.ndarray,
    y_values: list[float] | np.ndarray,
    max_points: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce display points while retaining local minima and maxima."""
    x_array = np.asarray(x_values)
    y_array = np.asarray(y_values)
    if len(x_array) <= max_points:
        return x_array, y_array
    bin_count = max(1, max_points // 2)
    edges = np.linspace(0, len(x_array), bin_count + 1, dtype=int)
    indices = {0, len(x_array) - 1}
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        segment = y_array[start:end]
        indices.add(start + int(np.argmin(segment)))
        indices.add(start + int(np.argmax(segment)))
    selected = np.asarray(sorted(indices))
    if len(selected) > max_points:
        selected = selected[np.linspace(0, len(selected) - 1, max_points, dtype=int)]
    return x_array[selected], y_array[selected]


class CartPoleGUI:
    PLOT_UPDATE_INTERVAL = 2.0
    BG = "#111827"
    PANEL = "#1f2937"
    FIELD = "#0f172a"
    FG = "#f3f4f6"
    MUTED = "#cbd5e1"
    ACCENT = "#60a5fa"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.workbench = CartPoleWorkbench()
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.animation_after: Optional[str] = None
        self.renderer = CartPoleRenderer()
        self.animation_observation: Optional[Any] = None
        self.animation_frame: Optional[Any] = None
        self.animation_step = 0
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.busy = False
        self.comparison_history: dict[str, dict[int, list[EpisodeMetric]]] = {}
        self.comparison_evaluation_history: dict[
            str, dict[int, list[tuple[int, int, EvaluationResult]]]
        ] = {}
        self.comparison_workbenches: dict[tuple[str, int], CartPoleWorkbench] = {}
        self.comparison_progress: dict[str, int] = {}
        self.active_comparison_algorithms: tuple[str, ...] = ()
        self.comparison_running = False
        self.live_training_history: list[EpisodeMetric] = []
        self.evaluation_history: list[tuple[int, int, EvaluationResult]] = []
        self.best_evaluation: Optional[tuple[int, int, EvaluationResult]] = None
        self.best_checkpoint_directory = tempfile.TemporaryDirectory(prefix="cartpole-best-")
        self.best_checkpoint_base = Path(self.best_checkpoint_directory.name) / "best_model"
        self.best_checkpoint_lock = threading.Lock()
        self._last_plot_update = 0.0
        self._variables()
        self._window()
        self._layout()
        self._show_initial_frame()
        self._refresh_plot()
        self.root.after_idle(self._initialize_visible_layout)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _variables(self) -> None:
        defaults = DQNConfig()
        values = {
            "total_timesteps": defaults.total_timesteps,
            "learning_rate": defaults.learning_rate,
            "buffer_size": defaults.buffer_size,
            "learning_starts": defaults.learning_starts,
            "batch_size": defaults.batch_size,
            "tau": defaults.tau,
            "gamma": defaults.gamma,
            "train_freq": defaults.train_freq,
            "gradient_steps": defaults.gradient_steps,
            "target_update_interval": defaults.target_update_interval,
            "exploration_fraction": defaults.exploration_fraction,
            "exploration_initial_eps": defaults.exploration_initial_eps,
            "exploration_final_eps": defaults.exploration_final_eps,
            "max_grad_norm": defaults.max_grad_norm,
            "seed": defaults.seed,
            "net_arch": ",".join(str(value) for value in defaults.net_arch),
            "activation": defaults.activation,
            "optimizer": defaults.optimizer,
            "optimizer_eps": defaults.optimizer_eps,
            "optimizer_weight_decay": defaults.optimizer_weight_decay,
        }
        self.vars = {name: tk.StringVar(value=str(value)) for name, value in values.items()}
        self.evaluation_episodes = tk.StringVar(value="10")
        self.evaluation_interval = tk.StringVar(value="5000")
        self.algorithm = tk.StringVar(value="DQN")
        self.compare_algorithms = {name: tk.BooleanVar(value=True) for name in ALGORITHMS}
        self.animation_enabled = tk.BooleanVar(value=True)
        self.animation_delay = tk.StringVar(value="20")
        self.status = tk.StringVar(value="Bereit")
        self.summary = tk.StringVar(value="Noch kein Training ausgeführt.")
        self.observation_text = tk.StringVar(value="Observation: —")
        self.progress = tk.DoubleVar(value=0.0)

    def _window(self) -> None:
        self.root.title("CartPole DQN Workbench")
        usable_width = max(1000, self.root.winfo_screenwidth() - 40)
        usable_height = max(700, self.root.winfo_screenheight() - 70)
        width = min(1440, usable_width)
        height = min(900, usable_height)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(1200, width), min(760, height))
        self.root.configure(background=self.BG)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=self.BG, foreground=self.FG)
        style.configure("TFrame", background=self.BG)
        style.configure("TLabel", background=self.BG, foreground=self.FG)
        style.configure("Title.TLabel", background=self.BG, foreground="#ffffff", font=("TkDefaultFont", 20, "bold"))
        style.configure("TLabelframe", background=self.BG, foreground=self.FG, bordercolor="#475569")
        style.configure("TLabelframe.Label", background=self.BG, foreground="#ffffff", font=("TkDefaultFont", 10, "bold"))
        style.configure("TButton", background="#334155", foreground=self.FG, bordercolor="#64748b", padding=4)
        style.map(
            "TButton",
            background=[("active", "#475569"), ("pressed", "#1e3a5f"), ("disabled", "#1f2937")],
            foreground=[("disabled", "#94a3b8")],
        )
        style.configure("TEntry", fieldbackground=self.FIELD, foreground=self.FG, insertcolor=self.FG, bordercolor="#64748b")
        style.configure("TCombobox", fieldbackground=self.FIELD, background="#334155", foreground=self.FG, arrowcolor=self.FG)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.FIELD)],
            foreground=[("readonly", self.FG)],
            selectbackground=[("readonly", self.FIELD)],
            selectforeground=[("readonly", self.FG)],
        )
        style.configure("TCheckbutton", background=self.BG, foreground=self.FG)
        style.map("TCheckbutton", background=[("active", self.BG)], foreground=[("disabled", "#94a3b8")])
        style.configure("TProgressbar", troughcolor=self.FIELD, background=self.ACCENT, bordercolor="#475569")
        style.configure("TPanedwindow", background="#475569")
        self.root.option_add("*TCombobox*Listbox.background", self.FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", self.FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#2563eb")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

    def _layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="CartPole DQN Workbench", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="DQN mit Stable-Baselines3 – die Stange möglichst lange balancieren",
        ).pack(side="left", padx=(16, 0), pady=(7, 0))
        ttk.Button(header, text="Bedienungsanleitung", command=self.instructions).pack(side="right")

        # Vertical orientation creates a horizontal, draggable divider.
        self.splitter = ttk.Panedwindow(outer, orient="vertical")
        self.splitter.pack(fill="both", expand=True)
        upper = ttk.Frame(self.splitter)
        lower = ttk.Frame(self.splitter)
        self.splitter.add(upper, weight=1)
        self.splitter.add(lower, weight=1)
        upper.columnconfigure(0, minsize=810)
        upper.columnconfigure(1, weight=1)
        upper.rowconfigure(0, weight=1)
        controls = ttk.Frame(upper, width=810, padding=(0, 0, 8, 0))
        controls.grid(row=0, column=0, sticky="nsew")
        controls.grid_propagate(False)
        self.controls_panel = controls
        self._build_controls(controls)
        self._build_environment(upper)

        lower.columnconfigure(0, weight=1)
        lower.rowconfigure(0, weight=1)
        self._build_lower(lower)

    def _initialize_visible_layout(self) -> None:
        """Choose a useful initial sash and verify that no essential widget is clipped."""
        self.root.update_idletasks()
        available = self.splitter.winfo_height()
        if available <= 1:
            self.root.after(20, self._initialize_visible_layout)
            return
        required_upper = max(
            (child.winfo_y() + child.winfo_reqheight() for child in self.controls_panel.winfo_children()),
            default=330,
        ) + 12
        lower_minimum = 200
        sash = max(int(available * 0.42), required_upper)
        sash = min(sash, max(280, available - lower_minimum))
        self.splitter.sashpos(0, sash)
        self.root.update_idletasks()
        issues = self.layout_visibility_issues()
        if issues:
            self.status.set("Layout-Warnung: nicht vollständig sichtbar: " + ", ".join(issues))

    def layout_visibility_issues(self) -> list[str]:
        """Return names of essential controls that are unmapped or clipped."""
        self.root.update_idletasks()
        left = self.controls_panel.winfo_rootx()
        top = self.controls_panel.winfo_rooty()
        right = left + self.controls_panel.winfo_width()
        bottom = top + self.controls_panel.winfo_height()
        issues: list[str] = []

        def descendants(widget: tk.Misc) -> list[tk.Misc]:
            result: list[tk.Misc] = []
            for child in widget.winfo_children():
                result.append(child)
                result.extend(descendants(child))
            return result

        for widget in descendants(self.controls_panel):
            if not isinstance(widget, (ttk.Entry, ttk.Combobox, ttk.Button, ttk.Checkbutton, ttk.Progressbar)):
                continue
            name = widget.winfo_class()
            if not widget.winfo_ismapped():
                issues.append(name)
                continue
            x = widget.winfo_rootx()
            y = widget.winfo_rooty()
            if x < left or y < top or x + widget.winfo_width() > right or y + widget.winfo_height() > bottom:
                issues.append(name)

        for name, widget in (
            ("Visualisierung", self.image_label),
            ("Diagramm", self.plot_canvas.get_tk_widget()),
        ):
            if not widget.winfo_ismapped() or widget.winfo_width() <= 1 or widget.winfo_height() <= 1:
                issues.append(name)
        legend = self.axes.get_legend()
        if legend is not None:
            self.plot_canvas.draw()
            legend_box = legend.get_window_extent(self.plot_canvas.get_renderer())
            figure_box = self.figure.bbox
            if (
                legend_box.x0 < figure_box.x0 or legend_box.y0 < figure_box.y0
                or legend_box.x1 > figure_box.x1 or legend_box.y1 > figure_box.y1
            ):
                issues.append("Legende")
            x_label_box = self.axes.xaxis.label.get_window_extent(
                self.plot_canvas.get_renderer()
            )
            if (
                x_label_box.x0 < figure_box.x0 or x_label_box.y0 < figure_box.y0
                or x_label_box.x1 > figure_box.x1 or x_label_box.y1 > figure_box.y1
            ):
                issues.append("X-Achsenbeschriftung")
        return issues

    def _entry(self, parent: ttk.Frame, row: int, label: str, name: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 5), pady=1)
        ttk.Entry(parent, textvariable=self.vars[name], width=12).grid(row=row, column=1, sticky="ew", pady=1)

    def _build_controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure((0, 1), weight=1, uniform="parameter")
        parent.columnconfigure(2, weight=1, uniform="parameter")
        parent.rowconfigure(0, weight=1)
        left_parameters = ttk.Frame(parent)
        right_parameters = ttk.Frame(parent)
        left_parameters.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        right_parameters.grid(row=0, column=1, sticky="nsew", padx=4)

        training = ttk.LabelFrame(left_parameters, text="Training und Replay Buffer", padding=6)
        training.pack(fill="x", pady=(0, 4))
        ttk.Label(training, text="Algorithmus").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=1)
        ttk.Combobox(
            training, textvariable=self.algorithm, values=ALGORITHMS,
            state="readonly", width=11,
        ).grid(row=0, column=1, sticky="ew", pady=1)
        for row, (label, name) in enumerate((
            ("Trainingsschritte N", "total_timesteps"),
            ("Lernstart t₀", "learning_starts"),
            ("Replay Buffer |D|", "buffer_size"),
            ("Batch-Größe B", "batch_size"),
            ("Trainingsfrequenz fₜ", "train_freq"),
            ("Gradientenschritte G", "gradient_steps"),
            ("Zufallsstart s", "seed"),
        ), start=1):
            self._entry(training, row, label, name)
        training.columnconfigure(1, weight=1)

        learning = ttk.LabelFrame(right_parameters, text="Lernen und Target Network", padding=6)
        learning.pack(fill="x", pady=(0, 4))
        for row, (label, name) in enumerate((
            ("Lernrate α", "learning_rate"),
            ("Diskontfaktor γ", "gamma"),
            ("Soft-Update τ", "tau"),
            ("Target-Intervall C", "target_update_interval"),
            ("Max. Gradientennorm ‖g‖ₘₐₓ", "max_grad_norm"),
        )):
            self._entry(learning, row, label, name)
        learning.columnconfigure(1, weight=1)

        exploration = ttk.LabelFrame(left_parameters, text="Exploration (ε-greedy)", padding=6)
        exploration.pack(fill="x", pady=4)
        for row, (label, name) in enumerate((
            ("Startwert ε₀", "exploration_initial_eps"),
            ("Endwert εₘᵢₙ", "exploration_final_eps"),
            ("Abklinganteil fₑ", "exploration_fraction"),
        )):
            self._entry(exploration, row, label, name)
        exploration.columnconfigure(1, weight=1)

        network = ttk.LabelFrame(right_parameters, text="Neuronales Netz und Optimizer", padding=6)
        network.pack(fill="x", pady=4)
        self._entry(network, 0, "Hidden Layers h", "net_arch")
        ttk.Label(network, text="Aktivierung φ").grid(row=1, column=0, sticky="w", pady=1)
        ttk.Combobox(network, textvariable=self.vars["activation"], values=tuple(ACTIVATIONS), state="readonly", width=11).grid(row=1, column=1, sticky="ew")
        ttk.Label(network, text="Optimizer opt").grid(row=2, column=0, sticky="w", pady=1)
        ttk.Combobox(network, textvariable=self.vars["optimizer"], values=tuple(OPTIMIZERS), state="readonly", width=11).grid(row=2, column=1, sticky="ew")
        self._entry(network, 3, "Numerische Stabilität εₒₚₜ", "optimizer_eps")
        self._entry(network, 4, "Gewichtszerfall λ", "optimizer_weight_decay")
        network.columnconfigure(1, weight=1)

        evaluation = ttk.LabelFrame(left_parameters, text="Evaluation und Vergleich", padding=6)
        evaluation.pack(fill="x", pady=(4, 0))
        ttk.Label(evaluation, text="Evaluations-Episoden M").grid(row=0, column=0, sticky="w")
        ttk.Entry(evaluation, textvariable=self.evaluation_episodes, width=10).grid(row=0, column=1, sticky="ew")
        ttk.Label(evaluation, text="Auto-Eval-Intervall E").grid(row=1, column=0, sticky="w")
        ttk.Entry(evaluation, textvariable=self.evaluation_interval, width=10).grid(row=1, column=1, sticky="ew")
        ttk.Label(evaluation, text="Algorithmen im Vergleich").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        algorithm_choices = ttk.Frame(evaluation)
        algorithm_choices.grid(row=3, column=0, columnspan=2, sticky="w", padx=(8, 0))
        for name in ALGORITHMS:
            ttk.Checkbutton(
                algorithm_choices,
                text=name,
                variable=self.compare_algorithms[name],
            ).pack(side="left", padx=(0, 10))
        evaluation.columnconfigure(1, weight=1)

        animation = ttk.LabelFrame(right_parameters, text="Animation", padding=6)
        animation.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(
            animation,
            text="Nach Evaluation anzeigen",
            variable=self.animation_enabled,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(animation, text="Intervall Δt (ms)").grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Entry(animation, textvariable=self.animation_delay, width=10).grid(
            row=1, column=1, sticky="ew", pady=(3, 0)
        )
        ttk.Label(
            animation,
            text="Eine sichtbare Episode startet direkt über den Animationsbutton.",
            wraplength=215,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        animation.columnconfigure(1, weight=1)

        actions = ttk.LabelFrame(parent, text="Steuerung", padding=6)
        actions.grid(row=0, column=2, sticky="new", padx=(4, 0))
        self.train_button = ttk.Button(actions, text="Training starten / fortsetzen", command=self.start_training)
        self.stop_button = ttk.Button(actions, text="Stoppen", command=self.stop, state="disabled")
        self.eval_button = ttk.Button(actions, text="Deterministisch evaluieren", command=self.start_evaluation)
        self.animate_button = ttk.Button(actions, text="Gelernte Policy animieren", command=self.animate_policy)
        self.compare_button = ttk.Button(actions, text="Algorithmen vergleichen", command=self.start_comparison)
        self.restore_best_button = ttk.Button(
            actions, text="Bestes Modell wiederherstellen",
            command=self.restore_best_model, state="disabled",
        )
        self.reset_button = ttk.Button(actions, text="Neues Modell", command=self.reset_model)
        self.save_button = ttk.Button(actions, text="Modell und Replay Buffer speichern", command=self.save_model)
        self.load_button = ttk.Button(actions, text="Modell und Replay Buffer laden", command=self.load_model)
        for index, button in enumerate((
            self.train_button,
            self.stop_button,
            self.eval_button,
            self.animate_button,
            self.compare_button,
            self.restore_best_button,
            self.reset_button,
            self.save_button,
            self.load_button,
        )):
            button.grid(row=index, column=0, sticky="ew", padx=3, pady=2, ipady=2)
        actions.columnconfigure(0, weight=1)
        ttk.Progressbar(actions, variable=self.progress, maximum=100).grid(row=9, column=0, sticky="ew", padx=3, pady=(6, 2))
        ttk.Label(
            actions,
            textvariable=self.status,
            foreground=self.ACCENT,
            justify="left",
            wraplength=225,
        ).grid(row=10, column=0, sticky="ew", padx=3, pady=(3, 2))

    def _build_environment(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Offizieller Gymnasium-RGB-Frame", padding=8)
        frame.grid(row=0, column=1, sticky="nsew")
        self.image_label = ttk.Label(frame, anchor="center")
        self.image_label.pack(fill="both", expand=True)
        ttk.Label(frame, textvariable=self.observation_text, justify="left", font=("TkFixedFont", 10)).pack(anchor="w", pady=(5, 0))

    def _build_lower(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=4)
        parent.columnconfigure(1, weight=1, minsize=280)
        parent.rowconfigure(0, weight=1)
        chart = ttk.LabelFrame(parent, text="Training / Vergleich", padding=5)
        chart.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        info = ttk.LabelFrame(parent, text="Summary", padding=10)
        info.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.figure = Figure(figsize=(9, 3), dpi=100, facecolor=self.BG)
        self.figure.subplots_adjust(left=0.09, right=0.82, bottom=0.24, top=0.88)
        self.axes = self.figure.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(info, textvariable=self.summary, justify="left", font="TkFixedFont").pack(
            anchor="nw", fill="x"
        )

    def _config(self) -> DQNConfig:
        try:
            seed_text = self.vars["seed"].get().strip()
            layers = tuple(int(value.strip()) for value in self.vars["net_arch"].get().split(",") if value.strip())
            config = DQNConfig(
                algorithm=self.algorithm.get(),
                total_timesteps=int(self.vars["total_timesteps"].get()),
                learning_rate=float(self.vars["learning_rate"].get()),
                buffer_size=int(self.vars["buffer_size"].get()),
                learning_starts=int(self.vars["learning_starts"].get()),
                batch_size=int(self.vars["batch_size"].get()), tau=float(self.vars["tau"].get()),
                gamma=float(self.vars["gamma"].get()), train_freq=int(self.vars["train_freq"].get()),
                gradient_steps=int(self.vars["gradient_steps"].get()),
                target_update_interval=int(self.vars["target_update_interval"].get()),
                exploration_fraction=float(self.vars["exploration_fraction"].get()),
                exploration_initial_eps=float(self.vars["exploration_initial_eps"].get()),
                exploration_final_eps=float(self.vars["exploration_final_eps"].get()),
                max_grad_norm=float(self.vars["max_grad_norm"].get()),
                seed=int(seed_text) if seed_text else None, net_arch=layers,
                activation=self.vars["activation"].get(), optimizer=self.vars["optimizer"].get(),
                optimizer_eps=float(self.vars["optimizer_eps"].get()),
                optimizer_weight_decay=float(self.vars["optimizer_weight_decay"].get()),
            )
            config.validate()
            return config
        except ValueError as error:
            raise ValueError(f"Parameter ungültig: {error}") from error

    def _automatic_evaluation_settings(self) -> tuple[int, int]:
        try:
            episodes = int(self.evaluation_episodes.get())
            interval = int(self.evaluation_interval.get())
        except ValueError as error:
            raise ValueError("Evaluation M und Auto-Eval-Intervall E müssen Ganzzahlen sein.") from error
        if episodes <= 0 or interval <= 0:
            raise ValueError("Evaluation M und Auto-Eval-Intervall E müssen positiv sein.")
        return episodes, interval

    def _set_busy(self, busy: bool, text: str) -> None:
        self.busy = busy
        self.status.set(text)
        state = "disabled" if busy else "normal"
        for button in (
            self.train_button,
            self.eval_button,
            self.animate_button,
            self.compare_button,
            self.restore_best_button,
            self.reset_button,
            self.save_button,
            self.load_button,
        ):
            button.configure(state=state)
        self.stop_button.configure(state="normal" if busy else "disabled")
        if not busy:
            self.restore_best_button.configure(
                state="normal" if self.best_evaluation is not None else "disabled"
            )

    def start_training(self) -> None:
        if self.busy:
            return
        self._stop_animation()
        try:
            config = self._config()
            evaluation_episodes, evaluation_interval = self._automatic_evaluation_settings()
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root)
            return
        if (
            self.workbench.model is not None
            and config.model_signature() != self.workbench.config.model_signature()
        ):
            if not messagebox.askyesno("Lernzustand zurücksetzen", "Geänderte Modellparameter erfordern ein neues Modell. Fortfahren?", parent=self.root):
                return
            self.workbench.close()
            self.workbench = CartPoleWorkbench(config)
            self.evaluation_history.clear()
            self._clear_best_checkpoint()
        else:
            self.workbench.config = config
        self._clear_comparison()
        self.comparison_running = False
        self.stop_event.clear()
        self.progress.set(0)
        self.live_training_history = list(self.workbench.training_history)
        self._set_busy(True, f"Läuft – {config.algorithm}-Training")
        self._last_plot_update = time.monotonic()
        self.worker = threading.Thread(
            target=self._train_worker,
            args=(evaluation_episodes, evaluation_interval),
            daemon=True,
        )
        self.worker.start()
        self.root.after(50, self._poll_events)

    def _train_worker(self, evaluation_episodes: int, evaluation_interval: int) -> None:
        try:
            metrics = self.workbench.train(
                self.stop_event, self.events,
                evaluation_interval=evaluation_interval,
                evaluation_episodes=evaluation_episodes,
                evaluation_seed=self.workbench.config.seed,
                best_model_callback=self._save_best_checkpoint,
            )
            self.events.put(("training_done", metrics))
        except Exception as error:
            self.events.put(("error", error))

    def stop(self) -> None:
        if self.busy:
            self.stop_event.set()
            self.status.set("Stoppen angefordert …")

    def start_comparison(self) -> None:
        if self.busy:
            return
        self._stop_animation()
        try:
            base_config = self._config()
            evaluation_episodes, evaluation_interval = self._automatic_evaluation_settings()
            repetitions = 1
        except ValueError as error:
            messagebox.showerror("Ungültiger Vergleich", str(error), parent=self.root)
            return
        algorithms = [name for name in ALGORITHMS if self.compare_algorithms[name].get()]
        if len(algorithms) < 2:
            messagebox.showerror(
                "Ungültiger Vergleich",
                "Wähle mindestens zwei Algorithmen für den Vergleich aus.",
                parent=self.root,
            )
            return
        existing = next(iter(self.comparison_workbenches.values()), None)
        if existing is not None and existing.config.model_signature() != base_config.model_signature():
            if not messagebox.askyesno(
                "Vergleich neu starten",
                "Geänderte Modellparameter sind nicht mit den bisherigen Vergleichsläufen kompatibel. "
                "Bisherigen Vergleich verwerfen?",
                parent=self.root,
            ):
                return
            self._clear_comparison()
        for name in algorithms:
            runs = self.comparison_history.setdefault(name, {})
            for run in range(1, repetitions + 1):
                runs.setdefault(run, [])
                self.comparison_evaluation_history.setdefault(name, {}).setdefault(run, [])
                key = (name, run)
                seed = None if base_config.seed is None else base_config.seed + run - 1
                config = replace(base_config, algorithm=name, seed=seed)
                if key not in self.comparison_workbenches:
                    self.comparison_workbenches[key] = CartPoleWorkbench(config)
                else:
                    self.comparison_workbenches[key].config = config
        self.comparison_progress = {name: 0 for name in algorithms}
        self.active_comparison_algorithms = tuple(algorithms)
        self.comparison_running = True
        self.stop_event.clear()
        self.progress.set(0)
        self._set_busy(True, f"Läuft – {' und '.join(algorithms)}")
        self._refresh_comparison_plot()
        self._last_plot_update = time.monotonic()
        self._comparison_summary()
        self.worker = threading.Thread(
            target=self._comparison_worker,
            args=(base_config, algorithms, repetitions, evaluation_episodes, evaluation_interval),
            daemon=True,
        )
        self.worker.start()
        self.root.after(50, self._poll_events)

    def _comparison_worker(
        self, base_config: DQNConfig, algorithms: list[str], repetitions: int,
        evaluation_episodes: int, evaluation_interval: int,
    ) -> None:
        total_runs = len(algorithms) * repetitions
        total_steps = total_runs * base_config.total_timesteps
        start_barrier = threading.Barrier(len(algorithms))
        errors: queue.Queue[Exception] = queue.Queue()

        def train_algorithm(algorithm: str) -> None:
            try:
                start_barrier.wait()
                for run in range(1, repetitions + 1):
                    if self.stop_event.is_set():
                        return
                    # Same repetition uses the same seed for every algorithm.
                    workbench = self.comparison_workbenches[(algorithm, run)]
                    self.events.put(("comparison_run", (algorithm, run, repetitions)))
                    workbench.train(
                        self.stop_event,
                        self.events,
                        series_name=(algorithm, run),
                        progress_offset=(run - 1) * base_config.total_timesteps,
                        comparison_total=total_steps,
                        evaluation_interval=evaluation_interval,
                        evaluation_episodes=evaluation_episodes,
                        evaluation_seed=workbench.config.seed,
                    )
            except Exception as error:
                self.stop_event.set()
                errors.put(error)

        try:
            workers = [
                threading.Thread(target=train_algorithm, args=(algorithm,), daemon=True)
                for algorithm in algorithms
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            if not errors.empty():
                raise errors.get()
            self.events.put(("comparison_done", self.stop_event.is_set()))
        except Exception as error:
            self.events.put(("error", error))

    def start_evaluation(self) -> None:
        if self.busy:
            return
        self._stop_animation()
        if self.workbench.model is None:
            messagebox.showinfo("Kein Modell", "Trainiere oder lade zuerst ein Modell.", parent=self.root)
            return
        try:
            episodes = int(self.evaluation_episodes.get())
            if episodes <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ungültiger Wert", "Evaluations-Episoden muss eine positive Ganzzahl sein.", parent=self.root)
            return
        self._set_busy(True, "Läuft – deterministische Evaluation")
        evaluation_seed = self.workbench.config.seed
        history = self.live_training_history or self.workbench.training_history
        evaluation_episode = history[-1].episode if history else 0
        evaluation_timesteps = self.workbench.model.num_timesteps
        self.worker = threading.Thread(
            target=self._evaluation_worker,
            args=(episodes, evaluation_seed, evaluation_episode, evaluation_timesteps),
            daemon=True,
        )
        self.worker.start()
        self.root.after(50, self._poll_events)

    def _evaluation_worker(
        self, episodes: int, seed: Optional[int], evaluation_episode: int,
        evaluation_timesteps: int,
    ) -> None:
        try:
            result = self.workbench.evaluate(episodes, seed)
            self._save_best_checkpoint(evaluation_episode, evaluation_timesteps, result)
            self.events.put(
                ("evaluation_done", (evaluation_episode, evaluation_timesteps, result))
            )
        except Exception as error:
            self.events.put(("error", error))

    def animate_policy(self) -> None:
        if self.busy:
            return
        if self.workbench.model is None:
            messagebox.showinfo("Kein Modell", "Trainiere oder lade zuerst ein Modell.", parent=self.root)
            return
        self._start_animation(self.workbench.config.seed)

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "episode":
                metric: EpisodeMetric = payload
                self.live_training_history.append(metric)
                self.status.set(f"Läuft – Episode {metric.episode}, Reward {metric.reward:.0f}")
                self._training_summary(self.live_training_history)
                if time.monotonic() - self._last_plot_update >= self.PLOT_UPDATE_INTERVAL:
                    self._refresh_plot()
                    self._last_plot_update = time.monotonic()
            elif kind == "progress":
                self.progress.set(min(100, 100 * int(payload) / max(1, self.workbench.config.total_timesteps)))
            elif kind == "comparison_episode":
                self._set_comparison_running_status()
                (algorithm, run), metric = payload
                self.comparison_history[algorithm][run].append(metric)
                self._comparison_summary()
                if time.monotonic() - self._last_plot_update >= self.PLOT_UPDATE_INTERVAL:
                    self._refresh_comparison_plot()
                    self._last_plot_update = time.monotonic()
            elif kind == "automatic_evaluation":
                evaluation_episode, evaluation_timesteps, result = payload
                self.evaluation_history.append((evaluation_episode, evaluation_timesteps, result))
                self._training_summary(self.live_training_history)
            elif kind == "comparison_evaluation":
                self._set_comparison_running_status()
                (algorithm, run), evaluation_episode, evaluation_timesteps, result = payload
                self.comparison_evaluation_history[algorithm][run].append(
                    (evaluation_episode, evaluation_timesteps, result)
                )
                self._comparison_summary()
            elif kind == "comparison_progress":
                self._set_comparison_running_status()
                (algorithm, _run), completed, total = payload
                self.comparison_progress[algorithm] = completed
                all_completed = sum(self.comparison_progress.values())
                self.progress.set(min(100, 100 * all_completed / max(1, total)))
            elif kind == "comparison_run":
                self._set_comparison_running_status()
            elif kind == "comparison_done":
                stopped = bool(payload)
                self.comparison_running = False
                self.progress.set(100 if not stopped else self.progress.get())
                self._set_busy(False, "Gestoppt – Vergleich" if stopped else "Abgeschlossen – Vergleich")
                self._refresh_comparison_plot()
                self._comparison_summary()
            elif kind == "training_done":
                self.progress.set(100)
                self.live_training_history = list(self.workbench.training_history)
                self._set_busy(False, "Gestoppt" if self.stop_event.is_set() else "Abgeschlossen – Training")
                self._refresh_plot()
                self._training_summary()
            elif kind == "evaluation_done":
                evaluation_episode, evaluation_timesteps, result = payload
                self.evaluation_history.append((evaluation_episode, evaluation_timesteps, result))
                self._set_busy(False, "Abgeschlossen – Evaluation")
                self._training_summary(self.live_training_history or self.workbench.training_history)
                self._refresh_plot()
                if self.animation_enabled.get():
                    self._start_animation(self.workbench.config.seed)
            elif kind == "error":
                self.comparison_running = False
                self._set_busy(False, "Fehler")
                messagebox.showerror("Fehler", str(payload), parent=self.root)
        if self.busy:
            self.root.after(50, self._poll_events)

    def _set_comparison_running_status(self) -> None:
        if self.comparison_running and self.active_comparison_algorithms:
            self.status.set(f"Läuft – {' und '.join(self.active_comparison_algorithms)}")

    def _refresh_plot(self) -> None:
        self.axes.clear()
        history = self.live_training_history or self.workbench.training_history
        if history:
            episodes = [item.episode for item in history]
            rewards = [item.reward for item in history]
            raw_x, raw_y = downsample_minmax(episodes, rewards)
            self.axes.plot(raw_x, raw_y, color="#93c5fd", linewidth=1, alpha=0.20, label="Reward")
            window = min(20, len(rewards))
            if window > 1:
                smooth = [sum(rewards[max(0, i - window + 1):i + 1]) / min(window, i + 1) for i in range(len(rewards))]
                self.axes.plot(episodes, smooth, color="#1d4ed8", linewidth=1.6, label=f"Ø {window}")
        self.axes.set_title("Episoden-Reward")
        self.axes.set_xlabel("Episode")
        self.axes.set_ylabel("Reward / Schritte")
        self._add_cartpole_goal_line()
        self._place_legend()
        self._style_axes()
        self.plot_canvas.draw_idle()

    def _refresh_comparison_plot(self) -> None:
        self.axes.clear()
        colors = {"DQN": "#60a5fa", "DDQN": "#f87171"}
        for algorithm, runs in self.comparison_history.items():
            populated = [metrics for metrics in runs.values() if metrics]
            if not populated:
                continue
            for metrics in populated:
                raw_x, raw_y = downsample_minmax(
                    [item.episode for item in metrics],
                    [item.reward for item in metrics],
                )
                self.axes.plot(
                    raw_x, raw_y,
                    color=colors.get(algorithm), alpha=0.08, linewidth=0.7,
                )
            if len(populated) == 1:
                x_values = [item.episode for item in populated[0]]
                means = [item.reward for item in populated[0]]
                deviations = [0.0] * len(means)
            else:
                x_values = sorted({item.episode for metrics in populated for item in metrics})
                means = []
                deviations = []
                run_arrays = [
                    (
                        np.asarray([item.episode for item in metrics], dtype=float),
                        np.asarray([item.reward for item in metrics], dtype=float),
                    )
                    for metrics in populated
                ]
                for x_value in x_values:
                    samples = [
                        float(np.interp(x_value, run_x, run_y))
                        for run_x, run_y in run_arrays
                        if run_x[0] <= x_value <= run_x[-1]
                    ]
                    means.append(float(np.mean(samples)))
                    deviations.append(float(np.std(samples)))
            x_array = np.asarray(x_values)
            mean_array = np.asarray(means)
            deviation_array = np.asarray(deviations)
            rolling_mean = rolling_average(mean_array, 20)
            self.axes.plot(
                x_array, mean_array, color=colors.get(algorithm), linewidth=1,
                alpha=0.22,
            )
            self.axes.plot(
                x_array, rolling_mean, color=colors.get(algorithm), linewidth=1.6,
                label=algorithm,
            )
            if len(populated) > 1:
                self.axes.fill_between(
                    x_array, mean_array - deviation_array, mean_array + deviation_array,
                    color=colors.get(algorithm), alpha=0.07,
                )
        self.axes.set_title("Live-Algorithmenvergleich")
        self.axes.set_xlabel("Episode")
        self.axes.set_ylabel("Episoden-Reward")
        self._add_cartpole_goal_line()
        self._place_legend()
        self._style_axes()
        self.plot_canvas.draw_idle()

    def _add_cartpole_goal_line(self) -> None:
        self.axes.axhline(
            500,
            color="#fbbf24",
            linestyle="--",
            linewidth=1.4,
            label="Ziel-Reward",
        )
        self.axes.set_ylim(bottom=0, top=525)

    def _style_axes(self) -> None:
        self.axes.set_facecolor(self.FIELD)
        self.axes.title.set_color(self.FG)
        self.axes.xaxis.label.set_color(self.MUTED)
        self.axes.yaxis.label.set_color(self.MUTED)
        self.axes.tick_params(axis="both", colors=self.MUTED)
        for spine in self.axes.spines.values():
            spine.set_color("#64748b")
        self.axes.grid(color="#475569", alpha=0.45)
        legend = self.axes.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(self.PANEL)
            legend.get_frame().set_edgecolor("#64748b")
            for text_item in legend.get_texts():
                text_item.set_color(self.FG)

    def _place_legend(self) -> None:
        self.axes.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
        )

    def _comparison_summary(self) -> None:
        state = "läuft" if self.comparison_running else "abgeschlossen"
        statistics: dict[
            str, tuple[int, int, float, float, Optional[float], Optional[float]]
        ] = {}
        for algorithm, runs in self.comparison_history.items():
            populated = [metrics for metrics in runs.values() if metrics]
            if not populated:
                continue
            all_metrics = [metric for metrics in populated for metric in metrics]
            completed_steps = sum(metrics[-1].timesteps for metrics in populated)
            success = sum(metric.success for metric in all_metrics) / len(all_metrics)
            evaluations = [
                values[-1][2].mean_reward
                for values in self.comparison_evaluation_history.get(algorithm, {}).values()
                if values
            ]
            all_evaluations = [
                result.mean_reward
                for values in self.comparison_evaluation_history.get(algorithm, {}).values()
                for _episode, _timesteps, result in values
            ]
            statistics[algorithm] = (
                len(all_metrics), completed_steps,
                float(np.mean([metric.reward for metric in all_metrics])), success,
                float(np.mean(evaluations)) if evaluations else None,
                max(all_evaluations) if all_evaluations else None,
            )
        algorithms = list(statistics)
        if not algorithms:
            self.summary.set(f"Algorithmenvergleich – {state}\nNoch keine vollständige Episode.")
            return
        label_width = 18
        value_width = 9
        header = "Statistik".ljust(label_width) + "".join(
            algorithm.rjust(value_width) for algorithm in algorithms
        )
        lines = [f"Vergleich – {state}", "", header, "─" * len(header)]
        rows = (
            ("Episoden", lambda values: str(values[0])),
            ("Schritte", lambda values: str(values[1])),
            ("Durchschn. Reward", lambda values: f"{values[2]:.1f}"),
            ("Erfolgsrate", lambda values: f"{values[3]:.1%}"),
            ("Letzte det. Eval", lambda values: "—" if values[4] is None else f"{values[4]:.1f}"),
            ("Beste det. Eval", lambda values: "—" if values[5] is None else f"{values[5]:.1f}"),
        )
        for label, formatter in rows:
            lines.append(label.ljust(label_width) + "".join(
                formatter(statistics[algorithm]).rjust(value_width)
                for algorithm in algorithms
            ))
        self.summary.set("\n".join(lines))

    def _training_summary(self, history: Optional[list[EpisodeMetric]] = None) -> None:
        history = history if history is not None else self.workbench.training_history
        success = sum(item.success for item in history) / len(history) if history else None
        average_reward = sum(item.reward for item in history) / len(history) if history else None
        algorithm = self.workbench.config.algorithm
        latest_evaluation = self.evaluation_history[-1][2].mean_reward if self.evaluation_history else None
        best_evaluation = self.best_evaluation[2].mean_reward if self.best_evaluation else None
        label_width = 18
        value_width = 9
        header = "Statistik".ljust(label_width) + algorithm.rjust(value_width)
        values = (
            ("Episoden", str(len(history))),
            ("Schritte", str(self.workbench.model.num_timesteps if self.workbench.model else 0)),
            ("Durchschn. Reward", "—" if average_reward is None else f"{average_reward:.1f}"),
            ("Erfolgsrate", "—" if success is None else f"{success:.1%}"),
            ("Letzte det. Eval", "—" if latest_evaluation is None else f"{latest_evaluation:.1f}"),
            ("Beste det. Eval", "—" if best_evaluation is None else f"{best_evaluation:.1f}"),
        )
        lines = ["Training", "", header, "─" * len(header)]
        lines.extend(label.ljust(label_width) + value.rjust(value_width) for label, value in values)
        self.summary.set("\n".join(lines))

    def _evaluation_summary(self, result: EvaluationResult) -> None:
        self.summary.set(
            f"Deterministische Evaluation ({result.episodes} Episoden)\n"
            f"Mittlerer Reward: {result.mean_reward:.2f}\n"
            f"Standardabweichung: {result.reward_std:.2f}\n"
            f"Mittlere Episodenlänge: {result.mean_length:.2f}\n"
            f"Erfolgsrate (500 Schritte): {result.success_rate:.1%}"
        )

    def _start_animation(self, seed: Optional[int]) -> None:
        self._stop_animation()
        self.animation_observation, _info, self.animation_frame = self.renderer.reset(seed)
        self.animation_step = 0
        self.status.set("Läuft – Animation")
        self._animation_tick()

    def _animation_tick(self) -> None:
        if self.animation_observation is None or self.animation_frame is None or self.workbench.model is None:
            return
        action, _ = self.workbench.model.predict(self.animation_observation, deterministic=True)
        observation = tuple(float(value) for value in self.animation_observation)
        next_observation, _reward, terminated, truncated, _info, next_frame = self.renderer.step(int(action))
        self.animation_step += 1
        self._show_frame(self.animation_frame)
        self._show_observation(observation, int(action), self.animation_step)
        if terminated or truncated:
            self._stop_animation()
            self.status.set("Abgeschlossen – Animation")
            return
        self.animation_observation = next_observation
        self.animation_frame = next_frame
        try:
            delay = max(1, min(5000, int(self.animation_delay.get())))
        except ValueError:
            delay = 20
        self.animation_after = self.root.after(delay, self._animation_tick)

    def _stop_animation(self) -> None:
        if self.animation_after is not None:
            self.root.after_cancel(self.animation_after)
            self.animation_after = None
        self.animation_observation = None
        self.animation_frame = None

    def _show_initial_frame(self) -> None:
        observation, _info, frame = self.renderer.reset(self.workbench.config.seed)
        self._show_frame(frame)
        self._show_observation(tuple(float(v) for v in observation), None, 0)

    def _show_frame(self, frame: Any) -> None:
        image = Image.fromarray(frame)
        image.thumbnail((780, 390), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)

    def _show_observation(self, observation: tuple[float, float, float, float], action: Optional[int], step: int) -> None:
        position, velocity, angle, angular_velocity = observation
        action_text = "—" if action is None else ("links" if action == 0 else "rechts")
        self.observation_text.set(
            f"Schritt: {step}   Action: {action_text}\n"
            f"Wagenposition: {position:+.3f} m   Wagengeschwindigkeit: {velocity:+.3f} m/s\n"
            f"Stangenwinkel: {angle * 180 / 3.141592653589793:+.2f}°   "
            f"Winkelgeschwindigkeit: {angular_velocity * 180 / 3.141592653589793:+.2f}°/s"
        )

    def _save_best_checkpoint(
        self, episode: int, timesteps: int, result: EvaluationResult
    ) -> None:
        """Persist the exact evaluated state; called synchronously by the worker."""
        with self.best_checkpoint_lock:
            if (
                self.best_evaluation is not None
                and result.mean_reward <= self.best_evaluation[2].mean_reward
            ):
                return
            self.workbench.save(self.best_checkpoint_base)
            self.best_evaluation = (episode, timesteps, result)

    def _clear_best_checkpoint(self) -> None:
        with self.best_checkpoint_lock:
            self.best_evaluation = None
        if hasattr(self, "restore_best_button"):
            self.restore_best_button.configure(state="disabled")

    def restore_best_model(self) -> None:
        if self.busy or self.best_evaluation is None:
            return
        self._stop_animation()
        try:
            restored = CartPoleWorkbench.load(self.best_checkpoint_base)
        except Exception as error:
            messagebox.showerror("Wiederherstellen fehlgeschlagen", str(error), parent=self.root)
            return
        best_episode, best_timesteps, best_result = self.best_evaluation
        retained_history = [
            metric for metric in (self.live_training_history or self.workbench.training_history)
            if metric.episode <= best_episode
        ]
        restored.training_history = list(retained_history)
        self.workbench.close()
        self.workbench = restored
        self.live_training_history = list(retained_history)
        self.evaluation_history = [
            item for item in self.evaluation_history if item[1] <= best_timesteps
        ]
        if not self.evaluation_history or self.evaluation_history[-1][1] != best_timesteps:
            self.evaluation_history.append((best_episode, best_timesteps, best_result))
        self._write_config(restored.config)
        self._training_summary(self.live_training_history)
        self._refresh_plot()
        self.status.set(f"Bestes Modell wiederhergestellt – Eval-Reward {best_result.mean_reward:.1f}")

    def reset_model(self) -> None:
        if self.busy:
            return
        self._stop_animation()
        if not messagebox.askyesno("Neues Modell", "Aktuellen Lernzustand verwerfen?", parent=self.root):
            return
        try:
            config = self._config()
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root)
            return
        self.workbench.close()
        self.workbench = CartPoleWorkbench(config)
        self.evaluation_history.clear()
        self._clear_best_checkpoint()
        self._clear_comparison()
        self.progress.set(0)
        self.summary.set("Neues, untrainiertes DQN-Modell vorbereitet.")
        self.status.set("Bereit")
        self._refresh_plot()

    def save_model(self) -> None:
        if self.workbench.model is None:
            messagebox.showinfo("Kein Modell", "Es gibt noch kein Modell zum Speichern.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Modell speichern", defaultextension=".zip", filetypes=(("SB3-Modell", "*.zip"),))
        if not path:
            return
        try:
            paths = self.workbench.save(path)
            self.status.set(f"Abgeschlossen – gespeichert als {paths[0].name}")
        except Exception as error:
            messagebox.showerror("Speichern fehlgeschlagen", str(error), parent=self.root)

    def load_model(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, title="SB3-Modell laden", filetypes=(("SB3-Modell", "*.zip"),))
        if not path:
            return
        try:
            loaded = CartPoleWorkbench.load(path)
        except Exception as error:
            messagebox.showerror("Laden fehlgeschlagen", str(error), parent=self.root)
            return
        self.workbench.close()
        self.workbench = loaded
        self.evaluation_history.clear()
        self._clear_best_checkpoint()
        self._write_config(loaded.config)
        self._refresh_plot()
        self.summary.set(f"Modell geladen. Trainingsschritte: {loaded.model.num_timesteps if loaded.model else 0}")
        self.status.set("Bereit – Modell und Replay Buffer geladen")

    def _write_config(self, config: DQNConfig) -> None:
        for name, value in config.__dict__.items():
            if name == "algorithm":
                self.algorithm.set(str(value))
                continue
            if name == "net_arch":
                value = ",".join(str(item) for item in value)
            if value is None:
                value = ""
            self.vars[name].set(str(value))

    def instructions(self) -> None:
        text = (
            "1. DQN- und Netzwerkparameter wählen.\n"
            "2. Training starten; Stoppen beendet es kontrolliert.\n"
            "3. Deterministisch evaluieren. Evaluation lernt nicht.\n"
            "4. Mit ‚Gelernte Policy animieren‘ sofort eine sichtbare Episode starten.\n"
            "5. Ein Erfolg bedeutet 500 balancierte Schritte.\n"
            "6. Modell und Replay Buffer gemeinsam speichern, um später weiterzutrainieren.\n\n"
            "CartPole gibt pro Schritt +1 Reward. Die Episode endet, wenn Wagen oder Stange den erlaubten Bereich verlassen."
        )
        messagebox.showinfo("Bedienungsanleitung", text, parent=self.root)

    def close(self) -> None:
        self.stop_event.set()
        self._stop_animation()
        self.renderer.close()
        self.workbench.close()
        self._clear_comparison()
        self.best_checkpoint_directory.cleanup()
        self.root.destroy()

    def _clear_comparison(self) -> None:
        for workbench in self.comparison_workbenches.values():
            workbench.close()
        self.comparison_workbenches.clear()
        self.comparison_history.clear()
        self.comparison_evaluation_history.clear()
        self.comparison_progress.clear()
        self.active_comparison_algorithms = ()
