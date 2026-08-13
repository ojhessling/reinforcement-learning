"""Dark Tkinter GUI for the Acrobot Rainbow-DDQN workbench."""

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

from acrobot_logic import (
    ACTIVATIONS, ALGORITHMS, DISTRIBUTIONAL_ALGORITHMS, DUELING_ALGORITHMS, MULTISTEP_ALGORITHMS,
    NOISY_ALGORITHMS, OPTIMIZERS, PER_ALGORITHMS,
    AcrobotConfig, AcrobotWorkbench, EpisodeMetric, EvaluationResult, angles_degrees,
)
from acrobot_render import AcrobotRenderer


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


ACTION_LABELS = ("Drehmoment −1", "kein Drehmoment", "Drehmoment +1")
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


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class AcrobotGUI:
    BG, PANEL, FIELD = "#111827", "#1f2937", "#0f172a"
    FG, MUTED, ACCENT = "#f3f4f6", "#cbd5e1", "#60a5fa"
    PLOT_INTERVAL = 2.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.workbench = AcrobotWorkbench()
        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.busy = False
        self.renderer = AcrobotRenderer()
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.last_frame: Optional[np.ndarray] = None
        self.animation_after: Optional[str] = None
        self.animation_observation: Optional[np.ndarray] = None
        self.animation_step = 0
        self.animation_reward = 0.0
        self.live_history: list[EpisodeMetric] = []
        self.comparison_history: dict[str, list[EpisodeMetric]] = {}
        self.comparison_evaluations: dict[str, list[EvaluationResult]] = {}
        self.comparison_workbenches: dict[str, AcrobotWorkbench] = {}
        self.active_algorithms: tuple[str, ...] = ()
        self.evaluations: list[EvaluationResult] = []
        self.best_evaluation: Optional[EvaluationResult] = None
        self.best_checkpoint = tempfile.TemporaryDirectory(prefix="acrobot-best-")
        self.best_base = Path(self.best_checkpoint.name) / "best"
        self.last_plot = 0.0
        self.progress_by_algorithm: dict[Optional[str], int] = {}
        self._export_snapshot_key: Optional[tuple[Any, ...]] = None
        self._export_stamp: Optional[str] = None
        self._variables()
        self._window()
        self._layout()
        self._show_initial_frame()
        self._refresh_training_plot()
        self.root.after_idle(self._initialize_layout)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _variables(self) -> None:
        defaults = AcrobotConfig()
        self.values: dict[str, tk.StringVar] = {}
        for name, value in defaults.__dict__.items():
            if name == "algorithm":
                continue
            if isinstance(value, tuple):
                value = ",".join(map(str, value))
            self.values[name] = tk.StringVar(value="" if value is None else str(value))
        self.algorithm = tk.StringVar(value=defaults.algorithm)
        self.evaluation_episodes = tk.StringVar(value="10")
        self.compare = {name: tk.BooleanVar(value=True) for name in ALGORITHMS}
        self.animation_delay = tk.StringVar(value="20")
        self.status = tk.StringVar(value="Bereit")
        self.progress = tk.DoubleVar(value=0)
        self.summary = tk.StringVar(value="Noch kein Training ausgeführt.")
        self.observation = tk.StringVar(value="θ₁: —   θ₂: —")

    def _window(self) -> None:
        width = min(1560, max(1240, self.root.winfo_screenwidth() - 40))
        height = min(960, max(830, self.root.winfo_screenheight() - 70))
        self.root.title("Acrobot Rainbow-DDQN Workbench")
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(1240, width), min(830, height))
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
        style.configure("TProgressbar", troughcolor=self.FIELD, background=self.ACCENT)
        self.root.option_add("*TCombobox*Listbox.background", self.FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", self.FG)

    def _layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="Acrobot Rainbow-DDQN Workbench", font=("TkDefaultFont", 18, "bold")).pack(side="left")
        ttk.Label(header, text="Sieben Double-DQN-Erweiterungen – freies Ende über die Zielhöhe schwingen").pack(side="left", padx=16)
        ttk.Button(header, text="Bedienungsanleitung", command=self.instructions).pack(side="right")
        self.splitter = ttk.Panedwindow(outer, orient="vertical")
        self.splitter.pack(fill="both", expand=True)
        upper, lower = ttk.Frame(self.splitter), ttk.Frame(self.splitter)
        self.splitter.add(upper, weight=1); self.splitter.add(lower, weight=1)
        upper.columnconfigure(0, minsize=760); upper.columnconfigure(1, weight=1); upper.rowconfigure(0, weight=1)
        self.controls = ttk.Frame(upper, width=760, padding=(0, 0, 8, 0))
        self.controls.grid(row=0, column=0, sticky="nsew"); self.controls.grid_propagate(False)
        self._controls(self.controls)
        self._environment(upper)
        lower.columnconfigure(0, weight=3); lower.columnconfigure(1, weight=1, minsize=620); lower.rowconfigure(0, weight=1)
        chart = ttk.LabelFrame(lower, text="Training / Vergleich", padding=5)
        chart.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        info = ttk.LabelFrame(lower, text="Summary", padding=8)
        info.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        chart_header = ttk.Frame(chart); chart_header.pack(fill="x")
        ttk.Button(chart_header, text="Diagramm exportieren (PNG)", command=self.export_chart).pack(side="right")
        self.figure = Figure(figsize=(9, 3), dpi=100, facecolor=self.BG)
        self.figure.subplots_adjust(left=.10, right=.80, bottom=.24, top=.78)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        info_header = ttk.Frame(info); info_header.pack(fill="x")
        ttk.Button(info_header, text="Summary exportieren (TXT)", command=self.export_summary).pack(side="right")
        ttk.Label(info, textvariable=self.summary, justify="left", font="TkFixedFont").pack(anchor="nw", pady=(4, 0))

    def _entry(self, parent: ttk.Frame, row: int, label: str, name: str) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 5), pady=1)
        entry = ttk.Entry(parent, textvariable=self.values[name], width=8)
        entry.grid(row=row, column=1, sticky="ew", pady=1)
        return entry

    def _controls(self, parent: ttk.Frame) -> None:
        parent.columnconfigure((0, 1, 2), weight=1, uniform="columns")
        left, middle, actions = ttk.Frame(parent), ttk.Frame(parent), ttk.LabelFrame(parent, text="Steuerung", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4)); middle.grid(row=0, column=1, sticky="nsew", padx=4)
        actions.grid(row=0, column=2, sticky="new", padx=(4, 0))
        training = ttk.LabelFrame(left, text="Training und Replay Buffer", padding=5); training.pack(fill="x", pady=(0, 3))
        ttk.Label(training, text="Algorithmus").grid(row=0, column=0, columnspan=2, sticky="w")
        combo = ttk.Combobox(training, textvariable=self.algorithm, values=ALGORITHMS, state="readonly")
        combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        combo.bind("<<ComboboxSelected>>", lambda _event: self._algorithm_fields())
        for row, pair in enumerate((("Trainingsschritte N", "total_timesteps"), ("Lernstart t₀", "learning_starts"),
                                    ("Replay Buffer |D|", "buffer_size"), ("Batch-Größe B", "batch_size"),
                                    ("Trainingsfrequenz fₜ", "train_freq"), ("Gradientenschritte G", "gradient_steps"),
                                    ("Zufallsstart s", "seed")), 2):
            self._entry(training, row, *pair)
        training.columnconfigure(1, weight=1)
        exploration = ttk.LabelFrame(left, text="Exploration und Evaluation", padding=5); exploration.pack(fill="x", pady=3)
        self.exploration_entries = [
            self._entry(exploration, row, label, name) for row, (label, name) in enumerate((
                ("Startwert ε₀", "exploration_initial_eps"), ("Endwert εₘᵢₙ", "exploration_final_eps"),
                ("Abklinganteil fₑ", "exploration_fraction")))
        ]
        ttk.Label(exploration, text="Eval-Episoden M").grid(row=3, column=0, sticky="w")
        ttk.Entry(exploration, textvariable=self.evaluation_episodes, width=8).grid(row=3, column=1, sticky="ew")
        variants_more = ttk.LabelFrame(middle, text="Variantenparameter", padding=5)
        # β-Anneal.-Schritte kann durchaus so groß wie das Trainingsbudget
        # werden (bis zu siebenstellig) und bekommt deshalb eine eigene volle
        # Zeile statt eines schmalen Spaltenpaars, damit der Wert nie
        # abgeschnitten wird.
        ttk.Label(variants_more, text="β-Steps").grid(row=0, column=0, sticky="w", padx=(0, 3), pady=1)
        wide_entry = ttk.Entry(variants_more, textvariable=self.values["per_beta_steps"], width=12)
        wide_entry.grid(row=0, column=1, columnspan=3, sticky="ew", pady=1)
        self.variant_entries: dict[str, ttk.Entry] = {"per_beta_steps": wide_entry}
        specifications = (
            ("Noisy σ₀", "noisy_sigma_0"),
            ("PER α", "per_alpha"), ("PER β₀", "per_beta_0"),
            ("PER ε", "per_epsilon"), ("Value hᵥ", "value_arch"),
            ("Advant hₐ", "advantage_arch"), ("n_step", "multistep_n"),
            ("n_atoms", "c51_atoms"), ("V_min", "c51_v_min"), ("V_max", "c51_v_max"),
        )
        for index, (label, name) in enumerate(specifications):
            row, group = divmod(index, 2); column = group * 2
            ttk.Label(variants_more, text=label).grid(row=row + 1, column=column, sticky="w", padx=(0, 3), pady=1)
            entry = ttk.Entry(variants_more, textvariable=self.values[name], width=9)
            entry.grid(row=row + 1, column=column + 1, sticky="ew", padx=(0, 5), pady=1)
            self.variant_entries[name] = entry
        variants_more.columnconfigure((1, 3), weight=1)
        learning = ttk.LabelFrame(middle, text="Lernen und Target Network", padding=5); learning.pack(fill="x", pady=(0, 3))
        for row, pair in enumerate((("Lernrate α", "learning_rate"), ("Diskontfaktor γ", "gamma"),
                                    ("Soft-Update τ", "tau"), ("Target-Intervall C", "target_update_interval"),
                                    ("Max. Gradientennorm", "max_grad_norm"))):
            self._entry(learning, row, *pair)
        network = ttk.LabelFrame(middle, text="Neuronales Netz und Optimizer", padding=5); network.pack(fill="x", pady=3)
        self._entry(network, 0, "Hidden Layers h", "net_arch")
        ttk.Label(network, text="Aktivierung φ").grid(row=1, column=0, sticky="w")
        ttk.Combobox(network, textvariable=self.values["activation"], values=tuple(ACTIVATIONS), state="readonly", width=11).grid(row=1, column=1, sticky="ew")
        ttk.Label(network, text="Optimizer opt").grid(row=2, column=0, sticky="w")
        ttk.Combobox(network, textvariable=self.values["optimizer"], values=tuple(OPTIMIZERS), state="readonly", width=11).grid(row=2, column=1, sticky="ew")
        self._entry(network, 3, "Stabilität εₒₚₜ", "optimizer_eps")
        self._entry(network, 4, "Gewichtszerfall λ", "optimizer_weight_decay")
        variants_more.pack(fill="x", pady=3)
        comparison = ttk.LabelFrame(left, text="Algorithmen im Vergleich", padding=5); comparison.pack(fill="x", pady=3)
        for index, name in enumerate(ALGORITHMS):
            ttk.Checkbutton(comparison, text=name, variable=self.compare[name]).grid(
                row=index // 2, column=index % 2, sticky="w", padx=(0, 8)
            )
        animation = ttk.LabelFrame(middle, text="Animation", padding=5); animation.pack(fill="x", pady=3)
        ttk.Label(animation, text="Intervall Δt (ms)").grid(row=0, column=0, sticky="w")
        ttk.Entry(animation, textvariable=self.animation_delay, width=11).grid(row=0, column=1, sticky="ew")
        self.buttons = []
        for text, command in (("Training starten / fortsetzen", self.start_training), ("Stoppen", self.stop),
                              ("Deterministisch evaluieren", self.start_evaluation), ("Policy animieren", self.animate),
                              ("Algorithmen vergleichen", self.start_comparison),
                              ("Bestes Modell wiederherstellen", self.restore_best), ("Neues Modell", self.reset),
                              ("Modell und Replay speichern", self.save), ("Modell und Replay laden", self.load)):
            button = ttk.Button(actions, text=text, command=command)
            button.pack(fill="x", pady=1, ipady=1); self.buttons.append(button)
        self.stop_button, self.best_button = self.buttons[1], self.buttons[5]
        self.stop_button.configure(state="disabled"); self.best_button.configure(state="disabled")
        ttk.Progressbar(actions, variable=self.progress, maximum=100).pack(fill="x", pady=(6, 2))
        ttk.Label(actions, textvariable=self.status, foreground=self.ACCENT, wraplength=240).pack(fill="x", pady=2)
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
        self.splitter.sashpos(0, min(max(required, int(available * .50)), available - 210))
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
        for label, widget in (("Visualisierung", self.image_label), ("Diagramm", self.canvas.get_tk_widget())):
            if not widget.winfo_ismapped() or widget.winfo_width() <= 1 or widget.winfo_height() <= 1:
                issues.append(label)
        return issues

    def _config(self) -> AcrobotConfig:
        try:
            data: dict[str, Any] = {}
            integers = {"total_timesteps", "buffer_size", "learning_starts", "batch_size", "train_freq", "gradient_steps", "target_update_interval", "per_beta_steps", "multistep_n", "c51_atoms"}
            tuples = {"net_arch", "value_arch", "advantage_arch"}
            strings = {"activation", "optimizer"}
            for name, variable in self.values.items():
                text = variable.get().strip()
                if name in tuples: data[name] = tuple(int(item.strip()) for item in text.split(",") if item.strip())
                elif name in integers: data[name] = int(text)
                elif name == "seed": data[name] = int(text) if text else None
                elif name in strings: data[name] = text
                else: data[name] = float(text)
            config = AcrobotConfig(algorithm=self.algorithm.get(), **data)
            config.validate(); return config
        except ValueError as error:
            raise ValueError(f"Ungültige Parameter: {error}") from error

    def _evaluation_settings(self) -> int:
        episodes = int(self.evaluation_episodes.get())
        if episodes <= 0: raise ValueError("Evaluations-Episoden M müssen positiv sein.")
        return episodes

    def _set_busy(self, busy: bool, text: str) -> None:
        self.busy = busy; self.status.set(text)
        for index, button in enumerate(self.buttons):
            button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy else "disabled")
        if not busy and self.best_evaluation is None: self.best_button.configure(state="disabled")

    def start_training(self) -> None:
        if self.busy: return
        try: config = self._config()
        except ValueError as error: messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root); return
        if self.workbench.model is not None and config.signature() != self.workbench.config.signature():
            if not messagebox.askyesno("Neues Modell", "Geänderte Modellparameter erfordern einen Reset. Fortfahren?", parent=self.root): return
            self.workbench.close(); self.workbench = AcrobotWorkbench(config); self.evaluations.clear(); self.best_evaluation = None
        else: self.workbench.config = config
        self._clear_comparison(); self.live_history = list(self.workbench.history); self.stop_event.clear(); self.progress.set(0)
        self.progress_by_algorithm = {None: 0}
        self._set_busy(True, f"Läuft – {config.algorithm}")
        self.worker = threading.Thread(target=self._train_worker, daemon=True); self.worker.start(); self.root.after(50, self._poll)

    def _train_worker(self) -> None:
        try:
            self.workbench.train(self.stop_event, self.events)
            self.events.put(("training_done", self.stop_event.is_set()))
        except Exception as error: self.events.put(("error", error))

    def start_comparison(self) -> None:
        if self.busy: return
        try: config = self._config()
        except ValueError as error: messagebox.showerror("Ungültiger Vergleich", str(error), parent=self.root); return
        algorithms = [name for name in ALGORITHMS if self.compare[name].get()]
        if len(algorithms) < 2: messagebox.showerror("Ungültiger Vergleich", "Wähle mindestens zwei Algorithmen.", parent=self.root); return
        if self.comparison_workbenches and next(iter(self.comparison_workbenches.values())).config.signature() != config.signature(): self._clear_comparison()
        for name in algorithms:
            self.comparison_history.setdefault(name, []); self.comparison_evaluations.setdefault(name, [])
            self.comparison_workbenches.setdefault(name, AcrobotWorkbench(replace(config, algorithm=name)))
        self.active_algorithms = tuple(algorithms); self.stop_event.clear(); self.progress.set(0)
        self.progress_by_algorithm = {name: 0 for name in algorithms}
        self._set_busy(True, f"Läuft – {' / '.join(algorithms)}"); self._refresh_comparison_plot(); self._comparison_summary()
        self.worker = threading.Thread(target=self._comparison_worker, args=(algorithms,), daemon=True); self.worker.start(); self.root.after(50, self._poll)

    def _comparison_worker(self, algorithms: list[str]) -> None:
        errors: queue.Queue = queue.Queue(); barrier = threading.Barrier(len(algorithms))
        def run(name: str) -> None:
            try:
                barrier.wait(); self.comparison_workbenches[name].train(self.stop_event, self.events, name)
            except Exception as error: errors.put(error); self.stop_event.set()
        threads = [threading.Thread(target=run, args=(name,), daemon=True) for name in algorithms]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.events.put(("error", errors.get())) if not errors.empty() else self.events.put(("comparison_done", self.stop_event.is_set()))

    def start_evaluation(self) -> None:
        if self.busy or self.workbench.model is None: return
        try: episodes = self._evaluation_settings()
        except ValueError as error: messagebox.showerror("Ungültige Evaluation", str(error), parent=self.root); return
        self._set_busy(True, "Läuft – deterministische Evaluation")
        self.worker = threading.Thread(target=self._evaluation_worker, args=(episodes,), daemon=True); self.worker.start(); self.root.after(50, self._poll)

    def _evaluation_worker(self, episodes: int) -> None:
        try:
            result = self.workbench.evaluate(episodes, self.workbench.config.seed or 0); self._save_best(0, self.workbench.model.num_timesteps, result)
            self.events.put(("manual_evaluation", result))
        except Exception as error: self.events.put(("error", error))

    def _save_best(self, _episode: int, _steps: int, result: EvaluationResult) -> None:
        if self.best_evaluation is None or result.mean_reward > self.best_evaluation.mean_reward:
            self.workbench.save(self.best_base); self.best_evaluation = result

    def restore_best(self) -> None:
        if self.busy or self.best_evaluation is None: return
        try: restored = AcrobotWorkbench.load(self.best_base)
        except Exception as error: messagebox.showerror("Wiederherstellen fehlgeschlagen", str(error), parent=self.root); return
        self.workbench.close(); self.workbench = restored; self.live_history = list(restored.history); self.status.set("Bestes Modell wiederhergestellt")

    def stop(self) -> None:
        if self.busy: self.stop_event.set(); self.status.set("Stoppen angefordert …")

    def _poll(self) -> None:
        while True:
            try: kind, payload = self.events.get_nowait()
            except queue.Empty: break
            if kind == "episode":
                self.live_history.append(payload); self._training_summary()
                if time.monotonic() - self.last_plot >= self.PLOT_INTERVAL: self._refresh_training_plot(); self.last_plot = time.monotonic()
            elif kind == "comparison_episode":
                name, metric = payload; self.comparison_history[name].append(metric); self._comparison_summary()
                if time.monotonic() - self.last_plot >= self.PLOT_INTERVAL: self._refresh_comparison_plot(); self.last_plot = time.monotonic()
            elif kind == "evaluation": self.evaluations.append(payload[2]); self._training_summary()
            elif kind == "comparison_evaluation": self.comparison_evaluations[payload[0]].append(payload[3]); self._comparison_summary()
            elif kind == "progress":
                name, elapsed = payload
                self.progress_by_algorithm[name] = max(elapsed, self.progress_by_algorithm.get(name, 0))
                completed = min(self.progress_by_algorithm.values()) if self.active_algorithms else self.progress_by_algorithm[name]
                self.progress.set(min(100, 100 * completed / self.workbench.config.total_timesteps))
                if self.active_algorithms: self.status.set(f"Läuft – {' / '.join(self.active_algorithms)}")
            elif kind == "training_done": self._set_busy(False, "Gestoppt" if payload else "Abgeschlossen – Training"); self._refresh_training_plot(); self._training_summary()
            elif kind == "comparison_done": self._set_busy(False, "Gestoppt – Vergleich" if payload else "Abgeschlossen – Vergleich"); self._refresh_comparison_plot(); self._comparison_summary()
            elif kind == "manual_evaluation": self.evaluations.append(payload); self._set_busy(False, "Abgeschlossen – Evaluation"); self._training_summary()
            elif kind == "error": self._set_busy(False, "Fehler"); messagebox.showerror("Fehler", str(payload), parent=self.root)
        if self.busy: self.root.after(50, self._poll)

    def _style_plot(self) -> None:
        self.axes.set_facecolor(self.FIELD); self.axes.tick_params(colors=self.MUTED); self.axes.grid(color="#475569", alpha=.4)
        for spine in self.axes.spines.values(): spine.set_color("#64748b")
        self.axes.title.set_color(self.FG); self.axes.xaxis.label.set_color(self.MUTED); self.axes.yaxis.label.set_color(self.MUTED)
        legend = self.axes.legend(loc="center left", bbox_to_anchor=(1.02, .5), borderaxespad=0)
        legend.get_frame().set_facecolor(self.PANEL); legend.get_frame().set_edgecolor("#64748b")
        for text in legend.get_texts(): text.set_color(self.FG)

    def _refresh_training_plot(self) -> None:
        self.axes.clear(); history = self.live_history or self.workbench.history
        if history:
            episodes = [item.episode for item in history]; rewards = [item.reward for item in history]
            x, y = downsample_minmax(episodes, rewards); self.axes.plot(x, y, color="#60a5fa", alpha=.15, linewidth=.8)
            self.axes.plot(episodes, rolling_average(rewards), color="#60a5fa", linewidth=1.6, label=self.workbench.config.algorithm)
        self.axes.axhline(-100, color="#fbbf24", linestyle="--", linewidth=1.1, label="Referenz -100 (gelöst)")
        self.axes.set(title="Episoden-Reward", xlabel="Episode", ylabel="Reward"); self._style_plot(); self.canvas.draw_idle()

    def _refresh_comparison_plot(self) -> None:
        self.axes.clear(); colors = dict(zip(ALGORITHMS, COMPARISON_COLORS))
        for name, history in self.comparison_history.items():
            if not history: continue
            episodes, rewards = [item.episode for item in history], [item.reward for item in history]
            x, y = downsample_minmax(episodes, rewards); self.axes.plot(x, y, color=colors[name], alpha=.08, linewidth=.7)
            self.axes.plot(episodes, rolling_average(rewards), color=colors[name], linewidth=1.6, label=name)
        self.axes.axhline(-100, color="#fbbf24", linestyle="--", linewidth=1.1, label="Referenz -100 (gelöst)")
        self.axes.set(title="Live-Algorithmenvergleich", xlabel="Episode", ylabel="Reward"); self._style_plot(); self.canvas.draw_idle()

    def _table(self, title: str, statistics: dict[str, tuple[Any, ...]]) -> None:
        if not statistics: self.summary.set(title + "\n\nNoch keine vollständige Episode."); return
        algorithms = list(statistics); width, value_width = 10, 11
        header = "Statistik".ljust(width) + "".join(name[:value_width].rjust(value_width) for name in algorithms)
        labels = ("Episoden", "Schritte", "Ø Reward", "Erfolgsrate")
        lines = [title, "", header, "─" * len(header)]
        for index, label in enumerate(labels): lines.append(label.ljust(width) + "".join(str(statistics[name][index]).rjust(value_width) for name in algorithms))
        self.summary.set("\n".join(lines))

    def _training_summary(self) -> None:
        history = self.live_history or self.workbench.history
        stats = (len(history), self.workbench.model.num_timesteps if self.workbench.model else 0,
                 "—" if not history else f"{np.mean([x.reward for x in history]):.1f}",
                 "—" if not history else f"{np.mean([x.success for x in history]):.1%}")
        self._table("Training", {self.workbench.config.algorithm: stats})

    def _comparison_summary(self) -> None:
        stats = {}
        for name, history in self.comparison_history.items():
            if not history: continue
            stats[name] = (len(history), history[-1].timesteps, f"{np.mean([x.reward for x in history]):.1f}",
                           f"{np.mean([x.success for x in history]):.1%}")
        self._table("Vergleich", stats)

    def _config_snapshot_lines(self) -> list[str]:
        if self.active_algorithms:
            base = next(iter(self.comparison_workbenches.values())).config
            lines = [f"Verglichene Algorithmen: {', '.join(self.active_algorithms)}"]
            values = {key: value for key, value in asdict(base).items() if key != "algorithm"}
        else:
            lines = []
            values = asdict(self.workbench.config)
        lines.extend(f"{key}: {value}" for key, value in values.items())
        return lines

    def _export_snapshot(self) -> tuple[str, tuple[Any, ...]]:
        """Kennzeichnet den aktuell sichtbaren Trainings-/Vergleichsstand.

        PNG- und TXT-Export teilen sich denselben Dateinamensstamm, solange
        sich zwischen beiden Exporten keine weitere Episode geändert hat –
        so bleiben Diagramm und zugehörige Konfigurationsdatei eindeutig
        zuordenbar, ohne sich auf zufällig gleiche Uhrzeiten zu verlassen.
        """
        if self.active_algorithms:
            slug = "vergleich-" + "-".join(slugify(name) for name in self.active_algorithms)
            key = ("comparison", self.active_algorithms, tuple(
                len(self.comparison_history.get(name, ())) for name in self.active_algorithms
            ))
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
        return f"acrobot_{slug}_{self._export_stamp}"

    def export_chart(self) -> None:
        EXPORT_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.root, initialdir=EXPORT_DIR, defaultextension=".png",
            filetypes=(("PNG-Bild", "*.png"),), initialfile=f"{self._export_base_name()}.png",
        )
        if not path: return
        self.figure.savefig(path, facecolor=self.figure.get_facecolor())
        self.status.set(f"Diagramm exportiert: {Path(path).name}")

    def export_summary(self) -> None:
        EXPORT_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.root, initialdir=EXPORT_DIR, defaultextension=".txt",
            filetypes=(("Textdatei", "*.txt"),), initialfile=f"{self._export_base_name()}_config.txt",
        )
        if not path: return
        text = self.summary.get() + "\n\nKonfiguration:\n" + "\n".join(self._config_snapshot_lines()) + "\n"
        Path(path).write_text(text, encoding="utf-8")
        self.status.set(f"Summary exportiert: {Path(path).name}")

    def _show_initial_frame(self) -> None:
        observation, _info, frame = self.renderer.reset(self.workbench.config.seed); self.animation_observation = observation; self._show_frame(frame)
        theta1, theta2 = angles_degrees(observation)
        self.observation.set(f"θ₁: {theta1:+.1f}°   θ₂: {theta2:+.1f}°   θ̇₁: {observation[4]:+.2f}   θ̇₂: {observation[5]:+.2f}")

    def _show_frame(self, frame: np.ndarray) -> None:
        self.last_frame = np.asarray(frame)
        width = max(220, self.image_label.winfo_width() - 12)
        height = max(160, self.image_label.winfo_height() - 12)
        image = Image.fromarray(frame); image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image); self.image_label.configure(image=self.photo)

    def animate(self) -> None:
        if self.busy or self.workbench.model is None: return
        self._stop_animation(); observation, _info, frame = self.renderer.reset(self.workbench.config.seed)
        self.animation_observation, self.animation_step, self.animation_reward = observation, 0, 0.0; self._show_frame(frame); self._animate_step()

    def _animate_step(self) -> None:
        with self.workbench.deterministic_policy(): action, _ = self.workbench.model.predict(self.animation_observation, deterministic=True)
        observation, reward, terminated, truncated, _info, frame = self.renderer.step(int(action))
        self.animation_observation = observation; self.animation_step += 1; self.animation_reward += reward; self._show_frame(frame)
        theta1, theta2 = angles_degrees(observation)
        self.observation.set(
            f"Schritt: {self.animation_step}   Action: {ACTION_LABELS[int(action)]}   Reward: {self.animation_reward:.0f}\n"
            f"θ₁: {theta1:+.1f}°   θ₂: {theta2:+.1f}°   θ̇₁: {observation[4]:+.2f}   θ̇₂: {observation[5]:+.2f}"
        )
        if terminated or truncated: self.status.set("Abgeschlossen – Animation"); return
        self.animation_after = self.root.after(max(1, int(self.animation_delay.get())), self._animate_step)

    def _stop_animation(self) -> None:
        if self.animation_after: self.root.after_cancel(self.animation_after); self.animation_after = None

    def reset(self) -> None:
        if self.busy or not messagebox.askyesno("Neues Modell", "Aktuellen Lernzustand verwerfen?", parent=self.root): return
        try: config = self._config()
        except ValueError as error: messagebox.showerror("Fehler", str(error), parent=self.root); return
        self.workbench.close(); self.workbench = AcrobotWorkbench(config); self.live_history.clear(); self.evaluations.clear(); self.best_evaluation = None
        self._clear_comparison(); self._refresh_training_plot(); self._training_summary(); self.status.set("Bereit")

    def save(self) -> None:
        if self.workbench.model is None: return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".zip", filetypes=(("SB3-Modell", "*.zip"),))
        if path: self.workbench.save(path); self.status.set("Abgeschlossen – Modell gespeichert")

    def load(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, filetypes=(("SB3-Modell", "*.zip"),))
        if not path: return
        try: loaded = AcrobotWorkbench.load(path)
        except Exception as error: messagebox.showerror("Laden fehlgeschlagen", str(error), parent=self.root); return
        self.workbench.close(); self.workbench = loaded
        self.algorithm.set(loaded.config.algorithm)
        for name, value in loaded.config.__dict__.items():
            if name == "algorithm":
                continue
            if isinstance(value, tuple):
                value = ",".join(map(str, value))
            self.values[name].set("" if value is None else str(value))
        self.live_history = list(loaded.history)
        self.evaluations.clear(); self.best_evaluation = None
        self.status.set("Bereit – Modell geladen"); self._algorithm_fields(); self._training_summary()

    def _clear_comparison(self) -> None:
        for workbench in self.comparison_workbenches.values(): workbench.close()
        self.comparison_workbenches.clear(); self.comparison_history.clear(); self.comparison_evaluations.clear(); self.active_algorithms = ()

    def instructions(self) -> None:
        messagebox.showinfo(
            "Bedienungsanleitung",
            "1. Algorithmus und Parameter wählen.\n"
            "2. Trainieren oder mehrere Varianten vergleichen.\n"
            "3. Reward -1 je Schritt, 0 bei Zielerreichung: weniger negativ ist besser.\n"
            "4. Ziel ist, das freie Ende der Kette über die Zielhöhe zu schwingen, bevor 500 Schritte erreicht sind.\n"
            "5. Evaluation lernt nicht; NoisyNet-Varianten verwenden dabei kein Rauschen.\n"
            "6. Rainbow DDQN kombiniert Noisy-, PER-, Dueling-, Multi-Step- und C51-Baustein gleichzeitig.\n"
            "7. Bestes Modell kann vollständig wiederhergestellt werden.",
            parent=self.root,
        )

    def close(self) -> None:
        self.stop_event.set(); self._stop_animation(); self.renderer.close(); self.workbench.close(); self._clear_comparison(); self.best_checkpoint.cleanup(); self.root.destroy()
