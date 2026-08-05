"""Tkinter GUI for the Gridworld reinforcement-learning laboratory."""

from __future__ import annotations

import csv
import queue
import re
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence, Tuple

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from gridworld_logic import (
    ACTION_NAMES,
    ACTIONS,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Agent,
    BasePolicy,
    ComparisonResult,
    GridWorld,
    MonteCarloPolicy,
    Transition,
    compare_policies,
    create_policy,
    moving_average,
)

POLICY_NAMES = ("Monte Carlo", "SARSA", "Expected SARSA", "Q-Learning")
POLICY_COLORS = {
    "Monte Carlo": "#2563eb",
    "SARSA": "#ea580c",
    "Expected SARSA": "#16a34a",
    "Q-Learning": "#7c3aed",
}
ARROWS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}
CSV_FIELDS = (
    "episode", "step", "state_x", "state_y", "action", "action_name",
    "next_state_x", "next_state_y", "reward", "done",
    "termination_reason", "policy",
)


class GridGUI:
    def __init__(
        self,
        root: tk.Tk,
        environment: GridWorld,
        agent: Agent,
        policies: Dict[str, BasePolicy],
    ) -> None:
        self.root = root
        self.environment = environment
        self.agent = agent
        self.policies = policies
        self.training_running = False
        self.stop_requested = False
        self.episodes_remaining = 0
        self.episodes_target = 0
        self.animation_running = False
        self.manual_mode = False
        self.comparison_results: List[ComparisonResult] = []
        self.comparison_queue: queue.Queue = queue.Queue()
        self.comparison_cancel = threading.Event()
        self.comparison_running = False
        self._create_variables()
        self._configure_window()
        self._build_layout()
        self._update_parameter_visibility()
        self._refresh_all()

    def _configure_window(self) -> None:
        self.root.title("Model-Free Gridworld RL Lab")
        width = min(1380, max(1100, self.root.winfo_screenwidth() - 80))
        height = min(920, max(720, self.root.winfo_screenheight() - 100))
        self.root.geometry("{}x{}".format(width, height))
        self.root.minsize(1050, 700)
        self.root.configure(bg="#eef2f7")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#eef2f7")
        style.configure("TLabelframe", background="#f8fafc")
        style.configure("TLabelframe.Label", background="#eef2f7", font=("TkDefaultFont", 10, "bold"))
        style.configure("TLabel", background="#eef2f7", foreground="#111827")
        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure("Accent.TButton", foreground="white", background="#2563eb", padding=7)
        style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("disabled", "#93c5fd")])
        style.configure("Treeview", rowheight=24, background="white", fieldbackground="white")

    def _create_variables(self) -> None:
        env = self.environment
        self.width_var = tk.StringVar(value=str(env.width))
        self.height_var = tk.StringVar(value=str(env.height))
        self.start_x_var = tk.StringVar(value=str(env.start[0]))
        self.start_y_var = tk.StringVar(value=str(env.start[1]))
        self.goal_x_var = tk.StringVar(value=str(env.goal[0]))
        self.goal_y_var = tk.StringVar(value=str(env.goal[1]))
        self.blocked_var = tk.StringVar(value=", ".join("({},{})".format(*cell) for cell in env.blocked))
        self.seed_var = tk.StringVar(value="42")
        self.policy_var = tk.StringVar(value=self.agent.policy.name)
        self.episodes_var = tk.StringVar(value="100")
        self.max_steps_var = tk.StringVar(value=str(env.max_steps))
        self.alpha_var = tk.StringVar(value="0.1")
        self.gamma_var = tk.StringVar(value="0.9")
        self.use_defaults_var = tk.BooleanVar(value=True)
        self.epsilon_start_var = tk.StringVar(value="1.0")
        self.epsilon_min_var = tk.StringVar(value="0.05")
        self.epsilon_decay_var = tk.StringVar(value="0.001")
        self.status_var = tk.StringVar(value="Bereit")
        self.stats_var = tk.StringVar()
        self.progress_var = tk.DoubleVar(value=0)
        self.compare_selected = {name: tk.BooleanVar(value=True) for name in POLICY_NAMES}
        self.compare_episodes_var = tk.StringVar(value="500")
        self.compare_repetitions_var = tk.StringVar(value="20")
        self.compare_seed_var = tk.StringVar(value="42")
        self.compare_ci_var = tk.BooleanVar(value=True)
        self.compare_status_var = tk.StringVar(value="Wähle Methoden und starte den Vergleich.")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Model-Free Gridworld RL Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Tabellarische RL-Verfahren interaktiv trainieren, untersuchen und vergleichen.").pack(anchor="w", pady=(0, 8))
        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, width=330)
        right = ttk.Frame(body)
        body.add(left, weight=0)
        body.add(right, weight=1)
        self._build_controls(left)
        self._build_workspace(right)

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable, width: int = 8) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        return entry

    def _build_controls(self, parent: ttk.Frame) -> None:
        grid_box = ttk.LabelFrame(parent, text="Grid-Konfiguration", padding=8)
        grid_box.pack(fill="x", pady=(0, 7))
        dimensions = ttk.Frame(grid_box)
        dimensions.pack(fill="x")
        for col, (label, var) in enumerate((("Breite", self.width_var), ("Höhe", self.height_var))):
            ttk.Label(dimensions, text=label).grid(row=0, column=col * 2, sticky="w")
            ttk.Entry(dimensions, textvariable=var, width=5).grid(row=0, column=col * 2 + 1, padx=(3, 8))
        points = ttk.Frame(grid_box)
        points.pack(fill="x", pady=4)
        ttk.Label(points, text="Start X/Y").grid(row=0, column=0, sticky="w")
        ttk.Entry(points, textvariable=self.start_x_var, width=4).grid(row=0, column=1)
        ttk.Entry(points, textvariable=self.start_y_var, width=4).grid(row=0, column=2, padx=(3, 9))
        ttk.Label(points, text="Ziel X/Y").grid(row=0, column=3, sticky="w")
        ttk.Entry(points, textvariable=self.goal_x_var, width=4).grid(row=0, column=4)
        ttk.Entry(points, textvariable=self.goal_y_var, width=4).grid(row=0, column=5, padx=(3, 0))
        ttk.Label(grid_box, text="Blockiert: (x,y), (x,y)").pack(anchor="w")
        ttk.Entry(grid_box, textvariable=self.blocked_var).pack(fill="x", pady=(0, 4))
        seed_row = ttk.Frame(grid_box)
        seed_row.pack(fill="x")
        ttk.Label(seed_row, text="Random Seed").pack(side="left")
        ttk.Entry(seed_row, textvariable=self.seed_var, width=8).pack(side="left", padx=5)
        self.apply_grid_button = ttk.Button(grid_box, text="Grid anwenden und zurücksetzen", command=self.apply_grid)
        self.apply_grid_button.pack(fill="x", pady=(5, 0))

        training_box = ttk.LabelFrame(parent, text="Training", padding=8)
        training_box.pack(fill="x", pady=(0, 7))
        ttk.Label(training_box, text="Policy").grid(row=0, column=0, sticky="w")
        self.policy_box = ttk.Combobox(training_box, textvariable=self.policy_var, values=POLICY_NAMES, state="readonly", width=18)
        self.policy_box.grid(row=0, column=1, sticky="ew", pady=2)
        self.policy_box.bind("<<ComboboxSelected>>", self._policy_changed)
        self._entry_row(training_box, 1, "Episoden", self.episodes_var)
        self._entry_row(training_box, 2, "Max. Schritte", self.max_steps_var)
        self.alpha_label = ttk.Label(training_box, text="Alpha")
        self.alpha_label.grid(row=3, column=0, sticky="w", padx=(0, 6), pady=2)
        self.alpha_entry = ttk.Entry(training_box, textvariable=self.alpha_var, width=8)
        self.alpha_entry.grid(row=3, column=1, sticky="ew", pady=2)
        self._entry_row(training_box, 4, "Gamma", self.gamma_var)
        self.defaults_check = ttk.Checkbutton(training_box, text="Algorithmus-Standardwerte", variable=self.use_defaults_var, command=self._update_parameter_visibility)
        self.defaults_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=3)
        self.custom_epsilon_frame = ttk.Frame(training_box)
        self.custom_epsilon_frame.grid(row=6, column=0, columnspan=2, sticky="ew")
        self._entry_row(self.custom_epsilon_frame, 0, "Epsilon Start", self.epsilon_start_var)
        self._entry_row(self.custom_epsilon_frame, 1, "Epsilon Min", self.epsilon_min_var)
        self._entry_row(self.custom_epsilon_frame, 2, "Epsilon Decay", self.epsilon_decay_var)
        self.apply_training_button = ttk.Button(training_box, text="Parameter anwenden und Training zurücksetzen", command=self.apply_training)
        self.apply_training_button.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        training_box.columnconfigure(1, weight=1)

        manual = ttk.LabelFrame(parent, text="Manuelle Demo", padding=7)
        manual.pack(fill="x", pady=(0, 7))
        ttk.Button(manual, text="↑", command=lambda: self.manual_step(UP), width=5).grid(row=0, column=1)
        ttk.Button(manual, text="←", command=lambda: self.manual_step(LEFT), width=5).grid(row=1, column=0)
        ttk.Button(manual, text="↓", command=lambda: self.manual_step(DOWN), width=5).grid(row=1, column=1)
        ttk.Button(manual, text="→", command=lambda: self.manual_step(RIGHT), width=5).grid(row=1, column=2)
        self.manual_buttons = list(manual.winfo_children())
        self.root.bind("<Up>", lambda _event: self.manual_step(UP))
        self.root.bind("<Down>", lambda _event: self.manual_step(DOWN))
        self.root.bind("<Left>", lambda _event: self.manual_step(LEFT))
        self.root.bind("<Right>", lambda _event: self.manual_step(RIGHT))

        actions = ttk.LabelFrame(parent, text="Agent", padding=8)
        actions.pack(fill="x")
        self.single_step_button = ttk.Button(actions, text="Einzelschritt", command=self.single_step)
        self.single_step_button.pack(fill="x", pady=2)
        self.single_episode_button = ttk.Button(actions, text="Eine Episode ausführen", command=self.single_episode)
        self.single_episode_button.pack(fill="x", pady=2)
        self.train_button = ttk.Button(actions, text="N Episoden trainieren", style="Accent.TButton", command=self.start_training)
        self.train_button.pack(fill="x", pady=2)
        self.stop_button = ttk.Button(actions, text="Training stoppen", command=self.stop_training, state="disabled")
        self.stop_button.pack(fill="x", pady=2)
        self.evaluate_button = ttk.Button(actions, text="Gelernte Policy ausführen", command=self.run_greedy_evaluation)
        self.evaluate_button.pack(fill="x", pady=2)
        ttk.Progressbar(actions, variable=self.progress_var, maximum=100).pack(fill="x", pady=5)
        tables = ttk.Frame(actions)
        tables.pack(fill="x")
        ttk.Button(tables, text="Value-Tabelle", command=self.open_value_dialog).pack(side="left", expand=True, fill="x")
        ttk.Button(tables, text="Q-Tabelle & Policy", command=self.open_q_dialog).pack(side="left", expand=True, fill="x", padx=(4, 0))

    def _build_workspace(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)
        experiment = ttk.Frame(self.notebook, padding=7)
        comparison = ttk.Frame(self.notebook, padding=7)
        self.notebook.add(experiment, text="Experiment")
        self.notebook.add(comparison, text="Methoden vergleichen")
        experiment.columnconfigure(0, weight=1)
        experiment.rowconfigure(0, weight=3)
        experiment.rowconfigure(1, weight=2)
        self.grid_canvas = tk.Canvas(experiment, bg="white", highlightthickness=1, highlightbackground="#cbd5e1")
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_canvas.bind("<Configure>", lambda _event: self.draw_grid())
        plot_frame = ttk.Frame(experiment)
        plot_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.return_figure = Figure(figsize=(7, 3), dpi=100)
        self.return_axes = self.return_figure.add_subplot(111)
        self.return_canvas = FigureCanvasTkAgg(self.return_figure, master=plot_frame)
        self.return_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        side = ttk.Frame(plot_frame)
        side.pack(side="right", fill="y", padx=(8, 0))
        ttk.Label(side, textvariable=self.stats_var, justify="left", wraplength=240).pack(anchor="w")
        ttk.Button(side, text="Diagramm speichern", command=self.save_return_plot).pack(fill="x", pady=5)
        ttk.Label(side, textvariable=self.status_var, justify="left", wraplength=240).pack(anchor="w")
        self._build_comparison(comparison)

    def _build_comparison(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill="x")
        choices = ttk.Frame(controls)
        choices.pack(side="left")
        for index, name in enumerate(POLICY_NAMES):
            ttk.Checkbutton(choices, text=name, variable=self.compare_selected[name]).grid(row=index // 2, column=index % 2, sticky="w", padx=4)
        fields = ttk.Frame(controls)
        fields.pack(side="left", padx=15)
        for column, (label, variable) in enumerate((("Episoden", self.compare_episodes_var), ("Wiederholungen", self.compare_repetitions_var), ("Seed", self.compare_seed_var))):
            ttk.Label(fields, text=label).grid(row=0, column=column)
            ttk.Entry(fields, textvariable=variable, width=8).grid(row=1, column=column, padx=3)
        self.compare_button = ttk.Button(controls, text="Vergleich starten", style="Accent.TButton", command=self.start_comparison)
        self.compare_button.pack(side="left", padx=3)
        self.compare_stop_button = ttk.Button(controls, text="Stoppen", command=self.stop_comparison, state="disabled")
        self.compare_stop_button.pack(side="left", padx=3)
        ttk.Button(controls, text="Zurücksetzen", command=self.reset_comparison).pack(side="left", padx=3)
        ttk.Checkbutton(controls, text="95-%-Konfidenzintervalle", variable=self.compare_ci_var, command=self.draw_comparison).pack(side="right")
        content = ttk.Frame(parent)
        content.pack(fill="both", expand=True, pady=(7, 0))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)
        self.compare_figure = Figure(figsize=(7, 5), dpi=100)
        self.compare_axes = self.compare_figure.add_subplot(111)
        self.compare_canvas = FigureCanvasTkAgg(self.compare_figure, master=content)
        self.compare_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        table_frame = ttk.Frame(content)
        table_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        columns = ("policy", "return", "length", "success", "ci")
        self.compare_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        for name, title, width in (("policy", "Policy", 125), ("return", "Return", 70), ("length", "Länge", 60), ("success", "Erfolg", 60), ("ci", "95 % CI", 70)):
            self.compare_table.heading(name, text=title)
            self.compare_table.column(name, width=width, anchor="center")
        self.compare_table.pack(fill="both", expand=True)
        ttk.Label(table_frame, textvariable=self.compare_status_var, wraplength=380, justify="left").pack(fill="x", pady=5)
        self.draw_comparison()

    def _parse_blocked(self, text: str) -> Tuple[Tuple[int, int], ...]:
        if not text.strip():
            return ()
        matches = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text)
        cleaned = re.sub(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", "", text)
        if cleaned.replace(",", "").strip() or not matches:
            raise ValueError("Blockierte Zellen im Format (x,y), (x,y) eingeben.")
        return tuple((int(x), int(y)) for x, y in matches)

    def _seed(self, variable: tk.StringVar) -> Optional[int]:
        text = variable.get().strip()
        return int(text) if text else None

    def _policy_parameters(self) -> Dict[str, object]:
        alpha = float(self.alpha_var.get())
        gamma = float(self.gamma_var.get())
        epsilon_start = float(self.epsilon_start_var.get())
        epsilon_min = float(self.epsilon_min_var.get())
        epsilon_decay = float(self.epsilon_decay_var.get())
        if not 0 <= alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Alpha und Gamma müssen zwischen 0 und 1 liegen.")
        if not 0 <= epsilon_min <= epsilon_start <= 1 or epsilon_decay <= 0:
            raise ValueError("Ungültige Epsilon-Werte.")
        return {"alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start, "epsilon_min": epsilon_min, "epsilon_decay": epsilon_decay, "seed": self._seed(self.seed_var), "use_algorithm_defaults": self.use_defaults_var.get()}

    def apply_grid(self) -> None:
        try:
            candidate = GridWorld(
                width=int(self.width_var.get()), height=int(self.height_var.get()),
                start=(int(self.start_x_var.get()), int(self.start_y_var.get())),
                goal=(int(self.goal_x_var.get()), int(self.goal_y_var.get())),
                blocked=self._parse_blocked(self.blocked_var.get()),
                max_steps=int(self.max_steps_var.get()), seed=self._seed(self.seed_var),
            )
            policy = create_policy(self.policy_var.get(), **self._policy_parameters())
        except ValueError as error:
            messagebox.showerror("Ungültige Konfiguration", str(error))
            return
        self.environment = candidate
        self.policies = {name: create_policy(name, **self._policy_parameters()) for name in POLICY_NAMES}
        self.agent = Agent(self.environment, policy)
        self.manual_mode = False
        self.status_var.set("Grid angewendet und Training zurückgesetzt.")
        self._refresh_all()

    def apply_training(self) -> None:
        try:
            max_steps = int(self.max_steps_var.get())
            episodes = int(self.episodes_var.get())
            if episodes <= 0:
                raise ValueError("Episoden muss positiv sein.")
            config = self.environment_config()
            config["max_steps"] = max_steps
            candidate = GridWorld(**config)
            policy = create_policy(self.policy_var.get(), **self._policy_parameters())
        except ValueError as error:
            messagebox.showerror("Ungültige Parameter", str(error))
            return
        self.environment = candidate
        self.agent = Agent(candidate, policy)
        self.manual_mode = False
        self.status_var.set("Parameter angewendet und Training zurückgesetzt.")
        self._refresh_all()

    def environment_config(self) -> Dict[str, object]:
        return {"width": self.environment.width, "height": self.environment.height, "start": self.environment.start, "goal": self.environment.goal, "blocked": self.environment.blocked, "max_steps": self.environment.max_steps, "seed": self.environment.seed}

    def _update_parameter_visibility(self) -> None:
        if self.policy_var.get() == "Monte Carlo":
            self.alpha_label.grid_remove()
            self.alpha_entry.grid_remove()
        else:
            self.alpha_label.grid()
            self.alpha_entry.grid()
        if self.use_defaults_var.get():
            self.custom_epsilon_frame.grid_remove()
        else:
            self.custom_epsilon_frame.grid()

    def _policy_changed(self, _event: object = None) -> None:
        self._update_parameter_visibility()
        self.apply_training()

    def _switch_from_manual(self) -> None:
        if self.manual_mode:
            self.environment.reset()
            self.manual_mode = False
            self.agent.episode_active = False

    def manual_step(self, action: int) -> None:
        if self.training_running or self.animation_running:
            return
        if not self.manual_mode or self.environment.done:
            self.environment.reset()
            self.agent.cancel_episode()
            self.manual_mode = True
        state = self.environment.state
        next_state, reward, done, info = self.environment.step(action)
        transition = Transition(0, len(self.agent.current_trajectory) + 1, state, action, next_state, reward, done, info["termination_reason"], "Manuell")
        self.agent.current_trajectory.append(transition)
        self.agent.latest_trajectory = self.agent.current_trajectory[:]
        self.status_var.set("Manuell: {} → Reward {}{}".format(ACTION_NAMES[action], int(reward), " – Episode beendet" if done else ""))
        self._refresh_all()

    def single_step(self) -> None:
        if self.training_running or self.animation_running:
            return
        self._switch_from_manual()
        transition = self.agent.step(training=True)
        self.status_var.set("{}: {} → Reward {}".format(self.agent.policy.name, ACTION_NAMES[transition.action], int(transition.reward)))
        self._refresh_all()

    def single_episode(self) -> None:
        if self.training_running or self.animation_running:
            return
        self._switch_from_manual()
        self.animation_running = True
        self.agent.start_episode(training=True)
        self._animate_episode(training=True)

    def _animate_episode(self, training: bool) -> None:
        if not self.agent.episode_active:
            self.animation_running = False
            self._set_controls_enabled(True)
            reason = self.agent.latest_trajectory[-1].termination_reason if self.agent.latest_trajectory else ""
            self.status_var.set("{} abgeschlossen: {}".format("Trainingsepisode" if training else "Greedy-Auswertung", reason))
            self._refresh_all()
            return
        self._set_controls_enabled(False)
        self.agent.step(training=training)
        self._refresh_all()
        self.root.after(120, lambda: self._animate_episode(training))

    def start_training(self) -> None:
        if self.training_running or self.animation_running:
            return
        try:
            count = int(self.episodes_var.get())
            if count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ungültige Eingabe", "Episoden muss eine positive Ganzzahl sein.")
            return
        self._switch_from_manual()
        self.training_running = True
        self.stop_requested = False
        self.episodes_remaining = count
        self.episodes_target = count
        self.progress_var.set(0)
        self._set_controls_enabled(False)
        self.stop_button.configure(state="normal")
        self.root.after(1, self._training_chunk)

    def _training_chunk(self) -> None:
        if self.stop_requested or self.episodes_remaining <= 0:
            self.training_running = False
            self._set_controls_enabled(True)
            self.stop_button.configure(state="disabled")
            self.status_var.set("Training gestoppt." if self.stop_requested else "Training abgeschlossen.")
            self._refresh_all()
            return
        chunk = min(10, self.episodes_remaining)
        for _ in range(chunk):
            self.agent.run_episode(training=True)
        self.episodes_remaining -= chunk
        completed = self.episodes_target - self.episodes_remaining
        self.progress_var.set(100 * completed / self.episodes_target)
        self.status_var.set("Training: Episode {} von {}".format(completed, self.episodes_target))
        self._refresh_all()
        self.root.after(1, self._training_chunk)

    def stop_training(self) -> None:
        self.stop_requested = True

    def run_greedy_evaluation(self) -> None:
        if self.training_running or self.animation_running:
            return
        self._switch_from_manual()
        self.animation_running = True
        self.agent.start_episode(training=False)
        self._animate_episode(training=False)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (self.apply_grid_button, self.apply_training_button, self.single_step_button, self.single_episode_button, self.train_button, self.evaluate_button):
            widget.configure(state=state)
        self.policy_box.configure(state="readonly" if enabled else "disabled")
        for widget in self.manual_buttons:
            if isinstance(widget, ttk.Button):
                widget.configure(state=state)

    def _refresh_all(self) -> None:
        self.draw_grid()
        self.draw_return_plot()
        policy = self.agent.policy
        successes = sum(self.agent.successes)
        self.stats_var.set("Policy: {}\nEpisoden: {}\nEpsilon: {:.4f}\nErfolge: {} ({:.1%})\nLetzter Return: {}".format(policy.name, self.agent.episode_count, policy.epsilon, successes, successes / self.agent.episode_count if self.agent.episode_count else 0, self.agent.returns[-1] if self.agent.returns else "–"))

    def draw_grid(self) -> None:
        canvas = self.grid_canvas
        canvas.delete("all")
        env = self.environment
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 280)
        cell = min((width - 40) / env.width, (height - 40) / env.height)
        ox = (width - cell * env.width) / 2
        oy = (height - cell * env.height) / 2
        for y in range(env.height):
            for x in range(env.width):
                state = (x, y)
                fill = "#ffffff"
                if state in env.blocked:
                    fill = "#374151"
                elif state == env.start:
                    fill = "#bbf7d0"
                elif state == env.goal:
                    fill = "#fde68a"
                x1, y1 = ox + x * cell, oy + y * cell
                canvas.create_rectangle(x1, y1, x1 + cell, y1 + cell, fill=fill, outline="#94a3b8", width=2)
                canvas.create_text(x1 + 5, y1 + 5, text="{},{}".format(x, y), anchor="nw", fill="#94a3b8", font=("TkDefaultFont", 8))
                if state == env.start:
                    canvas.create_text(x1 + cell / 2, y1 + cell * 0.75, text="START", fill="#166534", font=("TkDefaultFont", 8, "bold"))
                if state == env.goal:
                    canvas.create_text(x1 + cell / 2, y1 + cell * 0.75, text="ZIEL", fill="#92400e", font=("TkDefaultFont", 8, "bold"))
        trajectory = self.agent.current_trajectory or self.agent.latest_trajectory
        for transition in trajectory:
            if transition.state == transition.next_state:
                x, y = transition.state
                cx, cy = ox + (x + 0.5) * cell, oy + (y + 0.5) * cell
                canvas.create_oval(cx - 9, cy - 9, cx + 9, cy + 9, outline="#ef4444", width=2)
            else:
                x1 = ox + (transition.state[0] + 0.5) * cell
                y1 = oy + (transition.state[1] + 0.5) * cell
                x2 = ox + (transition.next_state[0] + 0.5) * cell
                y2 = oy + (transition.next_state[1] + 0.5) * cell
                canvas.create_line(x1, y1, x2, y2, fill="#2563eb", width=3, arrow="last")
        x, y = env.state
        cx, cy = ox + (x + 0.5) * cell, oy + (y + 0.5) * cell
        radius = min(cell * 0.22, 22)
        canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill="#dc2626", outline="white", width=2)
        canvas.create_text(cx, cy, text="A", fill="white", font=("TkDefaultFont", 12, "bold"))

    def draw_return_plot(self) -> None:
        axes = self.return_axes
        axes.clear()
        axes.set_title("Episode-Returns")
        axes.set_xlabel("Episode")
        axes.set_ylabel("Return")
        axes.grid(True, alpha=0.25)
        if self.agent.returns:
            x = list(range(1, len(self.agent.returns) + 1))
            color = POLICY_COLORS[self.agent.policy.name]
            axes.plot(x, self.agent.returns, color=color, alpha=0.25, linewidth=1, label="Episode-Return")
            axes.plot(x, moving_average(self.agent.returns), color=color, linewidth=2.5, label="Ø letzte 20")
            axes.legend(fontsize=8)
        else:
            axes.text(0.5, 0.5, "Trainiere den Agenten, um die Lernkurve zu sehen.", ha="center", va="center", transform=axes.transAxes, color="#64748b")
        self.return_figure.tight_layout()
        self.return_canvas.draw_idle()

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    def export_trajectory(self) -> None:
        trajectory = self.agent.latest_trajectory or self.agent.current_trajectory
        if not trajectory:
            messagebox.showinfo("Kein Export", "Es ist noch keine Trajektorie vorhanden.")
            return
        export_dir = Path(__file__).parent / "exports"
        export_dir.mkdir(exist_ok=True)
        policy = re.sub(r"[^a-z0-9]+", "-", trajectory[0].policy.lower()).strip("-")
        path = export_dir / "{}-episode-{}-{}.csv".format(policy, trajectory[0].episode, self._timestamp())
        try:
            with path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(item.csv_row() for item in trajectory)
        except OSError as error:
            messagebox.showerror("Export fehlgeschlagen", str(error))
            return
        messagebox.showinfo("Export abgeschlossen", str(path))

    def save_return_plot(self) -> None:
        plot_dir = Path(__file__).parent / "plots"
        plot_dir.mkdir(exist_ok=True)
        default = "{}-{}.png".format(self.agent.policy.name.lower().replace(" ", "-"), self._timestamp())
        path = filedialog.asksaveasfilename(initialdir=plot_dir, initialfile=default, defaultextension=".png", filetypes=(("PNG", "*.png"),))
        if path:
            try:
                self.return_figure.savefig(path, dpi=150)
            except OSError as error:
                messagebox.showerror("Speichern fehlgeschlagen", str(error))

    def _value_dialog_canvas(self, title: str) -> Tuple[tk.Toplevel, tk.Canvas]:
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("760x620")
        canvas = tk.Canvas(dialog, bg="white")
        canvas.pack(fill="both", expand=True, padx=8, pady=8)
        buttons = ttk.Frame(dialog)
        buttons.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="CSV-Trajektorie exportieren", command=self.export_trajectory).pack(side="left")
        ttk.Button(buttons, text="Schließen", command=dialog.destroy).pack(side="right")
        return dialog, canvas

    def open_value_dialog(self) -> None:
        dialog, canvas = self._value_dialog_canvas("Value-Tabelle – V(s)")
        self._draw_value_grid(canvas, q_mode=False)
        ttk.Button(dialog, text="Aktualisieren", command=lambda: self._draw_value_grid(canvas, q_mode=False)).pack(pady=(0, 7))

    def open_q_dialog(self) -> None:
        dialog, canvas = self._value_dialog_canvas("Q-Tabelle und greedy Policy")
        self._draw_value_grid(canvas, q_mode=True)
        ttk.Button(dialog, text="Aktualisieren", command=lambda: self._draw_value_grid(canvas, q_mode=True)).pack(pady=(0, 7))

    def _draw_value_grid(self, canvas: tk.Canvas, q_mode: bool) -> None:
        canvas.delete("all")
        canvas.update_idletasks()
        env, policy = self.environment, self.agent.policy
        width, height = max(canvas.winfo_width(), 700), max(canvas.winfo_height(), 500)
        cell = min((width - 30) / env.width, (height - 30) / env.height)
        ox, oy = (width - cell * env.width) / 2, (height - cell * env.height) / 2
        for y in range(env.height):
            for x in range(env.width):
                state = (x, y)
                x1, y1 = ox + x * cell, oy + y * cell
                fill = "#374151" if state in env.blocked else "#ffffff"
                if state == env.start:
                    fill = "#bbf7d0"
                elif state == env.goal:
                    fill = "#fde68a"
                canvas.create_rectangle(x1, y1, x1 + cell, y1 + cell, fill=fill, outline="#64748b")
                if state in env.blocked:
                    continue
                if state == env.goal:
                    canvas.create_text(x1 + cell / 2, y1 + cell / 2, text="ZIEL", font=("TkDefaultFont", 10, "bold"))
                elif q_mode:
                    if not policy.is_state_visited(state) and all(value == 0 for value in policy.q[state]):
                        canvas.create_text(x1 + cell / 2, y1 + cell / 2, text="?", font=("TkDefaultFont", 18, "bold"), fill="#64748b")
                    else:
                        values = policy.q[state]
                        best = set(policy.best_actions(state))
                        text = "\n".join("{} {:.2f}{}".format(ARROWS[action], values[action], " *" if action in best else "") for action in ACTIONS)
                        canvas.create_text(x1 + cell / 2, y1 + cell / 2, text=text, font=("TkDefaultFont", 9), justify="left")
                else:
                    canvas.create_text(x1 + cell / 2, y1 + cell / 2, text="V(s)\n{:.3f}".format(policy.get_state_value(state)), font=("TkDefaultFont", 10, "bold"))

    def start_comparison(self) -> None:
        if self.comparison_running:
            return
        names = [name for name, selected in self.compare_selected.items() if selected.get()]
        try:
            episodes = int(self.compare_episodes_var.get())
            repetitions = int(self.compare_repetitions_var.get())
            seed = self._seed(self.compare_seed_var)
            if not names or episodes <= 0 or repetitions <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ungültiger Vergleich", "Mindestens eine Policy und positive Ganzzahlen angeben.")
            return
        self.comparison_running = True
        self.comparison_cancel.clear()
        self.compare_button.configure(state="disabled")
        self.compare_stop_button.configure(state="normal")
        self.compare_status_var.set("Vergleich wird berechnet …")
        config = self.environment_config()
        parameters = self._policy_parameters()

        def worker() -> None:
            try:
                results = compare_policies(names, config, episodes, repetitions, seed, alpha=float(parameters["alpha"]), gamma=float(parameters["gamma"]), cancelled=self.comparison_cancel.is_set)
                self.comparison_queue.put(("ok", results))
            except Exception as error:
                self.comparison_queue.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(80, self._poll_comparison)

    def _poll_comparison(self) -> None:
        try:
            status, payload = self.comparison_queue.get_nowait()
        except queue.Empty:
            self.root.after(80, self._poll_comparison)
            return
        self.comparison_running = False
        self.compare_button.configure(state="normal")
        self.compare_stop_button.configure(state="disabled")
        if status == "error":
            messagebox.showerror("Vergleich fehlgeschlagen", payload)
            return
        self.comparison_results = payload
        self.draw_comparison()
        if self.comparison_results:
            winner = self.comparison_results[0]
            suffix = " Der Lauf wurde vorzeitig gestoppt." if self.comparison_cancel.is_set() else ""
            self.compare_status_var.set("{} erzielt unter diesen Bedingungen den besten finalen Mittelwert. Ergebnisse hängen von Grid, Parametern und Zufall ab.{}".format(winner.policy, suffix))
        else:
            self.compare_status_var.set("Vergleich ohne vollständige Wiederholung gestoppt.")

    def stop_comparison(self) -> None:
        self.comparison_cancel.set()
        self.compare_status_var.set("Stop angefordert – aktuelle Wiederholung wird abgeschlossen …")

    def reset_comparison(self) -> None:
        if self.comparison_running:
            return
        self.comparison_results = []
        self.compare_status_var.set("Wähle Methoden und starte den Vergleich.")
        self.draw_comparison()

    def draw_comparison(self) -> None:
        axes = self.compare_axes
        axes.clear()
        axes.set_title("Methodenvergleich")
        axes.set_xlabel("Episode")
        axes.set_ylabel("Mittlerer Episode-Return")
        axes.grid(True, alpha=0.25)
        for row in self.compare_table.get_children():
            self.compare_table.delete(row)
        if not self.comparison_results:
            axes.text(0.5, 0.5, "Starte einen Vergleich, um Lernkurven zu sehen.", ha="center", va="center", transform=axes.transAxes, color="#64748b")
        else:
            for rank, result in enumerate(self.comparison_results):
                x = list(range(1, len(result.mean_returns) + 1))
                color = POLICY_COLORS[result.policy]
                smoothed = moving_average(result.mean_returns)
                axes.plot(x, result.mean_returns, color=color, alpha=0.22, linewidth=1)
                axes.plot(x, smoothed, color=color, linewidth=2.5, label=result.policy)
                if self.compare_ci_var.get() and len(result.mean_returns) > 0:
                    axes.fill_between(x, result.confidence_low, result.confidence_high, color=color, alpha=0.08)
                self.compare_table.insert("", "end", values=(result.policy, "{:.2f}".format(result.final_moving_average), "{:.1f}".format(result.mean_episode_length), "{:.1%}".format(result.success_rate), "±{:.2f}".format(result.final_confidence_margin)), tags=("winner",) if rank == 0 else ())
            axes.legend(fontsize=8)
            self.compare_table.tag_configure("winner", background="#dcfce7")
        self.compare_figure.tight_layout()
        self.compare_canvas.draw_idle()
