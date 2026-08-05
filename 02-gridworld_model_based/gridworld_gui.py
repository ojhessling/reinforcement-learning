"""Tkinter interface for the model-based Gridworld laboratory."""

from __future__ import annotations

import csv
import queue
import re
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional, Sequence

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from gridworld_logic import (
    ACTION_ARROWS, ACTION_NAMES, ACTIONS, DOWN, LEFT, RIGHT, UP,
    BasePlanner, ComparisonResult, ComparisonRunner, GridWorld, RolloutAgent,
    RolloutResult, Transition, VALUE_CSV_FIELDS, Q_CSV_FIELDS,
    create_planners, q_table_rows, value_table_rows,
)

METHODS = ("Value Iteration", "Q-Value Iteration")
COLORS = {"Value Iteration": "#2563eb", "Q-Value Iteration": "#ea580c"}


class GridGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.environment = GridWorld()
        self.seed: Optional[int] = 42
        self.planners = create_planners(self.environment, seed=self.seed)
        self.agent = RolloutAgent(self.environment, self.planners[METHODS[0]], self.seed)
        self.epsilon_decay_current = 0.995
        self.rollout_results: Dict[str, List[RolloutResult]] = {name: [] for name in METHODS}
        self.current_path: List[Transition] = []
        self.manual_mode = False
        self.planning = False
        self.stop_requested = False
        self.animation = False
        self.comparison_results: List[ComparisonResult] = []
        self.comparison_queue: queue.Queue = queue.Queue()
        self.comparison_cancel = threading.Event()
        self.comparison_running = False
        self._create_variables()
        self._configure_window()
        self._build_layout()
        self._refresh_all()

    @property
    def planner(self) -> BasePlanner:
        return self.planners[self.method_var.get()]

    def _create_variables(self) -> None:
        env = self.environment
        self.rows_var = tk.StringVar(value=str(env.rows))
        self.columns_var = tk.StringVar(value=str(env.columns))
        self.start_row_var = tk.StringVar(value=str(env.start[0]))
        self.start_column_var = tk.StringVar(value=str(env.start[1]))
        self.goal_row_var = tk.StringVar(value=str(env.goal[0]))
        self.goal_column_var = tk.StringVar(value=str(env.goal[1]))
        self.obstacles_var = tk.StringVar(value=", ".join(f"({r},{c})" for r, c in env.obstacles))
        self.max_steps_var = tk.StringVar(value=str(env.max_steps))
        self.seed_var = tk.StringVar(value="42")
        self.method_var = tk.StringVar(value=METHODS[0])
        self.compare_var = tk.BooleanVar(value=True)
        self.planner_max_iterations_var = tk.StringVar(value="1000")
        self.gamma_var = tk.StringVar(value="0.9")
        self.tolerance_var = tk.StringVar(value="0.0001")
        self.agent_loops_var = tk.StringVar(value="50")
        self.epsilon_greedy_var = tk.StringVar(value="0.1")
        self.epsilon_max_var = tk.StringVar(value="0.995")
        self.epsilon_min_var = tk.StringVar(value="0.01")
        self.epsilon_decay_var = tk.StringVar(value="0.05")
        self.policy_visible_var = tk.BooleanVar(value=True)
        self.cell_display_var = tk.StringVar(value="V(s) und beste Actions")
        self.status_var = tk.StringVar(
            value="Noch nicht geplant: Werte starten bei 0. Bitte einen Planner-Sweep ausführen."
        )
        self.summary_var = tk.StringVar()
        self.compare_status_var = tk.StringVar(value="Noch kein Methodenvergleich ausgeführt.")

    def _configure_window(self) -> None:
        self.root.title("Modellbasiertes Gridworld Lab")
        width = min(1500, max(1180, self.root.winfo_screenwidth() - 60))
        height = min(980, max(760, self.root.winfo_screenheight() - 80))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1100, 720)
        self.root.configure(bg="#eef2f7")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#eef2f7")
        style.configure("TLabelframe", background="#f8fafc")
        style.configure("TLabelframe.Label", background="#eef2f7", font=("TkDefaultFont", 10, "bold"))
        style.configure("TLabel", background="#eef2f7", foreground="#111827")
        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure("Accent.TButton", foreground="white", background="#2563eb", padding=6)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Modellbasiertes Gridworld Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Value Iteration und Q-Value Iteration verstehen und vergleichen").pack(anchor="w", pady=(0, 8))
        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, width=350)
        workspace = ttk.Frame(body)
        body.add(controls, weight=0)
        body.add(workspace, weight=1)
        self._build_controls(controls)
        self._build_workspace(workspace)

    def _entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable, width: int = 9) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 5), pady=2)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        return entry

    def _build_controls(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, width=350, bg="#eef2f7", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        panel = ttk.Frame(canvas)
        panel.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=panel, anchor="nw", width=330)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        grid = ttk.LabelFrame(panel, text="Grid-Konfiguration", padding=8)
        grid.pack(fill="x", pady=(0, 7))
        pairs = ttk.Frame(grid)
        pairs.pack(fill="x")
        self._entry(pairs, 0, "Rows", self.rows_var)
        self._entry(pairs, 1, "Columns", self.columns_var)
        self._entry(pairs, 2, "Start Row", self.start_row_var)
        self._entry(pairs, 3, "Start Column", self.start_column_var)
        self._entry(pairs, 4, "Goal Row", self.goal_row_var)
        self._entry(pairs, 5, "Goal Column", self.goal_column_var)
        self._entry(pairs, 6, "Max Steps", self.max_steps_var)
        self._entry(pairs, 7, "Random Seed", self.seed_var)
        pairs.columnconfigure(1, weight=1)
        ttk.Label(grid, text="Hindernisse: (row,column), ...").pack(anchor="w", pady=(4, 0))
        ttk.Entry(grid, textvariable=self.obstacles_var).pack(fill="x")
        self.apply_button = ttk.Button(grid, text="Grid anwenden und zurücksetzen", command=self.apply_settings)
        self.apply_button.pack(fill="x", pady=(6, 0))

        settings = ttk.LabelFrame(panel, text="Planung und Rollout", padding=8)
        settings.pack(fill="x", pady=(0, 7))
        ttk.Label(settings, text="Methode").grid(row=0, column=0, sticky="w")
        self.method_box = ttk.Combobox(settings, textvariable=self.method_var, values=METHODS, state="readonly", width=20)
        self.method_box.grid(row=0, column=1, sticky="ew")
        self.method_box.bind("<<ComboboxSelected>>", self._method_changed)
        ttk.Checkbutton(settings, text="Methodenvergleich", variable=self.compare_var).grid(row=1, column=0, columnspan=2, sticky="w")
        self._entry(settings, 2, "Planner Max Iterations", self.planner_max_iterations_var)
        self._entry(settings, 3, "Gamma", self.gamma_var)
        self._entry(settings, 4, "Tolerance", self.tolerance_var)
        self._entry(settings, 5, "Agent-Loops", self.agent_loops_var)
        self._entry(settings, 6, "Epsilon (greedy)", self.epsilon_greedy_var)
        self._entry(settings, 7, "Epsilon Max (decay)", self.epsilon_max_var)
        self._entry(settings, 8, "Epsilon Min (decay)", self.epsilon_min_var)
        self._entry(settings, 9, "Decay (decay)", self.epsilon_decay_var)
        ttk.Label(settings, text="Epsilon steuert nur den Agenten-Rollout,\nnicht Value Iteration.", foreground="#475569").grid(row=10, column=0, columnspan=2, sticky="w", pady=4)
        settings.columnconfigure(1, weight=1)

        view = ttk.LabelFrame(panel, text="Grid-Anzeige", padding=7)
        view.pack(fill="x", pady=(0, 7))
        ttk.Checkbutton(view, text="Policy anzeigen", variable=self.policy_visible_var, command=self.draw_grid).pack(anchor="w")
        ttk.Combobox(view, textvariable=self.cell_display_var, values=("Keine", "V(s)", "beste Actions", "V(s) und beste Actions"), state="readonly").pack(fill="x")
        self.cell_display_var.trace_add("write", lambda *_: self.draw_grid())

        manual = ttk.LabelFrame(panel, text="Manuelle Demo", padding=7)
        manual.pack(fill="x", pady=(0, 7))
        ttk.Button(manual, text="Up", command=lambda: self.manual_step(UP)).grid(row=0, column=1, sticky="ew")
        ttk.Button(manual, text="Left", command=lambda: self.manual_step(LEFT)).grid(row=1, column=0, sticky="ew")
        ttk.Button(manual, text="Down", command=lambda: self.manual_step(DOWN)).grid(row=1, column=1, sticky="ew")
        ttk.Button(manual, text="Right", command=lambda: self.manual_step(RIGHT)).grid(row=1, column=2, sticky="ew")
        self.manual_buttons = list(manual.winfo_children())
        for column in range(3): manual.columnconfigure(column, weight=1)

        actions = ttk.LabelFrame(panel, text="Steuerung", padding=8)
        actions.pack(fill="x")
        ttk.Button(
            actions,
            text="Bedienungsanleitung",
            command=self.open_instructions_dialog,
        ).pack(fill="x", pady=(0, 5))
        self.sweep_button = ttk.Button(actions, text="Planner single sweep", command=self.single_sweep)
        self.sweep_button.pack(fill="x", pady=2)
        self.run_planner_button = ttk.Button(actions, text="Planner bis Konvergenz", style="Accent.TButton", command=self.start_planning)
        self.run_planner_button.pack(fill="x", pady=2)
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(fill="x", pady=2)
        ttk.Button(actions, text="Reset", command=self.reset_experiment).pack(fill="x", pady=2)
        self.agent_buttons: List[ttk.Button] = []
        for text, command in (
            ("Agent single step", self.agent_single_step),
            ("Agent run n loops", self.agent_run_loops),
            ("Agent run episode", self.agent_run_episode),
            ("Greedy Policy ausführen", self.run_greedy),
        ):
            button = ttk.Button(actions, text=text, command=command)
            button.pack(fill="x", pady=2)
            self.agent_buttons.append(button)
        tables = ttk.Frame(actions)
        tables.pack(fill="x", pady=(3, 0))
        ttk.Button(tables, text="Value-Table", command=self.open_value_dialog).pack(side="left", fill="x", expand=True)
        ttk.Button(tables, text="Q-Table", command=self.open_q_dialog).pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _build_workspace(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=2)
        grid_frame = ttk.Frame(parent)
        grid_frame.grid(row=0, column=0, sticky="nsew")
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.rowconfigure(0, weight=1)
        self.grid_canvas = tk.Canvas(grid_frame, bg="white", highlightbackground="#cbd5e1", highlightthickness=1)
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_canvas.bind("<Configure>", lambda _e: self.draw_grid())
        side = ttk.Frame(grid_frame, width=270, padding=(8, 0))
        side.grid(row=0, column=1, sticky="ns")
        ttk.Label(side, text="Summary", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        ttk.Label(side, textvariable=self.summary_var, justify="left", wraplength=250).pack(anchor="w", pady=5)
        ttk.Label(side, textvariable=self.status_var, justify="left", wraplength=250, foreground="#1d4ed8").pack(anchor="w")

        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(7, 0))
        self.figures: Dict[str, Figure] = {}
        self.axes = {}
        self.plot_canvases = {}
        for name in ("Reward", "Konvergenz", "Methodenvergleich"):
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=name)
            figure = Figure(figsize=(7, 3), dpi=100)
            axes = figure.add_subplot(111)
            canvas = FigureCanvasTkAgg(figure, master=frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.figures[name], self.axes[name], self.plot_canvases[name] = figure, axes, canvas
        compare_bar = ttk.Frame(self.notebook.nametowidget(self.notebook.tabs()[2]))
        compare_bar.pack(fill="x", before=self.plot_canvases["Methodenvergleich"].get_tk_widget())
        ttk.Button(compare_bar, text="Methodenvergleich starten", command=self.start_comparison).pack(side="left", padx=4, pady=3)
        ttk.Label(compare_bar, textvariable=self.compare_status_var).pack(side="left", padx=8)

    def _parse_obstacles(self, text: str) -> Sequence[tuple[int, int]]:
        if not text.strip(): return ()
        matches = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text)
        residue = re.sub(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", "", text).replace(",", "").strip()
        if residue or not matches:
            raise ValueError("Hindernisse bitte als (row,column), ... eingeben.")
        return tuple((int(row), int(column)) for row, column in matches)

    def _parameters(self) -> tuple[int, float, float, float, float, float, float]:
        iterations = int(self.planner_max_iterations_var.get())
        gamma, tolerance = float(self.gamma_var.get()), float(self.tolerance_var.get())
        epsilon_greedy = float(self.epsilon_greedy_var.get())
        epsilon_max, epsilon_min = float(self.epsilon_max_var.get()), float(self.epsilon_min_var.get())
        epsilon_decay = float(self.epsilon_decay_var.get())
        if iterations <= 0 or tolerance <= 0 or not 0 <= gamma <= 1:
            raise ValueError("Planer-Iterationen, Gamma oder Tolerance sind ungültig.")
        if not 0 <= epsilon_greedy <= 1 or not 0 <= epsilon_min <= epsilon_max <= 1 or not 0 <= epsilon_decay <= 1:
            raise ValueError("Epsilon-Werte müssen zwischen 0 und 1 liegen; Min darf Max nicht übersteigen.")
        return iterations, gamma, tolerance, epsilon_greedy, epsilon_max, epsilon_min, epsilon_decay

    def apply_settings(self) -> None:
        try:
            _, gamma, tolerance, _, epsilon_max, _, _ = self._parameters()
            seed = int(self.seed_var.get()) if self.seed_var.get().strip() else None
            environment = GridWorld(
                int(self.rows_var.get()), int(self.columns_var.get()),
                (int(self.start_row_var.get()), int(self.start_column_var.get())),
                (int(self.goal_row_var.get()), int(self.goal_column_var.get())),
                self._parse_obstacles(self.obstacles_var.get()), int(self.max_steps_var.get()), seed,
            )
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root)
            return
        self.environment, self.seed = environment, seed
        self.planners = create_planners(environment, gamma, tolerance, seed)
        self.agent = RolloutAgent(environment, self.planner, seed)
        self.epsilon_decay_current = epsilon_max
        self.rollout_results = {name: [] for name in METHODS}
        self.current_path = []
        self.comparison_results = []
        self.status_var.set("Grid und Parameter wurden übernommen.")
        self._refresh_all()

    def reset_experiment(self) -> None:
        try: _, _, _, _, epsilon_max, _, _ = self._parameters()
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root); return
        for planner in self.planners.values(): planner.reset()
        self.agent.set_planner(self.planner)
        self.epsilon_decay_current = epsilon_max
        self.rollout_results = {name: [] for name in METHODS}
        self.current_path = []
        self.comparison_results = []
        self.status_var.set("Experiment zurückgesetzt.")
        self._refresh_all()

    def _method_changed(self, _event=None) -> None:
        self.agent.set_planner(self.planner)
        self.current_path = []
        self.status_var.set(f"Methode: {self.planner.name}")
        self._refresh_all()

    def single_sweep(self) -> None:
        if self.planner.converged: return
        result = self.planner.single_sweep()
        self.status_var.set(f"Sweep {result.iteration}: Delta {result.delta:.6g}")
        self._refresh_all()

    def start_planning(self) -> None:
        try: maximum = int(self.planner_max_iterations_var.get())
        except ValueError: messagebox.showerror("Ungültige Eingabe", "Planner Max Iterations muss ganzzahlig sein."); return
        if maximum <= 0 or self.planner.converged: return
        self.planning, self.stop_requested = True, False
        self._planning_remaining = maximum
        self._set_busy(True)
        self._planning_tick()

    def _planning_tick(self) -> None:
        if self.stop_requested or self._planning_remaining <= 0 or self.planner.converged:
            self.planning = False
            self._set_busy(False)
            self.status_var.set("Planung abgebrochen." if self.stop_requested else ("Planner konvergiert." if self.planner.converged else "Iterationslimit erreicht."))
            self._refresh_all(); return
        for _ in range(min(10, self._planning_remaining)):
            if self.stop_requested or self.planner.converged: break
            self.planner.single_sweep(); self._planning_remaining -= 1
        self._refresh_all()
        self.root.after(10, self._planning_tick)

    def cancel(self) -> None:
        self.stop_requested = True
        self.comparison_cancel.set()

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.sweep_button.configure(state=state)
        self.run_planner_button.configure(state=state)
        self.apply_button.configure(state=state)
        for button in self.agent_buttons + self.manual_buttons: button.configure(state=state)
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def manual_step(self, action: int) -> None:
        if self.planning or self.animation: return
        if not self.manual_mode or self.environment.done:
            self.environment.reset(); self.current_path = []; self.manual_mode = True
        state = self.environment.current_state
        next_state, reward, done, info = self.environment.step(action)
        self.current_path.append(Transition(self.environment.step_count, state, action, next_state, reward, done, info["termination_reason"]))
        self.status_var.set("Manuelle Demo: Ziel erreicht." if done else "Manuelle Demo – beeinflusst keine Planung.")
        self.draw_grid()

    def _prepare_agent(self) -> None:
        if self.planner.iteration == 0:
            raise RuntimeError(
                "Die Value-Tabelle wurde noch nicht berechnet. Bitte zuerst "
                "„Planner single sweep“ oder „Planner bis Konvergenz“ ausführen."
            )
        if self.manual_mode:
            self.agent.reset(); self.current_path = []; self.manual_mode = False
        self.agent.planner = self.planner

    def agent_single_step(self) -> None:
        try: epsilon = self._parameters()[3]
        except ValueError as error: messagebox.showerror("Ungültige Eingabe", str(error)); return
        try:
            self._prepare_agent()
        except RuntimeError as error:
            messagebox.showinfo("Planner zuerst ausführen", str(error), parent=self.root)
            return
        transition = self.agent.single_step(epsilon)
        self.current_path = list(self.agent.transitions)
        if transition.done:
            self.status_var.set("Einzelschritt hat den Rollout beendet; keine Statistik oder Decay.")
        else: self.status_var.set("Ein Agentenschritt ausgeführt.")
        self._refresh_all()

    def agent_run_loops(self) -> None:
        try: loops = int(self.agent_loops_var.get()); self._parameters()
        except ValueError as error: messagebox.showerror("Ungültige Eingabe", str(error)); return
        if loops <= 0: messagebox.showerror("Ungültige Eingabe", "Agent-Loops muss positiv sein."); return
        self._start_animation(loops, self.epsilon_decay_current, True, False)

    def agent_run_episode(self) -> None:
        self._start_animation(self.environment.max_steps, self.epsilon_decay_current, True, True)

    def run_greedy(self) -> None:
        self._start_animation(self.environment.max_steps, 0.0, False, True)

    def _start_animation(self, count: int, epsilon: float, decay: bool, new_episode: bool) -> None:
        if self.animation: return
        try: self._parameters()
        except ValueError as error: messagebox.showerror("Ungültige Eingabe", str(error)); return
        self.stop_requested = False
        try:
            self._prepare_agent()
        except RuntimeError as error:
            messagebox.showinfo("Planner zuerst ausführen", str(error), parent=self.root)
            return
        if new_episode or self.environment.done: self.agent.reset()
        self.current_path = list(self.agent.transitions)
        self._animation_remaining, self._animation_epsilon, self._animation_decay = count, epsilon, decay
        self.animation = True; self._set_busy(True); self._animation_tick()

    def _animation_tick(self) -> None:
        if self.stop_requested or self._animation_remaining <= 0 or self.environment.done:
            completed = self.environment.done
            if completed:
                result = self.agent.current_result(self._animation_epsilon)
                self.rollout_results[self.planner.name].append(result)
                if self._animation_decay:
                    _, _, _, _, _, epsilon_min, decay = self._parameters()
                    self.epsilon_decay_current = max(epsilon_min, self.epsilon_decay_current * (1 - decay))
            self.animation = False; self.stop_requested = False; self._set_busy(False)
            self.status_var.set("Rollout abgeschlossen." if completed else "Rollout pausiert und kann fortgesetzt werden.")
            self._refresh_all(); return
        transition = self.agent.single_step(self._animation_epsilon)
        self.current_path = list(self.agent.transitions)
        self._animation_remaining -= 1
        self.draw_grid(); self.root.after(120, self._animation_tick)

    def start_comparison(self) -> None:
        if self.comparison_running: return
        try:
            maximum, gamma, tolerance, _, epsilon_max, epsilon_min, decay = self._parameters()
        except ValueError as error: messagebox.showerror("Ungültige Eingabe", str(error)); return
        self.comparison_running = True; self.comparison_cancel.clear()
        self.compare_status_var.set("Vergleich läuft …")
        runner = ComparisonRunner(self.environment, gamma, tolerance, maximum, 20, epsilon_max, epsilon_min, decay, self.seed)
        threading.Thread(target=lambda: self.comparison_queue.put(runner.run(self.comparison_cancel.is_set)), daemon=True).start()
        self.root.after(100, self._poll_comparison)

    def _poll_comparison(self) -> None:
        try: results = self.comparison_queue.get_nowait()
        except queue.Empty: self.root.after(100, self._poll_comparison); return
        self.comparison_running = False; self.comparison_results = results
        self.compare_status_var.set("Vergleich abgeschlossen." if len(results) == 2 else "Vergleich unvollständig.")
        self._draw_plots()

    def draw_grid(self) -> None:
        if not hasattr(self, "grid_canvas"): return
        canvas = self.grid_canvas; canvas.delete("all")
        width, height = max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 200)
        cell = max(30, min((width - 20) / self.environment.columns, (height - 20) / self.environment.rows))
        x0, y0 = (width - cell * self.environment.columns) / 2, (height - cell * self.environment.rows) / 2
        display = self.cell_display_var.get()
        path_states = {item.next_state for item in self.current_path}
        for row in range(self.environment.rows):
            for column in range(self.environment.columns):
                state = (row, column); x1, y1 = x0 + column * cell, y0 + row * cell
                fill = "#f8fafc"
                if state in self.environment.obstacles: fill = "#94a3b8"
                elif state == self.environment.start: fill = "#86efac"
                elif state == self.environment.goal: fill = "#fde047"
                elif state in path_states: fill = "#dbeafe"
                canvas.create_rectangle(x1, y1, x1 + cell, y1 + cell, fill=fill, outline="#64748b")
                canvas.create_text(x1 + 5, y1 + 5, text=f"{row},{column}", anchor="nw", fill="#64748b", font=("TkDefaultFont", 8))
                if state not in self.environment.obstacles and state != self.environment.goal:
                    best = self.planner.get_best_actions(state)
                    lines = []
                    if "V(s)" in display:
                        lines.append(
                            "V=?"
                            if self.planner.iteration == 0
                            else f"V={self.planner.get_value(state):.2f}"
                        )
                    if "beste Actions" in display or self.policy_visible_var.get(): lines.append("".join(ACTION_ARROWS[a] for a in best) if best else "?")
                    if display != "Keine" or self.policy_visible_var.get():
                        canvas.create_text(x1 + cell / 2, y1 + cell / 2, text="\n".join(lines), fill="#1e3a8a", font=("TkDefaultFont", max(8, int(cell / 8)), "bold"))
        for item in self.current_path:
            row, column = item.state
            nx, ny = x0 + (column + .5) * cell, y0 + (row + .5) * cell
            canvas.create_text(nx, ny + cell * .28, text=ACTION_ARROWS[item.action], fill="#2563eb", font=("TkDefaultFont", max(10, int(cell / 6)), "bold"))
        row, column = self.environment.current_state
        cx, cy = x0 + (column + .5) * cell, y0 + (row + .5) * cell
        radius = cell * .18
        canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius, fill="#2563eb", outline="white", width=2)
        canvas.create_text(cx, cy, text="A", fill="white", font=("TkDefaultFont", max(8, int(cell / 7)), "bold"))

    def _refresh_all(self) -> None:
        planner = self.planner
        last_delta = planner.history[-1].delta if planner.history else 0.0
        rollouts = self.rollout_results[planner.name]
        successes = sum(result.success for result in rollouts)
        average = sum(result.total_reward for result in rollouts) / len(rollouts) if rollouts else 0.0
        last = rollouts[-1] if rollouts else None
        self.summary_var.set(
            f"Methode: {planner.name}\nIterationen: {planner.iteration}\n"
            f"Konvergiert: {'Ja' if planner.converged else 'Nein'}\nDelta: {last_delta:.6g}\n"
            f"V(Start): {planner.get_value(self.environment.start):.4f}\n\n"
            f"Rollouts: {len(rollouts)}\nErfolge: {successes}\nØ Reward: {average:.2f}\n"
            f"Epsilon greedy: {self.epsilon_greedy_var.get()}\nEpsilon decay aktuell: {self.epsilon_decay_current:.4f}\n"
            f"Letzter Lauf: {last.steps if last else 0} Schritte / {last.total_reward if last else 0:.1f} Reward"
        )
        if not self.planning:
            disabled = "disabled" if planner.converged else "normal"
            self.sweep_button.configure(state=disabled); self.run_planner_button.configure(state=disabled)
        self.draw_grid(); self._draw_plots()

    def _draw_plots(self) -> None:
        ax = self.axes["Reward"]; ax.clear(); ax.set_title("Agenten-Reward je abgeschlossenem Rollout")
        for method, results in self.rollout_results.items():
            if results:
                x_values = list(range(1, len(results) + 1))
                rewards = [result.total_reward for result in results]
                running_average = [sum(rewards[:index]) / index for index in x_values]
                ax.plot(x_values, rewards, color=COLORS[method], linewidth=1, alpha=.35)
                ax.plot(x_values, running_average, color=COLORS[method], linewidth=3, label=f"{method} – Durchschnitt")
        ax.set_xlabel("Rollout"); ax.set_ylabel("Reward"); ax.grid(alpha=.25)
        if any(self.rollout_results.values()): ax.legend()
        self.plot_canvases["Reward"].draw_idle()
        ax = self.axes["Konvergenz"]; ax.clear(); ax.set_title("Konvergenz der Bellman-Sweeps")
        for method, planner in self.planners.items():
            if planner.history: ax.semilogy([r.iteration for r in planner.history], [max(r.delta, 1e-12) for r in planner.history], color=COLORS[method], label=method)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Delta (log)"); ax.grid(alpha=.25)
        if any(p.history for p in self.planners.values()): ax.legend()
        self.plot_canvases["Konvergenz"].draw_idle()
        ax = self.axes["Methodenvergleich"]; ax.clear(); ax.set_title("Methodenvergleich: Ø Rollout-Reward")
        if self.comparison_results:
            names = [r.method for r in self.comparison_results]
            means = [sum(x.total_reward for x in r.rollout_results)/len(r.rollout_results) if r.rollout_results else 0 for r in self.comparison_results]
            ax.bar(names, means, color=[COLORS[name] for name in names])
            labels = []
            for result in self.comparison_results:
                success = sum(x.success for x in result.rollout_results)
                labels.append(f"{result.method}: {len(result.sweep_results)} Sweeps, V(Start)={result.final_start_value:.3f}, {result.runtime_ms:.1f} ms, Erfolg {success}/{len(result.rollout_results)}")
            ax.set_xlabel(" | ".join(labels))
        ax.set_ylabel("Ø Reward"); ax.grid(axis="y", alpha=.25)
        self.plot_canvases["Methodenvergleich"].draw_idle()

    def _open_table(self, title: str, fields: Sequence[str], rows: List[Dict[str, object]], prefix: str) -> None:
        dialog = tk.Toplevel(self.root); dialog.title(title); dialog.geometry("1000x520")
        dialog.transient(self.root); dialog.grab_set()
        frame = ttk.Frame(dialog, padding=8); frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=fields, show="headings")
        for field in fields: tree.heading(field, text=field); tree.column(field, width=95, anchor="center")
        for row in rows: tree.insert("", "end", values=[row[field] for field in fields])
        ybar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview); xbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.grid(row=0, column=0, sticky="nsew"); ybar.grid(row=0, column=1, sticky="ns"); xbar.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1); frame.rowconfigure(0, weight=1)
        def export() -> None:
            target = filedialog.asksaveasfilename(parent=dialog, defaultextension=".csv", initialfile=f"{prefix}_{self.planner.name.lower().replace(' ', '_')}_{self.planner.iteration}.csv", filetypes=(("CSV-Datei", "*.csv"),))
            if not target: return
            with Path(target).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            messagebox.showinfo("Export abgeschlossen", f"CSV gespeichert:\n{target}", parent=dialog)
        buttons = ttk.Frame(dialog, padding=8); buttons.pack(fill="x")
        ttk.Button(buttons, text="Als CSV exportieren", command=export).pack(side="left")
        ttk.Button(buttons, text="Schließen", command=dialog.destroy).pack(side="right")
        dialog.wait_window()

    def open_value_dialog(self) -> None:
        self._open_table("Value-Table", VALUE_CSV_FIELDS, value_table_rows(self.planner), "values")

    def open_q_dialog(self) -> None:
        self._open_table("Q-Table", Q_CSV_FIELDS, q_table_rows(self.planner), "q_values")

    def open_instructions_dialog(self) -> None:
        """Show a concise German guide without changing the experiment."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Bedienungsanleitung")
        dialog.geometry("760x650")
        dialog.minsize(620, 480)
        dialog.transient(self.root)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="So verwendest du das modellbasierte Gridworld Lab",
            font=("TkDefaultFont", 15, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        instructions = """1. Grid konfigurieren

Rows und Columns bestimmen die Größe. Start, Goal und Hindernisse werden als Row und Column angegeben. Beispiel: (2,0). Mit „Grid anwenden und zurücksetzen“ werden alle Eingaben geprüft und das Experiment neu gestartet.

2. Planungsmethode auswählen

„Value Iteration“ berechnet V(s) direkt. „Q-Value Iteration“ berechnet Q(s,a) direkt und leitet daraus V(s) ab. Beide Methoden kennen das vollständige Environment-Modell.

3. Werte berechnen

Alle Value- und Q-Werte starten bei 0. Das ist der initiale Zustand und noch kein berechnetes Ergebnis.

• „Planner single sweep“ führt genau einen vollständigen Bellman-Sweep aus.
• „Planner bis Konvergenz“ berechnet weitere Sweeps, bis Tolerance erreicht oder das Iterationslimit ausgeschöpft ist.
• „Cancel“ beendet die Planung nach dem aktuellen vollständigen Sweep.

Agenten- und manuelle Bewegungen berechnen keine Values. Deshalb muss vor einem Agenten-Rollout zuerst der Planner ausgeführt werden.

4. Grid lesen

Start ist grün, Goal gelb, Hindernisse grau und der Agent blau. Unter „Zellanzeige“ können V(s), optimale Actions oder beide angezeigt werden. „Policy anzeigen“ blendet die optimalen Richtungspfeile ein. Mehrere Pfeile bedeuten, dass mehrere Actions gleich gut sind.

5. Agenten-Rollouts

• „Agent single step“ führt genau einen Policy-Schritt aus.
• „Agent run n loops“ führt höchstens die unter Agent-Loops eingestellte Anzahl Schritte aus und kann anschließend fortgesetzt werden.
• „Agent run episode“ läuft bis zum Goal oder Max Steps.
• „Greedy Policy ausführen“ verwendet immer die beste bekannte Action ohne Exploration.

Epsilon beeinflusst ausschließlich die sichtbaren Rollouts, niemals die Bellman-Berechnung. Epsilon (greedy) gilt für Einzelschritte. Der Decay-Wert gilt für automatische Rollouts und wird erst nach einem vollständig beendeten Rollout reduziert.

6. Manuelle Demo

Up, Down, Left und Right bewegen den Agenten manuell. Dieser Modus verändert weder Planner-Werte noch Rollout-Statistik oder Epsilon.

7. Ergebnisse untersuchen

Die Summary zeigt Iterationen, Konvergenz, Delta, V(Start), Rollout-Erfolge und Rewards. Die Tabs zeigen Reward-Verlauf, Planner-Konvergenz und den isolierten Methodenvergleich. Value-Table und Q-Table öffnen die vollständigen Tabellen und erlauben einen CSV-Export.

Empfohlener Ablauf

Grid anwenden → Methode wählen → Planner bis Konvergenz → Values und Policy ansehen → Greedy Policy ausführen → Methodenvergleich starten.
"""
        text_box = scrolledtext.ScrolledText(
            frame,
            wrap="word",
            font=("TkDefaultFont", 11),
            padx=10,
            pady=10,
            background="#f8fafc",
            foreground="#111827",
            relief="solid",
            borderwidth=1,
        )
        text_box.pack(fill="both", expand=True)
        text_box.insert("1.0", instructions)
        text_box.configure(state="disabled")

        ttk.Button(frame, text="Schließen", command=dialog.destroy).pack(
            anchor="e", pady=(10, 0)
        )
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.wait_window()
