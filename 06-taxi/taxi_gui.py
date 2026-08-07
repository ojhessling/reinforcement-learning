"""Tkinter GUI for the Taxi learning workbench."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from taxi_logic import (
    ACTION_NAMES, POLICY_CLASSES, EvaluationRunner, TabularAgent,
    TaxiEnvironment, TrainingRunner, create_policy,
)

PASSENGER_NAMES = {0: "Rot", 1: "Grün", 2: "Gelb", 3: "Blau", 4: "Im Taxi"}
DESTINATION_NAMES = {0: "Rot", 1: "Grün", 2: "Gelb", 3: "Blau"}


class TaxiGUI:
    def __init__(self, root: tk.Tk, environment: Optional[TaxiEnvironment] = None) -> None:
        self.root = root
        self.environment = environment or TaxiEnvironment()
        self.agent = TabularAgent(self.environment, create_policy("Q-Learning"))
        self.training_runner = TrainingRunner(self.agent)
        self.evaluation_runner = EvaluationRunner(self.agent)
        self.training_running = False
        self.animation_running = False
        self.stop_requested = False
        self._create_variables()
        self._configure_window()
        self._build_layout()
        self._settings_signature = self._input_signature()
        self._refresh_all()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _create_variables(self) -> None:
        self.method_var = tk.StringVar(value="Q-Learning")
        self.exploration_var = tk.StringVar(value="Epsilon konstant")
        self.rainy_var = tk.BooleanVar(value=False)
        self.fickle_var = tk.BooleanVar(value=False)
        self.action_mask_var = tk.BooleanVar(value=False)
        self.episodes_var = tk.StringVar(value="5000")
        self.max_steps_var = tk.StringVar(value="200")
        self.alpha_var = tk.StringVar(value="0.1")
        self.gamma_var = tk.StringVar(value="0.99")
        self.epsilon_var = tk.StringVar(value="0.1")
        self.epsilon_min_var = tk.StringVar(value="0.01")
        self.epsilon_decay_var = tk.StringVar(value="0.995")
        self.seed_var = tk.StringVar(value="42")
        self.animation_enabled_var = tk.BooleanVar(value=True)
        self.animation_delay_var = tk.StringVar(value="10")
        self.state_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Bereit – Einstellungen wählen und Training starten.")
        self.progress_var = tk.DoubleVar(value=0)

    def _configure_window(self) -> None:
        self.root.title("Taxi RL Workbench")
        width = min(1450, max(1100, self.root.winfo_screenwidth() - 70))
        height = min(930, max(720, self.root.winfo_screenheight() - 90))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1050, 680)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure("Accent.TButton", foreground="white", background="#2563eb", padding=7)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="Taxi RL Workbench", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Bedienungsanleitung", command=self.open_instructions).pack(side="right")
        ttk.Label(outer, text="Fahrgast abholen und sicher zum Ziel bringen").pack(anchor="w", pady=(0, 8))
        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, width=335)
        workspace = ttk.Frame(body)
        body.add(controls, weight=0)
        body.add(workspace, weight=1)
        self._build_controls(controls)
        self._build_workspace(workspace)

    def _entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 6))
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        return entry

    def _build_controls(self, parent: ttk.Frame) -> None:
        settings = ttk.LabelFrame(parent, text="Environment und Training", padding=9)
        settings.pack(fill="x", pady=(0, 8))
        ttk.Label(settings, text="Methode").grid(row=0, column=0, sticky="w")
        self.method_box = ttk.Combobox(settings, textvariable=self.method_var, values=tuple(POLICY_CLASSES), state="readonly")
        self.method_box.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(settings, text="Exploration").grid(row=1, column=0, sticky="w")
        self.exploration_box = ttk.Combobox(
            settings, textvariable=self.exploration_var,
            values=("Epsilon konstant", "Epsilon-Decay"), state="readonly",
        )
        self.exploration_box.grid(row=1, column=1, sticky="ew", pady=2)
        self.exploration_box.bind("<<ComboboxSelected>>", lambda _event: self._toggle_decay_fields())
        self.rainy_check = ttk.Checkbutton(settings, text="Rainy", variable=self.rainy_var)
        self.rainy_check.grid(row=2, column=0, columnspan=2, sticky="w")
        self.fickle_check = ttk.Checkbutton(settings, text="Fickle Passenger", variable=self.fickle_var)
        self.fickle_check.grid(row=3, column=0, columnspan=2, sticky="w")
        self.mask_check = ttk.Checkbutton(settings, text="Action Mask verwenden", variable=self.action_mask_var)
        self.mask_check.grid(row=4, column=0, columnspan=2, sticky="w")
        self._entry(settings, 5, "Episoden", self.episodes_var)
        self._entry(settings, 6, "Max. Schritte", self.max_steps_var)
        self._entry(settings, 7, "Alpha", self.alpha_var)
        self._entry(settings, 8, "Gamma", self.gamma_var)
        self._entry(settings, 9, "Epsilon", self.epsilon_var)
        self.epsilon_min_entry = self._entry(settings, 10, "Epsilon Minimum", self.epsilon_min_var)
        self.epsilon_decay_entry = self._entry(settings, 11, "Epsilon Decay", self.epsilon_decay_var)
        self._entry(settings, 12, "Random Seed", self.seed_var)
        self.animation_check = ttk.Checkbutton(
            settings, text="Animation anzeigen", variable=self.animation_enabled_var,
            command=self._toggle_animation_field,
        )
        self.animation_check.grid(row=13, column=0, columnspan=2, sticky="w")
        self.animation_delay_entry = self._entry(settings, 14, "Animationsintervall (ms)", self.animation_delay_var)
        settings.columnconfigure(1, weight=1)
        self._toggle_decay_fields()

        actions = ttk.LabelFrame(parent, text="Steuerung", padding=9)
        actions.pack(fill="x", pady=(0, 8))
        self.step_button = ttk.Button(actions, text="Einzelschritt trainieren", command=self.single_step)
        self.episode_button = ttk.Button(actions, text="Eine Episode trainieren", command=self.train_episode)
        self.train_button = ttk.Button(actions, text="N Episoden trainieren", style="Accent.TButton", command=self.start_training)
        self.stop_button = ttk.Button(actions, text="Stoppen", command=self.stop, state="disabled")
        self.evaluate_button = ttk.Button(actions, text="Gelernte Policy ausführen", command=self.evaluate)
        self.reset_button = ttk.Button(actions, text="Training zurücksetzen", command=self.reset_training)
        self.q_button = ttk.Button(actions, text="Q-Tabelle öffnen", command=self.open_q_table)
        for button in (
            self.step_button, self.episode_button, self.train_button, self.stop_button,
            self.evaluate_button, self.reset_button, self.q_button,
        ):
            button.pack(fill="x", pady=2)
        ttk.Progressbar(actions, variable=self.progress_var, maximum=100).pack(fill="x", pady=(6, 0))
        summary = ttk.LabelFrame(parent, text="Summary", padding=9)
        summary.pack(fill="x")
        ttk.Label(summary, textvariable=self.summary_var, justify="left", wraplength=295).pack(anchor="w")
        ttk.Label(summary, textvariable=self.status_var, justify="left", wraplength=295, foreground="#1d4ed8").pack(anchor="w", pady=(8, 0))

    def _build_workspace(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=2)
        environment_frame = ttk.LabelFrame(parent, text="Offizielles Gymnasium-Spielfeld", padding=10)
        environment_frame.grid(row=0, column=0, sticky="nsew")
        self.environment_image = ttk.Label(environment_frame, anchor="center")
        self.environment_image.pack(fill="both", expand=True)
        ttk.Label(environment_frame, textvariable=self.state_var, justify="left", font=("TkFixedFont", 10)).pack(pady=(5, 0))
        figure = Figure(figsize=(8, 3), dpi=100)
        self.return_axes = figure.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(figure, master=parent)
        self.plot_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(7, 0))

    def _read_parameters(self) -> Dict[str, object]:
        try:
            episodes = int(self.episodes_var.get())
            max_steps = int(self.max_steps_var.get())
            seed_text = self.seed_var.get().strip()
            seed = int(seed_text) if seed_text else None
            alpha = float(self.alpha_var.get())
            gamma = float(self.gamma_var.get())
            epsilon = float(self.epsilon_var.get())
            epsilon_min = float(self.epsilon_min_var.get())
            epsilon_decay = float(self.epsilon_decay_var.get())
        except ValueError as error:
            raise ValueError("Numerische Felder enthalten einen ungültigen Wert.") from error
        if episodes <= 0:
            raise ValueError("Episoden muss eine positive Ganzzahl sein.")
        policy_parameters = {
            "alpha": alpha, "gamma": gamma, "epsilon": epsilon,
            "epsilon_min": epsilon_min, "epsilon_decay": epsilon_decay,
            "exploration": self.exploration_var.get(),
            "use_action_mask": self.action_mask_var.get(), "seed": seed,
        }
        create_policy(self.method_var.get(), **policy_parameters)
        test_environment = TaxiEnvironment(
            self.rainy_var.get(), self.fickle_var.get(), max_steps, seed
        )
        test_environment.close()
        return {"episodes": episodes, "max_steps": max_steps, "seed": seed, "policy": policy_parameters}

    def _input_signature(self) -> tuple[object, ...]:
        return (
            self.method_var.get(), self.exploration_var.get(), self.rainy_var.get(),
            self.fickle_var.get(), self.action_mask_var.get(), self.max_steps_var.get(),
            self.alpha_var.get(), self.gamma_var.get(), self.epsilon_var.get(),
            self.epsilon_min_var.get(), self.epsilon_decay_var.get(), self.seed_var.get(),
        )

    def _sync_settings(self) -> bool:
        return True if self._input_signature() == self._settings_signature else self.apply_settings()

    def apply_settings(self) -> bool:
        if self.training_running or self.animation_running:
            return False
        try:
            parameters = self._read_parameters()
            environment = TaxiEnvironment(
                self.rainy_var.get(), self.fickle_var.get(),
                int(parameters["max_steps"]), parameters["seed"],
            )
            environment.render_rgb()
            policy = create_policy(self.method_var.get(), **parameters["policy"])
        except (TypeError, ValueError) as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root)
            return False
        self.environment.close()
        self.environment = environment
        self.agent = TabularAgent(environment, policy)
        self.training_runner = TrainingRunner(self.agent)
        self.evaluation_runner = EvaluationRunner(self.agent)
        self._settings_signature = self._input_signature()
        self.progress_var.set(0)
        self.status_var.set("Einstellungen übernommen; Lernzustand zurückgesetzt.")
        self._refresh_all()
        return True

    def _toggle_decay_fields(self) -> None:
        state = "normal" if self.exploration_var.get() == "Epsilon-Decay" else "disabled"
        self.epsilon_min_entry.configure(state=state)
        self.epsilon_decay_entry.configure(state=state)

    def _toggle_animation_field(self) -> None:
        self.animation_delay_entry.configure(state="normal" if self.animation_enabled_var.get() else "disabled")

    def _animation_delay(self) -> int:
        try:
            delay = int(self.animation_delay_var.get())
        except ValueError as error:
            raise ValueError("Animationsintervall muss eine ganze Zahl sein.") from error
        if not 1 <= delay <= 5000:
            raise ValueError("Animationsintervall muss zwischen 1 und 5000 ms liegen.")
        return delay

    def single_step(self) -> None:
        if self.training_running or self.animation_running or not self._sync_settings():
            return
        transition = self.agent.step(training=True)
        self.status_var.set(f"Schritt {transition.step}: {ACTION_NAMES[transition.action]}, Reward {transition.reward:.0f}.")
        self._refresh_all()

    def train_episode(self) -> None:
        self._start_visible_episode(True)

    def evaluate(self) -> None:
        if self.agent.episode_count == 0:
            messagebox.showinfo("Noch kein Training", "Trainiere zuerst mindestens eine Episode.", parent=self.root)
            return
        self._start_visible_episode(False)

    def _start_visible_episode(self, training: bool) -> None:
        if self.training_running or self.animation_running or not self._sync_settings():
            return
        if not self.animation_enabled_var.get():
            result = self.training_runner.run(1)[0] if training else self.evaluation_runner.run(1)[0]
            self._show_result(result, "Training" if training else "Evaluation")
            self._refresh_all()
            return
        try:
            self._animation_delay_ms = self._animation_delay()
        except ValueError as error:
            messagebox.showerror("Ungültige Animation", str(error), parent=self.root)
            return
        self.animation_running = True
        self.stop_requested = False
        self._animation_training = training
        self.agent.start_episode(training)
        self._set_busy(True)
        self._animation_tick()

    def _animation_tick(self) -> None:
        if self.stop_requested:
            self.agent.episode_active = False
        if not self.agent.episode_active:
            self.animation_running = False
            self._set_busy(False)
            if self.stop_requested:
                self.status_var.set("Animation gestoppt.")
            elif self.agent.latest_result:
                self._show_result(self.agent.latest_result, "Training" if self._animation_training else "Evaluation")
            self.stop_requested = False
            self._refresh_all()
            return
        self.agent.step(self._animation_training)
        self._refresh_environment()
        self.root.after(self._animation_delay_ms, self._animation_tick)

    def _show_result(self, result, kind: str) -> None:
        outcome = "Fahrgast abgeliefert" if result.success else "Max. Schritte erreicht"
        self.status_var.set(
            f"{kind}: {outcome}, Return {result.total_reward:.0f}, "
            f"{result.illegal_actions} illegale Actions."
        )

    def start_training(self) -> None:
        if self.training_running or self.animation_running or not self._sync_settings():
            return
        try:
            count = int(self.episodes_var.get())
            if count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ungültige Eingabe", "Episoden muss positiv sein.", parent=self.root)
            return
        self.training_running = True
        self.stop_requested = False
        self._training_target = count
        self._training_remaining = count
        self.progress_var.set(0)
        self._set_busy(True)
        self.root.after(1, self._training_chunk)

    def _training_chunk(self) -> None:
        if self.stop_requested or self._training_remaining <= 0:
            self.training_running = False
            self._set_busy(False)
            self.status_var.set("Training gestoppt." if self.stop_requested else "Training abgeschlossen.")
            self.stop_requested = False
            self._refresh_all()
            return
        count = min(20, self._training_remaining)
        self.training_runner.run(count)
        self._training_remaining -= count
        completed = self._training_target - self._training_remaining
        self.progress_var.set(100 * completed / self._training_target)
        self.status_var.set(f"Training: {completed} von {self._training_target} Episoden")
        if completed % 100 == 0 or self._training_remaining == 0:
            self._refresh_all()
        self.root.after(1, self._training_chunk)

    def stop(self) -> None:
        self.stop_requested = True

    def reset_training(self) -> None:
        if self.training_running or self.animation_running:
            return
        self.agent.reset_training()
        self.progress_var.set(0)
        self.status_var.set("Training zurückgesetzt.")
        self._refresh_all()

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for button in (self.step_button, self.episode_button, self.train_button, self.evaluate_button, self.reset_button, self.q_button):
            button.configure(state=state)
        for checkbox in (self.rainy_check, self.fickle_check, self.mask_check, self.animation_check):
            checkbox.configure(state=state)
        self.method_box.configure(state="disabled" if busy else "readonly")
        self.exploration_box.configure(state="disabled" if busy else "readonly")
        self.stop_button.configure(state="normal" if busy else "disabled")
        if not busy:
            self._toggle_decay_fields()
            self._toggle_animation_field()

    def _refresh_all(self) -> None:
        success_count = sum(self.agent.successes)
        recent = self.agent.successes[-100:]
        success_rate = sum(recent) / len(recent) if recent else 0
        illegal = sum(self.agent.illegal_actions)
        average = sum(self.agent.returns[-100:]) / len(self.agent.returns[-100:]) if self.agent.returns else 0
        self.summary_var.set(
            f"Methode: {self.agent.policy.name}\nEpisoden: {self.agent.episode_count}\n"
            f"Epsilon: {self.agent.policy.epsilon:.4f}\nErfolge: {success_count}\n"
            f"Erfolgsrate letzte 100: {success_rate:.1%}\n"
            f"Ø Return letzte 100: {average:.2f}\nIllegale Actions: {illegal}"
        )
        self._refresh_environment()
        self._draw_plot()

    def _refresh_environment(self) -> None:
        frame = self.environment.render_rgb()
        image = Image.fromarray(frame)
        image.thumbnail((850, 420), Image.Resampling.LANCZOS)
        self._environment_photo = ImageTk.PhotoImage(image)
        self.environment_image.configure(image=self._environment_photo)
        row, column, passenger, destination = self.environment.decode_observation(self.environment.observation)
        transition = self.agent.current_transitions[-1] if self.agent.current_transitions else None
        action = ACTION_NAMES[transition.action] if transition else "—"
        reward = f"{transition.reward:.0f}" if transition else "—"
        self.state_var.set(
            f"Taxi: ({row}, {column})   Fahrgast: {PASSENGER_NAMES[passenger]}   "
            f"Ziel: {DESTINATION_NAMES[destination]}   Action: {action}   Reward: {reward}"
        )

    def _draw_plot(self) -> None:
        axes = self.return_axes
        axes.clear()
        axes.set_title("Episode-Returns")
        axes.set_xlabel("Trainingsepisode")
        axes.set_ylabel("Return")
        axes.grid(alpha=0.25)
        if self.agent.returns:
            x_values = list(range(1, self.agent.episode_count + 1))
            axes.plot(x_values, self.agent.returns, color="#93c5fd", alpha=0.35, linewidth=1, label="Episode")
            axes.plot(x_values, self.agent.moving_average_returns(), color="#2563eb", linewidth=2.5, label="Ø letzte 100")
            axes.legend()
        else:
            axes.text(0.5, 0.5, "Noch keine Trainingsepisoden", ha="center", va="center", transform=axes.transAxes)
        self.plot_canvas.draw_idle()

    def open_q_table(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Q-Tabelle – {self.agent.policy.name}")
        dialog.geometry("1000x650")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill="both", expand=True)
        columns = ("state", "south", "north", "east", "west", "pickup", "dropoff", "best", "visits")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column.capitalize())
            tree.column(column, width=95, anchor="center")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for observation in range(500):
            visits = int(self.agent.policy.visit_counts[observation])
            values = self.agent.policy.q_values[observation]
            shown = [f"{value:.3f}" if visits else "—" for value in values]
            best = ", ".join(ACTION_NAMES[a] for a in self.agent.policy.best_actions(observation)) if visits else "?"
            tree.insert("", "end", values=(observation, *shown, best, visits))
        ttk.Button(dialog, text="Schließen", command=dialog.destroy).pack(pady=8)

    def open_instructions(self) -> None:
        messagebox.showinfo(
            "Bedienungsanleitung",
            "1. Methode, Exploration und Checkboxen wählen.\n"
            "2. N Episoden trainieren.\n"
            "3. Return und Q-Tabelle untersuchen.\n"
            "4. Gelernte Policy ohne Exploration ausführen.\n\n"
            "Jeder Schritt kostet -1, illegales Pickup/Dropoff -10 und eine "
            "erfolgreiche Ablieferung bringt +20. Die Action Mask verhindert "
            "ungültige Actions, wenn sie aktiviert ist.",
            parent=self.root,
        )

    def _close(self) -> None:
        self.stop_requested = True
        self.environment.close()
        self.root.destroy()
