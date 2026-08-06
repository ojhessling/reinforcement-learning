"""Tkinter GUI for the minimal CliffWalking learning workbench."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from cliff_walking_logic import (
    ACTION_ARROWS, ACTION_NAMES, POLICY_CLASSES, CliffWalkingEnvironment,
    TabularAgent, create_policy,
)

METHOD_COLORS = {
    "Q-Learning": "#2563eb",
    "SARSA": "#ea580c",
    "Expected SARSA": "#16a34a",
}


class CliffWalkingGUI:
    def __init__(
        self,
        root: tk.Tk,
        environment: Optional[CliffWalkingEnvironment] = None,
    ) -> None:
        self.root = root
        self.environment = environment or CliffWalkingEnvironment()
        self.agent = TabularAgent(self.environment, create_policy("SARSA"))
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
        self.method_var = tk.StringVar(value="SARSA")
        self.episodes_var = tk.StringVar(value="500")
        self.max_steps_var = tk.StringVar(value="500")
        self.alpha_var = tk.StringVar(value="0.5")
        self.gamma_var = tk.StringVar(value="0.99")
        self.epsilon_start_var = tk.StringVar(value="1.0")
        self.epsilon_min_var = tk.StringVar(value="0.05")
        self.epsilon_decay_var = tk.StringVar(value="0.995")
        self.seed_var = tk.StringVar(value="42")
        self.animation_enabled_var = tk.BooleanVar(value=True)
        self.animation_delay_var = tk.StringVar(value="10")
        self.status_var = tk.StringVar(value="Bereit – wähle eine Methode und starte das Training.")
        self.state_var = tk.StringVar()
        self.summary_var = tk.StringVar()
        self.progress_var = tk.DoubleVar(value=0.0)

    def _configure_window(self) -> None:
        self.root.title("CliffWalking RL Workbench – Minimalversion")
        width = min(1450, max(1120, self.root.winfo_screenwidth() - 70))
        height = min(930, max(720, self.root.winfo_screenheight() - 90))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1050, 680)
        self.root.configure(bg="#eef2f7")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#eef2f7")
        style.configure("TLabel", background="#eef2f7", foreground="#111827")
        style.configure("TLabelframe", background="#f8fafc")
        style.configure("TLabelframe.Label", background="#eef2f7", font=("TkDefaultFont", 10, "bold"))
        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure("Accent.TButton", foreground="white", background="#2563eb", padding=7)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="CliffWalking RL Workbench", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Bedienungsanleitung", command=self.open_instructions).pack(side="right")
        ttk.Label(outer, text="Q-Learning, SARSA und Expected SARSA am gefährlichen Cliff vergleichen").pack(anchor="w", pady=(0, 8))

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, width=330)
        workspace = ttk.Frame(body)
        body.add(controls, weight=0)
        body.add(workspace, weight=1)
        self._build_controls(controls)
        self._build_workspace(workspace)

    def _entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 6))
        ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=1, sticky="ew", pady=2)

    def _build_controls(self, parent: ttk.Frame) -> None:
        settings = ttk.LabelFrame(parent, text="Training und Environment", padding=9)
        settings.pack(fill="x", pady=(0, 8))
        ttk.Label(settings, text="Methode").grid(row=0, column=0, sticky="w")
        self.method_box = ttk.Combobox(
            settings, textvariable=self.method_var, values=tuple(POLICY_CLASSES),
            state="readonly", width=18,
        )
        self.method_box.grid(row=0, column=1, sticky="ew", pady=2)
        self.method_box.bind("<<ComboboxSelected>>", lambda _event: self.apply_settings())
        self._entry(settings, 1, "Episoden", self.episodes_var)
        self._entry(settings, 2, "Max. Schritte", self.max_steps_var)
        self._entry(settings, 3, "Alpha", self.alpha_var)
        self._entry(settings, 4, "Gamma", self.gamma_var)
        self._entry(settings, 5, "Epsilon Start", self.epsilon_start_var)
        self._entry(settings, 6, "Epsilon Min", self.epsilon_min_var)
        self._entry(settings, 7, "Epsilon Decay", self.epsilon_decay_var)
        self._entry(settings, 8, "Random Seed", self.seed_var)
        self.animation_checkbutton = ttk.Checkbutton(
            settings, text="Animation anzeigen",
            variable=self.animation_enabled_var,
            command=self._toggle_animation_controls,
        )
        self.animation_checkbutton.grid(row=9, column=0, columnspan=2, sticky="w", pady=(4, 2))
        ttk.Label(settings, text="Animationsintervall (ms)").grid(row=10, column=0, sticky="w", pady=2, padx=(0, 6))
        self.animation_delay_entry = ttk.Entry(settings, textvariable=self.animation_delay_var, width=12)
        self.animation_delay_entry.grid(row=10, column=1, sticky="ew", pady=2)
        self.apply_button = ttk.Button(settings, text="Einstellungen anwenden und zurücksetzen", command=self.apply_settings)
        self.apply_button.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        settings.columnconfigure(1, weight=1)

        actions = ttk.LabelFrame(parent, text="Steuerung", padding=9)
        actions.pack(fill="x", pady=(0, 8))
        self.step_button = ttk.Button(actions, text="Einzelschritt trainieren", command=self.single_step)
        self.episode_button = ttk.Button(actions, text="Eine Episode animieren", command=self.animate_training_episode)
        self.train_button = ttk.Button(actions, text="N Episoden trainieren", style="Accent.TButton", command=self.start_training)
        self.stop_button = ttk.Button(actions, text="Training stoppen", command=self.stop, state="disabled")
        self.evaluate_button = ttk.Button(actions, text="Gelernte Policy ausführen", command=self.animate_evaluation)
        self.reset_button = ttk.Button(actions, text="Training zurücksetzen", command=self.reset_training)
        self.q_button = ttk.Button(actions, text="Q-Tabelle öffnen", command=self.open_q_table)
        for button in (self.step_button, self.episode_button, self.train_button, self.stop_button, self.evaluate_button, self.reset_button, self.q_button):
            button.pack(fill="x", pady=2)
        ttk.Progressbar(actions, variable=self.progress_var, maximum=100).pack(fill="x", pady=(6, 0))

        summary = ttk.LabelFrame(parent, text="Summary", padding=9)
        summary.pack(fill="x")
        ttk.Label(summary, textvariable=self.summary_var, justify="left", wraplength=290).pack(anchor="w")
        ttk.Label(summary, textvariable=self.status_var, justify="left", wraplength=290, foreground="#1d4ed8").pack(anchor="w", pady=(8, 0))

    def _build_workspace(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=2)
        state_frame = ttk.LabelFrame(parent, text="Offizielles Gymnasium-Spielfeld", padding=18)
        state_frame.grid(row=0, column=0, sticky="nsew")
        self.environment_image = ttk.Label(state_frame, anchor="center")
        self.environment_image.pack(fill="both", expand=True)
        ttk.Label(
            state_frame, textvariable=self.state_var, justify="left",
            font=("TkFixedFont", 11),
        ).pack(anchor="center", pady=(8, 0))
        figure = Figure(figsize=(8, 3), dpi=100)
        self.return_axes = figure.add_subplot(111)
        self.return_canvas = FigureCanvasTkAgg(figure, master=parent)
        self.return_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(7, 0))

    def _read_parameters(self) -> Dict[str, object]:
        episodes = int(self.episodes_var.get())
        max_steps = int(self.max_steps_var.get())
        seed_text = self.seed_var.get().strip()
        seed = int(seed_text) if seed_text else None
        if episodes <= 0:
            raise ValueError("Episoden muss eine positive Ganzzahl sein.")
        policy_parameters: Dict[str, object] = {
            "alpha": float(self.alpha_var.get()),
            "gamma": float(self.gamma_var.get()),
            "epsilon_start": float(self.epsilon_start_var.get()),
            "epsilon_min": float(self.epsilon_min_var.get()),
            "epsilon_decay": float(self.epsilon_decay_var.get()),
            "seed": seed,
        }
        # Constructors perform the detailed range validation atomically.
        create_policy(self.method_var.get(), **policy_parameters)
        CliffWalkingEnvironment(max_steps=max_steps, seed=seed).close()
        return {"episodes": episodes, "max_steps": max_steps, "seed": seed, "policy": policy_parameters}

    def _input_signature(self) -> tuple[object, ...]:
        """Values that invalidate the current learning experiment."""
        return (
            self.method_var.get(), self.max_steps_var.get(), self.alpha_var.get(),
            self.gamma_var.get(), self.epsilon_start_var.get(), self.epsilon_min_var.get(),
            self.epsilon_decay_var.get(), self.seed_var.get(),
        )

    def _sync_settings(self) -> bool:
        if self._input_signature() == self._settings_signature:
            return True
        return self.apply_settings()

    def apply_settings(self) -> bool:
        if self.training_running or self.animation_running:
            return False
        try:
            parameters = self._read_parameters()
            environment = CliffWalkingEnvironment(
                max_steps=int(parameters["max_steps"]), seed=parameters["seed"]
            )
            # Initialize Gymnasium's Pygame surface before the current
            # renderer is closed. This avoids reinitializing Pygame beside Tk
            # on macOS/Python 3.13.
            environment.render_rgb()
            policy = create_policy(self.method_var.get(), **parameters["policy"])
        except (TypeError, ValueError) as error:
            messagebox.showerror("Ungültige Einstellungen", str(error), parent=self.root)
            return False
        self.environment.close()
        self.environment = environment
        self.agent = TabularAgent(environment, policy)
        self._settings_signature = self._input_signature()
        self.progress_var.set(0)
        self.status_var.set("Einstellungen übernommen; Training wurde zurückgesetzt.")
        self._refresh_all()
        return True

    def reset_training(self) -> None:
        if self.training_running or self.animation_running:
            return
        self.agent.reset_training()
        self.progress_var.set(0)
        self.status_var.set("Training zurückgesetzt.")
        self._refresh_all()

    def single_step(self) -> None:
        if self.training_running or self.animation_running:
            return
        if not self._sync_settings():
            return
        transition = self.agent.step(training=True)
        self.status_var.set(
            "Cliff-Fall: -100 und zurück zum Start."
            if transition.fell_into_cliff else f"Schritt {transition.step}: Reward {transition.reward:.0f}"
        )
        self._refresh_all()

    def animate_training_episode(self) -> None:
        self._start_animation(training=True)

    def animate_evaluation(self) -> None:
        if self.agent.episode_count == 0:
            messagebox.showinfo("Noch kein Training", "Trainiere zuerst mindestens eine Episode.", parent=self.root)
            return
        self._start_animation(training=False)

    def _toggle_animation_controls(self) -> None:
        self.animation_delay_entry.configure(
            state="normal" if self.animation_enabled_var.get() else "disabled"
        )

    def _animation_delay(self) -> int:
        try:
            delay = int(self.animation_delay_var.get())
        except ValueError as error:
            raise ValueError("Animationsintervall muss eine ganze Zahl sein.") from error
        if not 1 <= delay <= 5_000:
            raise ValueError("Animationsintervall muss zwischen 1 und 5000 ms liegen.")
        return delay

    def _start_animation(self, training: bool) -> None:
        if self.training_running or self.animation_running:
            return
        if not self._sync_settings():
            return
        if not self.animation_enabled_var.get():
            result = self.agent.run_episode(training=training)
            kind = "Trainingsepisode" if training else "Greedy-Auswertung"
            reason = "Ziel erreicht" if result.success else "Max. Schritte erreicht"
            self.status_var.set(
                f"{kind} ohne Animation: {reason}, Return {result.total_reward:.0f}, "
                f"Cliff-Fälle {result.cliff_falls}."
            )
            self._refresh_all()
            return
        try:
            self._animation_delay_ms = self._animation_delay()
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellung", str(error), parent=self.root)
            return
        self.animation_running = True
        self.stop_requested = False
        self._animation_training = training
        self.agent.start_episode(training=training)
        self._set_busy(True)
        self._animation_tick()

    def _animation_tick(self) -> None:
        if self.stop_requested:
            self.agent.episode_active = False
        if not self.agent.episode_active:
            self.animation_running = False
            self._set_busy(False)
            result = self.agent.latest_result
            if self.stop_requested:
                self.status_var.set("Animation gestoppt.")
            elif result is not None:
                kind = "Trainingsepisode" if self._animation_training else "Greedy-Auswertung"
                reason = "Ziel erreicht" if result.success else "Max. Schritte erreicht"
                self.status_var.set(f"{kind}: {reason}, Return {result.total_reward:.0f}, Cliff-Fälle {result.cliff_falls}.")
            self.stop_requested = False
            self._refresh_all()
            return
        transition = self.agent.step(training=self._animation_training)
        if transition.fell_into_cliff:
            self.status_var.set("Cliff-Fall: Reward -100, Agent kehrt zum Start zurück.")
        self._refresh_state()
        self.root.after(self._animation_delay_ms, self._animation_tick)

    def start_training(self) -> None:
        if self.training_running or self.animation_running:
            return
        if not self._sync_settings():
            return
        try:
            count = int(self.episodes_var.get())
            if count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ungültige Eingabe", "Episoden muss eine positive Ganzzahl sein.", parent=self.root)
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
        chunk = min(10, self._training_remaining)
        for _ in range(chunk):
            self.agent.run_episode(training=True)
        self._training_remaining -= chunk
        completed = self._training_target - self._training_remaining
        self.progress_var.set(100 * completed / self._training_target)
        self.status_var.set(f"Training: {completed} von {self._training_target} Episoden")
        self._refresh_all()
        self.root.after(1, self._training_chunk)

    def stop(self) -> None:
        self.stop_requested = True

    def _set_busy(self, busy: bool) -> None:
        normal = "disabled" if busy else "normal"
        for button in (self.apply_button, self.step_button, self.episode_button, self.train_button, self.evaluate_button, self.reset_button, self.q_button):
            button.configure(state=normal)
        self.method_box.configure(state="disabled" if busy else "readonly")
        self.animation_checkbutton.configure(state="disabled" if busy else "normal")
        if busy:
            self.animation_delay_entry.configure(state="disabled")
        else:
            self._toggle_animation_controls()
        self.stop_button.configure(state="normal" if busy else "disabled")

    def _refresh_all(self) -> None:
        returns = self.agent.returns
        successes = sum(self.agent.successes)
        average = sum(returns[-20:]) / len(returns[-20:]) if returns else 0.0
        falls = sum(self.agent.cliff_falls)
        self.summary_var.set(
            f"Methode: {self.agent.policy.name}\n"
            f"Episoden: {self.agent.episode_count}\n"
            f"Epsilon: {self.agent.policy.epsilon:.4f}\n"
            f"Erfolge: {successes}/{self.agent.episode_count}\n"
            f"Ø Return letzte 20: {average:.2f}\n"
            f"Cliff-Fälle: {falls}\n"
            f"Letzter Return: {returns[-1] if returns else '—'}"
        )
        self._refresh_state()
        self.draw_plot()

    def _refresh_state(self) -> None:
        frame = self.environment.render_rgb()
        image = Image.fromarray(frame)
        image.thumbnail((900, 360), Image.Resampling.LANCZOS)
        self._environment_photo = ImageTk.PhotoImage(image)
        self.environment_image.configure(image=self._environment_photo)
        observation = self.environment.observation
        row, column = self.environment.observation_to_position(observation)
        lines = [
            f"Observation: {observation}",
            f"Position:    ({row}, {column})",
            f"Schritt:     {self.environment.step_count}",
            f"Return:      {self.environment.total_reward:.0f}",
        ]
        if self.agent.current_transitions:
            transition = self.agent.current_transitions[-1]
            lines.extend((
                f"Action:      {ACTION_NAMES[transition.action]} ({transition.action})",
                f"Reward:      {transition.reward:.0f}",
                f"Cliff-Fall:  {'ja' if transition.fell_into_cliff else 'nein'}",
            ))
        else:
            lines.extend(("Action:      —", "Reward:      —", "Cliff-Fall:  nein"))
        self.state_var.set("\n".join(lines))

    def draw_plot(self) -> None:
        axes = self.return_axes
        axes.clear()
        axes.set_title("Episode-Returns")
        axes.set_xlabel("Trainingsepisode")
        axes.set_ylabel("Return")
        axes.grid(alpha=.25)
        if self.agent.returns:
            x_values = list(range(1, len(self.agent.returns) + 1))
            color = METHOD_COLORS[self.agent.policy.name]
            axes.plot(x_values, self.agent.returns, color=color, alpha=.25, linewidth=1, label="Episode-Return")
            axes.plot(x_values, self.agent.moving_average_returns(), color=color, linewidth=2.5, label="Ø letzte 20")
            axes.legend()
        else:
            axes.text(.5, .5, "Noch keine Trainingsepisoden", ha="center", va="center", transform=axes.transAxes, color="#64748b")
        self.return_canvas.draw_idle()

    def open_q_table(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Q-Tabelle – {self.agent.policy.name}")
        dialog.geometry("900x620")
        dialog.transient(self.root)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=10)
        frame.pack(fill="both", expand=True)
        columns = ("state", "row", "column", "up", "right", "down", "left", "best", "visits")
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            tree.heading(column, text=column.capitalize())
            tree.column(column, width=90, anchor="center")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        for observation in range(48):
            row, column = divmod(observation, 12)
            visits = int(self.agent.policy.visit_counts[observation])
            values = self.agent.policy.q_values[observation]
            shown = [f"{value:.3f}" if visits else "—" for value in values]
            best = "".join(ACTION_ARROWS[action] for action in self.agent.policy.best_actions(observation)) if visits else "?"
            tree.insert("", "end", values=(observation, row, column, *shown, best, visits))
        ttk.Button(dialog, text="Schließen", command=dialog.destroy).pack(pady=8)
        dialog.wait_window()

    def open_instructions(self) -> None:
        messagebox.showinfo(
            "Bedienungsanleitung",
            "1. Methode und Parameter einstellen.\n"
            "2. N Episoden trainieren; Änderungen werden automatisch übernommen.\n"
            "3. Zustandsanzeige und Return-Kurve untersuchen.\n"
            "4. Gelernte Policy ohne Exploration ausführen.\n\n"
            "Ein normaler Schritt kostet -1. Ein Cliff-Fall kostet -100 und setzt den Agenten "
            "zum Start zurück. SARSA lernt häufig einen sichereren Weg; Q-Learning häufig den "
            "kürzeren Weg nahe am Cliff.",
            parent=self.root,
        )

    def _close(self) -> None:
        self.stop_requested = True
        self.environment.close()
        self.root.destroy()
