"""Tkinter interface for the interactive Multi-Armed Bandit laboratory."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Dict, List, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from bandit_logic import (
    BanditEnvironment,
    BaseAgent,
    StrategyComparisonResult,
    compare_strategies,
    create_agent,
)


STRATEGIES = (
    "Epsilon Greedy",
    "Epsilon Decay",
    "Thompson Sampling",
    "UCB",
    "Boltzmann/Softmax",
)

STRATEGY_HELP = {
    "Epsilon Greedy": "Probiert mit Wahrscheinlichkeit Epsilon zufällig aus und nutzt sonst die bisher beste Option.",
    "Epsilon Decay": "Exploriert am Anfang viel und reduziert das zufällige Ausprobieren nach jedem Schritt.",
    "Thompson Sampling": "Zieht für jeden Banditen eine plausible Erfolgsrate und lernt aus Erfolgen und Misserfolgen.",
    "UCB": "Bevorzugt gute Banditen, gibt aber unsicheren Optionen gezielt eine faire Chance.",
    "Boltzmann/Softmax": "Wählt gute Banditen häufiger; die Temperatur bestimmt, wie stark Unterschiede gewichtet werden.",
}

STRATEGY_COLORS = {
    "Epsilon Greedy": "#2563eb",
    "Epsilon Decay": "#ea580c",
    "Thompson Sampling": "#16a34a",
    "UCB": "#9333ea",
    "Boltzmann/Softmax": "#0891b2",
}


class BanditCard:
    """Visual slot-machine card with compact statistics."""

    def __init__(self, parent: ttk.Frame, index: int, color: str, command: object) -> None:
        self.index = index
        self.color = color
        self.frame = ttk.Frame(parent, style="Card.TFrame", padding=10)
        self.frame.columnconfigure(0, weight=1)
        ttk.Label(
            self.frame,
            text="Bandit {}".format(index + 1),
            style="CardTitle.TLabel",
        ).grid(row=0, column=0, sticky="ew")

        self.canvas = tk.Canvas(
            self.frame, width=190, height=105, bg="#ffffff", highlightthickness=0
        )
        self.canvas.grid(row=1, column=0, pady=4)
        self.body = self.canvas.create_rectangle(32, 8, 158, 98, fill=color, outline="#1f2937", width=2)
        self.canvas.create_rectangle(48, 25, 142, 64, fill="#111827", outline="#f8fafc")
        self.display = self.canvas.create_text(
            95, 44, text="Bereit", fill="#ffffff", font=("TkDefaultFont", 12, "bold")
        )
        self.canvas.create_line(158, 34, 178, 22, fill="#374151", width=5)
        self.canvas.create_oval(171, 13, 185, 27, fill="#ef4444", outline="#7f1d1d")
        self.canvas.create_rectangle(55, 73, 135, 91, fill="#f8fafc", outline="#374151")
        self.canvas.create_text(95, 82, text="MAB", fill="#111827", font=("TkDefaultFont", 9, "bold"))

        self.probability_var = tk.StringVar(value="Wahrscheinlichkeit: verborgen")
        ttk.Label(self.frame, textvariable=self.probability_var, style="CardMeta.TLabel").grid(
            row=2, column=0
        )
        self.stats_var = tk.StringVar()
        ttk.Label(
            self.frame,
            textvariable=self.stats_var,
            style="CardMeta.TLabel",
            justify="center",
        ).grid(row=3, column=0, sticky="ew", pady=4)
        self.button = ttk.Button(self.frame, text="Ziehen", command=command)
        self.button.grid(row=4, column=0, sticky="ew")

    def update(self, stats: Dict[str, object], estimate: float) -> None:
        last = "–" if stats["last_reward"] is None else str(stats["last_reward"])
        self.stats_var.set(
            "Pulls: {}   Letzter Reward: {}\nRewards: {}   Erfolg: {:.1%}   Schätzung: {:.3f}".format(
                stats["pulls"], last, stats["reward_sum"], stats["success_rate"], estimate
            )
        )
        self.canvas.itemconfigure(
            self.display,
            text="Bereit" if stats["last_reward"] is None else "Reward {}".format(last),
        )

    def show_probability(self, probability: float, visible: bool) -> None:
        self.probability_var.set(
            "Wahre Chance: {:.0%}".format(probability)
            if visible
            else "Wahrscheinlichkeit: verborgen"
        )

    def flash(self, reward: int, selected_by_agent: bool = False) -> None:
        flash_color = "#22c55e" if reward else "#ef4444"
        self.canvas.itemconfigure(self.body, fill=flash_color, width=4 if selected_by_agent else 2)
        self.canvas.itemconfigure(self.display, text="Reward {}".format(reward))
        self.canvas.after(350, lambda: self.canvas.itemconfigure(self.body, fill=self.color, width=2))


class BanditGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Multi-Armed Bandit Lab")
        width = min(1280, max(1000, self.root.winfo_screenwidth() - 80))
        height = min(900, max(700, self.root.winfo_screenheight() - 100))
        self.root.geometry("{}x{}".format(width, height))
        self.root.minsize(1000, 700)
        self._configure_style()

        self.environment: Optional[BanditEnvironment] = None
        self.agent: Optional[BaseAgent] = None
        self._remaining_steps = 0
        self._run_total = 0
        self._running = False
        self.comparison_results: List[StrategyComparisonResult] = []
        self._comparison_queue: queue.Queue = queue.Queue()
        self._comparison_running = False

        self._create_variables()
        self._build_layout()
        self._rebuild_parameter_fields()
        self.reset()

    def _configure_style(self) -> None:
        self.root.configure(background="#eef2f7")
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#eef2f7")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#eef2f7", foreground="#111827")
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#111827", font=("TkDefaultFont", 13, "bold"), anchor="center")
        style.configure("CardMeta.TLabel", background="#ffffff", foreground="#374151", anchor="center")
        style.configure("Title.TLabel", font=("TkDefaultFont", 22, "bold"), foreground="#111827")
        style.configure("Subtitle.TLabel", font=("TkDefaultFont", 11), foreground="#4b5563")
        style.configure("Section.TLabel", font=("TkDefaultFont", 11, "bold"))
        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", padding=7, foreground="#ffffff", background="#2563eb")
        style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("disabled", "#93c5fd")])
        style.configure("TEntry", fieldbackground="#ffffff", foreground="#111827")
        style.configure("TCombobox", fieldbackground="#ffffff", foreground="#111827")
        style.configure("Treeview", rowheight=25, fieldbackground="#ffffff", background="#ffffff", foreground="#111827")
        style.configure("Treeview.Heading", font=("TkDefaultFont", 9, "bold"))

    def _create_variables(self) -> None:
        self.strategy_var = tk.StringVar(value=STRATEGIES[0])
        self.strategy_help_var = tk.StringVar(value=STRATEGY_HELP[STRATEGIES[0]])
        self.loops_var = tk.StringVar(value="100")
        self.memory_var = tk.StringVar(value="0")
        self.seed_var = tk.StringVar(value="")
        self.epsilon_var = tk.StringVar(value="0.1")
        self.epsilon_start_var = tk.StringVar(value="0.999")
        self.epsilon_min_var = tk.StringVar(value="0.001")
        self.epsilon_decay_var = tk.StringVar(value="0.05")
        self.temperature_var = tk.StringVar(value="1.0")
        self.show_probabilities_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Bereit")
        self.total_var = tk.StringVar(value="Agent-Pulls: 0 | Reward: 0")

        self.comparison_selected = {strategy: tk.BooleanVar(value=True) for strategy in STRATEGIES}
        self.comparison_steps_var = tk.StringVar(value="500")
        self.comparison_repetitions_var = tk.StringVar(value="20")
        self.comparison_seed_var = tk.StringVar(value="42")
        self.show_ci_var = tk.BooleanVar(value=True)
        self.comparison_status_var = tk.StringVar(value="Wähle Strategien und starte den Vergleich.")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Multi-Armed Bandit Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="Entdecke, wie ein Agent zwischen Ausprobieren und dem Nutzen seiner Erfahrung entscheidet.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(0, 8))

        settings = ttk.Frame(outer)
        settings.pack(fill="x", pady=(0, 8))
        ttk.Label(settings, text="Strategie", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.strategy_box = ttk.Combobox(settings, textvariable=self.strategy_var, values=STRATEGIES, state="readonly", width=21)
        self.strategy_box.grid(row=1, column=0, sticky="w", padx=(0, 12))
        self.strategy_box.bind("<<ComboboxSelected>>", self._strategy_changed)
        ttk.Label(settings, textvariable=self.strategy_help_var, wraplength=480, justify="left").grid(row=0, column=1, rowspan=2, sticky="w", padx=(0, 16))

        common = ttk.Frame(settings)
        common.grid(row=0, column=2, rowspan=2, sticky="e")
        ttk.Label(common, text="Agent Loops").grid(row=0, column=0, sticky="w")
        ttk.Entry(common, textvariable=self.loops_var, width=8).grid(row=1, column=0, padx=(0, 8))
        ttk.Label(common, text="Random Seed").grid(row=0, column=1, sticky="w")
        ttk.Entry(common, textvariable=self.seed_var, width=9).grid(row=1, column=1, padx=(0, 8))
        self.parameter_frame = ttk.Frame(settings)
        self.parameter_frame.grid(row=0, column=3, rowspan=2, sticky="e")
        settings.columnconfigure(1, weight=1)

        cards_frame = ttk.Frame(outer)
        cards_frame.pack(fill="x", pady=4)
        colors = ("#60a5fa", "#f59e0b", "#8b5cf6")
        self.cards: List[BanditCard] = []
        for index, color in enumerate(colors):
            cards_frame.columnconfigure(index, weight=1, uniform="cards")
            card = BanditCard(cards_frame, index, color, lambda action=index: self.manual_pull(action))
            card.frame.grid(row=0, column=index, sticky="nsew", padx=5)
            self.cards.append(card)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=7)
        self.step_button = ttk.Button(actions, text="Einzelschritt", command=self.agent_step)
        self.step_button.pack(side="left")
        self.run_button = ttk.Button(actions, text="Agent starten", style="Accent.TButton", command=self.run_agent)
        self.run_button.pack(side="left", padx=7)
        self.reset_button = ttk.Button(actions, text="Experiment zurücksetzen", command=self.reset)
        self.reset_button.pack(side="left")
        ttk.Checkbutton(actions, text="Wahre Wahrscheinlichkeiten anzeigen", variable=self.show_probabilities_var, command=self._update_probabilities).pack(side="right")
        ttk.Label(actions, textvariable=self.status_var).pack(side="right", padx=16)

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.current_tab = ttk.Frame(self.notebook, padding=7)
        self.comparison_tab = ttk.Frame(self.notebook, padding=7)
        self.notebook.add(self.current_tab, text="Aktueller Lauf")
        self.notebook.add(self.comparison_tab, text="Strategien vergleichen")
        self._build_current_tab()
        self._build_comparison_tab()

    def _build_current_tab(self) -> None:
        ttk.Label(self.current_tab, textvariable=self.total_var, style="Section.TLabel").pack(anchor="w")
        self.current_figure = Figure(figsize=(8, 3.1), dpi=100)
        self.current_axes = self.current_figure.add_subplot(111)
        self.current_canvas = FigureCanvasTkAgg(self.current_figure, master=self.current_tab)
        self.current_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_comparison_tab(self) -> None:
        controls = ttk.Frame(self.comparison_tab)
        controls.pack(fill="x")
        choices = ttk.Frame(controls)
        choices.grid(row=0, column=0, rowspan=2, sticky="w")
        for index, strategy in enumerate(STRATEGIES):
            ttk.Checkbutton(choices, text=strategy, variable=self.comparison_selected[strategy]).grid(row=index // 3, column=index % 3, sticky="w", padx=(0, 8))

        fields = ttk.Frame(controls)
        fields.grid(row=0, column=1, rowspan=2, padx=12)
        for column, (label, variable) in enumerate((("Schritte", self.comparison_steps_var), ("Wiederholungen", self.comparison_repetitions_var), ("Seed", self.comparison_seed_var))):
            ttk.Label(fields, text=label).grid(row=0, column=column)
            ttk.Entry(fields, textvariable=variable, width=8).grid(row=1, column=column, padx=3)

        self.compare_button = ttk.Button(controls, text="Vergleich starten", style="Accent.TButton", command=self.start_comparison)
        self.compare_button.grid(row=0, column=2, padx=4)
        ttk.Button(controls, text="Zurücksetzen", command=self.reset_comparison).grid(row=1, column=2, padx=4)
        ttk.Checkbutton(controls, text="95-%-Konfidenzintervalle", variable=self.show_ci_var, command=self._draw_comparison).grid(row=0, column=3, padx=8)
        controls.columnconfigure(0, weight=1)

        content = ttk.Frame(self.comparison_tab)
        content.pack(fill="both", expand=True, pady=(5, 0))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)
        self.comparison_figure = Figure(figsize=(7, 3), dpi=100)
        self.comparison_axes = self.comparison_figure.add_subplot(111)
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_figure, master=content)
        self.comparison_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        table_frame = ttk.Frame(content)
        table_frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        columns = ("strategy", "reward", "rate", "best", "regret")
        self.comparison_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        headings = (("strategy", "Strategie", 145), ("reward", "End-Reward", 80), ("rate", "pro Pull", 65), ("best", "Bandit 3", 70), ("regret", "Regret", 65))
        for name, title, width in headings:
            self.comparison_table.heading(name, text=title)
            self.comparison_table.column(name, width=width, anchor="center")
        self.comparison_table.pack(fill="both", expand=True)
        ttk.Label(table_frame, textvariable=self.comparison_status_var, wraplength=430, justify="left").pack(fill="x", pady=(5, 0))
        self._draw_comparison()

    def _rebuild_parameter_fields(self) -> None:
        for widget in self.parameter_frame.winfo_children():
            widget.destroy()
        mapping = {
            "Epsilon Greedy": (("Epsilon", self.epsilon_var), ("Last N", self.memory_var)),
            "Epsilon Decay": (("Start", self.epsilon_start_var), ("Minimum", self.epsilon_min_var), ("Decay", self.epsilon_decay_var), ("Last N", self.memory_var)),
            "Thompson Sampling": (),
            "UCB": (),
            "Boltzmann/Softmax": (("Temperatur", self.temperature_var), ("Last N", self.memory_var)),
        }
        for column, (label, variable) in enumerate(mapping[self.strategy_var.get()]):
            ttk.Label(self.parameter_frame, text=label).grid(row=0, column=column, sticky="w")
            ttk.Entry(self.parameter_frame, textvariable=variable, width=8).grid(row=1, column=column, padx=(0, 5))

    def _settings(self) -> Dict[str, object]:
        try:
            loops = int(self.loops_var.get())
            memory = int(self.memory_var.get())
            seed_text = self.seed_var.get().strip()
            seed = int(seed_text) if seed_text else None
            values = {
                "loops": loops, "memory": memory, "seed": seed,
                "epsilon": float(self.epsilon_var.get()),
                "epsilon_start": float(self.epsilon_start_var.get()),
                "epsilon_min": float(self.epsilon_min_var.get()),
                "epsilon_decay": float(self.epsilon_decay_var.get()),
                "temperature": float(self.temperature_var.get()),
            }
        except ValueError as error:
            raise ValueError("Bitte nur gültige Zahlen eingeben.") from error
        if loops <= 0 or memory < 0:
            raise ValueError("Loops muss positiv und Last N darf nicht negativ sein.")
        if not 0 <= values["epsilon"] <= 1:
            raise ValueError("Epsilon muss zwischen 0 und 1 liegen.")
        if not 0 <= values["epsilon_min"] <= values["epsilon_start"] <= 1:
            raise ValueError("Es muss 0 ≤ Minimum ≤ Start ≤ 1 gelten.")
        if not 0 < values["epsilon_decay"] <= 1 or values["temperature"] <= 0:
            raise ValueError("Decay muss in (0, 1] und Temperatur größer als 0 sein.")
        return values

    def _new_models(self) -> None:
        settings = self._settings()
        self.environment = BanditEnvironment(seed=settings["seed"])
        self.agent = create_agent(self.strategy_var.get(), 3, memory_limit=settings["memory"], seed=settings["seed"], epsilon=settings["epsilon"], epsilon_start=settings["epsilon_start"], epsilon_min=settings["epsilon_min"], epsilon_decay=settings["epsilon_decay"], temperature=settings["temperature"])

    def reset(self) -> None:
        self._running = False
        try:
            self._new_models()
        except ValueError as error:
            messagebox.showerror("Ungültige Einstellung", str(error))
            return
        self.status_var.set("Bereit")
        self._set_running(False)
        self._refresh_display()

    def _strategy_changed(self, _event: object = None) -> None:
        self.strategy_help_var.set(STRATEGY_HELP[self.strategy_var.get()])
        self._rebuild_parameter_fields()
        self.reset()

    def _update_probabilities(self) -> None:
        if self.environment:
            for card, probability in zip(self.cards, self.environment.probabilities):
                card.show_probability(probability, self.show_probabilities_var.get())

    def manual_pull(self, action: int) -> None:
        if not self.environment:
            return
        reward = self.environment.pull(action)
        self.cards[action].flash(reward)
        self.status_var.set("Manuell: Bandit {} → Reward {}".format(action + 1, reward))
        self._refresh_display()

    def _agent_step(self, refresh: bool = True) -> None:
        if not self.agent or not self.environment:
            return
        action = self.agent.select_action()
        reward = self.environment.pull(action)
        self.agent.update(action, reward)
        self.cards[action].flash(reward, selected_by_agent=True)
        if refresh:
            self.status_var.set("Agent: Bandit {} → Reward {}".format(action + 1, reward))
            self._refresh_display()

    def agent_step(self) -> None:
        self._agent_step()

    def run_agent(self) -> None:
        try:
            loops = int(self.loops_var.get())
            if loops <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ungültige Einstellung", "Agent Loops muss positiv sein.")
            return
        self._remaining_steps = loops
        self._run_total = loops
        self._running = True
        self._set_running(True)
        self.root.after(1, self._run_chunk)

    def _run_chunk(self) -> None:
        if not self._running:
            return
        chunk = min(20, self._remaining_steps)
        for _ in range(chunk):
            self._agent_step(refresh=False)
        self._remaining_steps -= chunk
        completed = self._run_total - self._remaining_steps
        self.status_var.set("Schritt {} von {}".format(completed, self._run_total))
        self._refresh_display()
        if self._remaining_steps:
            self.root.after(1, self._run_chunk)
        else:
            self._running = False
            self._set_running(False)
            self.status_var.set("Agent-Lauf abgeschlossen")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for card in self.cards:
            card.button.configure(state=state)
        self.step_button.configure(state=state)
        self.run_button.configure(state=state)
        self.strategy_box.configure(state="disabled" if running else "readonly")

    def _refresh_display(self) -> None:
        if not self.agent or not self.environment:
            return
        estimates = self.agent.estimated_values()
        for index, card in enumerate(self.cards):
            card.update(self.environment.statistics(index), estimates[index])
        self._update_probabilities()
        extra = ""
        if hasattr(self.agent, "epsilon"):
            extra = " | Epsilon: {:.4f}".format(self.agent.epsilon)
        elif hasattr(self.agent, "temperature"):
            extra = " | Temperatur: {:.3f}".format(self.agent.temperature)
        self.total_var.set("{} | Agent-Pulls: {} | kumulativer Reward: {}{}".format(self.strategy_var.get(), self.agent.total_pulls, self.agent.cumulative_reward, extra))
        self.current_axes.clear()
        self.current_axes.set_title("Kumulativer Reward des Agenten")
        self.current_axes.set_xlabel("Agent-Pulls")
        self.current_axes.set_ylabel("Kumulativer Reward")
        self.current_axes.grid(True, alpha=0.25)
        if self.agent.reward_history:
            self.current_axes.plot(range(1, len(self.agent.reward_history) + 1), self.agent.reward_history, color="#2563eb", linewidth=2)
        else:
            self.current_axes.text(0.5, 0.5, "Starte den Agenten, um den Lernverlauf zu sehen.", ha="center", va="center", transform=self.current_axes.transAxes, color="#6b7280")
        self.current_figure.tight_layout()
        self.current_canvas.draw_idle()

    def start_comparison(self) -> None:
        if self._comparison_running:
            return
        strategies = [name for name, variable in self.comparison_selected.items() if variable.get()]
        try:
            steps = int(self.comparison_steps_var.get())
            repetitions = int(self.comparison_repetitions_var.get())
            seed_text = self.comparison_seed_var.get().strip()
            seed = int(seed_text) if seed_text else None
            if not strategies or steps <= 0 or repetitions <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ungültiger Vergleich", "Wähle mindestens eine Strategie und gib positive Ganzzahlen ein.")
            return
        self._comparison_running = True
        self.compare_button.configure(state="disabled")
        self.comparison_status_var.set("Vergleich wird berechnet …")

        def worker() -> None:
            try:
                result = compare_strategies(strategies, steps, repetitions, seed)
                self._comparison_queue.put(("ok", result))
            except Exception as error:
                self._comparison_queue.put(("error", str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._poll_comparison)

    def _poll_comparison(self) -> None:
        try:
            status, payload = self._comparison_queue.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll_comparison)
            return
        self._comparison_running = False
        self.compare_button.configure(state="normal")
        if status == "error":
            messagebox.showerror("Vergleich fehlgeschlagen", payload)
            self.comparison_status_var.set("Der Vergleich konnte nicht berechnet werden.")
            return
        self.comparison_results = payload
        self._draw_comparison()
        winner = self.comparison_results[0]
        self.comparison_status_var.set("{} erzielte unter diesen Bedingungen den höchsten mittleren Reward. Ergebnisse können durch Zufall und Laufdauer schwanken.".format(winner.strategy))

    def reset_comparison(self) -> None:
        if self._comparison_running:
            return
        self.comparison_results = []
        self.comparison_status_var.set("Wähle Strategien und starte den Vergleich.")
        self._draw_comparison()

    def _draw_comparison(self) -> None:
        self.comparison_axes.clear()
        self.comparison_axes.set_title("Strategien im Vergleich")
        self.comparison_axes.set_xlabel("Agent-Pulls")
        self.comparison_axes.set_ylabel("Mittlerer kumulativer Reward")
        self.comparison_axes.grid(True, alpha=0.25)
        for item in self.comparison_table.get_children():
            self.comparison_table.delete(item)
        if not self.comparison_results:
            self.comparison_axes.text(0.5, 0.5, "Starte einen Vergleich, um die Lernkurven zu sehen.", ha="center", va="center", transform=self.comparison_axes.transAxes, color="#6b7280")
        else:
            steps = len(self.comparison_results[0].mean_rewards)
            x_values = list(range(1, steps + 1))
            for rank, result in enumerate(self.comparison_results):
                color = STRATEGY_COLORS[result.strategy]
                self.comparison_axes.plot(x_values, result.mean_rewards, label=result.strategy, color=color, linewidth=2.3 if rank == 0 else 1.8)
                if self.show_ci_var.get():
                    self.comparison_axes.fill_between(x_values, result.confidence_low, result.confidence_high, color=color, alpha=0.10)
                self.comparison_table.insert("", "end", values=(result.strategy, "{:.1f} ± {:.1f}".format(result.final_reward, result.final_confidence_margin), "{:.3f}".format(result.reward_per_pull), "{:.1%}".format(result.best_bandit_share), "{:.1f}".format(result.regret)), tags=("winner",) if rank == 0 else ())
            self.comparison_axes.plot(x_values, [step * 0.8 for step in x_values], linestyle="--", color="#111827", linewidth=1.3, label="Optimaler Erwartungswert")
            self.comparison_axes.legend(fontsize=8, loc="upper left")
            self.comparison_table.tag_configure("winner", background="#dcfce7")
        self.comparison_figure.tight_layout()
        self.comparison_canvas.draw_idle()
