"""Model-based planning logic for the Gridworld learning laboratory."""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

State = Tuple[int, int]

UP, DOWN, LEFT, RIGHT = range(4)
ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_NAMES = {UP: "Up", DOWN: "Down", LEFT: "Left", RIGHT: "Right"}
ACTION_ARROWS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


@dataclass(frozen=True)
class SweepResult:
    iteration: int
    delta: float
    start_value: float
    converged: bool


@dataclass(frozen=True)
class Transition:
    step: int
    state: State
    action: int
    next_state: State
    reward: float
    done: bool
    termination_reason: Optional[str]


@dataclass(frozen=True)
class RolloutResult:
    method: str
    transitions: List[Transition]
    total_reward: float
    steps: int
    success: bool
    epsilon: float


@dataclass(frozen=True)
class ComparisonResult:
    method: str
    sweep_results: List[SweepResult]
    converged: bool
    runtime_ms: float
    final_start_value: float
    rollout_results: List[RolloutResult]


class GridWorld:
    """Deterministic rectangular environment using (row, column) states."""

    def __init__(
        self,
        rows: int = 3,
        columns: int = 5,
        start: State = (2, 0),
        goal: State = (2, 4),
        obstacles: Sequence[State] = ((1, 2), (2, 2)),
        max_steps: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacles = tuple(tuple(cell) for cell in obstacles)
        self.max_steps = max_steps
        self.seed = seed
        self._rng = random.Random(seed)
        self._validate()
        self.reset()

    def _validate(self) -> None:
        if not 2 <= self.rows <= 8:
            raise ValueError("Rows muss zwischen 2 und 8 liegen.")
        if not 2 <= self.columns <= 10:
            raise ValueError("Columns muss zwischen 2 und 10 liegen.")
        if self.max_steps <= 0:
            raise ValueError("Max Steps muss eine positive Ganzzahl sein.")
        if self.start == self.goal:
            raise ValueError("Start und Ziel müssen verschieden sein.")
        cells = (self.start, self.goal) + self.obstacles
        if any(not self.in_bounds(cell) for cell in cells):
            raise ValueError("Start, Ziel und Hindernisse müssen innerhalb des Grids liegen.")
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Start und Ziel dürfen keine Hindernisse sein.")
        if len(set(self.obstacles)) != len(self.obstacles):
            raise ValueError("Hindernisse dürfen keine Duplikate enthalten.")
        distance = self.shortest_path_length()
        if distance is None:
            raise ValueError("Vom Start zum Ziel existiert kein Weg.")
        if distance > self.max_steps:
            raise ValueError(
                f"Der kürzeste Weg benötigt {distance} Schritte; Max Steps ist nur {self.max_steps}."
            )

    def in_bounds(self, state: State) -> bool:
        row, column = state
        return 0 <= row < self.rows and 0 <= column < self.columns

    @property
    def states(self) -> List[State]:
        blocked = set(self.obstacles)
        return [
            (row, column)
            for row in range(self.rows)
            for column in range(self.columns)
            if (row, column) not in blocked
        ]

    @property
    def planning_states(self) -> List[State]:
        return [state for state in self.states if state != self.goal]

    def shortest_path_length(self) -> Optional[int]:
        queue = deque([(self.start, 0)])
        visited = {self.start}
        blocked = set(self.obstacles)
        while queue:
            state, distance = queue.popleft()
            if state == self.goal:
                return distance
            for delta_row, delta_column in ACTION_DELTAS.values():
                candidate = (state[0] + delta_row, state[1] + delta_column)
                if self.in_bounds(candidate) and candidate not in blocked and candidate not in visited:
                    visited.add(candidate)
                    queue.append((candidate, distance + 1))
        return None

    def transition(self, state: State, action: int) -> Tuple[State, float, bool]:
        if not self.in_bounds(state) or state in self.obstacles:
            raise ValueError("Der Zustand ist ungültig oder blockiert.")
        if action not in ACTIONS:
            raise ValueError("Action muss 0, 1, 2 oder 3 sein.")
        if state == self.goal:
            return self.goal, 0.0, True
        delta_row, delta_column = ACTION_DELTAS[action]
        candidate = (state[0] + delta_row, state[1] + delta_column)
        if not self.in_bounds(candidate) or candidate in self.obstacles:
            candidate = state
        done = candidate == self.goal
        return candidate, (0.0 if done else -1.0), done

    def reset(self, seed: Optional[int] = None) -> State:
        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        self.current_state = self.start
        self.step_count = 0
        self.done = False
        return self.current_state

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Optional[str]]]:
        if self.done:
            raise RuntimeError("Der Rollout ist beendet. Vor dem nächsten Schritt reset() aufrufen.")
        next_state, reward, reached_goal = self.transition(self.current_state, action)
        self.current_state = next_state
        self.step_count += 1
        reason: Optional[str] = None
        if reached_goal:
            self.done = True
            reason = "goal_reached"
        elif self.step_count >= self.max_steps:
            self.done = True
            reason = "max_steps"
        return next_state, reward, self.done, {"termination_reason": reason}

    def clone(self) -> "GridWorld":
        return GridWorld(
            self.rows, self.columns, self.start, self.goal, self.obstacles,
            self.max_steps, self.seed,
        )


class BasePlanner:
    name = "Base"

    def __init__(
        self,
        environment: GridWorld,
        gamma: float = 0.9,
        tolerance: float = 0.0001,
        seed: Optional[int] = None,
    ) -> None:
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma muss zwischen 0 und 1 liegen.")
        if tolerance <= 0:
            raise ValueError("Tolerance muss größer als 0 sein.")
        self.environment = environment
        self.gamma = gamma
        self.tolerance = tolerance
        self.seed = seed
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.iteration = 0
        self.converged = False
        self.history: List[SweepResult] = []
        self._reset_tables()

    def _reset_tables(self) -> None:
        raise NotImplementedError

    def _perform_sweep(self) -> float:
        raise NotImplementedError

    def single_sweep(self) -> SweepResult:
        if self.converged and self.history:
            return self.history[-1]
        delta = self._perform_sweep()
        self.iteration += 1
        self.converged = delta < self.tolerance
        result = SweepResult(self.iteration, delta, self.get_value(self.environment.start), self.converged)
        self.history.append(result)
        return result

    def run(
        self,
        max_sweeps: int,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[SweepResult]:
        if max_sweeps <= 0:
            raise ValueError("max_sweeps muss positiv sein.")
        results: List[SweepResult] = []
        for _ in range(max_sweeps):
            if self.converged or (cancelled is not None and cancelled()):
                break
            results.append(self.single_sweep())
        return results

    def get_value(self, state: State) -> float:
        raise NotImplementedError

    def get_q_value(self, state: State, action: int) -> float:
        raise NotImplementedError

    def get_best_actions(self, state: State) -> List[int]:
        if state == self.environment.goal or state in self.environment.obstacles:
            return []
        if self.iteration == 0:
            return []
        action_values = [(action, self.get_q_value(state, action)) for action in ACTIONS]
        best_value = max(value for _, value in action_values)
        action_tolerance = max(self.tolerance, 1e-9)
        return [action for action, value in action_values if best_value - value <= action_tolerance]

    def get_greedy_action(self, state: State) -> int:
        best = self.get_best_actions(state)
        if not best:
            best = list(ACTIONS)
        return self._rng.choice(best)


class ValueIteration(BasePlanner):
    name = "Value Iteration"

    def _reset_tables(self) -> None:
        self.values = {state: 0.0 for state in self.environment.states}

    def _action_value(self, state: State, action: int, values: Dict[State, float]) -> float:
        next_state, reward, done = self.environment.transition(state, action)
        return reward if done else reward + self.gamma * values[next_state]

    def _perform_sweep(self) -> float:
        old = self.values.copy()
        new = old.copy()
        delta = 0.0
        for state in self.environment.planning_states:
            new[state] = max(self._action_value(state, action, old) for action in ACTIONS)
            delta = max(delta, abs(new[state] - old[state]))
        new[self.environment.goal] = 0.0
        self.values = new
        return delta

    def get_value(self, state: State) -> float:
        return self.values.get(state, 0.0)

    def get_q_value(self, state: State, action: int) -> float:
        if state == self.environment.goal or state in self.environment.obstacles:
            return 0.0
        return self._action_value(state, action, self.values)


class QValueIteration(BasePlanner):
    name = "Q-Value Iteration"

    def _reset_tables(self) -> None:
        self.q_values = {
            state: [0.0 for _ in ACTIONS] for state in self.environment.states
        }

    def _perform_sweep(self) -> float:
        old = {state: values.copy() for state, values in self.q_values.items()}
        new = {state: values.copy() for state, values in old.items()}
        delta = 0.0
        for state in self.environment.planning_states:
            for action in ACTIONS:
                next_state, reward, done = self.environment.transition(state, action)
                value = reward if done else reward + self.gamma * max(old[next_state])
                new[state][action] = value
                delta = max(delta, abs(value - old[state][action]))
        new[self.environment.goal] = [0.0 for _ in ACTIONS]
        self.q_values = new
        return delta

    def get_value(self, state: State) -> float:
        return max(self.q_values.get(state, [0.0]))

    def get_q_value(self, state: State, action: int) -> float:
        if action not in ACTIONS:
            raise ValueError("Action muss 0, 1, 2 oder 3 sein.")
        return self.q_values.get(state, [0.0] * 4)[action]


class RolloutAgent:
    """Executes a planner policy without changing its tables."""

    def __init__(
        self,
        environment: GridWorld,
        planner: BasePlanner,
        seed: Optional[int] = None,
    ) -> None:
        self.environment = environment
        self.planner = planner
        self.seed = seed
        self._tie_rng = random.Random(seed)
        self._explore_rng = random.Random(None if seed is None else seed + 1)
        self.reset()

    def set_planner(self, planner: BasePlanner) -> None:
        self.planner = planner
        self.reset()

    def reset(self) -> State:
        self.transitions: List[Transition] = []
        return self.environment.reset()

    def select_action(self, state: State, epsilon: float) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon muss zwischen 0 und 1 liegen.")
        if self._explore_rng.random() < epsilon:
            return self._explore_rng.choice(ACTIONS)
        best = self.planner.get_best_actions(state) or list(ACTIONS)
        return self._tie_rng.choice(best)

    def single_step(self, epsilon: float) -> Transition:
        if self.environment.done:
            self.reset()
        state = self.environment.current_state
        action = self.select_action(state, epsilon)
        next_state, reward, done, info = self.environment.step(action)
        transition = Transition(
            self.environment.step_count, state, action, next_state, reward, done,
            info["termination_reason"],
        )
        self.transitions.append(transition)
        return transition

    def current_result(self, epsilon: float) -> RolloutResult:
        return RolloutResult(
            self.planner.name,
            list(self.transitions),
            sum(item.reward for item in self.transitions),
            len(self.transitions),
            bool(self.transitions and self.transitions[-1].termination_reason == "goal_reached"),
            epsilon,
        )

    def run_steps(self, epsilon: float, steps: int) -> RolloutResult:
        if steps <= 0:
            raise ValueError("steps muss positiv sein.")
        if self.environment.done:
            self.reset()
        for _ in range(steps):
            transition = self.single_step(epsilon)
            if transition.done:
                break
        return self.current_result(epsilon)

    def run_episode(self, epsilon: float, max_steps: Optional[int] = None) -> RolloutResult:
        self.reset()
        limit = self.environment.max_steps if max_steps is None else max_steps
        return self.run_steps(epsilon, limit)

    def run_greedy_episode(self) -> RolloutResult:
        return self.run_episode(0.0, self.environment.max_steps)


def create_planners(
    environment: GridWorld,
    gamma: float = 0.9,
    tolerance: float = 0.0001,
    seed: Optional[int] = None,
) -> Dict[str, BasePlanner]:
    return {
        ValueIteration.name: ValueIteration(environment, gamma, tolerance, seed),
        QValueIteration.name: QValueIteration(environment, gamma, tolerance, seed),
    }


class ComparisonRunner:
    def __init__(
        self,
        environment: GridWorld,
        gamma: float,
        tolerance: float,
        max_iterations: int,
        rollout_count: int = 20,
        epsilon_max: float = 0.995,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        if max_iterations <= 0 or rollout_count <= 0:
            raise ValueError("Iterationen und Vergleichs-Rollouts müssen positiv sein.")
        self.environment = environment
        self.gamma = gamma
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.rollout_count = rollout_count
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed

    def run(self, cancelled: Optional[Callable[[], bool]] = None) -> List[ComparisonResult]:
        results: List[ComparisonResult] = []
        for index, planner_class in enumerate((ValueIteration, QValueIteration)):
            if cancelled is not None and cancelled():
                break
            environment = self.environment.clone()
            derived_seed = None if self.seed is None else self.seed + index * 1000
            planner = planner_class(environment, self.gamma, self.tolerance, derived_seed)
            started = time.perf_counter()
            sweeps = planner.run(self.max_iterations, cancelled)
            runtime_ms = (time.perf_counter() - started) * 1000.0
            rollouts: List[RolloutResult] = []
            epsilon = self.epsilon_max
            if cancelled is None or not cancelled():
                for rollout_index in range(self.rollout_count):
                    agent_seed = None if self.seed is None else self.seed + 10_000 + rollout_index
                    agent = RolloutAgent(environment, planner, agent_seed)
                    rollouts.append(agent.run_episode(epsilon, environment.max_steps))
                    epsilon = max(self.epsilon_min, epsilon * (1.0 - self.epsilon_decay))
            results.append(ComparisonResult(
                planner.name, sweeps, planner.converged, runtime_ms,
                planner.get_value(environment.start), rollouts,
            ))
        return results


VALUE_CSV_FIELDS = (
    "row", "column", "value", "is_start", "is_goal", "is_obstacle", "method", "iteration",
)
Q_CSV_FIELDS = (
    "row", "column", "q_up", "q_down", "q_left", "q_right", "best_actions",
    "is_start", "is_goal", "is_obstacle", "method", "iteration",
)


def value_table_rows(planner: BasePlanner) -> List[Dict[str, object]]:
    environment = planner.environment
    rows: List[Dict[str, object]] = []
    for row in range(environment.rows):
        for column in range(environment.columns):
            state = (row, column)
            blocked = state in environment.obstacles
            rows.append({
                "row": row, "column": column,
                "value": f"{planner.get_value(state):.6f}",
                "is_start": state == environment.start,
                "is_goal": state == environment.goal,
                "is_obstacle": blocked,
                "method": planner.name, "iteration": planner.iteration,
            })
    return rows


def q_table_rows(planner: BasePlanner) -> List[Dict[str, object]]:
    environment = planner.environment
    rows: List[Dict[str, object]] = []
    for row in range(environment.rows):
        for column in range(environment.columns):
            state = (row, column)
            blocked = state in environment.obstacles
            q_values = [planner.get_q_value(state, action) for action in ACTIONS]
            best = " ".join(ACTION_NAMES[action] for action in planner.get_best_actions(state))
            rows.append({
                "row": row, "column": column,
                "q_up": f"{q_values[UP]:.6f}", "q_down": f"{q_values[DOWN]:.6f}",
                "q_left": f"{q_values[LEFT]:.6f}", "q_right": f"{q_values[RIGHT]:.6f}",
                "best_actions": best,
                "is_start": state == environment.start,
                "is_goal": state == environment.goal,
                "is_obstacle": blocked,
                "method": planner.name, "iteration": planner.iteration,
            })
    return rows
