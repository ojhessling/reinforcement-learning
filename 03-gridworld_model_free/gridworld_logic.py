"""Tabular model-free reinforcement-learning logic for Gridworld."""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from statistics import mean, stdev
from typing import Callable, DefaultDict, Dict, List, Optional, Sequence, Tuple

State = Tuple[int, int]

UP, DOWN, LEFT, RIGHT = range(4)
ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_NAMES = {UP: "Up", DOWN: "Down", LEFT: "Left", RIGHT: "Right"}
ACTION_DELTAS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}


@dataclass
class Transition:
    episode: int
    step: int
    state: State
    action: int
    next_state: State
    reward: float
    done: bool
    termination_reason: Optional[str]
    policy: str

    def csv_row(self) -> Dict[str, object]:
        return {
            "episode": self.episode,
            "step": self.step,
            "state_x": self.state[0],
            "state_y": self.state[1],
            "action": self.action,
            "action_name": ACTION_NAMES[self.action],
            "next_state_x": self.next_state[0],
            "next_state_y": self.next_state[1],
            "reward": self.reward,
            "done": self.done,
            "termination_reason": self.termination_reason or "",
            "policy": self.policy,
        }


class GridWorld:
    """Deterministic rectangular Gridworld with blocked cells."""

    def __init__(
        self,
        width: int = 5,
        height: int = 3,
        start: State = (0, 2),
        goal: State = (4, 2),
        blocked: Sequence[State] = ((2, 1), (2, 2)),
        max_steps: int = 20,
        seed: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.blocked = tuple(tuple(cell) for cell in blocked)
        self.max_steps = max_steps
        self.seed = seed
        self._rng = random.Random(seed)
        self._validate()
        self.reset()

    def _validate(self) -> None:
        if not 2 <= self.width <= 15 or not 2 <= self.height <= 15:
            raise ValueError("Grid-Breite und -Höhe müssen zwischen 2 und 15 liegen.")
        if self.max_steps <= 0:
            raise ValueError("max_steps muss positiv sein.")
        if self.start == self.goal:
            raise ValueError("Start und Ziel müssen verschieden sein.")
        cells = (self.start, self.goal) + self.blocked
        if any(not self.in_bounds(cell) for cell in cells):
            raise ValueError("Start, Ziel und blockierte Zellen müssen im Grid liegen.")
        if self.start in self.blocked or self.goal in self.blocked:
            raise ValueError("Start und Ziel dürfen nicht blockiert sein.")
        if len(set(self.blocked)) != len(self.blocked):
            raise ValueError("Blockierte Zellen dürfen keine Duplikate enthalten.")
        distance = self.shortest_path_length()
        if distance is None:
            raise ValueError("Das Ziel ist vom Start aus nicht erreichbar.")
        if distance > self.max_steps:
            raise ValueError(
                "Der kürzeste Weg benötigt {} Schritte, max_steps ist aber {}.".format(
                    distance, self.max_steps
                )
            )

    def in_bounds(self, state: State) -> bool:
        return 0 <= state[0] < self.width and 0 <= state[1] < self.height

    def shortest_path_length(self) -> Optional[int]:
        queue = deque([(self.start, 0)])
        visited = {self.start}
        blocked = set(self.blocked)
        while queue:
            state, distance = queue.popleft()
            if state == self.goal:
                return distance
            for dx, dy in ACTION_DELTAS.values():
                candidate = (state[0] + dx, state[1] + dy)
                if self.in_bounds(candidate) and candidate not in blocked and candidate not in visited:
                    visited.add(candidate)
                    queue.append((candidate, distance + 1))
        return None

    @property
    def states(self) -> List[State]:
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.blocked
        ]

    def reset(self, seed: Optional[int] = None) -> State:
        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        self.state = self.start
        self.steps = 0
        self.done = False
        return self.state

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("Die Episode ist beendet. Vor einem weiteren Schritt reset() aufrufen.")
        if action not in ACTIONS:
            raise ValueError("action muss 0, 1, 2 oder 3 sein.")
        dx, dy = ACTION_DELTAS[action]
        candidate = (self.state[0] + dx, self.state[1] + dy)
        if self.in_bounds(candidate) and candidate not in self.blocked:
            self.state = candidate
        self.steps += 1
        reward = 0.0 if self.state == self.goal else -1.0
        reason: Optional[str] = None
        if self.state == self.goal:
            self.done = True
            reason = "goal_reached"
        elif self.steps >= self.max_steps:
            self.done = True
            reason = "max_steps"
        return self.state, reward, self.done, {"termination_reason": reason}


class BasePolicy:
    """Common epsilon-greedy tabular policy functionality."""

    name = "Base"

    def __init__(
        self,
        actions: Sequence[int] = ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.001,
        seed: Optional[int] = None,
        use_algorithm_defaults: bool = True,
    ) -> None:
        if not actions:
            raise ValueError("Mindestens eine Action ist erforderlich.")
        if not 0 <= alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("alpha und gamma müssen zwischen 0 und 1 liegen.")
        if not 0 <= epsilon_min <= epsilon_start <= 1 or epsilon_decay <= 0:
            raise ValueError("Ungültige Epsilon-Parameter.")
        self.actions = tuple(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_algorithm_defaults = use_algorithm_defaults
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.q: DefaultDict[State, List[float]] = defaultdict(
            lambda: [0.0] * len(self.actions)
        )
        self.state_visits: DefaultDict[State, int] = defaultdict(int)
        self.episode = 0
        self.epsilon = self.epsilon_for_episode(0)

    def epsilon_for_episode(self, episode: int) -> float:
        if self.use_algorithm_defaults:
            return max(self.epsilon_min, self.default_epsilon(episode))
        return max(
            self.epsilon_min,
            self.epsilon_start * math.exp(-self.epsilon_decay * episode),
        )

    def default_epsilon(self, episode: int) -> float:
        return 1.0 / (1.0 + 0.001 * episode)

    def set_episode(self, episode: int) -> None:
        self.episode = episode
        self.epsilon = self.epsilon_for_episode(episode)

    def best_actions(self, state: State) -> List[int]:
        values = self.q[state]
        best = max(values)
        return [action for action, value in zip(self.actions, values) if value == best]

    def select_action(self, state: State, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        return self._rng.choice(self.best_actions(state))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: Optional[int],
        done: bool,
    ) -> None:
        raise NotImplementedError

    def end_episode(self, trajectory: Sequence[Transition]) -> None:
        return None

    def get_q_value(self, state: State, action: int) -> float:
        return self.q[state][action]

    def get_state_value(self, state: State) -> float:
        return max(self.q[state])

    def get_best_action(self, state: State) -> int:
        return self._rng.choice(self.best_actions(state))

    def is_state_visited(self, state: State) -> bool:
        return self.state_visits[state] > 0


class MonteCarloPolicy(BasePolicy):
    name = "Monte Carlo"

    def reset(self) -> None:
        super().reset()
        self.return_counts: DefaultDict[Tuple[State, int], int] = defaultdict(int)

    def update(self, state: State, action: int, reward: float, next_state: State, next_action: Optional[int], done: bool) -> None:
        return None

    def end_episode(self, trajectory: Sequence[Transition]) -> None:
        episode_return = 0.0
        for transition in reversed(trajectory):
            episode_return = transition.reward + self.gamma * episode_return
            key = (transition.state, transition.action)
            self.return_counts[key] += 1
            count = self.return_counts[key]
            old = self.q[transition.state][transition.action]
            self.q[transition.state][transition.action] = old + (episode_return - old) / count
            self.state_visits[transition.state] += 1


class SarsaPolicy(BasePolicy):
    name = "SARSA"

    def update(self, state: State, action: int, reward: float, next_state: State, next_action: Optional[int], done: bool) -> None:
        future = 0.0 if done else self.q[next_state][int(next_action)]
        current = self.q[state][action]
        self.q[state][action] += self.alpha * (reward + self.gamma * future - current)
        self.state_visits[state] += 1


class ExpectedSarsaPolicy(BasePolicy):
    name = "Expected SARSA"

    def default_epsilon(self, episode: int) -> float:
        return math.exp(-0.0005 * episode)

    def expected_q(self, state: State) -> float:
        values = self.q[state]
        best = self.best_actions(state)
        exploration = self.epsilon / len(self.actions)
        exploitation = (1.0 - self.epsilon) / len(best)
        return sum(
            value * (exploration + (exploitation if action in best else 0.0))
            for action, value in zip(self.actions, values)
        )

    def update(self, state: State, action: int, reward: float, next_state: State, next_action: Optional[int], done: bool) -> None:
        future = 0.0 if done else self.expected_q(next_state)
        current = self.q[state][action]
        self.q[state][action] += self.alpha * (reward + self.gamma * future - current)
        self.state_visits[state] += 1


class QLearningPolicy(BasePolicy):
    name = "Q-Learning"

    def default_epsilon(self, episode: int) -> float:
        return math.exp(-0.005 * episode)

    def update(self, state: State, action: int, reward: float, next_state: State, next_action: Optional[int], done: bool) -> None:
        future = 0.0 if done else max(self.q[next_state])
        current = self.q[state][action]
        self.q[state][action] += self.alpha * (reward + self.gamma * future - current)
        self.state_visits[state] += 1


POLICY_CLASSES = {
    "Monte Carlo": MonteCarloPolicy,
    "SARSA": SarsaPolicy,
    "Expected SARSA": ExpectedSarsaPolicy,
    "Q-Learning": QLearningPolicy,
}


def create_policy(name: str, **kwargs: object) -> BasePolicy:
    try:
        return POLICY_CLASSES[name](**kwargs)
    except KeyError as error:
        raise ValueError("Unbekannte Policy: {}".format(name)) from error


class Agent:
    """Runs episodes and injects transitions into the selected policy."""

    def __init__(self, environment: GridWorld, policy: BasePolicy) -> None:
        self.environment = environment
        self.policy = policy
        self.reset_training()

    def set_policy(self, policy: BasePolicy) -> None:
        self.policy = policy
        self.reset_training()

    def reset_training(self) -> None:
        self.policy.reset()
        self.episode_count = 0
        self.returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.successes: List[bool] = []
        self.current_trajectory: List[Transition] = []
        self.latest_trajectory: List[Transition] = []
        self.episode_active = False
        self.training = True
        self.current_return = 0.0
        self._cached_action: Optional[int] = None

    def start_episode(self, training: bool = True) -> State:
        state = self.environment.reset()
        self.current_trajectory = []
        self.current_return = 0.0
        self.episode_active = True
        self.training = training
        self._cached_action = None
        if training:
            self.policy.set_episode(self.episode_count)
        return state

    def step(self, training: bool = True, action: Optional[int] = None) -> Transition:
        if not self.episode_active:
            self.start_episode(training=training)
        state = self.environment.state
        if action is None:
            action = self._cached_action
            if action is None:
                action = self.policy.select_action(state, greedy=not training)
        next_state, reward, done, info = self.environment.step(action)
        next_action: Optional[int] = None
        if not done and isinstance(self.policy, SarsaPolicy):
            next_action = self.policy.select_action(next_state, greedy=not training)
        transition = Transition(
            episode=self.episode_count + 1,
            step=len(self.current_trajectory) + 1,
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            done=done,
            termination_reason=info["termination_reason"],
            policy=self.policy.name,
        )
        self.current_trajectory.append(transition)
        self.current_return += reward
        if training:
            self.policy.update(state, action, reward, next_state, next_action, done)
        self._cached_action = next_action
        if done:
            self.end_episode()
        return transition

    def end_episode(self) -> None:
        if not self.episode_active:
            return
        if self.training:
            self.policy.end_episode(self.current_trajectory)
            self.episode_count += 1
            self.returns.append(self.current_return)
            self.episode_lengths.append(len(self.current_trajectory))
            self.successes.append(
                bool(self.current_trajectory)
                and self.current_trajectory[-1].termination_reason == "goal_reached"
            )
        self.latest_trajectory = self.current_trajectory[:]
        self.episode_active = False
        self._cached_action = None

    def run_episode(self, training: bool = True) -> List[Transition]:
        self.start_episode(training=training)
        while self.episode_active:
            self.step(training=training)
        return self.latest_trajectory

    def cancel_episode(self) -> None:
        """Discard an unfinished episode without changing learning statistics."""
        self.current_trajectory = []
        self.current_return = 0.0
        self.episode_active = False
        self._cached_action = None


@dataclass
class ComparisonResult:
    policy: str
    mean_returns: List[float]
    confidence_low: List[float]
    confidence_high: List[float]
    final_moving_average: float
    mean_episode_length: float
    success_rate: float
    final_confidence_margin: float


def moving_average(values: Sequence[float], window: int = 20) -> List[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(mean(values[start : index + 1]))
    return result


def compare_policies(
    policy_names: Sequence[str],
    environment_config: Dict[str, object],
    episodes: int = 500,
    repetitions: int = 20,
    base_seed: Optional[int] = None,
    alpha: float = 0.1,
    gamma: float = 0.9,
    cancelled: Optional[Callable[[], bool]] = None,
) -> List[ComparisonResult]:
    if not policy_names or episodes <= 0 or repetitions <= 0:
        raise ValueError("Policies, episodes und repetitions müssen gültig sein.")
    seed = base_seed if base_seed is not None else random.SystemRandom().randrange(2**31)
    all_returns: Dict[str, List[List[float]]] = {name: [] for name in policy_names}
    all_lengths: Dict[str, List[float]] = {name: [] for name in policy_names}
    all_success: Dict[str, List[float]] = {name: [] for name in policy_names}
    for repetition in range(repetitions):
        if cancelled and cancelled():
            break
        for name in policy_names:
            stable = sum((index + 1) * ord(char) for index, char in enumerate(name))
            run_seed = seed + repetition * 104729 + stable
            config = dict(environment_config)
            config["seed"] = run_seed
            environment = GridWorld(**config)
            policy = create_policy(name, alpha=alpha, gamma=gamma, seed=run_seed)
            agent = Agent(environment, policy)
            for _ in range(episodes):
                agent.run_episode(training=True)
            all_returns[name].append(agent.returns)
            all_lengths[name].append(mean(agent.episode_lengths))
            all_success[name].append(mean(agent.successes))
    completed = min((len(runs) for runs in all_returns.values()), default=0)
    if completed == 0:
        return []
    results = []
    for name in policy_names:
        runs = all_returns[name][:completed]
        means, lows, highs = [], [], []
        for episode in range(episodes):
            values = [run[episode] for run in runs]
            average = mean(values)
            margin = 1.96 * stdev(values) / math.sqrt(completed) if completed > 1 else 0.0
            means.append(average)
            lows.append(average - margin)
            highs.append(average + margin)
        final_values = [mean(run[-min(20, len(run)) :]) for run in runs]
        final_average = mean(final_values)
        final_margin = (
            1.96 * stdev(final_values) / math.sqrt(completed) if completed > 1 else 0.0
        )
        results.append(
            ComparisonResult(
                policy=name,
                mean_returns=means,
                confidence_low=lows,
                confidence_high=highs,
                final_moving_average=final_average,
                mean_episode_length=mean(all_lengths[name][:completed]),
                success_rate=mean(all_success[name][:completed]),
                final_confidence_margin=final_margin,
            )
        )
    return sorted(results, key=lambda item: item.final_moving_average, reverse=True)
