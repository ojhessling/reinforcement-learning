"""Tabular reinforcement-learning logic for Gymnasium FrozenLake."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Dict, List, Optional, Tuple, Type

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np

LEFT, DOWN, RIGHT, UP = range(4)
ACTIONS = (LEFT, DOWN, RIGHT, UP)
ACTION_NAMES = {LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up"}
ACTION_ARROWS = {LEFT: "←", DOWN: "↓", RIGHT: "→", UP: "↑"}
MAP_SIZES = {"4x4": 4, "8x8": 8}


@dataclass(frozen=True)
class Transition:
    observation: int
    action: int
    next_observation: int
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    step: int


@dataclass(frozen=True)
class EpisodeResult:
    method: str
    transitions: List[Transition]
    total_reward: float
    steps: int
    success: bool
    termination_reason: str
    epsilon: float


class FrozenLakeEnvironment:
    """Adapter around Gymnasium's unmodified FrozenLake-v1 environment."""

    def __init__(
        self,
        map_name: str = "4x4",
        is_slippery: bool = True,
        max_steps: int = 100,
        seed: Optional[int] = 42,
    ) -> None:
        if map_name not in MAP_SIZES:
            raise ValueError("Karte muss 4x4 oder 8x8 sein.")
        if max_steps <= 0:
            raise ValueError("Max. Schritte muss eine positive Ganzzahl sein.")
        self.map_name = map_name
        self.size = MAP_SIZES[map_name]
        self.is_slippery = bool(is_slippery)
        self.max_steps = max_steps
        self.seed = seed
        self._environment = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=self.is_slippery,
            render_mode="rgb_array",
        )
        if seed is not None:
            self._environment.action_space.seed(seed)
        self.reset(seed=seed)

    @property
    def state_count(self) -> int:
        return self.size * self.size

    def observation_to_position(self, observation: int) -> Tuple[int, int]:
        if not 0 <= int(observation) < self.state_count:
            raise ValueError("Observation liegt außerhalb der gewählten Karte.")
        return divmod(int(observation), self.size)

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.seed = seed
        observation, _ = self._environment.reset(seed=seed)
        self.observation = int(observation)
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        return self.observation

    def step(self, action: int) -> Transition:
        if self.done:
            raise RuntimeError("Episode beendet; vor dem nächsten Schritt reset() aufrufen.")
        if action not in ACTIONS:
            raise ValueError("Action muss 0, 1, 2 oder 3 sein.")
        observation = self.observation
        next_observation, reward, terminated, gym_truncated, _ = self._environment.step(action)
        self.step_count += 1
        truncated = bool(gym_truncated or (self.step_count >= self.max_steps and not terminated))
        self.observation = int(next_observation)
        self.total_reward += float(reward)
        self.done = bool(terminated or truncated)
        return Transition(
            observation=observation,
            action=action,
            next_observation=self.observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=truncated,
            done=self.done,
            step=self.step_count,
        )

    def render_rgb(self) -> np.ndarray:
        return np.asarray(self._environment.render())

    def close(self) -> None:
        self._environment.close()


class BasePolicy:
    name = "Base"

    def __init__(
        self,
        state_count: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        exploration: str = "Epsilon konstant",
        seed: Optional[int] = 42,
    ) -> None:
        if state_count <= 0:
            raise ValueError("Anzahl Zustände muss positiv sein.")
        if not 0 < alpha <= 1:
            raise ValueError("Alpha muss größer als 0 und höchstens 1 sein.")
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma muss zwischen 0 und 1 liegen.")
        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon muss zwischen 0 und 1 liegen.")
        if not 0 <= epsilon_min <= epsilon:
            raise ValueError("Epsilon Minimum muss zwischen 0 und Epsilon liegen.")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("Epsilon Decay muss größer als 0 und höchstens 1 sein.")
        if exploration not in ("Epsilon konstant", "Epsilon-Decay"):
            raise ValueError("Unbekannte Exploration.")
        self.state_count = state_count
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.exploration = exploration
        self.seed = seed
        self._rng = random.Random(seed)
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        self.q_values = np.zeros((self.state_count, 4), dtype=float)
        self.visit_counts = np.zeros(self.state_count, dtype=int)
        self.epsilon = self.epsilon_start
        self.episodes_completed = 0

    def best_actions(self, observation: int) -> List[int]:
        values = self.q_values[int(observation)]
        best = float(np.max(values))
        return [index for index, value in enumerate(values) if np.isclose(value, best)]

    def select_action(self, observation: int, training: bool = True) -> int:
        if training and self._rng.random() < self.epsilon:
            return self._rng.choice(ACTIONS)
        return self._rng.choice(self.best_actions(observation))

    def expected_q(self, observation: int) -> float:
        values = self.q_values[observation]
        best = self.best_actions(observation)
        random_probability = self.epsilon / len(ACTIONS)
        greedy_probability = (1 - self.epsilon) / len(best)
        return float(sum(
            value * (random_probability + (greedy_probability if action in best else 0))
            for action, value in enumerate(values)
        ))

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        next_action: Optional[int],
        done: bool,
    ) -> None:
        raise NotImplementedError

    def end_episode(self) -> None:
        self.episodes_completed += 1
        if self.exploration == "Epsilon-Decay":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningPolicy(BasePolicy):
    name = "Q-Learning"

    def update(self, observation: int, action: int, reward: float, next_observation: int, next_action: Optional[int], done: bool) -> None:
        future = 0.0 if done else float(np.max(self.q_values[next_observation]))
        target = reward + self.gamma * future
        self.q_values[observation, action] += self.alpha * (target - self.q_values[observation, action])
        self.visit_counts[observation] += 1


class SarsaPolicy(BasePolicy):
    name = "SARSA"

    def update(self, observation: int, action: int, reward: float, next_observation: int, next_action: Optional[int], done: bool) -> None:
        if not done and next_action is None:
            raise ValueError("SARSA benötigt next_action.")
        future = 0.0 if done else float(self.q_values[next_observation, int(next_action)])
        target = reward + self.gamma * future
        self.q_values[observation, action] += self.alpha * (target - self.q_values[observation, action])
        self.visit_counts[observation] += 1


class ExpectedSarsaPolicy(BasePolicy):
    name = "Expected SARSA"

    def update(self, observation: int, action: int, reward: float, next_observation: int, next_action: Optional[int], done: bool) -> None:
        future = 0.0 if done else self.expected_q(next_observation)
        target = reward + self.gamma * future
        self.q_values[observation, action] += self.alpha * (target - self.q_values[observation, action])
        self.visit_counts[observation] += 1


POLICY_CLASSES: Dict[str, Type[BasePolicy]] = {
    policy.name: policy for policy in (QLearningPolicy, SarsaPolicy, ExpectedSarsaPolicy)
}


def create_policy(name: str, **parameters: object) -> BasePolicy:
    try:
        return POLICY_CLASSES[name](**parameters)
    except KeyError as error:
        raise ValueError(f"Unbekannte Methode: {name}") from error


class TabularAgent:
    def __init__(self, environment: FrozenLakeEnvironment, policy: BasePolicy) -> None:
        self.environment = environment
        self.policy = policy
        self.returns: List[float] = []
        self.successes: List[bool] = []
        self.episode_lengths: List[int] = []
        self.current_transitions: List[Transition] = []
        self.latest_result: Optional[EpisodeResult] = None
        self.episode_active = False
        self.training = True
        self._next_action: Optional[int] = None

    @property
    def episode_count(self) -> int:
        return len(self.returns)

    def start_episode(self, training: bool = True, seed: Optional[int] = None) -> int:
        observation = self.environment.reset(seed=seed)
        self.current_transitions = []
        self.episode_active = True
        self.training = training
        self._next_action = None
        return observation

    def step(self, training: Optional[bool] = None) -> Transition:
        if training is None:
            training = self.training
        if not self.episode_active:
            self.start_episode(training=training)
        observation = self.environment.observation
        action = self._next_action
        if action is None:
            action = self.policy.select_action(observation, training=training)
        transition = self.environment.step(action)
        next_action = None
        if not transition.done and isinstance(self.policy, SarsaPolicy):
            next_action = self.policy.select_action(transition.next_observation, training=training)
        if training:
            self.policy.update(
                observation, action, transition.reward, transition.next_observation,
                next_action, transition.done,
            )
        self._next_action = next_action
        self.current_transitions.append(transition)
        if transition.done:
            self._finish_episode(training)
        return transition

    def _finish_episode(self, training: bool) -> EpisodeResult:
        success = bool(self.current_transitions[-1].reward > 0)
        reason = "goal_reached" if success else (
            "max_steps" if self.current_transitions[-1].truncated else "hole"
        )
        result = EpisodeResult(
            method=self.policy.name,
            transitions=list(self.current_transitions),
            total_reward=sum(item.reward for item in self.current_transitions),
            steps=len(self.current_transitions),
            success=success,
            termination_reason=reason,
            epsilon=0.0 if not training else self.policy.epsilon,
        )
        if training:
            self.returns.append(result.total_reward)
            self.successes.append(success)
            self.episode_lengths.append(result.steps)
            self.policy.end_episode()
        self.latest_result = result
        self.episode_active = False
        self._next_action = None
        return result

    def run_episode(self, training: bool = True, seed: Optional[int] = None) -> EpisodeResult:
        self.start_episode(training=training, seed=seed)
        while self.episode_active:
            self.step(training=training)
        assert self.latest_result is not None
        return self.latest_result

    def moving_success_rate(self, window: int = 100) -> List[float]:
        return [
            mean(self.successes[max(0, index - window + 1):index + 1])
            for index in range(len(self.successes))
        ]

    def reset_training(self) -> None:
        self.policy.reset(self.policy.seed)
        self.returns.clear()
        self.successes.clear()
        self.episode_lengths.clear()
        self.current_transitions = []
        self.latest_result = None
        self.episode_active = False
        self._next_action = None
        self.environment.reset(seed=self.environment.seed)


class TrainingRunner:
    def __init__(self, agent: TabularAgent) -> None:
        self.agent = agent

    def run(
        self,
        episodes: int,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[EpisodeResult]:
        if episodes <= 0:
            raise ValueError("Episoden muss eine positive Ganzzahl sein.")
        results = []
        for _ in range(episodes):
            if should_stop and should_stop():
                break
            results.append(self.agent.run_episode(training=True))
        return results


class EvaluationRunner:
    def __init__(self, agent: TabularAgent) -> None:
        self.agent = agent

    def run(self, episodes: int = 1) -> List[EpisodeResult]:
        if episodes <= 0:
            raise ValueError("Evaluationsepisoden muss positiv sein.")
        return [self.agent.run_episode(training=False) for _ in range(episodes)]
