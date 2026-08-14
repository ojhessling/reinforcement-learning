"""Tabular reinforcement-learning logic for Gymnasium Taxi."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np

SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF = range(6)
ACTIONS = tuple(range(6))
ACTION_NAMES = {
    SOUTH: "South", NORTH: "North", EAST: "East", WEST: "West",
    PICKUP: "Pickup", DROPOFF: "Dropoff",
}


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
    next_action_mask: Tuple[int, ...]


@dataclass(frozen=True)
class EpisodeResult:
    method: str
    transitions: List[Transition]
    total_reward: float
    steps: int
    success: bool
    illegal_actions: int
    termination_reason: str
    epsilon: float


class TaxiEnvironment:
    def __init__(
        self,
        is_rainy: bool = False,
        fickle_passenger: bool = False,
        max_steps: int = 200,
        seed: Optional[int] = 42,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("Max. Schritte muss eine positive Ganzzahl sein.")
        self.is_rainy = bool(is_rainy)
        self.fickle_passenger = bool(fickle_passenger)
        self.max_steps = max_steps
        self.seed = seed
        self._environment = gym.make(
            "Taxi-v3",
            is_rainy=self.is_rainy,
            fickle_passenger=self.fickle_passenger,
            render_mode="rgb_array",
        )
        if seed is not None:
            self._environment.action_space.seed(seed)
        self.reset(seed=seed)

    @staticmethod
    def decode_observation(observation: int) -> Tuple[int, int, int, int]:
        if not 0 <= int(observation) < 500:
            raise ValueError("Observation muss zwischen 0 und 499 liegen.")
        value = int(observation)
        destination = value % 4
        value //= 4
        passenger = value % 5
        value //= 5
        taxi_column = value % 5
        taxi_row = value // 5
        return taxi_row, taxi_column, passenger, destination

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.seed = seed
        observation, info = self._environment.reset(seed=seed)
        self.observation = int(observation)
        self.action_mask = tuple(int(value) for value in info["action_mask"])
        self.step_count = 0
        self.total_reward = 0.0
        self.illegal_actions = 0
        self.done = False
        return self.observation

    def step(self, action: int) -> Transition:
        if self.done:
            raise RuntimeError("Episode beendet; vor dem nächsten Schritt reset() aufrufen.")
        if action not in ACTIONS:
            raise ValueError("Action muss zwischen 0 und 5 liegen.")
        observation = self.observation
        next_observation, reward, terminated, gym_truncated, info = self._environment.step(action)
        self.step_count += 1
        truncated = bool(gym_truncated or (self.step_count >= self.max_steps and not terminated))
        if float(reward) == -10.0:
            self.illegal_actions += 1
        self.observation = int(next_observation)
        self.action_mask = tuple(int(value) for value in info["action_mask"])
        self.total_reward += float(reward)
        self.done = bool(terminated or truncated)
        return Transition(
            observation, action, self.observation, float(reward), bool(terminated),
            truncated, self.done, self.step_count, self.action_mask,
        )

    def render_rgb(self) -> np.ndarray:
        return np.asarray(self._environment.render())

    def close(self) -> None:
        self._environment.close()


class BasePolicy:
    name = "Base"

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        exploration: str = "Epsilon konstant",
        use_action_mask: bool = False,
        seed: Optional[int] = 42,
    ) -> None:
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
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.exploration = exploration
        self.use_action_mask = bool(use_action_mask)
        self.seed = seed
        self._rng = random.Random(seed)
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
            self._rng.seed(seed)
        self.q_values = np.zeros((500, 6), dtype=float)
        self.visit_counts = np.zeros(500, dtype=int)
        self.epsilon = self.epsilon_start
        self.episodes_completed = 0

    def valid_actions(self, action_mask: Optional[Sequence[int]]) -> Tuple[int, ...]:
        if not self.use_action_mask or action_mask is None:
            return ACTIONS
        valid = tuple(index for index, allowed in enumerate(action_mask) if allowed)
        if not valid:
            raise ValueError("Action Mask enthält keine gültige Action.")
        return valid

    def best_actions(self, observation: int, action_mask: Optional[Sequence[int]] = None) -> List[int]:
        valid = self.valid_actions(action_mask)
        best = max(float(self.q_values[observation, action]) for action in valid)
        return [action for action in valid if np.isclose(self.q_values[observation, action], best)]

    def select_action(
        self,
        observation: int,
        training: bool = True,
        action_mask: Optional[Sequence[int]] = None,
    ) -> int:
        valid = self.valid_actions(action_mask)
        if training and self._rng.random() < self.epsilon:
            return self._rng.choice(valid)
        return self._rng.choice(self.best_actions(observation, action_mask))

    def expected_q(self, observation: int, action_mask: Optional[Sequence[int]]) -> float:
        valid = self.valid_actions(action_mask)
        best = self.best_actions(observation, action_mask)
        random_probability = self.epsilon / len(valid)
        greedy_probability = (1 - self.epsilon) / len(best)
        return float(sum(
            self.q_values[observation, action]
            * (random_probability + (greedy_probability if action in best else 0))
            for action in valid
        ))

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        next_action: Optional[int],
        next_action_mask: Optional[Sequence[int]],
        done: bool,
    ) -> None:
        raise NotImplementedError

    def end_episode(self) -> None:
        self.episodes_completed += 1
        if self.exploration == "Epsilon-Decay":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningPolicy(BasePolicy):
    name = "Q-Learning"

    def update(self, observation: int, action: int, reward: float, next_observation: int, next_action: Optional[int], next_action_mask: Optional[Sequence[int]], done: bool) -> None:
        future = 0.0 if done else max(
            float(self.q_values[next_observation, candidate])
            for candidate in self.valid_actions(next_action_mask)
        )
        target = reward + self.gamma * future
        self.q_values[observation, action] += self.alpha * (target - self.q_values[observation, action])
        self.visit_counts[observation] += 1


class SarsaPolicy(BasePolicy):
    name = "SARSA"

    def update(self, observation: int, action: int, reward: float, next_observation: int, next_action: Optional[int], next_action_mask: Optional[Sequence[int]], done: bool) -> None:
        if not done and next_action is None:
            raise ValueError("SARSA benötigt next_action.")
        future = 0.0 if done else float(self.q_values[next_observation, int(next_action)])
        target = reward + self.gamma * future
        self.q_values[observation, action] += self.alpha * (target - self.q_values[observation, action])
        self.visit_counts[observation] += 1


class ExpectedSarsaPolicy(BasePolicy):
    name = "Expected SARSA"

    def update(self, observation: int, action: int, reward: float, next_observation: int, next_action: Optional[int], next_action_mask: Optional[Sequence[int]], done: bool) -> None:
        future = 0.0 if done else self.expected_q(next_observation, next_action_mask)
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
    def __init__(self, environment: TaxiEnvironment, policy: BasePolicy) -> None:
        self.environment = environment
        self.policy = policy
        self.returns: List[float] = []
        self.successes: List[bool] = []
        self.episode_lengths: List[int] = []
        self.illegal_actions: List[int] = []
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
            action = self.policy.select_action(
                observation, training, self.environment.action_mask
            )
        transition = self.environment.step(action)
        next_action = None
        if not transition.done and isinstance(self.policy, SarsaPolicy):
            next_action = self.policy.select_action(
                transition.next_observation, training, transition.next_action_mask
            )
        if training:
            self.policy.update(
                observation, action, transition.reward, transition.next_observation,
                next_action, transition.next_action_mask, transition.done,
            )
        self._next_action = next_action
        self.current_transitions.append(transition)
        if transition.done:
            self._finish_episode(training)
        return transition

    def _finish_episode(self, training: bool) -> EpisodeResult:
        success = bool(self.current_transitions[-1].terminated)
        reason = "goal_reached" if success else "max_steps"
        result = EpisodeResult(
            self.policy.name,
            list(self.current_transitions),
            sum(item.reward for item in self.current_transitions),
            len(self.current_transitions),
            success,
            sum(item.reward == -10 for item in self.current_transitions),
            reason,
            0.0 if not training else self.policy.epsilon,
        )
        if training:
            self.returns.append(result.total_reward)
            self.successes.append(success)
            self.episode_lengths.append(result.steps)
            self.illegal_actions.append(result.illegal_actions)
            self.policy.end_episode()
        self.latest_result = result
        self.episode_active = False
        self._next_action = None
        return result

    def run_episode(self, training: bool = True, seed: Optional[int] = None) -> EpisodeResult:
        self.start_episode(training, seed)
        while self.episode_active:
            self.step(training)
        assert self.latest_result is not None
        return self.latest_result

    def moving_average_returns(self, window: int = 100) -> List[float]:
        return [mean(self.returns[max(0, i - window + 1):i + 1]) for i in range(len(self.returns))]

    def reset_training(self) -> None:
        self.policy.reset(self.policy.seed)
        self.returns.clear()
        self.successes.clear()
        self.episode_lengths.clear()
        self.illegal_actions.clear()
        self.current_transitions = []
        self.latest_result = None
        self.episode_active = False
        self._next_action = None
        self.environment.reset(seed=self.environment.seed)


class TrainingRunner:
    def __init__(self, agent: TabularAgent) -> None:
        self.agent = agent

    def run(self, episodes: int, should_stop: Optional[Callable[[], bool]] = None) -> List[EpisodeResult]:
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
