"""Tabular reinforcement-learning logic for Gymnasium CliffWalking."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple, Type

# Gymnasium uses Pygame to generate rgb_array frames. The dummy video driver
# keeps that renderer offscreen and avoids a second macOS GUI event loop next
# to Tkinter.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np

UP, RIGHT, DOWN, LEFT = range(4)
ACTIONS = (UP, RIGHT, DOWN, LEFT)
ACTION_NAMES = {UP: "Up", RIGHT: "Right", DOWN: "Down", LEFT: "Left"}
ACTION_ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}


@dataclass(frozen=True)
class StepResult:
    observation: int
    action: int
    next_observation: int
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    fell_into_cliff: bool
    termination_reason: Optional[str]
    step: int


@dataclass(frozen=True)
class EpisodeResult:
    method: str
    episode: int
    transitions: List[StepResult]
    total_reward: float
    steps: int
    cliff_falls: int
    success: bool
    termination_reason: str
    epsilon: float
    seed: Optional[int]


class CliffWalkingEnvironment:
    """Small adapter around Gymnasium's CliffWalking-v1 environment."""

    rows = 4
    columns = 12
    start_position = (3, 0)
    goal_position = (3, 11)
    cliff_positions = tuple((3, column) for column in range(1, 11))

    def __init__(
        self,
        max_steps: int = 500,
        seed: Optional[int] = 42,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("Max. Schritte muss eine positive Ganzzahl sein.")
        self.max_steps = max_steps
        self.seed = seed
        self._environment = gym.make("CliffWalking-v1", render_mode="rgb_array")
        if seed is not None:
            self._environment.action_space.seed(seed)
        self.reset(seed=seed)

    @staticmethod
    def observation_to_position(observation: int) -> Tuple[int, int]:
        if not 0 <= int(observation) < 48:
            raise ValueError("Observation muss zwischen 0 und 47 liegen.")
        return divmod(int(observation), 12)

    @staticmethod
    def position_to_observation(position: Tuple[int, int]) -> int:
        row, column = position
        if not 0 <= row < 4 or not 0 <= column < 12:
            raise ValueError("Position liegt außerhalb des 4-x-12-Grids.")
        return row * 12 + column

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.seed = seed
        observation, _ = self._environment.reset(seed=seed)
        self.observation = int(observation)
        self.step_count = 0
        self.total_reward = 0.0
        self.cliff_falls = 0
        self.done = False
        return self.observation

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Die Episode ist beendet. Vor dem nächsten Schritt reset() aufrufen.")
        if action not in ACTIONS:
            raise ValueError("Action muss 0, 1, 2 oder 3 sein.")
        observation = self.observation
        next_observation, reward, terminated, gym_truncated, _ = self._environment.step(action)
        self.step_count += 1
        max_steps_reached = self.step_count >= self.max_steps and not terminated
        truncated = bool(gym_truncated or max_steps_reached)
        fell_into_cliff = float(reward) == -100.0
        if fell_into_cliff:
            self.cliff_falls += 1
        self.observation = int(next_observation)
        self.total_reward += float(reward)
        self.done = bool(terminated or truncated)
        reason: Optional[str] = None
        if terminated:
            reason = "goal_reached"
        elif truncated:
            reason = "max_steps"
        return StepResult(
            observation, action, self.observation, float(reward), bool(terminated),
            truncated, self.done, fell_into_cliff, reason, self.step_count,
        )

    def close(self) -> None:
        self._environment.close()

    def render_rgb(self) -> np.ndarray:
        """Return Gymnasium's official CliffWalking visualization."""
        return np.asarray(self._environment.render())

    def configuration(self) -> Dict[str, object]:
        return {
            "max_steps": self.max_steps,
            "seed": self.seed,
        }


class BaseTabularPolicy:
    name = "Base"

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = 42,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("Alpha muss größer als 0 und höchstens 1 sein.")
        if not 0 <= gamma <= 1:
            raise ValueError("Gamma muss zwischen 0 und 1 liegen.")
        if not 0 <= epsilon_min <= epsilon_start <= 1:
            raise ValueError("Es muss 0 <= Epsilon Min <= Epsilon Start <= 1 gelten.")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("Epsilon Decay muss größer als 0 und höchstens 1 sein.")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.q_values = np.zeros((48, 4), dtype=float)
        self.visit_counts = np.zeros(48, dtype=int)
        self.epsilon = self.epsilon_start
        self.episodes_completed = 0

    def best_actions(self, observation: int) -> List[int]:
        values = self.q_values[int(observation)]
        best = float(np.max(values))
        return [action for action, value in enumerate(values) if abs(float(value) - best) <= 1e-10]

    def select_action(self, observation: int, training: bool = True) -> int:
        if training and self._rng.random() < self.epsilon:
            return self._rng.choice(ACTIONS)
        return self._rng.choice(self.best_actions(observation))

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

    def expected_q(self, observation: int) -> float:
        values = self.q_values[int(observation)]
        best = self.best_actions(observation)
        probability_random = self.epsilon / len(ACTIONS)
        probability_best = (1.0 - self.epsilon) / len(best)
        return float(sum(
            value * (probability_random + (probability_best if action in best else 0.0))
            for action, value in enumerate(values)
        ))

    def end_episode(self) -> None:
        self.episodes_completed += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class QLearningPolicy(BaseTabularPolicy):
    name = "Q-Learning"

    def update(self, observation: int, action: int, reward: float, next_observation: int, next_action: Optional[int], done: bool) -> None:
        future = 0.0 if done else float(np.max(self.q_values[next_observation]))
        target = reward + self.gamma * future
        self.q_values[observation, action] += self.alpha * (target - self.q_values[observation, action])
        self.visit_counts[observation] += 1


class SarsaPolicy(BaseTabularPolicy):
    name = "SARSA"

    def update(self, observation: int, action: int, reward: float, next_observation: int, next_action: Optional[int], done: bool) -> None:
        if not done and next_action is None:
            raise ValueError("SARSA benötigt für einen nicht terminalen Schritt next_action.")
        future = 0.0 if done else float(self.q_values[next_observation, int(next_action)])
        target = reward + self.gamma * future
        self.q_values[observation, action] += self.alpha * (target - self.q_values[observation, action])
        self.visit_counts[observation] += 1


class ExpectedSarsaPolicy(BaseTabularPolicy):
    name = "Expected SARSA"

    def update(self, observation: int, action: int, reward: float, next_observation: int, next_action: Optional[int], done: bool) -> None:
        future = 0.0 if done else self.expected_q(next_observation)
        target = reward + self.gamma * future
        self.q_values[observation, action] += self.alpha * (target - self.q_values[observation, action])
        self.visit_counts[observation] += 1


POLICY_CLASSES: Dict[str, Type[BaseTabularPolicy]] = {
    QLearningPolicy.name: QLearningPolicy,
    SarsaPolicy.name: SarsaPolicy,
    ExpectedSarsaPolicy.name: ExpectedSarsaPolicy,
}


def create_policy(name: str, **parameters: object) -> BaseTabularPolicy:
    try:
        return POLICY_CLASSES[name](**parameters)
    except KeyError as error:
        raise ValueError(f"Unbekannte Methode: {name}") from error


class TabularAgent:
    def __init__(self, environment: CliffWalkingEnvironment, policy: BaseTabularPolicy) -> None:
        self.environment = environment
        self.policy = policy
        self.returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.cliff_falls: List[int] = []
        self.successes: List[bool] = []
        self.current_transitions: List[StepResult] = []
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

    def step(self, training: Optional[bool] = None) -> StepResult:
        if training is None:
            training = self.training
        if not self.episode_active:
            self.start_episode(training=training)
        observation = self.environment.observation
        action = self._next_action
        if action is None:
            action = self.policy.select_action(observation, training=training)
        transition = self.environment.step(action)
        next_action: Optional[int] = None
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
        total_reward = sum(item.reward for item in self.current_transitions)
        cliff_falls = sum(item.fell_into_cliff for item in self.current_transitions)
        reason = self.current_transitions[-1].termination_reason or "unknown"
        result = EpisodeResult(
            self.policy.name,
            self.episode_count + (1 if training else 0),
            list(self.current_transitions),
            total_reward,
            len(self.current_transitions),
            cliff_falls,
            reason == "goal_reached",
            reason,
            0.0 if not training else self.policy.epsilon,
            self.environment.seed,
        )
        if training:
            self.returns.append(total_reward)
            self.episode_lengths.append(result.steps)
            self.cliff_falls.append(cliff_falls)
            self.successes.append(result.success)
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

    def moving_average_returns(self, window: int = 20) -> List[float]:
        return [mean(self.returns[max(0, index - window + 1):index + 1]) for index in range(len(self.returns))]

    def reset_training(self) -> None:
        self.policy.reset()
        self.returns.clear()
        self.episode_lengths.clear()
        self.cliff_falls.clear()
        self.successes.clear()
        self.current_transitions = []
        self.latest_result = None
        self.episode_active = False
        self.environment.reset(seed=self.environment.seed)
