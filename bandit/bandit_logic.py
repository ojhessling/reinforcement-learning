"""Core logic for the Multi-Armed Bandit demo.

This module intentionally has no GUI dependencies, so the algorithms can be
used and tested independently from Tkinter.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Deque, Dict, List, Optional, Sequence


class BanditEnvironment:
    """An environment containing Bernoulli bandits."""

    def __init__(
        self,
        probabilities: Sequence[float] = (0.2, 0.5, 0.8),
        seed: Optional[int] = None,
    ) -> None:
        if not probabilities:
            raise ValueError("At least one bandit is required.")
        if any(probability < 0 or probability > 1 for probability in probabilities):
            raise ValueError("Bandit probabilities must be between 0 and 1.")
        self.probabilities = tuple(float(value) for value in probabilities)
        self._rng = random.Random(seed)
        self.reset(seed)

    @property
    def number_of_bandits(self) -> int:
        return len(self.probabilities)

    def pull(self, action: int) -> int:
        if action < 0 or action >= self.number_of_bandits:
            raise IndexError("Bandit index is out of range.")
        reward = int(self._rng.random() < self.probabilities[action])
        self.pull_counts[action] += 1
        self.reward_sums[action] += reward
        self.last_rewards[action] = reward
        return reward

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
        self.pull_counts = [0] * self.number_of_bandits
        self.reward_sums = [0] * self.number_of_bandits
        self.last_rewards: List[Optional[int]] = [None] * self.number_of_bandits

    def statistics(self, action: int) -> Dict[str, object]:
        pulls = self.pull_counts[action]
        rewards = self.reward_sums[action]
        return {
            "pulls": pulls,
            "last_reward": self.last_rewards[action],
            "reward_sum": rewards,
            "success_rate": rewards / pulls if pulls else 0.0,
        }


class BaseAgent(ABC):
    """Common interface and accounting shared by all strategies."""

    def __init__(
        self,
        number_of_bandits: int,
        memory_limit: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        if number_of_bandits <= 0:
            raise ValueError("number_of_bandits must be positive.")
        if memory_limit < 0:
            raise ValueError("memory_limit cannot be negative.")
        self.number_of_bandits = number_of_bandits
        self.memory_limit = memory_limit
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        max_length = self.memory_limit or None
        self.action_rewards: List[Deque[int]] = [
            deque(maxlen=max_length) for _ in range(self.number_of_bandits)
        ]
        self.action_counts = [0] * self.number_of_bandits
        self.action_reward_sums = [0] * self.number_of_bandits
        self.total_pulls = 0
        self.cumulative_reward = 0
        self.reward_history: List[int] = []

    @abstractmethod
    def select_action(self) -> int:
        """Return the index of the next bandit to pull."""

    def update(self, action: int, reward: int) -> None:
        if action < 0 or action >= self.number_of_bandits:
            raise IndexError("Bandit index is out of range.")
        if reward not in (0, 1):
            raise ValueError("Reward must be 0 or 1.")
        self.action_counts[action] += 1
        self.action_reward_sums[action] += reward
        self.action_rewards[action].append(reward)
        self.total_pulls += 1
        self.cumulative_reward += reward
        self.reward_history.append(self.cumulative_reward)

    def estimated_values(self) -> List[float]:
        return [
            sum(rewards) / len(rewards) if rewards else 0.0
            for rewards in self.action_rewards
        ]

    def _random_best_action(self, values: Sequence[float]) -> int:
        best_value = max(values)
        candidates = [index for index, value in enumerate(values) if value == best_value]
        return self._rng.choice(candidates)


class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, number_of_bandits: int, epsilon: float = 0.1, **kwargs: object) -> None:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1.")
        self.epsilon = float(epsilon)
        super().__init__(number_of_bandits, **kwargs)

    def select_action(self) -> int:
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.number_of_bandits)
        return self._random_best_action(self.estimated_values())


class EpsilonDecayAgent(BaseAgent):
    def __init__(
        self,
        number_of_bandits: int,
        epsilon_start: float = 0.999,
        epsilon_min: float = 0.001,
        epsilon_decay: float = 0.05,
        **kwargs: object,
    ) -> None:
        if not 0 <= epsilon_min <= epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be greater than 0 and at most 1.")
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = self.epsilon_start
        super().__init__(number_of_bandits, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.epsilon = self.epsilon_start

    def select_action(self) -> int:
        if self._rng.random() < self.epsilon:
            action = self._rng.randrange(self.number_of_bandits)
        else:
            action = self._random_best_action(self.estimated_values())
        self.epsilon = max(self.epsilon_min, self.epsilon * (1 - self.epsilon_decay))
        return action


class ThompsonSamplingAgent(BaseAgent):
    def __init__(self, number_of_bandits: int, **kwargs: object) -> None:
        kwargs["memory_limit"] = 0
        super().__init__(number_of_bandits, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.alpha = [1] * self.number_of_bandits
        self.beta = [1] * self.number_of_bandits

    def select_action(self) -> int:
        samples = [
            self._rng.betavariate(self.alpha[index], self.beta[index])
            for index in range(self.number_of_bandits)
        ]
        return self._random_best_action(samples)

    def update(self, action: int, reward: int) -> None:
        super().update(action, reward)
        self.alpha[action] += reward
        self.beta[action] += 1 - reward

    def estimated_values(self) -> List[float]:
        return [
            alpha / (alpha + beta) for alpha, beta in zip(self.alpha, self.beta)
        ]


class UCBAgent(BaseAgent):
    def __init__(self, number_of_bandits: int, **kwargs: object) -> None:
        kwargs["memory_limit"] = 0
        super().__init__(number_of_bandits, **kwargs)

    def select_action(self) -> int:
        for action, count in enumerate(self.action_counts):
            if count == 0:
                return action
        values = self.estimated_values()
        scores = [
            value + math.sqrt(2 * math.log(self.total_pulls) / self.action_counts[action])
            for action, value in enumerate(values)
        ]
        return self._random_best_action(scores)


class SoftmaxAgent(BaseAgent):
    def __init__(
        self, number_of_bandits: int, temperature: float = 1.0, **kwargs: object
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")
        self.temperature = float(temperature)
        super().__init__(number_of_bandits, **kwargs)

    def select_action(self) -> int:
        scaled = [value / self.temperature for value in self.estimated_values()]
        maximum = max(scaled)
        weights = [math.exp(value - maximum) for value in scaled]
        return self._rng.choices(range(self.number_of_bandits), weights=weights, k=1)[0]


def create_agent(
    strategy: str,
    number_of_bandits: int,
    memory_limit: int = 0,
    seed: Optional[int] = None,
    **parameters: float,
) -> BaseAgent:
    """Create an agent from the user-facing strategy name."""

    common = {
        "number_of_bandits": number_of_bandits,
        "memory_limit": memory_limit,
        "seed": seed,
    }
    if strategy == "Epsilon Greedy":
        return EpsilonGreedyAgent(epsilon=parameters.get("epsilon", 0.1), **common)
    if strategy == "Epsilon Decay":
        return EpsilonDecayAgent(
            epsilon_start=parameters.get("epsilon_start", 0.999),
            epsilon_min=parameters.get("epsilon_min", 0.001),
            epsilon_decay=parameters.get("epsilon_decay", 0.05),
            **common,
        )
    if strategy == "Thompson Sampling":
        return ThompsonSamplingAgent(**common)
    if strategy == "UCB":
        return UCBAgent(**common)
    if strategy == "Boltzmann/Softmax":
        return SoftmaxAgent(temperature=parameters.get("temperature", 1.0), **common)
    raise ValueError("Unknown strategy: {}".format(strategy))


@dataclass
class StrategyComparisonResult:
    strategy: str
    mean_rewards: List[float]
    confidence_low: List[float]
    confidence_high: List[float]
    final_reward: float
    final_confidence_margin: float
    reward_per_pull: float
    best_bandit_share: float
    regret: float


def compare_strategies(
    strategies: Sequence[str],
    steps: int = 500,
    repetitions: int = 20,
    seed: Optional[int] = None,
    probabilities: Sequence[float] = (0.2, 0.5, 0.8),
) -> List[StrategyComparisonResult]:
    """Compare strategies under reproducible, shared random conditions.

    Each repetition uses one random value per time step. Every strategy sees
    those same values, while its chosen action determines the probability
    threshold. Results are averaged over repetitions.
    """

    if not strategies:
        raise ValueError("At least one strategy must be selected.")
    if steps <= 0 or repetitions <= 0:
        raise ValueError("steps and repetitions must be positive.")
    if not probabilities or any(value < 0 or value > 1 for value in probabilities):
        raise ValueError("Probabilities must be between 0 and 1.")

    base_seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
    histories: Dict[str, List[List[int]]] = {strategy: [] for strategy in strategies}
    best_shares: Dict[str, List[float]] = {strategy: [] for strategy in strategies}

    for repetition in range(repetitions):
        reward_rng = random.Random(base_seed + repetition * 104729)
        shared_random_values = [reward_rng.random() for _ in range(steps)]
        for strategy in strategies:
            stable_strategy_id = sum(
                (index + 1) * ord(character)
                for index, character in enumerate(strategy)
            )
            agent_seed = base_seed + repetition * 104729 + stable_strategy_id
            agent = create_agent(strategy, len(probabilities), seed=agent_seed)
            for step, random_value in enumerate(shared_random_values):
                action = agent.select_action()
                reward = int(random_value < probabilities[action])
                agent.update(action, reward)
            histories[strategy].append(agent.reward_history[:])
            best_shares[strategy].append(agent.action_counts[-1] / steps)

    results = []
    optimal_probability = max(probabilities)
    for strategy in strategies:
        strategy_histories = histories[strategy]
        means = []
        lows = []
        highs = []
        for step in range(steps):
            values = [history[step] for history in strategy_histories]
            average = mean(values)
            margin = 1.96 * stdev(values) / math.sqrt(repetitions) if repetitions > 1 else 0.0
            means.append(average)
            lows.append(max(0.0, average - margin))
            highs.append(average + margin)
        final_values = [history[-1] for history in strategy_histories]
        final_average = mean(final_values)
        final_margin = (
            1.96 * stdev(final_values) / math.sqrt(repetitions)
            if repetitions > 1
            else 0.0
        )
        results.append(
            StrategyComparisonResult(
                strategy=strategy,
                mean_rewards=means,
                confidence_low=lows,
                confidence_high=highs,
                final_reward=final_average,
                final_confidence_margin=final_margin,
                reward_per_pull=final_average / steps,
                best_bandit_share=mean(best_shares[strategy]),
                regret=steps * optimal_probability - final_average,
            )
        )
    return sorted(results, key=lambda result: result.final_reward, reverse=True)
