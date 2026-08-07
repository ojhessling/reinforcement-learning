import unittest
from unittest.mock import Mock, patch

import numpy as np

from frozenlake_logic import (
    DOWN,
    LEFT,
    RIGHT,
    ExpectedSarsaPolicy,
    FrozenLakeEnvironment,
    QLearningPolicy,
    SarsaPolicy,
    TabularAgent,
    TrainingRunner,
    EvaluationRunner,
    create_policy,
)


class EnvironmentTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "environment"):
            self.environment.close()

    @patch("frozenlake_logic.gym.make")
    def test_environment_configuration(self, make_mock):
        wrapped = Mock()
        wrapped.reset.return_value = (0, {})
        wrapped.action_space = Mock()
        make_mock.return_value = wrapped
        self.environment = FrozenLakeEnvironment("8x8", False, 200, 7)
        make_mock.assert_called_once_with(
            "FrozenLake-v1", map_name="8x8", is_slippery=False,
            render_mode="rgb_array",
        )
        self.assertEqual(self.environment.state_count, 64)

    def test_deterministic_transition_and_hole(self):
        self.environment = FrozenLakeEnvironment("4x4", False, 20, 1)
        first = self.environment.step(RIGHT)
        self.assertEqual(first.next_observation, 1)
        self.assertEqual(first.reward, 0)
        hole = self.environment.step(DOWN)
        self.assertEqual(hole.next_observation, 5)
        self.assertTrue(hole.terminated)
        self.assertEqual(hole.reward, 0)

    def test_goal_reward_is_one(self):
        self.environment = FrozenLakeEnvironment("4x4", False, 20, 1)
        result = None
        for action in (DOWN, DOWN, RIGHT, DOWN, RIGHT, RIGHT):
            result = self.environment.step(action)
        self.assertIsNotNone(result)
        self.assertTrue(result.terminated)
        self.assertEqual(result.reward, 1)

    def test_max_steps_truncates(self):
        self.environment = FrozenLakeEnvironment("4x4", False, 1, 1)
        result = self.environment.step(LEFT)
        self.assertTrue(result.truncated)
        self.assertTrue(result.done)

    def test_position_conversion_depends_on_map(self):
        self.environment = FrozenLakeEnvironment("8x8", False, 20, 1)
        self.assertEqual(self.environment.observation_to_position(63), (7, 7))
        with self.assertRaises(ValueError):
            self.environment.observation_to_position(64)


class PolicyTests(unittest.TestCase):
    def test_q_learning_update(self):
        policy = QLearningPolicy(16, alpha=0.5, gamma=0.9, seed=1)
        policy.q_values[1, RIGHT] = 4
        policy.update(0, DOWN, 0, 1, None, False)
        self.assertAlmostEqual(policy.q_values[0, DOWN], 1.8)

    def test_sarsa_uses_next_action(self):
        policy = SarsaPolicy(16, alpha=1, gamma=0.5, seed=1)
        policy.q_values[1, LEFT] = 8
        policy.update(0, DOWN, 0, 1, LEFT, False)
        self.assertEqual(policy.q_values[0, DOWN], 4)

    def test_expected_sarsa_handles_ties(self):
        policy = ExpectedSarsaPolicy(16, epsilon=0.2, seed=1)
        policy.q_values[1] = [4, 4, 0, 0]
        self.assertAlmostEqual(policy.expected_q(1), 3.6)

    def test_done_update_does_not_bootstrap(self):
        for policy_class in (QLearningPolicy, SarsaPolicy, ExpectedSarsaPolicy):
            policy = policy_class(16, alpha=1, gamma=0.99, seed=1)
            policy.q_values[1] = [10] * 4
            policy.update(0, DOWN, 1, 1, None, True)
            self.assertEqual(policy.q_values[0, DOWN], 1)

    def test_constant_and_decay_exploration(self):
        constant = QLearningPolicy(16, epsilon=0.2, exploration="Epsilon konstant")
        decay = QLearningPolicy(
            16, epsilon=0.2, epsilon_min=0.1, epsilon_decay=0.5,
            exploration="Epsilon-Decay",
        )
        constant.end_episode()
        decay.end_episode()
        self.assertEqual(constant.epsilon, 0.2)
        self.assertEqual(decay.epsilon, 0.1)

    def test_factory_creates_all_methods(self):
        for name in ("Q-Learning", "SARSA", "Expected SARSA"):
            self.assertEqual(create_policy(name, state_count=16).name, name)


class RunnerTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "environment"):
            self.environment.close()

    def test_evaluation_does_not_change_training(self):
        self.environment = FrozenLakeEnvironment("4x4", False, 20, 2)
        policy = QLearningPolicy(16, seed=2)
        agent = TabularAgent(self.environment, policy)
        path = {0: DOWN, 4: DOWN, 8: RIGHT, 9: DOWN, 13: RIGHT, 14: RIGHT}
        for state, action in path.items():
            policy.q_values[state, action] = 10
        before = policy.q_values.copy()
        result = EvaluationRunner(agent).run(1)[0]
        self.assertTrue(result.success)
        self.assertEqual(agent.episode_count, 0)
        np.testing.assert_array_equal(before, policy.q_values)

    def test_training_runner_collects_results(self):
        self.environment = FrozenLakeEnvironment("4x4", False, 20, 3)
        agent = TabularAgent(self.environment, QLearningPolicy(16, seed=3))
        results = TrainingRunner(agent).run(3)
        self.assertEqual(len(results), 3)
        self.assertEqual(agent.episode_count, 3)

    def test_q_learning_learns_deterministic_map(self):
        self.environment = FrozenLakeEnvironment("4x4", False, 100, 7)
        policy = QLearningPolicy(
            16, alpha=0.5, gamma=0.99, epsilon=1.0,
            epsilon_min=0.02, epsilon_decay=0.995,
            exploration="Epsilon-Decay", seed=7,
        )
        agent = TabularAgent(self.environment, policy)
        TrainingRunner(agent).run(2500)
        results = EvaluationRunner(agent).run(10)
        self.assertTrue(all(result.success for result in results))


if __name__ == "__main__":
    unittest.main()
