import unittest
from unittest.mock import Mock, patch

import numpy as np

from cliff_walking_logic import (
    ACTIONS, DOWN, RIGHT, UP, CliffWalkingEnvironment, ExpectedSarsaPolicy,
    QLearningPolicy, SarsaPolicy, TabularAgent, create_policy,
)


class EnvironmentTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "environment"):
            self.environment.close()

    def test_position_conversion(self):
        self.assertEqual(CliffWalkingEnvironment.position_to_observation((3, 0)), 36)
        self.assertEqual(CliffWalkingEnvironment.observation_to_position(47), (3, 11))
        with self.assertRaises(ValueError):
            CliffWalkingEnvironment.position_to_observation((4, 0))

    @patch("cliff_walking_logic.gym.make")
    def test_environment_uses_required_gymnasium_factory_call(self, make_mock):
        gym_environment = Mock()
        gym_environment.reset.return_value = (36, {})
        gym_environment.action_space = Mock()
        make_mock.return_value = gym_environment

        self.environment = CliffWalkingEnvironment(max_steps=10, seed=1)

        make_mock.assert_called_once_with("CliffWalking-v1", render_mode="rgb_array")

    def test_normal_step_has_minus_one_reward(self):
        self.environment = CliffWalkingEnvironment(max_steps=10, seed=1)
        result = self.environment.step(UP)
        self.assertEqual(result.reward, -1.0)
        self.assertEqual(result.next_observation, 24)
        self.assertFalse(result.done)

    def test_cliff_fall_returns_to_start_without_termination(self):
        self.environment = CliffWalkingEnvironment(max_steps=10, seed=1)
        result = self.environment.step(RIGHT)
        self.assertEqual(result.reward, -100.0)
        self.assertTrue(result.fell_into_cliff)
        self.assertEqual(result.next_observation, 36)
        self.assertFalse(result.done)

    def test_goal_terminates_episode(self):
        self.environment = CliffWalkingEnvironment(max_steps=30, seed=1)
        for action in [UP] + [RIGHT] * 11 + [DOWN]:
            result = self.environment.step(action)
        self.assertTrue(result.terminated)
        self.assertEqual(result.termination_reason, "goal_reached")

    def test_max_steps_truncates_episode(self):
        self.environment = CliffWalkingEnvironment(max_steps=2, seed=1)
        self.environment.step(UP)
        result = self.environment.step(UP)
        self.assertTrue(result.truncated)
        self.assertEqual(result.termination_reason, "max_steps")
        with self.assertRaises(RuntimeError):
            self.environment.step(UP)


class PolicyTests(unittest.TestCase):
    def test_q_learning_update(self):
        policy = QLearningPolicy(alpha=0.5, gamma=0.9, seed=1)
        policy.q_values[1, RIGHT] = 4.0
        policy.update(0, UP, -1.0, 1, None, False)
        self.assertAlmostEqual(policy.q_values[0, UP], 1.3)

    def test_sarsa_uses_given_next_action(self):
        policy = SarsaPolicy(alpha=1.0, gamma=0.5, seed=1)
        policy.q_values[1, RIGHT] = 8.0
        policy.q_values[1, DOWN] = 2.0
        policy.update(0, UP, -1.0, 1, DOWN, False)
        self.assertEqual(policy.q_values[0, UP], 0.0)

    def test_expected_sarsa_handles_ties(self):
        policy = ExpectedSarsaPolicy(epsilon_start=0.2, epsilon_min=0.0, seed=1)
        policy.q_values[1] = [4.0, 4.0, 0.0, 0.0]
        self.assertAlmostEqual(policy.expected_q(1), 3.6)

    def test_terminal_update_does_not_bootstrap(self):
        for policy_class in (QLearningPolicy, SarsaPolicy, ExpectedSarsaPolicy):
            policy = policy_class(alpha=1.0, gamma=0.99, seed=1)
            policy.q_values[1] = [100.0] * 4
            policy.update(0, UP, -1.0, 1, None, True)
            self.assertEqual(policy.q_values[0, UP], -1.0)

    def test_epsilon_decay_stops_at_minimum(self):
        policy = QLearningPolicy(epsilon_start=0.2, epsilon_min=0.1, epsilon_decay=0.5)
        for _ in range(5):
            policy.end_episode()
        self.assertEqual(policy.epsilon, 0.1)

    def test_factory_creates_all_methods(self):
        for name in ("Q-Learning", "SARSA", "Expected SARSA"):
            self.assertEqual(create_policy(name).name, name)


class AgentTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "environment"):
            self.environment.close()

    def test_evaluation_does_not_change_training(self):
        self.environment = CliffWalkingEnvironment(max_steps=30, seed=2)
        policy = QLearningPolicy(seed=2)
        agent = TabularAgent(self.environment, policy)
        policy.q_values[36, UP] = 10
        for observation in range(24, 35):
            policy.q_values[observation, RIGHT] = 10
        policy.q_values[35, DOWN] = 10
        before = policy.q_values.copy()
        result = agent.run_episode(training=False, seed=2)
        self.assertTrue(result.success)
        self.assertEqual(agent.episode_count, 0)
        np.testing.assert_array_equal(policy.q_values, before)

    def test_training_episode_collects_statistics(self):
        self.environment = CliffWalkingEnvironment(max_steps=5, seed=3)
        agent = TabularAgent(self.environment, QLearningPolicy(seed=3))
        result = agent.run_episode(training=True, seed=3)
        self.assertEqual(agent.episode_count, 1)
        self.assertEqual(agent.returns[-1], result.total_reward)
        self.assertEqual(agent.episode_lengths[-1], result.steps)

    def test_q_learning_can_learn_a_successful_policy(self):
        self.environment = CliffWalkingEnvironment(max_steps=200, seed=7)
        policy = QLearningPolicy(
            alpha=0.5, gamma=1.0, epsilon_start=1.0,
            epsilon_min=0.05, epsilon_decay=0.995, seed=7,
        )
        agent = TabularAgent(self.environment, policy)
        for _ in range(1200):
            agent.run_episode(training=True)
        evaluations = [agent.run_episode(training=False, seed=100 + index) for index in range(5)]
        self.assertTrue(all(result.success for result in evaluations))
        self.assertTrue(all(result.steps <= 20 for result in evaluations))


if __name__ == "__main__":
    unittest.main()
