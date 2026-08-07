import unittest
from unittest.mock import Mock, patch

import numpy as np

from taxi_logic import (
    DROPOFF, EAST, NORTH, PICKUP, SOUTH, EvaluationRunner,
    ExpectedSarsaPolicy, QLearningPolicy, SarsaPolicy, TabularAgent,
    TaxiEnvironment, TrainingRunner, create_policy,
)


class EnvironmentTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "environment"):
            self.environment.close()

    @patch("taxi_logic.gym.make")
    def test_environment_configuration(self, make_mock):
        wrapped = Mock()
        wrapped.reset.return_value = (123, {"action_mask": np.ones(6, dtype=np.int8)})
        wrapped.action_space = Mock()
        make_mock.return_value = wrapped
        self.environment = TaxiEnvironment(True, True, 200, 7)
        make_mock.assert_called_once_with(
            "Taxi-v3", is_rainy=True, fickle_passenger=True,
            render_mode="rgb_array",
        )

    def test_decode_observation(self):
        self.assertEqual(TaxiEnvironment.decode_observation(0), (0, 0, 0, 0))
        self.assertEqual(TaxiEnvironment.decode_observation(499), (4, 4, 4, 3))
        with self.assertRaises(ValueError):
            TaxiEnvironment.decode_observation(500)

    def test_illegal_pickup_has_minus_ten_reward(self):
        self.environment = TaxiEnvironment(False, False, 200, 1)
        result = self.environment.step(PICKUP)
        if result.reward != -10:
            result = self.environment.step(DROPOFF)
        self.assertEqual(result.reward, -10)
        self.assertEqual(self.environment.illegal_actions, 1)

    def test_max_steps_truncates(self):
        self.environment = TaxiEnvironment(False, False, 1, 1)
        result = self.environment.step(SOUTH)
        self.assertTrue(result.truncated)
        self.assertTrue(result.done)

    def test_reset_provides_action_mask(self):
        self.environment = TaxiEnvironment(False, False, 200, 2)
        self.assertEqual(len(self.environment.action_mask), 6)
        self.assertTrue(any(self.environment.action_mask))


class PolicyTests(unittest.TestCase):
    def test_q_learning_update(self):
        policy = QLearningPolicy(alpha=0.5, gamma=0.9, seed=1)
        policy.q_values[1, EAST] = 4
        policy.update(0, SOUTH, -1, 1, None, None, False)
        self.assertAlmostEqual(policy.q_values[0, SOUTH], 1.3)

    def test_sarsa_uses_given_next_action(self):
        policy = SarsaPolicy(alpha=1, gamma=0.5, seed=1)
        policy.q_values[1, NORTH] = 8
        policy.update(0, SOUTH, -1, 1, NORTH, None, False)
        self.assertEqual(policy.q_values[0, SOUTH], 3)

    def test_expected_sarsa_handles_ties(self):
        policy = ExpectedSarsaPolicy(epsilon=0.2, seed=1)
        policy.q_values[1] = [4, 4, 0, 0, 0, 0]
        self.assertAlmostEqual(policy.expected_q(1, None), 3.4666666667)

    def test_terminal_update_does_not_bootstrap(self):
        for policy_class in (QLearningPolicy, SarsaPolicy, ExpectedSarsaPolicy):
            policy = policy_class(alpha=1, gamma=0.99, seed=1)
            policy.q_values[1] = [100] * 6
            policy.update(0, SOUTH, 20, 1, None, None, True)
            self.assertEqual(policy.q_values[0, SOUTH], 20)

    def test_action_mask_limits_exploration_and_greedy_actions(self):
        mask = (0, 1, 0, 0, 1, 0)
        policy = QLearningPolicy(epsilon=1, use_action_mask=True, seed=1)
        selected = {policy.select_action(0, True, mask) for _ in range(100)}
        self.assertEqual(selected, {NORTH, PICKUP})
        policy.q_values[1, DROPOFF] = 100
        policy.q_values[1, NORTH] = 5
        self.assertEqual(policy.best_actions(1, mask), [NORTH])

    def test_q_learning_bootstrap_respects_action_mask(self):
        policy = QLearningPolicy(alpha=1, gamma=1, use_action_mask=True)
        policy.q_values[1] = [100, 5, 0, 0, 0, 0]
        policy.update(0, SOUTH, 0, 1, None, (0, 1, 0, 0, 0, 0), False)
        self.assertEqual(policy.q_values[0, SOUTH], 5)

    def test_expected_sarsa_ignores_masked_actions(self):
        policy = ExpectedSarsaPolicy(epsilon=0.2, use_action_mask=True)
        policy.q_values[1] = [4, 0, 100, 100, 100, 100]
        self.assertAlmostEqual(policy.expected_q(1, (1, 1, 0, 0, 0, 0)), 3.6)

    def test_constant_and_decay_exploration(self):
        constant = QLearningPolicy(epsilon=0.2, exploration="Epsilon konstant")
        decay = QLearningPolicy(
            epsilon=0.2, epsilon_min=0.1, epsilon_decay=0.5,
            exploration="Epsilon-Decay",
        )
        constant.end_episode()
        decay.end_episode()
        self.assertEqual(constant.epsilon, 0.2)
        self.assertEqual(decay.epsilon, 0.1)

    def test_factory_creates_all_methods(self):
        for name in ("Q-Learning", "SARSA", "Expected SARSA"):
            self.assertEqual(create_policy(name).name, name)


class RunnerTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "environment"):
            self.environment.close()

    def test_evaluation_does_not_change_learning_state(self):
        self.environment = TaxiEnvironment(False, False, 20, 3)
        policy = QLearningPolicy(use_action_mask=True, seed=3)
        agent = TabularAgent(self.environment, policy)
        before = policy.q_values.copy()
        EvaluationRunner(agent).run(1)
        self.assertEqual(agent.episode_count, 0)
        np.testing.assert_array_equal(before, policy.q_values)

    def test_training_runner_collects_results(self):
        self.environment = TaxiEnvironment(False, False, 20, 4)
        agent = TabularAgent(self.environment, QLearningPolicy(seed=4))
        results = TrainingRunner(agent).run(3)
        self.assertEqual(len(results), 3)
        self.assertEqual(agent.episode_count, 3)

    def test_all_methods_learn_with_action_mask(self):
        for policy_class in (QLearningPolicy, SarsaPolicy, ExpectedSarsaPolicy):
            with self.subTest(method=policy_class.name):
                environment = TaxiEnvironment(False, False, 200, 7)
                try:
                    policy = policy_class(
                        alpha=0.5, gamma=0.99, epsilon=1,
                        epsilon_min=0.02, epsilon_decay=0.995,
                        exploration="Epsilon-Decay", use_action_mask=True, seed=7,
                    )
                    agent = TabularAgent(environment, policy)
                    TrainingRunner(agent).run(3000)
                    results = EvaluationRunner(agent).run(20)
                    self.assertGreaterEqual(sum(result.success for result in results), 18)
                finally:
                    environment.close()


if __name__ == "__main__":
    unittest.main()
