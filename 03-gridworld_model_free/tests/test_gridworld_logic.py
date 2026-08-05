import unittest

from gridworld_logic import (
    ACTIONS,
    DOWN,
    RIGHT,
    UP,
    Agent,
    ExpectedSarsaPolicy,
    GridWorld,
    MonteCarloPolicy,
    QLearningPolicy,
    SarsaPolicy,
    Transition,
    compare_policies,
    state_table_rows,
)


class GridWorldTests(unittest.TestCase):
    def test_default_shortest_path_fits_limit(self):
        environment = GridWorld()
        self.assertEqual(environment.shortest_path_length(), 8)

    def test_wall_and_blocked_cell_keep_state(self):
        environment = GridWorld()
        state, reward, done, _ = environment.step(DOWN)
        self.assertEqual(state, (0, 2))
        self.assertEqual(reward, -1)
        self.assertFalse(done)

    def test_goal_ends_episode_and_rejects_more_steps(self):
        environment = GridWorld(width=2, height=2, start=(0, 0), goal=(1, 0), blocked=(), max_steps=2)
        state, reward, done, info = environment.step(RIGHT)
        self.assertEqual(state, (1, 0))
        self.assertEqual(reward, 0)
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "goal_reached")
        with self.assertRaises(RuntimeError):
            environment.step(RIGHT)

    def test_max_steps_ends_episode(self):
        environment = GridWorld(width=2, height=2, start=(0, 0), goal=(1, 1), blocked=(), max_steps=2)
        environment.step(DOWN)
        _, _, done, info = environment.step(DOWN)
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "max_steps")

    def test_rejects_unreachable_and_too_short_limit(self):
        with self.assertRaises(ValueError):
            GridWorld(width=3, height=3, start=(0, 1), goal=(2, 1), blocked=((1, 0), (1, 1), (1, 2)), max_steps=10)
        with self.assertRaises(ValueError):
            GridWorld(width=3, height=3, start=(0, 0), goal=(2, 2), blocked=(), max_steps=3)

    def test_rejects_invalid_grid_data(self):
        invalid = (
            {"width": 1},
            {"start": (0, 0), "goal": (0, 0)},
            {"blocked": ((0, 2),)},
            {"blocked": ((1, 1), (1, 1))},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                GridWorld(**arguments)


class PolicyTests(unittest.TestCase):
    def test_q_learning_update(self):
        policy = QLearningPolicy(alpha=0.5, gamma=0.9, seed=1)
        policy.q[(1, 0)] = [1.0, 2.0, 3.0, 4.0]
        policy.update((0, 0), RIGHT, -1, (1, 0), None, False)
        self.assertAlmostEqual(policy.q[(0, 0)][RIGHT], 1.3)

    def test_terminal_update_has_no_future_value(self):
        policy = QLearningPolicy(alpha=0.5, gamma=0.9)
        policy.q[(1, 0)] = [100.0] * 4
        policy.update((0, 0), RIGHT, 0, (1, 0), None, True)
        self.assertEqual(policy.q[(0, 0)][RIGHT], 0.0)

    def test_sarsa_uses_given_next_action(self):
        policy = SarsaPolicy(alpha=1.0, gamma=0.5)
        policy.q[(1, 0)] = [1.0, 2.0, 3.0, 4.0]
        policy.update((0, 0), RIGHT, -1, (1, 0), DOWN, False)
        self.assertEqual(policy.q[(0, 0)][RIGHT], 0.0)

    def test_expected_sarsa_distributes_ties(self):
        policy = ExpectedSarsaPolicy(alpha=1.0, gamma=1.0, epsilon_start=0.2, epsilon_min=0.2, use_algorithm_defaults=False)
        policy.epsilon = 0.2
        policy.q[(1, 0)] = [4.0, 4.0, 0.0, 0.0]
        self.assertAlmostEqual(policy.expected_q((1, 0)), 3.6)

    def test_monte_carlo_every_visit_sample_average(self):
        policy = MonteCarloPolicy(gamma=1.0)
        trajectory = [
            Transition(1, 1, (0, 0), RIGHT, (1, 0), -1, False, None, policy.name),
            Transition(1, 2, (0, 0), RIGHT, (1, 0), 0, True, "goal_reached", policy.name),
        ]
        policy.end_episode(trajectory)
        self.assertEqual(policy.return_counts[((0, 0), RIGHT)], 2)
        self.assertAlmostEqual(policy.q[(0, 0)][RIGHT], -0.5)

    def test_algorithm_specific_epsilon(self):
        self.assertGreater(SarsaPolicy().epsilon_for_episode(1000), QLearningPolicy().epsilon_for_episode(1000))
        self.assertGreater(ExpectedSarsaPolicy().epsilon_for_episode(1000), QLearningPolicy().epsilon_for_episode(1000))


class AgentAndComparisonTests(unittest.TestCase):
    def test_state_tables_distinguish_unvisited_and_learned_values(self):
        environment = GridWorld(
            width=3, height=2, start=(0, 0), goal=(2, 0), blocked=((1, 1),), max_steps=4
        )
        policy = QLearningPolicy(seed=2)
        initial = state_table_rows(environment, policy)
        start = next(row for row in initial if (row["x"], row["y"]) == environment.start)
        goal = next(row for row in initial if (row["x"], row["y"]) == environment.goal)
        obstacle = next(row for row in initial if (row["x"], row["y"]) == (1, 1))
        self.assertIsNone(start["value"])
        self.assertEqual(start["status"], "Start – unbesucht")
        self.assertEqual(goal["value"], 0.0)
        self.assertEqual(goal["status"], "Ziel")
        self.assertIsNone(obstacle["value"])

        policy.q[environment.start][RIGHT] = -2.5
        policy.state_visits[environment.start] = 3
        learned = state_table_rows(environment, policy)
        start = next(row for row in learned if (row["x"], row["y"]) == environment.start)
        self.assertEqual(start["q_right"], -2.5)
        self.assertEqual(start["value"], 0.0)
        self.assertEqual(start["visits"], 3)
        self.assertEqual(start["status"], "Start – gelernt")

    def test_sarsa_cached_action_is_executed_next(self):
        environment = GridWorld(width=3, height=2, start=(0, 0), goal=(2, 0), blocked=(), max_steps=5)
        policy = SarsaPolicy(
            epsilon_start=0,
            epsilon_min=0,
            seed=3,
            use_algorithm_defaults=False,
        )
        agent = Agent(environment, policy)
        policy.q[(0, 0)][RIGHT] = 2
        policy.q[(1, 0)][RIGHT] = 3
        first = agent.step(training=True)
        second = agent.step(training=True)
        self.assertEqual(first.action, RIGHT)
        self.assertEqual(second.action, RIGHT)

    def test_greedy_evaluation_does_not_change_training(self):
        environment = GridWorld(width=2, height=2, start=(0, 0), goal=(1, 0), blocked=(), max_steps=2)
        policy = QLearningPolicy(seed=2)
        agent = Agent(environment, policy)
        policy.q[(0, 0)][RIGHT] = 2
        before = [row[:] for row in policy.q.values()]
        trajectory = agent.run_episode(training=False)
        self.assertEqual(trajectory[-1].termination_reason, "goal_reached")
        self.assertEqual(agent.episode_count, 0)
        self.assertEqual([row[:] for row in policy.q.values()], before)

    def test_greedy_evaluation_without_path_stops_at_policy_cycle(self):
        """Regression: a cyclic learned policy must not crash the evaluation."""
        environment = GridWorld(
            width=3,
            height=2,
            start=(0, 0),
            goal=(2, 0),
            blocked=(),
            max_steps=4,
        )
        policy = QLearningPolicy(seed=7)
        agent = Agent(environment, policy)
        policy.q[(0, 0)][UP] = 10.0
        before_q = [row[:] for row in policy.q.values()]

        trajectory = agent.run_episode(training=False)

        self.assertLess(len(trajectory), environment.max_steps)
        self.assertEqual(trajectory[-1].termination_reason, "policy_cycle")
        self.assertFalse(agent.episode_active)
        self.assertEqual(agent.episode_count, 0)
        self.assertEqual(agent.returns, [])
        self.assertEqual([row[:] for row in policy.q.values()], before_q)

    def test_comparison_is_reproducible_and_isolated(self):
        config = {"width": 3, "height": 2, "start": (0, 0), "goal": (2, 0), "blocked": (), "max_steps": 6}
        first = compare_policies(["SARSA", "Q-Learning"], config, episodes=15, repetitions=3, base_seed=9)
        second = compare_policies(["SARSA", "Q-Learning"], config, episodes=15, repetitions=3, base_seed=9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(len(result.mean_returns) == 15 for result in first))


if __name__ == "__main__":
    unittest.main()
