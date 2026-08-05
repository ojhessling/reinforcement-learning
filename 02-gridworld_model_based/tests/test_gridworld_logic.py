"""Tests for the model-based Gridworld logic."""

import unittest

from gridworld_logic import (
    ACTIONS, DOWN, LEFT, RIGHT, UP, ComparisonRunner, GridWorld,
    QValueIteration, RolloutAgent, ValueIteration, q_table_rows,
    value_table_rows,
)


class GridWorldTests(unittest.TestCase):
    def test_default_grid_is_reachable(self):
        environment = GridWorld()
        self.assertEqual(environment.shortest_path_length(), 8)
        self.assertEqual(environment.start, (2, 0))

    def test_action_mapping_uses_row_column(self):
        environment = GridWorld(obstacles=())
        self.assertEqual(environment.transition((1, 1), UP)[0], (0, 1))
        self.assertEqual(environment.transition((1, 1), DOWN)[0], (2, 1))
        self.assertEqual(environment.transition((1, 1), LEFT)[0], (1, 0))
        self.assertEqual(environment.transition((1, 1), RIGHT)[0], (1, 2))

    def test_wall_and_obstacle_keep_state(self):
        environment = GridWorld()
        self.assertEqual(environment.transition((0, 0), UP), ((0, 0), -1.0, False))
        self.assertEqual(environment.transition((1, 1), RIGHT), ((1, 1), -1.0, False))

    def test_goal_transition_is_terminal_and_free(self):
        environment = GridWorld()
        self.assertEqual(environment.transition((2, 3), RIGHT), ((2, 4), 0.0, True))
        for action in ACTIONS:
            self.assertEqual(environment.transition(environment.goal, action), (environment.goal, 0.0, True))

    def test_transition_is_pure(self):
        environment = GridWorld()
        before = (environment.current_state, environment.step_count)
        environment.transition((0, 0), RIGHT)
        self.assertEqual((environment.current_state, environment.step_count), before)

    def test_step_refuses_action_after_done(self):
        environment = GridWorld(rows=2, columns=2, start=(0, 0), goal=(0, 1), obstacles=(), max_steps=2)
        environment.step(RIGHT)
        with self.assertRaises(RuntimeError):
            environment.step(RIGHT)

    def test_max_steps_terminates_rollout(self):
        environment = GridWorld(rows=2, columns=2, start=(0, 0), goal=(1, 1), obstacles=(), max_steps=2)
        environment.step(UP)
        _, _, done, info = environment.step(UP)
        self.assertTrue(done)
        self.assertEqual(info["termination_reason"], "max_steps")

    def test_validation_rejects_invalid_grids(self):
        invalid = (
            dict(rows=1), dict(columns=11), dict(start=(2, 0), goal=(2, 0)),
            dict(obstacles=((1, 2), (1, 2))),
            dict(obstacles=((0, 2), (1, 2), (2, 2))),
            dict(max_steps=2),
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                GridWorld(**parameters)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.environment = GridWorld()

    def test_value_iteration_uses_synchronous_sweep(self):
        planner = ValueIteration(self.environment)
        result = planner.single_sweep()
        self.assertEqual(result.iteration, 1)
        self.assertEqual(planner.get_value(self.environment.start), -1.0)
        self.assertEqual(planner.get_value((2, 3)), 0.0)

    def test_regression_displayed_start_value_changes_after_sweep(self):
        """Regression: the value source used by GUI/table must not stay zero."""
        for planner_class in (ValueIteration, QValueIteration):
            with self.subTest(planner=planner_class.__name__):
                planner = planner_class(self.environment)
                before = next(
                    row for row in value_table_rows(planner)
                    if (row["row"], row["column"]) == self.environment.start
                )
                self.assertEqual(before["value"], "0.000000")

                result = planner.single_sweep()
                after = next(
                    row for row in value_table_rows(planner)
                    if (row["row"], row["column"]) == self.environment.start
                )
                self.assertEqual(result.start_value, -1.0)
                self.assertEqual(after["value"], "-1.000000")

    def test_q_value_iteration_uses_synchronous_sweep(self):
        planner = QValueIteration(self.environment)
        planner.single_sweep()
        self.assertEqual(planner.get_q_value(self.environment.start, RIGHT), -1.0)
        self.assertEqual(planner.get_q_value((2, 3), RIGHT), 0.0)

    def test_planners_converge_to_same_values_and_actions(self):
        value = ValueIteration(self.environment, gamma=0.9, seed=7)
        q_value = QValueIteration(self.environment, gamma=0.9, seed=7)
        value.run(1000); q_value.run(1000)
        self.assertTrue(value.converged and q_value.converged)
        for state in self.environment.states:
            self.assertAlmostEqual(value.get_value(state), q_value.get_value(state), places=8)
            self.assertEqual(value.get_best_actions(state), q_value.get_best_actions(state))

    def test_terminal_value_remains_zero(self):
        for planner_class in (ValueIteration, QValueIteration):
            planner = planner_class(self.environment)
            planner.run(100)
            self.assertEqual(planner.get_value(self.environment.goal), 0.0)

    def test_run_respects_iteration_limit_and_cancellation(self):
        planner = ValueIteration(self.environment, tolerance=1e-30)
        self.assertEqual(len(planner.run(2)), 2)
        self.assertEqual(planner.run(20, lambda: True), [])

    def test_sweep_after_convergence_is_no_op(self):
        planner = ValueIteration(self.environment)
        planner.run(1000)
        iteration = planner.iteration
        result = planner.single_sweep()
        self.assertEqual(result.iteration, iteration)
        self.assertEqual(planner.iteration, iteration)

    def test_reset_clears_tables_and_history(self):
        planner = QValueIteration(self.environment)
        planner.run(3); planner.reset()
        self.assertEqual(planner.iteration, 0)
        self.assertEqual(planner.history, [])
        self.assertTrue(all(planner.get_q_value(state, action) == 0 for state in self.environment.states for action in ACTIONS))

    def test_policy_is_unknown_before_first_sweep(self):
        planner = ValueIteration(self.environment)
        self.assertEqual(planner.get_best_actions(self.environment.start), [])

    def test_tie_tolerance_includes_nearly_equal_actions(self):
        environment = GridWorld(rows=2, columns=2, start=(0, 0), goal=(1, 1), obstacles=(), max_steps=3)
        planner = QValueIteration(environment, tolerance=0.001)
        planner.iteration = 1
        planner.q_values[(0, 0)] = [0.0, 1.0, 0.0, 0.9995]
        self.assertEqual(planner.get_best_actions((0, 0)), [DOWN, RIGHT])


class RolloutAndComparisonTests(unittest.TestCase):
    def setUp(self):
        self.environment = GridWorld()
        self.planner = ValueIteration(self.environment, seed=10)
        self.planner.run(1000)

    def test_greedy_rollout_reaches_goal_on_shortest_path(self):
        result = RolloutAgent(self.environment, self.planner, seed=10).run_greedy_episode()
        self.assertTrue(result.success)
        self.assertEqual(result.steps, self.environment.shortest_path_length())
        self.assertEqual(result.total_reward, -(result.steps - 1))

    def test_rollout_does_not_change_planner(self):
        iteration = self.planner.iteration
        values = self.planner.values.copy()
        RolloutAgent(self.environment, self.planner, seed=5).run_episode(0.5)
        self.assertEqual(self.planner.iteration, iteration)
        self.assertEqual(self.planner.values, values)

    def test_run_steps_can_continue_same_rollout(self):
        agent = RolloutAgent(self.environment, self.planner, seed=1)
        first = agent.run_steps(0.0, 2)
        second = agent.run_steps(0.0, 2)
        self.assertEqual(first.steps, 2)
        self.assertEqual(second.steps, 4)

    def test_seed_makes_rollouts_reproducible(self):
        first = RolloutAgent(self.environment.clone(), self.planner, 88).run_episode(0.7)
        second = RolloutAgent(self.environment.clone(), self.planner, 88).run_episode(0.7)
        self.assertEqual([x.action for x in first.transitions], [x.action for x in second.transitions])

    def test_comparison_is_isolated_and_complete(self):
        iteration = self.planner.iteration
        results = ComparisonRunner(self.environment, .9, .0001, 1000, rollout_count=3, seed=4).run()
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.converged for result in results))
        self.assertTrue(all(len(result.rollout_results) == 3 for result in results))
        self.assertEqual(self.planner.iteration, iteration)

    def test_comparison_cancellation(self):
        results = ComparisonRunner(self.environment, .9, .0001, 10, seed=4).run(lambda: True)
        self.assertEqual(results, [])

    def test_table_rows_include_obstacles_and_six_decimals(self):
        values = value_table_rows(self.planner)
        q_values = q_table_rows(self.planner)
        self.assertEqual(len(values), self.environment.rows * self.environment.columns)
        self.assertEqual(len(q_values), len(values))
        obstacle = next(row for row in values if (row["row"], row["column"]) == (1, 2))
        self.assertTrue(obstacle["is_obstacle"])
        self.assertRegex(obstacle["value"], r"^-?\d+\.\d{6}$")


if __name__ == "__main__":
    unittest.main()
