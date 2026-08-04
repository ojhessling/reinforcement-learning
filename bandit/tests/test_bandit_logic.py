import unittest

from bandit_logic import (
    BanditEnvironment,
    EpsilonDecayAgent,
    EpsilonGreedyAgent,
    SoftmaxAgent,
    ThompsonSamplingAgent,
    UCBAgent,
    compare_strategies,
    create_agent,
)


class BanditEnvironmentTests(unittest.TestCase):
    def test_seed_makes_rewards_reproducible(self):
        first = BanditEnvironment(seed=42)
        second = BanditEnvironment(seed=42)
        self.assertEqual(
            [first.pull(2) for _ in range(20)],
            [second.pull(2) for _ in range(20)],
        )

    def test_statistics_are_updated(self):
        environment = BanditEnvironment(probabilities=(1.0,), seed=1)
        self.assertEqual(environment.pull(0), 1)
        self.assertEqual(environment.statistics(0)["success_rate"], 1.0)


class AgentTests(unittest.TestCase):
    def test_memory_keeps_only_last_n_rewards_for_estimate(self):
        agent = EpsilonGreedyAgent(1, epsilon=0, memory_limit=2, seed=1)
        for reward in (1, 0, 0):
            agent.update(0, reward)
        self.assertEqual(agent.estimated_values(), [0.0])
        self.assertEqual(agent.cumulative_reward, 1)

    def test_ucb_pulls_every_bandit_first(self):
        agent = UCBAgent(3, seed=1)
        actions = []
        for _ in range(3):
            action = agent.select_action()
            actions.append(action)
            agent.update(action, 0)
        self.assertEqual(actions, [0, 1, 2])

    def test_decay_never_goes_below_minimum(self):
        agent = EpsilonDecayAgent(
            3, epsilon_start=1.0, epsilon_min=0.2, epsilon_decay=0.5, seed=1
        )
        for _ in range(20):
            agent.select_action()
        self.assertEqual(agent.epsilon, 0.2)

    def test_thompson_updates_beta_parameters(self):
        agent = ThompsonSamplingAgent(2, seed=1)
        agent.update(0, 1)
        agent.update(1, 0)
        self.assertEqual(agent.alpha, [2, 1])
        self.assertEqual(agent.beta, [1, 2])

    def test_all_factories_produce_valid_actions(self):
        for strategy in (
            "Epsilon Greedy",
            "Epsilon Decay",
            "Thompson Sampling",
            "UCB",
            "Boltzmann/Softmax",
        ):
            agent = create_agent(strategy, 3, seed=5)
            self.assertIn(agent.select_action(), range(3))

    def test_softmax_rejects_invalid_temperature(self):
        with self.assertRaises(ValueError):
            SoftmaxAgent(3, temperature=0)

    def test_comparison_is_reproducible_and_complete(self):
        strategies = ["Epsilon Greedy", "UCB"]
        first = compare_strategies(strategies, steps=80, repetitions=5, seed=42)
        second = compare_strategies(strategies, steps=80, repetitions=5, seed=42)
        self.assertEqual(first, second)
        self.assertEqual({result.strategy for result in first}, set(strategies))
        self.assertTrue(all(len(result.mean_rewards) == 80 for result in first))

    def test_comparison_validates_inputs(self):
        with self.assertRaises(ValueError):
            compare_strategies([], steps=10, repetitions=2)
        with self.assertRaises(ValueError):
            compare_strategies(["UCB"], steps=0, repetitions=2)

    def test_strategy_result_does_not_depend_on_other_selections(self):
        alone = compare_strategies(["UCB"], steps=60, repetitions=4, seed=7)[0]
        together = compare_strategies(
            ["Epsilon Greedy", "UCB"], steps=60, repetitions=4, seed=7
        )
        compared_ucb = next(result for result in together if result.strategy == "UCB")
        self.assertEqual(alone, compared_ucb)

    def test_strategies_learn_to_prefer_best_bandit(self):
        for agent_class in (EpsilonGreedyAgent, ThompsonSamplingAgent, UCBAgent):
            environment = BanditEnvironment(seed=123)
            if agent_class is EpsilonGreedyAgent:
                agent = agent_class(3, epsilon=0.1, seed=123)
            else:
                agent = agent_class(3, seed=123)
            for _ in range(3000):
                action = agent.select_action()
                agent.update(action, environment.pull(action))
            self.assertEqual(max(range(3), key=agent.action_counts.__getitem__), 2)


if __name__ == "__main__":
    unittest.main()
