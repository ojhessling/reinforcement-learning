import queue
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from mountaincar_logic import (
    ALGORITHMS,
    DuelingDQNPolicy,
    MountainCarCallback,
    MountainCarConfig,
    MountainCarWorkbench,
    NoisyDQNPolicy,
    NoisyLinear,
    PrioritizedReplayBuffer,
    double_dqn_next_values,
    make_mountaincar_env,
)
from mountaincar_render import MountainCarRenderer


class ConfigTests(unittest.TestCase):
    def test_defaults_match_rl_zoo_mountaincar_profile(self):
        config = MountainCarConfig()
        self.assertEqual(config.total_timesteps, 120_000)
        self.assertEqual(config.learning_rate, 0.004)
        self.assertEqual(config.buffer_size, 10_000)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.gamma, 0.98)
        self.assertEqual(config.train_freq, 16)
        self.assertEqual(config.gradient_steps, 8)
        self.assertEqual(config.target_update_interval, 600)
        self.assertEqual(config.net_arch, (64, 64))

    def test_every_algorithm_validates(self):
        for algorithm in ALGORITHMS:
            MountainCarConfig(algorithm=algorithm).validate()

    def test_double_dqn_separates_selection_and_evaluation(self):
        online = torch.tensor([[1.0, 8.0, 2.0]])
        target = torch.tensor([[9.0, 3.0, 4.0]])
        torch.testing.assert_close(double_dqn_next_values(online, target), torch.tensor([[3.0]]))

    def test_noisy_variant_disables_epsilon_greedy(self):
        kwargs = MountainCarConfig(algorithm="Noisy DDQN").model_kwargs()
        self.assertEqual(kwargs["exploration_initial_eps"], 0.0)
        self.assertEqual(kwargs["exploration_final_eps"], 0.0)
        self.assertEqual(kwargs["exploration_fraction"], 0.0)

    def test_callback_uses_terminal_observation_for_success(self):
        callback = MountainCarCallback(threading.Event())
        callback.model = Mock(num_timesteps=1)
        callback.num_timesteps = 1
        callback.locals = {
            "rewards": np.array([-1.0]), "dones": np.array([True]),
            "new_obs": np.array([[-0.5, 0.0]]),
            "infos": [{"terminal_observation": np.array([0.5, 0.01])}],
        }
        callback._on_step()
        self.assertTrue(callback.metrics[0].success)
        self.assertEqual(callback.metrics[0].max_position, 0.5)


class EnvironmentTests(unittest.TestCase):
    @patch("mountaincar_logic.gymnasium.make")
    def test_required_environment_factory(self, make_mock):
        make_mock.return_value = Mock()
        make_mountaincar_env()
        make_mock.assert_called_once_with("MountainCar-v0", render_mode="rgb_array")

    def test_spaces_reward_and_time_limit(self):
        env = make_mountaincar_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=3)
            self.assertEqual(observation.shape, (2,))
            self.assertEqual(env.action_space.n, 3)
            self.assertEqual(env.spec.max_episode_steps, 200)
            _observation, reward, _terminated, _truncated, _info = env.step(1)
            self.assertEqual(reward, -1.0)
        finally:
            env.close()

    def test_isolated_renderer_returns_official_rgb_frames(self):
        renderer = MountainCarRenderer()
        try:
            observation, _info, frame = renderer.reset(seed=3)
            self.assertEqual(observation.shape, (2,))
            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[2], 3)
            next_observation, reward, _terminated, _truncated, _info, next_frame = renderer.step(2)
            self.assertEqual(next_observation.shape, (2,))
            self.assertEqual(reward, -1.0)
            self.assertEqual(next_frame.shape, frame.shape)
        finally:
            renderer.close()


class ArchitectureTests(unittest.TestCase):
    def test_noisy_linear_changes_with_noise_and_is_stable_without_noise(self):
        layer = NoisyLinear(2, 3)
        values = torch.ones(1, 2)
        first = layer(values)
        layer.reset_noise()
        second = layer(values)
        self.assertFalse(torch.equal(first, second))
        layer.noise_enabled = False
        torch.testing.assert_close(layer(values), layer(values))

    def test_dueling_output_has_action_dimension_and_gradients(self):
        workbench = MountainCarWorkbench(
            MountainCarConfig(algorithm="Dueling DDQN", total_timesteps=32, learning_starts=0)
        )
        try:
            model = workbench.create_model()
            output = model.q_net(torch.zeros(4, 2, device=model.device))
            self.assertEqual(tuple(output.shape), (4, 3))
            output.sum().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.q_net.parameters()))
        finally:
            workbench.close()


class PERTests(unittest.TestCase):
    def test_priority_updates_add_epsilon(self):
        underlying = Mock(buffer_size=10)
        wrapper = PrioritizedReplayBuffer(underlying, alpha=0.6, epsilon=1e-6)
        wrapper.update_priorities(np.array([2, 5]), torch.tensor([[3.0], [-4.0]]))
        self.assertAlmostEqual(float(wrapper.priorities[2]), 3.000001, places=5)
        self.assertAlmostEqual(float(wrapper.priorities[5]), 4.000001, places=5)

    def test_priority_can_decrease_and_duplicate_uses_largest_new_value(self):
        underlying = Mock(buffer_size=10)
        wrapper = PrioritizedReplayBuffer(underlying, alpha=0.6, epsilon=1e-6)
        wrapper.priorities[2] = 20.0
        wrapper.update_priorities(np.array([2, 2]), torch.tensor([[1.0], [3.0]]))
        self.assertAlmostEqual(float(wrapper.priorities[2]), 3.000001, places=5)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = MountainCarConfig(
            total_timesteps=240, buffer_size=500, learning_starts=16,
            batch_size=16, train_freq=4, gradient_steps=1,
            target_update_interval=32, net_arch=(32,), seed=7,
            per_beta_steps=240, value_arch=(16,), advantage_arch=(16,),
        )

    def test_all_variants_create_and_train(self):
        for algorithm in ALGORITHMS:
            workbench = MountainCarWorkbench(replace(self.config, algorithm=algorithm))
            try:
                metrics = workbench.train()
                self.assertIsNotNone(workbench.model)
                self.assertGreater(workbench.model.num_timesteps, 0)
                self.assertGreater(len(metrics), 0)
                if algorithm == "Noisy DDQN":
                    self.assertIsInstance(workbench.model.policy, NoisyDQNPolicy)
                if algorithm == "Dueling DDQN":
                    self.assertIsInstance(workbench.model.policy, DuelingDQNPolicy)
            finally:
                workbench.close()

    def test_evaluation_does_not_change_parameters_or_replay(self):
        workbench = MountainCarWorkbench(self.config)
        try:
            workbench.train()
            before = {name: value.detach().clone() for name, value in workbench.model.policy.state_dict().items()}
            size = workbench.model.replay_buffer.size()
            result = workbench.evaluate(2, seed=10)
            self.assertEqual(result.episodes, 2)
            self.assertEqual(workbench.model.replay_buffer.size(), size)
            for name, value in before.items():
                torch.testing.assert_close(value, workbench.model.policy.state_dict()[name])
        finally:
            workbench.close()

    def test_save_load_and_continue(self):
        workbench = MountainCarWorkbench(self.config)
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "mountaincar"
                workbench.save(base)
                loaded = MountainCarWorkbench.load(base)
                try:
                    old = loaded.model.num_timesteps
                    loaded.train()
                    self.assertGreater(loaded.model.num_timesteps, old)
                finally:
                    loaded.close()
        finally:
            workbench.close()


if __name__ == "__main__":
    unittest.main()
