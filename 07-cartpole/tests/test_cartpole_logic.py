import queue
import os
from dataclasses import replace
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from cartpole_logic import (
    DDQN,
    CartPoleWorkbench,
    DQNConfig,
    EpisodeMetric,
    EvaluationResult,
    WorkbenchCallback,
    double_dqn_next_values,
    make_cartpole_env,
)
from cartpole_render import CartPoleRenderer
from cartpole_gui import downsample_minmax, rolling_average


class ConfigTests(unittest.TestCase):
    def test_display_downsampling_limits_points_and_keeps_extremes(self):
        x_values = np.arange(10_000)
        y_values = np.zeros(10_000)
        y_values[1234] = 500
        y_values[6789] = -100
        reduced_x, reduced_y = downsample_minmax(x_values, y_values, max_points=2_000)
        self.assertLessEqual(len(reduced_x), 2_000)
        self.assertIn(500, reduced_y)
        self.assertIn(-100, reduced_y)
        self.assertEqual(reduced_x[0], 0)
        self.assertEqual(reduced_x[-1], 9_999)

    def test_rolling_average_uses_last_twenty_values(self):
        values = np.arange(1, 26, dtype=float)
        averaged = rolling_average(values)
        self.assertEqual(averaged[0], 1.0)
        self.assertEqual(averaged[19], 10.5)
        self.assertEqual(averaged[24], 15.5)

    def test_defaults_match_stable_cartpole_profile(self):
        config = DQNConfig()
        self.assertEqual(config.total_timesteps, 100_000)
        self.assertEqual(config.learning_rate, 0.0005)
        self.assertEqual(config.buffer_size, 200_000)
        self.assertEqual(config.learning_starts, 1_000)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.train_freq, 4)
        self.assertEqual(config.gradient_steps, 1)
        self.assertEqual(config.target_update_interval, 1_000)
        self.assertEqual(config.exploration_fraction, 0.20)
        self.assertEqual(config.exploration_final_eps, 0.05)
        self.assertEqual(config.net_arch, (64, 64))
        self.assertEqual(config.optimizer_eps, 1e-5)

    def test_model_kwargs_include_network_and_optimizer(self):
        kwargs = DQNConfig(net_arch=(32, 16), activation="Tanh", optimizer="AdamW").model_kwargs()
        self.assertEqual(kwargs["policy_kwargs"]["net_arch"], [32, 16])
        self.assertIs(kwargs["policy_kwargs"]["activation_fn"], torch.nn.Tanh)
        self.assertIs(kwargs["policy_kwargs"]["optimizer_class"], torch.optim.AdamW)

    def test_rejects_invalid_parameter_relations(self):
        with self.assertRaisesRegex(ValueError, "Batch-Größe"):
            DQNConfig(buffer_size=10, batch_size=32).validate()
        with self.assertRaisesRegex(ValueError, "Learning Starts"):
            DQNConfig(total_timesteps=10, learning_starts=20).validate()
        with self.assertRaisesRegex(ValueError, "Final Epsilon"):
            DQNConfig(exploration_initial_eps=0.1, exploration_final_eps=0.2).validate()

    def test_config_roundtrip(self):
        original = DQNConfig(net_arch=(32, 16), seed=None)
        restored = DQNConfig.from_dict({**original.__dict__, "net_arch": [32, 16]})
        self.assertEqual(restored, original)

    def test_training_budget_does_not_change_model_signature(self):
        self.assertEqual(
            DQNConfig(total_timesteps=1_000).model_signature(),
            DQNConfig(total_timesteps=20_000).model_signature(),
        )

    def test_double_dqn_selects_online_action_and_uses_target_value(self):
        online = torch.tensor([[1.0, 5.0], [7.0, 2.0]])
        target = torch.tensor([[10.0, 2.0], [3.0, 9.0]])
        values = double_dqn_next_values(online, target)
        torch.testing.assert_close(values, torch.tensor([[2.0], [3.0]]))


class EnvironmentTests(unittest.TestCase):
    @patch("cartpole_logic.gymnasium.make")
    def test_factory_uses_required_environment(self, make_mock):
        make_mock.return_value = Mock()
        make_cartpole_env()
        make_mock.assert_called_once_with("CartPole-v1", render_mode="rgb_array")

    def test_real_environment_spaces_and_time_limit(self):
        env = make_cartpole_env()
        try:
            observation, _ = env.reset(seed=7)
            self.assertEqual(observation.shape, (4,))
            self.assertEqual(env.action_space.n, 2)
            self.assertEqual(env.spec.max_episode_steps, 500)
            _next, reward, terminated, truncated, _info = env.step(0)
            self.assertEqual(reward, 1.0)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
        finally:
            env.close()


class CallbackTests(unittest.TestCase):
    @patch("cartpole_logic.evaluate_policy")
    def test_automatic_evaluation_is_emitted_at_interval(self, evaluate_mock):
        result = Mock(mean_reward=450.0)
        evaluate_mock.return_value = result
        output = queue.Queue()
        best_callback = Mock()
        callback = WorkbenchCallback(
            threading.Event(), output, evaluation_interval=100,
            evaluation_episodes=20, evaluation_seed=42,
            best_model_callback=best_callback,
        )
        callback.model = Mock(exploration_rate=0.1, num_timesteps=0)
        callback._on_training_start()
        callback.num_timesteps = 100
        callback.locals = {
            "rewards": np.array([1.0]), "dones": np.array([False]), "infos": [{}]
        }
        callback._on_step()
        kind, payload = output.get_nowait()
        self.assertEqual(kind, "automatic_evaluation")
        self.assertEqual(payload, (0, 100, result))
        evaluate_mock.assert_called_once_with(callback.model, 20, 42)
        best_callback.assert_called_once_with(0, 100, result)

    def test_stop_event_requests_clean_sb3_stop(self):
        event = threading.Event()
        event.set()
        callback = WorkbenchCallback(event)
        callback.locals = {"rewards": np.array([1.0]), "dones": np.array([False]), "infos": [{}]}
        self.assertFalse(callback._on_step())

    def test_completed_episode_is_reported(self):
        output = queue.Queue()
        callback = WorkbenchCallback(threading.Event(), output)
        callback.model = Mock(exploration_rate=0.2)
        callback.num_timesteps = 500
        callback.locals = {"rewards": np.array([1.0]), "dones": np.array([True]), "infos": [{"TimeLimit.truncated": True}]}
        callback._length = 499
        callback._reward = 499
        self.assertTrue(callback._on_step())
        self.assertEqual(callback.episode_metrics[0].reward, 500)
        self.assertTrue(callback.episode_metrics[0].success)

    def test_comparison_episode_is_streamed_with_series_name(self):
        output = queue.Queue()
        callback = WorkbenchCallback(
            threading.Event(), output, series_name=("DDQN", 2)
        )
        callback.model = Mock(exploration_rate=0.1)
        callback.num_timesteps = 20
        callback.locals = {
            "rewards": np.array([1.0]), "dones": np.array([True]), "infos": [{}]
        }
        callback._on_step()
        kind, payload = output.get_nowait()
        self.assertEqual(kind, "comparison_episode")
        self.assertEqual(payload[0], ("DDQN", 2))

    def test_comparison_progress_identifies_parallel_algorithm(self):
        output = queue.Queue()
        callback = WorkbenchCallback(
            threading.Event(), output, series_name=("DQN", 2),
            progress_offset=10_000, comparison_total=40_000,
        )
        callback.model = Mock(exploration_rate=0.1, num_timesteps=0)
        callback._on_training_start()
        callback.num_timesteps = 100
        callback.locals = {
            "rewards": np.array([1.0]), "dones": np.array([False]), "infos": [{}]
        }
        callback._on_step()
        kind, payload = output.get_nowait()
        self.assertEqual(kind, "comparison_progress")
        self.assertEqual(payload, (("DQN", 2), 10_100, 40_000))


class IsolatedRendererTests(unittest.TestCase):
    def test_render_process_returns_official_rgb_frames(self):
        renderer = CartPoleRenderer()
        try:
            observation, _info, frame = renderer.reset(seed=42)
            self.assertEqual(observation.shape, (4,))
            self.assertEqual(frame.shape, (400, 600, 3))
            next_observation, reward, terminated, truncated, _info, next_frame = renderer.step(0)
            self.assertEqual(next_observation.shape, (4,))
            self.assertEqual(reward, 1.0)
            self.assertFalse(terminated and truncated)
            self.assertEqual(next_frame.shape, frame.shape)
        finally:
            renderer.close()


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = DQNConfig(
            total_timesteps=250,
            buffer_size=1_000,
            learning_starts=10,
            batch_size=16,
            target_update_interval=100,
            net_arch=(32,),
            seed=3,
        )
        self.workbench = CartPoleWorkbench(self.config)

    def tearDown(self):
        self.workbench.close()

    def test_training_and_evaluation_do_not_mutate_model_during_eval(self):
        self.workbench.train()
        before = {name: value.detach().cpu().clone() for name, value in self.workbench.model.policy.state_dict().items()}
        replay_size = self.workbench.model.replay_buffer.size()
        result = self.workbench.evaluate(3, seed=10)
        after = self.workbench.model.policy.state_dict()
        self.assertEqual(result.episodes, 3)
        self.assertEqual(len(result.rewards), 3)
        self.assertEqual(self.workbench.model.replay_buffer.size(), replay_size)
        for name, value in before.items():
            torch.testing.assert_close(value, after[name].detach().cpu())

    def test_single_training_executes_at_least_requested_environment_steps(self):
        self.workbench.train()
        self.assertGreaterEqual(self.workbench.model.num_timesteps, self.config.total_timesteps)
        self.assertLess(
            self.workbench.model.num_timesteps,
            self.config.total_timesteps + self.config.train_freq,
        )

    def test_ddqn_uses_sb3_model_and_trains(self):
        ddqn = CartPoleWorkbench(replace(self.config, algorithm="DDQN"))
        try:
            ddqn.train()
            self.assertIsInstance(ddqn.model, DDQN)
            self.assertGreater(ddqn.model._n_updates, 0)
            self.assertGreater(len(ddqn.training_history), 0)
        finally:
            ddqn.close()

    def test_animation_uses_separate_environment(self):
        self.workbench.train()
        training_env = self.workbench.training_env
        steps = self.workbench.animation_episode(seed=11)
        self.assertIs(self.workbench.training_env, training_env)
        self.assertGreater(len(steps), 0)
        self.assertEqual(steps[0].frame.ndim, 3)
        self.assertEqual(len(steps[0].observation), 4)

    def test_save_load_and_continue_training(self):
        self.workbench.train()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "cartpole_dqn"
            paths = self.workbench.save(base)
            self.assertTrue(all(path.exists() for path in paths))
            loaded = CartPoleWorkbench.load(base)
            try:
                old_steps = loaded.model.num_timesteps
                old_replay_size = loaded.model.replay_buffer.size()
                loaded.train()
                self.assertGreater(loaded.model.num_timesteps, old_steps)
                self.assertGreaterEqual(loaded.model.replay_buffer.size(), old_replay_size)
            finally:
                loaded.close()


@unittest.skipUnless(os.environ.get("RUN_GUI_TESTS") == "1", "GUI-Smoke-Test benötigt RUN_GUI_TESTS=1 und ein Display")
class GUIVisibilityTests(unittest.TestCase):
    def test_all_essential_widgets_are_visible_at_start(self):
        import tkinter as tk
        from cartpole_gui import CartPoleGUI

        root = tk.Tk()
        gui = CartPoleGUI(root)
        try:
            root.update_idletasks()
            root.update()
            gui._initialize_visible_layout()
            metric = EpisodeMetric(1, 100.0, 100, False, 100, 0.1)
            evaluation = EvaluationResult(10, 120.0, 5.0, 120.0, 0.0, (120.0,), (120,))
            gui.comparison_history = {"DQN": {1: [metric]}, "DDQN": {1: [metric]}}
            gui.comparison_evaluation_history = {
                "DQN": {1: [(1, 100, evaluation)]},
                "DDQN": {1: [(1, 100, evaluation)]},
            }
            gui._refresh_comparison_plot()
            root.update_idletasks()
            root.update()
            self.assertEqual(gui.layout_visibility_issues(), [])
        finally:
            gui.close()


if __name__ == "__main__":
    unittest.main()
