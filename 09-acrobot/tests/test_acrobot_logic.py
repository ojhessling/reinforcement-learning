import pickle
import queue
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer

from acrobot_logic import (
    ALGORITHMS,
    DISTRIBUTIONAL_ALGORITHMS,
    DUELING_ALGORITHMS,
    MULTISTEP_ALGORITHMS,
    NOISY_ALGORITHMS,
    PER_ALGORITHMS,
    AcrobotCallback,
    AcrobotConfig,
    AcrobotDQNPolicy,
    AcrobotWorkbench,
    NoisyLinear,
    NStepReplayBuffer,
    PrioritizedReplayBuffer,
    RainbowCapableQNetwork,
    angles_degrees,
    double_dqn_next_values,
    make_acrobot_env,
    project_categorical,
)
from acrobot_render import AcrobotRenderer


def _raw_buffer(size: int = 50) -> ReplayBuffer:
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)
    return ReplayBuffer(size, observation_space, action_space, device="cpu", n_envs=1, handle_timeout_termination=True)


def _transition(reward: float, done: bool, truncated: bool = False):
    obs = np.zeros((1, 6), dtype=np.float32)
    next_obs = np.full((1, 6), reward, dtype=np.float32)
    action = np.array([[1]])
    infos = [{"TimeLimit.truncated": True}] if truncated else [{}]
    return obs, next_obs, action, np.array([reward], dtype=np.float32), np.array([done]), infos


def _has_per(buffer) -> bool:
    while isinstance(buffer, NStepReplayBuffer):
        buffer = buffer.buffer
    return isinstance(buffer, PrioritizedReplayBuffer)


class ConfigTests(unittest.TestCase):
    def test_defaults_match_rl_zoo_acrobot_profile(self):
        config = AcrobotConfig()
        self.assertEqual(config.total_timesteps, 100_000)
        self.assertEqual(config.learning_rate, 6.3e-4)
        self.assertEqual(config.buffer_size, 50_000)
        self.assertEqual(config.learning_starts, 0)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.train_freq, 4)
        self.assertEqual(config.target_update_interval, 250)
        self.assertEqual(config.exploration_fraction, 0.12)
        self.assertEqual(config.exploration_final_eps, 0.1)
        self.assertEqual(config.net_arch, (256, 256))
        self.assertEqual(config.multistep_n, 3)
        self.assertEqual(config.c51_atoms, 51)
        self.assertEqual(config.c51_v_min, -500.0)
        self.assertEqual(config.c51_v_max, 0.0)

    def test_all_seven_algorithms_validate(self):
        self.assertEqual(len(ALGORITHMS), 7)
        for algorithm in ALGORITHMS:
            AcrobotConfig(algorithm=algorithm).validate()

    def test_double_dqn_separates_selection_and_evaluation(self):
        online = torch.tensor([[1.0, 8.0, 2.0]])
        target = torch.tensor([[9.0, 3.0, 4.0]])
        torch.testing.assert_close(double_dqn_next_values(online, target), torch.tensor([[3.0]]))

    def test_noisy_variants_disable_epsilon_greedy(self):
        for algorithm in NOISY_ALGORITHMS:
            kwargs = AcrobotConfig(algorithm=algorithm).model_kwargs()
            self.assertEqual(kwargs["exploration_initial_eps"], 0.0)
            self.assertEqual(kwargs["exploration_final_eps"], 0.0)
            self.assertEqual(kwargs["exploration_fraction"], 0.0)

    def test_non_noisy_variants_keep_epsilon_greedy(self):
        for algorithm in set(ALGORITHMS) - NOISY_ALGORITHMS:
            kwargs = AcrobotConfig(algorithm=algorithm).model_kwargs()
            self.assertEqual(kwargs["exploration_fraction"], 0.12)

    def test_rainbow_activates_all_five_extension_bausteine(self):
        kwargs = AcrobotConfig(algorithm="Rainbow DDQN").model_kwargs()
        self.assertTrue(kwargs["policy_kwargs"]["noisy"])
        self.assertTrue(kwargs["policy_kwargs"]["dueling"])
        self.assertTrue(kwargs["policy_kwargs"]["distributional"])
        self.assertEqual(kwargs["exploration_fraction"], 0.0)
        self.assertIn("Rainbow DDQN", PER_ALGORITHMS)
        self.assertIn("Rainbow DDQN", MULTISTEP_ALGORITHMS)

    def test_invalid_multistep_and_c51_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            AcrobotConfig(multistep_n=0).validate()
        with self.assertRaises(ValueError):
            AcrobotConfig(c51_atoms=1).validate()
        with self.assertRaises(ValueError):
            AcrobotConfig(c51_v_min=0.0, c51_v_max=0.0).validate()

    def test_callback_uses_truncation_flag_for_success(self):
        callback = AcrobotCallback(threading.Event())
        callback.model = Mock(num_timesteps=1)
        callback.num_timesteps = 1
        callback.locals = {"rewards": np.array([0.0]), "dones": np.array([True]), "infos": [{}]}
        callback._on_step()
        self.assertTrue(callback.metrics[0].success)

    def test_callback_marks_truncated_episode_as_failure(self):
        callback = AcrobotCallback(threading.Event())
        callback.model = Mock(num_timesteps=1)
        callback.num_timesteps = 1
        callback.locals = {
            "rewards": np.array([-1.0]), "dones": np.array([True]),
            "infos": [{"TimeLimit.truncated": True}],
        }
        callback._on_step()
        self.assertFalse(callback.metrics[0].success)

    def test_angles_degrees_recovers_zero_angle(self):
        theta1, theta2 = angles_degrees(np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(theta1, 0.0)
        self.assertAlmostEqual(theta2, 0.0)


class EnvironmentTests(unittest.TestCase):
    @patch("acrobot_logic.gymnasium.make")
    def test_required_environment_factory(self, make_mock):
        make_mock.return_value = Mock()
        make_acrobot_env()
        make_mock.assert_called_once_with("Acrobot-v1", render_mode="rgb_array")

    def test_spaces_reward_and_time_limit(self):
        env = make_acrobot_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=3)
            self.assertEqual(observation.shape, (6,))
            self.assertEqual(env.action_space.n, 3)
            self.assertEqual(env.spec.max_episode_steps, 500)
            _observation, reward, _terminated, _truncated, _info = env.step(1)
            self.assertEqual(reward, -1.0)
        finally:
            env.close()

    def test_isolated_renderer_returns_official_rgb_frames(self):
        renderer = AcrobotRenderer()
        try:
            observation, _info, frame = renderer.reset(seed=3)
            self.assertEqual(observation.shape, (6,))
            self.assertEqual(frame.ndim, 3)
            self.assertEqual(frame.shape[2], 3)
            next_observation, reward, _terminated, _truncated, _info, next_frame = renderer.step(2)
            self.assertEqual(next_observation.shape, (6,))
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

    def _model_for(self, algorithm: str, **overrides):
        config = AcrobotConfig(algorithm=algorithm, total_timesteps=32, learning_starts=0, **overrides)
        return AcrobotWorkbench(config)

    def test_ddqn_network_has_no_extension_bausteine(self):
        workbench = self._model_for("DDQN")
        try:
            model = workbench.create_model()
            self.assertFalse(model.q_net.noisy)
            self.assertFalse(model.q_net.dueling)
            self.assertFalse(model.q_net.distributional)
            self.assertFalse(any(isinstance(module, NoisyLinear) for module in model.q_net.modules()))
        finally:
            workbench.close()

    def test_dueling_output_has_action_dimension_and_gradients(self):
        workbench = self._model_for("Dueling DDQN")
        try:
            model = workbench.create_model()
            self.assertIsInstance(model.q_net, RainbowCapableQNetwork)
            self.assertTrue(model.q_net.dueling)
            output = model.q_net(torch.zeros(4, 6, device=model.device))
            self.assertEqual(tuple(output.shape), (4, 3))
            output.sum().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.q_net.parameters()))
        finally:
            workbench.close()

    def test_c51_network_outputs_a_normalised_distribution_per_action(self):
        workbench = self._model_for("C51 DDQN", net_arch=(8,), c51_atoms=11, c51_v_min=-5.0, c51_v_max=5.0)
        try:
            model = workbench.create_model()
            self.assertTrue(model.q_net.distributional)
            distribution = model.q_net.distribution(torch.zeros(4, 6, device=model.device))
            self.assertEqual(tuple(distribution.shape), (4, 3, 11))
            torch.testing.assert_close(distribution.sum(dim=-1), torch.ones(4, 3, device=model.device), atol=1e-5, rtol=1e-5)
            q_values = model.q_net(torch.zeros(4, 6, device=model.device))
            self.assertEqual(tuple(q_values.shape), (4, 3))
        finally:
            workbench.close()

    def test_rainbow_network_combines_noisy_dueling_and_distributional(self):
        workbench = self._model_for("Rainbow DDQN", net_arch=(8,), c51_atoms=9, c51_v_min=-5.0, c51_v_max=5.0)
        try:
            model = workbench.create_model()
            self.assertIsInstance(model.policy, AcrobotDQNPolicy)
            self.assertTrue(model.q_net.noisy)
            self.assertTrue(model.q_net.dueling)
            self.assertTrue(model.q_net.distributional)
            self.assertTrue(any(isinstance(module, NoisyLinear) for module in model.q_net.value_stream.modules()))
            self.assertTrue(any(isinstance(module, NoisyLinear) for module in model.q_net.advantage_stream.modules()))
            values = torch.zeros(1, 6, device=model.device)
            first = model.q_net(values)
            model.q_net.reset_noise()
            second = model.q_net(values)
            self.assertFalse(torch.equal(first, second))
        finally:
            workbench.close()


class CategoricalProjectionTests(unittest.TestCase):
    def test_projected_probability_mass_is_conserved(self):
        atoms = torch.linspace(-10.0, 10.0, 21)
        next_probs = torch.softmax(torch.randn(6, 21), dim=1)
        rewards = torch.randn(6, 1)
        dones = torch.zeros(6, 1)
        target = project_categorical(next_probs, rewards, dones, 0.99, atoms)
        torch.testing.assert_close(target.sum(dim=1), torch.ones(6), atol=1e-5, rtol=1e-5)

    def test_exact_atom_hit_keeps_full_mass_without_loss(self):
        atoms = torch.linspace(-2.0, 2.0, 5)  # -2,-1,0,1,2
        next_probs = torch.zeros(1, 5)
        next_probs[0, 2] = 1.0  # all mass on atom value 0
        rewards = torch.tensor([[1.0]])
        dones = torch.tensor([[0.0]])
        # Bellman target z = 1.0 + 1.0 * 1.0 * 0.0 = 1.0, which is exactly atom index 3.
        target = project_categorical(next_probs, rewards, dones, 1.0, atoms)
        self.assertAlmostEqual(float(target[0, 3]), 1.0, places=5)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=5)

    def test_terminal_transitions_ignore_the_bootstrap_distribution(self):
        atoms = torch.linspace(-2.0, 2.0, 5)
        next_probs = torch.softmax(torch.randn(3, 5), dim=1)
        rewards = torch.tensor([[-1.0], [0.0], [2.0]])
        dones = torch.ones(3, 1)
        target = project_categorical(next_probs, rewards, dones, 0.99, atoms)
        for index in range(3):
            expected = float(rewards[index, 0].clamp(atoms[0], atoms[-1]))
            mean_value = float((target[index] * atoms).sum())
            self.assertAlmostEqual(mean_value, expected, places=4)


class NStepReplayBufferTests(unittest.TestCase):
    def test_full_window_accumulates_n_step_return_with_gamma_power_n_discount(self):
        raw = _raw_buffer()
        buffer = NStepReplayBuffer(raw, n_step=3, gamma=0.5)
        for reward in (1.0, 2.0, 4.0):
            buffer.add(*_transition(reward, done=False))
        self.assertEqual(raw.pos, 1)
        expected_reward = 1.0 + 0.5 * 2.0 + 0.25 * 4.0
        self.assertAlmostEqual(float(raw.rewards[0, 0]), expected_reward, places=5)
        self.assertAlmostEqual(float(buffer.discounts[0]), 0.5 ** 3, places=5)

    def test_steady_state_slides_one_transition_per_step(self):
        raw = _raw_buffer()
        buffer = NStepReplayBuffer(raw, n_step=2, gamma=0.9)
        for reward in (1.0, 1.0, 1.0, 1.0):
            buffer.add(*_transition(reward, done=False))
        # 4 Schritte, Fenstergröße 2 -> ab Schritt 2 ein Emit je weiterem Schritt: 3 Inserts.
        self.assertEqual(raw.pos, 3)

    def test_episode_end_flushes_shortened_trailing_windows(self):
        raw = _raw_buffer()
        buffer = NStepReplayBuffer(raw, n_step=3, gamma=0.5)
        buffer.add(*_transition(1.0, done=False))
        buffer.add(*_transition(2.0, done=True))
        self.assertEqual(raw.pos, 2)
        self.assertAlmostEqual(float(raw.rewards[0, 0]), 1.0 + 0.5 * 2.0, places=5)
        self.assertAlmostEqual(float(raw.rewards[1, 0]), 2.0, places=5)
        self.assertTrue(bool(raw.dones[0, 0]))
        self.assertTrue(bool(raw.dones[1, 0]))

    def test_propagates_truncation_flag_for_correct_bootstrap_handling(self):
        raw = _raw_buffer()
        buffer = NStepReplayBuffer(raw, n_step=1, gamma=0.9)
        buffer.add(*_transition(-1.0, done=True, truncated=True))
        self.assertTrue(bool(raw.timeouts[0, 0]))

    def test_combines_with_per_using_the_same_slot_for_priority_and_discount(self):
        raw = _raw_buffer()
        per = PrioritizedReplayBuffer(raw, alpha=0.6, epsilon=1e-6)
        buffer = NStepReplayBuffer(per, n_step=1, gamma=0.9)
        buffer.add(*_transition(1.0, done=False))
        self.assertEqual(raw.pos, 1)
        self.assertGreater(per.priorities[0], 0.0)
        self.assertAlmostEqual(float(buffer.discounts[0]), 0.9, places=5)

    def test_sample_attaches_discounts_and_forwards_priority_updates(self):
        raw = _raw_buffer()
        per = PrioritizedReplayBuffer(raw, alpha=0.6, epsilon=1e-6)
        buffer = NStepReplayBuffer(per, n_step=1, gamma=0.9)
        for _ in range(5):
            buffer.add(*_transition(1.0, done=False))
        samples, indices, weights = buffer.sample(4, beta=0.4)
        self.assertIsNotNone(samples.discounts)
        self.assertEqual(tuple(samples.discounts.shape), (4, 1))
        buffer.update_priorities(indices, torch.ones(4, 1))
        self.assertTrue((per.priorities[indices] > 0).all())

    def test_sample_without_per_draws_uniformly_and_still_attaches_discounts(self):
        raw = _raw_buffer()
        buffer = NStepReplayBuffer(raw, n_step=1, gamma=0.9)
        for _ in range(5):
            buffer.add(*_transition(1.0, done=False))
        samples, indices, weights = buffer.sample(4)
        self.assertEqual(weights, 1.0)
        self.assertEqual(len(indices), 4)
        self.assertIsNotNone(samples.discounts)

    def test_wrapper_chain_survives_pickle_roundtrip(self):
        raw = _raw_buffer()
        wrapped = NStepReplayBuffer(PrioritizedReplayBuffer(raw, 0.6, 1e-6), n_step=2, gamma=0.9)
        for reward in (1.0, 2.0, 3.0, 4.0):
            wrapped.add(*_transition(reward, done=False))
        restored = pickle.loads(pickle.dumps(wrapped))
        self.assertEqual(restored.size(), wrapped.size())
        self.assertIsInstance(restored, NStepReplayBuffer)
        self.assertIsInstance(restored.buffer, PrioritizedReplayBuffer)


class DistributionalDDQNTests(unittest.TestCase):
    def test_target_distribution_is_normalised_and_uses_double_dqn_action_selection(self):
        workbench = AcrobotWorkbench(AcrobotConfig(
            algorithm="C51 DDQN", total_timesteps=32, learning_starts=0,
            net_arch=(8,), c51_atoms=11, c51_v_min=-5.0, c51_v_max=5.0,
        ))
        try:
            model = workbench.create_model()
            batch = 6
            replay_data = SimpleNamespace(
                next_observations=torch.zeros(batch, 6, device=model.device),
                rewards=torch.zeros(batch, 1, device=model.device),
                dones=torch.zeros(batch, 1, device=model.device),
            )
            target = model._distributional_target(replay_data, model.gamma)
            self.assertEqual(tuple(target.shape), (batch, 11))
            torch.testing.assert_close(
                target.sum(dim=1), torch.ones(batch, device=model.device), atol=1e-4, rtol=1e-4
            )
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
        # total_timesteps liegt bewusst über max_episode_steps (500), damit
        # jeder kurze Testlauf mindestens eine abgeschlossene Episode
        # (spätestens per Truncation) liefert.
        self.config = AcrobotConfig(
            total_timesteps=520, buffer_size=600, learning_starts=16,
            batch_size=16, train_freq=4, gradient_steps=1,
            target_update_interval=64, net_arch=(32,), seed=7,
            per_beta_steps=520, value_arch=(16,), advantage_arch=(16,),
            multistep_n=3, c51_atoms=11, c51_v_min=-500.0, c51_v_max=0.0,
        )

    def test_all_seven_variants_create_and_train_with_correct_bausteine(self):
        for algorithm in ALGORITHMS:
            workbench = AcrobotWorkbench(replace(self.config, algorithm=algorithm))
            try:
                metrics = workbench.train()
                self.assertIsNotNone(workbench.model)
                self.assertGreater(workbench.model.num_timesteps, 0)
                self.assertGreater(len(metrics), 0)
                self.assertEqual(algorithm in NOISY_ALGORITHMS, workbench.model.q_net.noisy)
                self.assertEqual(algorithm in DUELING_ALGORITHMS, workbench.model.q_net.dueling)
                self.assertEqual(algorithm in DISTRIBUTIONAL_ALGORITHMS, workbench.model.q_net.distributional)
                self.assertEqual(algorithm in MULTISTEP_ALGORITHMS, isinstance(workbench.model.replay_buffer, NStepReplayBuffer))
                self.assertEqual(algorithm in PER_ALGORITHMS, _has_per(workbench.model.replay_buffer))
                for parameter in workbench.model.policy.parameters():
                    self.assertFalse(torch.isnan(parameter).any(), f"NaN-Parameter nach Training von {algorithm}")
            finally:
                workbench.close()

    def test_evaluation_does_not_change_parameters_or_replay(self):
        workbench = AcrobotWorkbench(self.config)
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

    def test_evaluation_without_success_reports_missing_steps_to_goal(self):
        workbench = AcrobotWorkbench(self.config)
        try:
            workbench.train()
            result = workbench.evaluate(2, seed=10)
            if result.success_rate == 0.0:
                self.assertIsNone(result.mean_steps_to_goal)
                self.assertIsNone(result.best_steps_to_goal)
        finally:
            workbench.close()

    def test_save_load_and_continue_for_every_algorithm(self):
        for algorithm in ALGORITHMS:
            workbench = AcrobotWorkbench(replace(self.config, algorithm=algorithm))
            try:
                workbench.train()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory) / "acrobot"
                    workbench.save(base)
                    loaded = AcrobotWorkbench.load(base)
                    try:
                        old = loaded.model.num_timesteps
                        loaded.train()
                        self.assertGreater(loaded.model.num_timesteps, old)
                        self.assertEqual(type(loaded.model.replay_buffer), type(workbench.model.replay_buffer))
                    finally:
                        loaded.close()
            finally:
                workbench.close()


if __name__ == "__main__":
    unittest.main()
