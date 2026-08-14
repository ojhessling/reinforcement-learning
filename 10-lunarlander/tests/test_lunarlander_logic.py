import math
import pickle
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

from lunarlander_logic import (
    ALGORITHMS,
    DISTRIBUTIONAL_ALGORITHMS,
    DUELING_ALGORITHMS,
    ENV_ID,
    MULTISTEP_ALGORITHMS,
    NOISY_ALGORITHMS,
    PER_ALGORITHMS,
    SOLVED_RETURN,
    LunarLanderCallback,
    LunarLanderConfig,
    LunarLanderDQNPolicy,
    LunarLanderWorkbench,
    NStepReplayBuffer,
    NoisyLinear,
    PrioritizedReplayBuffer,
    RainbowCapableQNetwork,
    double_dqn_next_values,
    episode_outcome,
    make_lunarlander_env,
    observation_readout,
    project_categorical,
)
from lunarlander_render import LunarLanderRenderer


def _raw_buffer(size: int = 50) -> ReplayBuffer:
    observation_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)
    return ReplayBuffer(size, observation_space, action_space, device="cpu", n_envs=1,
                        handle_timeout_termination=True)


def _transition(reward: float, done: bool, truncated: bool = False):
    obs = np.zeros((1, 8), dtype=np.float32)
    next_obs = np.full((1, 8), reward, dtype=np.float32)
    action = np.array([[1]])
    infos = [{"TimeLimit.truncated": True}] if truncated else [{}]
    return obs, next_obs, action, np.array([reward], dtype=np.float32), np.array([done]), infos


def _has_per(buffer) -> bool:
    while isinstance(buffer, NStepReplayBuffer):
        buffer = buffer.buffer
    return isinstance(buffer, PrioritizedReplayBuffer)


class ConfigTests(unittest.TestCase):
    def test_defaults_match_rl_zoo_lunarlander_profile(self):
        config = LunarLanderConfig()
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

    def test_c51_range_covers_the_two_sided_lunarlander_return(self):
        config = LunarLanderConfig()
        self.assertEqual(config.c51_atoms, 51)
        self.assertLess(config.c51_v_min, -SOLVED_RETURN)
        self.assertGreater(config.c51_v_max, SOLVED_RETURN)

    def test_all_seven_algorithms_validate(self):
        self.assertEqual(len(ALGORITHMS), 7)
        for algorithm in ALGORITHMS:
            LunarLanderConfig(algorithm=algorithm).validate()

    def test_double_dqn_separates_selection_and_evaluation(self):
        online = torch.tensor([[1.0, 8.0, 2.0, 0.5]])
        target = torch.tensor([[9.0, 3.0, 4.0, 7.0]])
        torch.testing.assert_close(double_dqn_next_values(online, target), torch.tensor([[3.0]]))

    def test_noisy_variants_disable_epsilon_greedy(self):
        for algorithm in NOISY_ALGORITHMS:
            kwargs = LunarLanderConfig(algorithm=algorithm).model_kwargs()
            self.assertEqual(kwargs["exploration_initial_eps"], 0.0)
            self.assertEqual(kwargs["exploration_final_eps"], 0.0)
            self.assertEqual(kwargs["exploration_fraction"], 0.0)

    def test_non_noisy_variants_keep_epsilon_greedy(self):
        for algorithm in set(ALGORITHMS) - NOISY_ALGORITHMS:
            kwargs = LunarLanderConfig(algorithm=algorithm).model_kwargs()
            self.assertEqual(kwargs["exploration_fraction"], 0.12)

    def test_rainbow_activates_all_five_extension_bausteine(self):
        kwargs = LunarLanderConfig(algorithm="Rainbow DDQN").model_kwargs()
        self.assertTrue(kwargs["policy_kwargs"]["noisy"])
        self.assertTrue(kwargs["policy_kwargs"]["dueling"])
        self.assertTrue(kwargs["policy_kwargs"]["distributional"])
        self.assertIn("Rainbow DDQN", PER_ALGORITHMS)
        self.assertIn("Rainbow DDQN", MULTISTEP_ALGORITHMS)

    def test_validation_messages_name_field_value_and_range(self):
        cases = (
            (dict(batch_size=999_999), ("Batch-Größe B", "999999")),
            (dict(gamma=1.5), ("Diskontfaktor γ", "1.5")),
            (dict(multistep_n=0), ("Schritte n", "0")),
            (dict(c51_atoms=1), ("Atome n_atoms", "1")),
            (dict(c51_v_min=500.0, c51_v_max=-500.0), ("V_min/V_max", "500.0")),
            (dict(learning_starts=-1), ("Lernstart t₀", "-1")),
        )
        for overrides, (field, value) in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError) as context:
                    LunarLanderConfig(**overrides).validate()
                message = str(context.exception)
                self.assertIn(field, message)
                self.assertIn(value, message)
                self.assertIn("Gültig", message)


class OutcomeTests(unittest.TestCase):
    def test_safe_landing_is_detected_from_the_terminal_reward(self):
        landed, crashed, solved = episode_outcome(100.0, truncated=False, episode_return=250.0)
        self.assertTrue(landed); self.assertFalse(crashed); self.assertTrue(solved)

    def test_crash_is_detected_from_the_terminal_reward(self):
        landed, crashed, solved = episode_outcome(-100.0, truncated=False, episode_return=-150.0)
        self.assertFalse(landed); self.assertTrue(crashed); self.assertFalse(solved)

    def test_time_limit_is_neither_landing_nor_crash(self):
        landed, crashed, _ = episode_outcome(-0.5, truncated=True, episode_return=50.0)
        self.assertFalse(landed); self.assertFalse(crashed)

    def test_solved_uses_the_official_threshold_of_200(self):
        self.assertTrue(episode_outcome(100.0, False, SOLVED_RETURN)[2])
        self.assertFalse(episode_outcome(100.0, False, SOLVED_RETURN - 0.1)[2])

    def test_landing_below_threshold_counts_as_landed_but_not_solved(self):
        landed, _, solved = episode_outcome(100.0, truncated=False, episode_return=120.0)
        self.assertTrue(landed)
        self.assertFalse(solved)

    def test_observation_readout_converts_angle_and_angular_velocity(self):
        observation = np.array([0.1, 0.2, 0.3, 0.4, math.pi / 2, 1.0, 1.0, 0.0], dtype=np.float32)
        values = observation_readout(observation)
        self.assertAlmostEqual(values["angle_degrees"], 90.0, places=4)
        self.assertAlmostEqual(values["angular_velocity"], 2.5, places=4)
        self.assertTrue(values["left_leg"])
        self.assertFalse(values["right_leg"])


class CallbackTests(unittest.TestCase):
    def _callback_with(self, reward: float, info: dict) -> LunarLanderCallback:
        callback = LunarLanderCallback(threading.Event())
        callback.model = Mock(num_timesteps=1)
        callback.num_timesteps = 1
        callback.locals = {"rewards": np.array([reward]), "dones": np.array([True]), "infos": [info]}
        callback._on_step()
        return callback

    def test_landing_episode_is_recorded_as_landed(self):
        metric = self._callback_with(100.0, {}).metrics[0]
        self.assertTrue(metric.landed); self.assertFalse(metric.crashed)

    def test_crash_episode_is_recorded_as_crashed(self):
        metric = self._callback_with(-100.0, {}).metrics[0]
        self.assertTrue(metric.crashed); self.assertFalse(metric.landed)

    def test_truncated_episode_is_neither_landed_nor_crashed(self):
        metric = self._callback_with(-0.4, {"TimeLimit.truncated": True}).metrics[0]
        self.assertFalse(metric.landed); self.assertFalse(metric.crashed)


class EnvironmentTests(unittest.TestCase):
    @patch("lunarlander_logic.gymnasium.make")
    def test_required_environment_factory(self, make_mock):
        make_mock.return_value = Mock()
        make_lunarlander_env()
        make_mock.assert_called_once_with(ENV_ID, render_mode="rgb_array")

    def test_spaces_reward_and_time_limit(self):
        env = make_lunarlander_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=3)
            self.assertEqual(observation.shape, (8,))
            self.assertEqual(env.action_space.n, 4)
            self.assertEqual(env.spec.max_episode_steps, 1000)
            self.assertEqual(env.spec.reward_threshold, SOLVED_RETURN)
        finally:
            env.close()

    def test_default_environment_arguments_are_untouched(self):
        env = make_lunarlander_env(render_mode=None)
        try:
            unwrapped = env.unwrapped
            self.assertFalse(unwrapped.continuous)
            self.assertEqual(unwrapped.gravity, -10.0)
            self.assertFalse(unwrapped.enable_wind)
            self.assertEqual(unwrapped.wind_power, 15.0)
            self.assertEqual(unwrapped.turbulence_power, 1.5)
        finally:
            env.close()

    def test_isolated_renderer_returns_official_rgb_frames(self):
        renderer = LunarLanderRenderer()
        try:
            observation, _info, frame = renderer.reset(seed=3)
            self.assertEqual(observation.shape, (8,))
            self.assertEqual(frame.shape, (400, 600, 3))
            next_observation, _reward, _terminated, _truncated, _info, next_frame = renderer.step(2)
            self.assertEqual(next_observation.shape, (8,))
            self.assertEqual(next_frame.shape, frame.shape)
        finally:
            renderer.close()


class ArchitectureTests(unittest.TestCase):
    def _workbench(self, algorithm: str, **overrides):
        config = LunarLanderConfig(algorithm=algorithm, total_timesteps=32, learning_starts=0, **overrides)
        return LunarLanderWorkbench(config)

    def test_noisy_linear_changes_with_noise_and_is_stable_without_noise(self):
        layer = NoisyLinear(2, 3)
        values = torch.ones(1, 2)
        first = layer(values)
        layer.reset_noise()
        self.assertFalse(torch.equal(first, layer(values)))
        layer.noise_enabled = False
        torch.testing.assert_close(layer(values), layer(values))

    def test_plain_ddqn_network_has_no_extension_bausteine(self):
        workbench = self._workbench("DDQN")
        try:
            model = workbench.create_model()
            self.assertFalse(model.q_net.noisy)
            self.assertFalse(model.q_net.dueling)
            self.assertFalse(model.q_net.distributional)
            self.assertFalse(any(isinstance(module, NoisyLinear) for module in model.q_net.modules()))
        finally:
            workbench.close()

    def test_dueling_output_has_action_dimension_and_gradients(self):
        workbench = self._workbench("Dueling DDQN")
        try:
            model = workbench.create_model()
            self.assertIsInstance(model.q_net, RainbowCapableQNetwork)
            output = model.q_net(torch.zeros(4, 8, device=model.device))
            self.assertEqual(tuple(output.shape), (4, 4))
            output.sum().backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.q_net.parameters()))
        finally:
            workbench.close()

    def test_c51_network_outputs_a_normalised_distribution_per_action(self):
        workbench = self._workbench("C51 DDQN", net_arch=(8,), c51_atoms=11, c51_v_min=-50.0, c51_v_max=50.0)
        try:
            model = workbench.create_model()
            distribution = model.q_net.distribution(torch.zeros(4, 8, device=model.device))
            self.assertEqual(tuple(distribution.shape), (4, 4, 11))
            torch.testing.assert_close(distribution.sum(dim=-1), torch.ones(4, 4, device=model.device),
                                       atol=1e-5, rtol=1e-5)
            self.assertEqual(tuple(model.q_net(torch.zeros(4, 8, device=model.device)).shape), (4, 4))
        finally:
            workbench.close()

    def test_rainbow_network_combines_noisy_dueling_and_distributional(self):
        workbench = self._workbench("Rainbow DDQN", net_arch=(8,), c51_atoms=9, c51_v_min=-50.0, c51_v_max=50.0)
        try:
            model = workbench.create_model()
            self.assertIsInstance(model.policy, LunarLanderDQNPolicy)
            self.assertTrue(model.q_net.noisy)
            self.assertTrue(model.q_net.dueling)
            self.assertTrue(model.q_net.distributional)
            self.assertTrue(any(isinstance(m, NoisyLinear) for m in model.q_net.value_stream.modules()))
            self.assertTrue(any(isinstance(m, NoisyLinear) for m in model.q_net.advantage_stream.modules()))
            values = torch.zeros(1, 8, device=model.device)
            first = model.q_net(values)
            model.q_net.reset_noise()
            self.assertFalse(torch.equal(first, model.q_net(values)))
        finally:
            workbench.close()


class CategoricalProjectionTests(unittest.TestCase):
    def test_projected_probability_mass_is_conserved(self):
        atoms = torch.linspace(-400.0, 400.0, 51)
        next_probs = torch.softmax(torch.randn(6, 51), dim=1)
        target = project_categorical(next_probs, torch.randn(6, 1) * 50, torch.zeros(6, 1), 0.99, atoms)
        torch.testing.assert_close(target.sum(dim=1), torch.ones(6), atol=1e-5, rtol=1e-5)

    def test_exact_atom_hit_keeps_full_mass_without_loss(self):
        atoms = torch.linspace(-2.0, 2.0, 5)  # -2,-1,0,1,2
        next_probs = torch.zeros(1, 5)
        next_probs[0, 2] = 1.0  # gesamte Masse auf Atomwert 0
        # Bellman-Ziel z = 1.0 + 1.0 * 1.0 * 0.0 = 1.0, also exakt Atomindex 3.
        target = project_categorical(next_probs, torch.tensor([[1.0]]), torch.tensor([[0.0]]), 1.0, atoms)
        self.assertAlmostEqual(float(target[0, 3]), 1.0, places=5)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=5)

    def test_terminal_transitions_ignore_the_bootstrap_distribution(self):
        atoms = torch.linspace(-2.0, 2.0, 5)
        next_probs = torch.softmax(torch.randn(3, 5), dim=1)
        rewards = torch.tensor([[-1.0], [0.0], [2.0]])
        target = project_categorical(next_probs, rewards, torch.ones(3, 1), 0.99, atoms)
        for index in range(3):
            self.assertAlmostEqual(float((target[index] * atoms).sum()), float(rewards[index, 0]), places=4)


class NStepReplayBufferTests(unittest.TestCase):
    def test_full_window_accumulates_n_step_return_with_gamma_power_n_discount(self):
        raw = _raw_buffer()
        buffer = NStepReplayBuffer(raw, n_step=3, gamma=0.5)
        for reward in (1.0, 2.0, 4.0):
            buffer.add(*_transition(reward, done=False))
        self.assertEqual(raw.pos, 1)
        self.assertAlmostEqual(float(raw.rewards[0, 0]), 1.0 + 0.5 * 2.0 + 0.25 * 4.0, places=5)
        self.assertAlmostEqual(float(buffer.discounts[0]), 0.5 ** 3, places=5)

    def test_steady_state_slides_one_transition_per_step(self):
        raw = _raw_buffer()
        buffer = NStepReplayBuffer(raw, n_step=2, gamma=0.9)
        for _ in range(4):
            buffer.add(*_transition(1.0, done=False))
        self.assertEqual(raw.pos, 3)

    def test_episode_end_flushes_shortened_trailing_windows(self):
        raw = _raw_buffer()
        buffer = NStepReplayBuffer(raw, n_step=3, gamma=0.5)
        buffer.add(*_transition(1.0, done=False))
        buffer.add(*_transition(2.0, done=True))
        self.assertEqual(raw.pos, 2)
        self.assertAlmostEqual(float(raw.rewards[0, 0]), 1.0 + 0.5 * 2.0, places=5)
        self.assertAlmostEqual(float(raw.rewards[1, 0]), 2.0, places=5)

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
        samples, indices, _weights = buffer.sample(4, beta=0.4)
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
        self.assertIsInstance(restored.buffer, PrioritizedReplayBuffer)


class PERTests(unittest.TestCase):
    def test_priority_updates_add_epsilon(self):
        wrapper = PrioritizedReplayBuffer(Mock(buffer_size=10), alpha=0.6, epsilon=1e-6)
        wrapper.update_priorities(np.array([2, 5]), torch.tensor([[3.0], [-4.0]]))
        self.assertAlmostEqual(float(wrapper.priorities[2]), 3.000001, places=5)
        self.assertAlmostEqual(float(wrapper.priorities[5]), 4.000001, places=5)

    def test_priority_can_decrease_and_duplicate_uses_largest_new_value(self):
        wrapper = PrioritizedReplayBuffer(Mock(buffer_size=10), alpha=0.6, epsilon=1e-6)
        wrapper.priorities[2] = 20.0
        wrapper.update_priorities(np.array([2, 2]), torch.tensor([[1.0], [3.0]]))
        self.assertAlmostEqual(float(wrapper.priorities[2]), 3.000001, places=5)


class DistributionalTargetTests(unittest.TestCase):
    def test_target_distribution_is_normalised(self):
        workbench = LunarLanderWorkbench(LunarLanderConfig(
            algorithm="C51 DDQN", total_timesteps=32, learning_starts=0,
            net_arch=(8,), c51_atoms=11, c51_v_min=-50.0, c51_v_max=50.0,
        ))
        try:
            model = workbench.create_model()
            replay_data = SimpleNamespace(
                next_observations=torch.zeros(6, 8, device=model.device),
                rewards=torch.zeros(6, 1, device=model.device),
                dones=torch.zeros(6, 1, device=model.device),
            )
            target = model._distributional_target(replay_data, model.gamma)
            self.assertEqual(tuple(target.shape), (6, 11))
            torch.testing.assert_close(target.sum(dim=1), torch.ones(6, device=model.device),
                                       atol=1e-4, rtol=1e-4)
        finally:
            workbench.close()


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        # LunarLander-Episoden enden meist deutlich vor dem Limit von 1000
        # Schritten (Absturz), sodass dieses Budget zuverlässig mindestens
        # eine vollständige Episode liefert.
        self.config = LunarLanderConfig(
            total_timesteps=600, buffer_size=800, learning_starts=32,
            batch_size=16, train_freq=8, gradient_steps=1,
            target_update_interval=64, net_arch=(32,), seed=7,
            per_beta_steps=600, value_arch=(16,), advantage_arch=(16,),
            multistep_n=3, c51_atoms=11, c51_v_min=-400.0, c51_v_max=400.0,
        )

    def test_all_seven_variants_train_with_the_correct_bausteine(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                workbench = LunarLanderWorkbench(replace(self.config, algorithm=algorithm))
                try:
                    metrics = workbench.train()
                    self.assertGreater(workbench.model.num_timesteps, 0)
                    self.assertGreater(len(metrics), 0)
                    self.assertEqual(algorithm in NOISY_ALGORITHMS, workbench.model.q_net.noisy)
                    self.assertEqual(algorithm in DUELING_ALGORITHMS, workbench.model.q_net.dueling)
                    self.assertEqual(algorithm in DISTRIBUTIONAL_ALGORITHMS, workbench.model.q_net.distributional)
                    self.assertEqual(algorithm in MULTISTEP_ALGORITHMS,
                                     isinstance(workbench.model.replay_buffer, NStepReplayBuffer))
                    self.assertEqual(algorithm in PER_ALGORITHMS, _has_per(workbench.model.replay_buffer))
                    for parameter in workbench.model.policy.parameters():
                        self.assertFalse(torch.isnan(parameter).any(), f"NaN nach Training von {algorithm}")
                finally:
                    workbench.close()

    def test_evaluation_does_not_change_parameters_or_replay(self):
        workbench = LunarLanderWorkbench(self.config)
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

    def test_evaluation_reports_all_lunarlander_rates(self):
        workbench = LunarLanderWorkbench(self.config)
        try:
            workbench.train()
            result = workbench.evaluate(3, seed=5)
            for rate in (result.landing_rate, result.solved_rate, result.crash_rate):
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 1.0)
            self.assertLessEqual(result.landing_rate + result.crash_rate, 1.0)
            self.assertGreater(result.mean_length, 0)
        finally:
            workbench.close()

    def test_automatic_evaluation_runs_at_the_configured_interval(self):
        workbench = LunarLanderWorkbench(replace(self.config, total_timesteps=300))
        seen: list = []
        try:
            workbench.train(evaluation_interval=100, evaluation_episodes=1,
                            best_callback=lambda episode, steps, result: seen.append(result))
            self.assertGreaterEqual(len(seen), 2)
        finally:
            workbench.close()

    def test_save_load_and_continue_for_every_algorithm(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                workbench = LunarLanderWorkbench(replace(self.config, algorithm=algorithm))
                try:
                    workbench.train()
                    with tempfile.TemporaryDirectory() as directory:
                        base = Path(directory) / "lunarlander"
                        workbench.save(base)
                        loaded = LunarLanderWorkbench.load(base)
                        try:
                            old = loaded.model.num_timesteps
                            loaded.train()
                            self.assertGreater(loaded.model.num_timesteps, old)
                            self.assertEqual(type(loaded.model.replay_buffer),
                                             type(workbench.model.replay_buffer))
                        finally:
                            loaded.close()
                finally:
                    workbench.close()

    def test_incompatible_checkpoint_is_rejected_with_a_clear_message(self):
        workbench = LunarLanderWorkbench(self.config)
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "lunarlander"
                workbench.save(base)
                metadata = base.with_name(base.name + "_metadata.json")
                metadata.write_text('{"format": 1, "environment": "CartPole-v1", "config": {}}', encoding="utf-8")
                with self.assertRaises(ValueError) as context:
                    LunarLanderWorkbench.load(base)
                self.assertIn("CartPole-v1", str(context.exception))
        finally:
            workbench.close()


if __name__ == "__main__":
    unittest.main()
