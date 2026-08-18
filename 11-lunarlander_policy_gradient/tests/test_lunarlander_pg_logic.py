import inspect
import math
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from lunarlander_pg_logic import (
    ACTION_DIM,
    DEFAULT_NET_ARCH,
    DEFAULT_TOTAL_TIMESTEPS,
    ALGORITHM_CLASSES,
    ALGORITHMS,
    CONTINUOUS,
    ENV_ID,
    OFF_POLICY_ALGORITHMS,
    SOLVED_RETURN,
    LinearSchedule,
    LunarLanderPGCallback,
    LunarLanderPGConfig,
    LunarLanderPGWorkbench,
    action_readout,
    config_differences,
    default_config,
    episode_outcome,
    make_action_noise,
    make_lunarlander_env,
    observation_readout,
)
from lunarlander_pg_render import LunarLanderRenderer


def _small(algorithm: str, **overrides) -> LunarLanderPGConfig:
    """Kleine, schnell laufende Variante des jeweiligen Profils."""
    base = dict(
        total_timesteps=400, batch_size=16, actor_arch=(32,), critic_arch=(32,), seed=7,
    )
    if algorithm == "PPO":
        base.update(n_steps=64)
    else:
        base.update(buffer_size=1_000, learning_starts=32, train_freq=16, gradient_steps=1)
    base.update(overrides)
    return replace(default_config(algorithm), **base)


class ProfileTests(unittest.TestCase):
    """Die Defaults stammen aus den `LunarLanderContinuous-v3`-Profilen des Zoo."""

    def test_ppo_profile_matches_rl_zoo(self):
        config = default_config("PPO")
        self.assertEqual(config.n_steps, 1024)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.gae_lambda, 0.98)
        self.assertEqual(config.gamma, 0.999)
        self.assertEqual(config.n_epochs, 4)
        self.assertEqual(config.ent_coef, 0.01)
        self.assertEqual(config.activation, "Tanh")

    def test_td3_profile_matches_rl_zoo(self):
        config = default_config("TD3")
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.gamma, 0.98)
        self.assertEqual(config.buffer_size, 200_000)
        self.assertEqual(config.learning_starts, 10_000)
        self.assertEqual(config.action_noise, "normal")
        self.assertEqual(config.action_noise_sigma, 0.1)

    def test_sac_profile_matches_rl_zoo_including_linear_learning_rate(self):
        config = default_config("SAC")
        self.assertEqual(config.learning_rate, 7.3e-4)
        self.assertEqual(config.learning_rate_schedule, "linear fallend")
        self.assertEqual(config.buffer_size, 1_000_000)
        self.assertEqual(config.tau, 0.01)
        self.assertEqual(config.ent_coef_mode, "auto")

    def test_every_algorithm_has_its_own_profile_and_validates(self):
        self.assertEqual(len(ALGORITHMS), 3)
        for algorithm in ALGORITHMS:
            config = default_config(algorithm)
            config.validate()
            self.assertEqual(config.algorithm, algorithm)

    def test_all_algorithms_share_budget_and_network_size(self):
        """Gemeinsames Budget und gemeinsame Netzgröße machen den Vergleich
        ohne Zutun fair; die getunten Hyperparameter bleiben verfahrenseigen."""
        self.assertEqual(DEFAULT_TOTAL_TIMESTEPS, 100_000)
        self.assertEqual(DEFAULT_NET_ARCH, (64, 64))
        for algorithm in ALGORITHMS:
            config = default_config(algorithm)
            self.assertEqual(config.total_timesteps, DEFAULT_TOTAL_TIMESTEPS)
            self.assertEqual(config.actor_arch, DEFAULT_NET_ARCH)
            self.assertEqual(config.critic_arch, DEFAULT_NET_ARCH)

    def test_unknown_algorithm_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            default_config("DQN")
        self.assertIn("DQN", str(context.exception))

    def test_linear_schedule_falls_from_initial_value_to_zero(self):
        schedule = LinearSchedule(7.3e-4)
        self.assertAlmostEqual(schedule(1.0), 7.3e-4)
        self.assertAlmostEqual(schedule(0.5), 3.65e-4)
        self.assertAlmostEqual(schedule(0.0), 0.0)
        self.assertIsInstance(default_config("SAC").learning_rate_value(), LinearSchedule)
        self.assertEqual(default_config("PPO").learning_rate_value(), 3e-4)


class ConfigTests(unittest.TestCase):
    def test_model_kwargs_only_use_arguments_the_algorithm_knows(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                parameters = inspect.signature(ALGORITHM_CLASSES[algorithm].__init__).parameters
                for key in default_config(algorithm).model_kwargs():
                    self.assertIn(key, parameters, f"{algorithm} kennt '{key}' nicht")

    def test_ppo_has_no_replay_parameters_and_off_policy_no_rollout_parameters(self):
        ppo = default_config("PPO").model_kwargs()
        for key in ("buffer_size", "learning_starts", "tau", "action_noise"):
            self.assertNotIn(key, ppo)
        for algorithm in OFF_POLICY_ALGORITHMS:
            kwargs = default_config(algorithm).model_kwargs()
            for key in ("n_steps", "n_epochs", "gae_lambda", "clip_range", "max_grad_norm"):
                self.assertNotIn(key, kwargs)

    def test_uses_marks_only_the_parameters_of_the_chosen_algorithm(self):
        ppo, td3, sac = (default_config(name) for name in ALGORITHMS)
        self.assertTrue(ppo.uses("n_steps"))
        self.assertFalse(ppo.uses("buffer_size"))
        self.assertTrue(td3.uses("policy_delay"))
        self.assertFalse(td3.uses("ent_coef_mode"))
        self.assertFalse(td3.uses("use_sde"))
        self.assertTrue(sac.uses("ent_coef_mode"))
        self.assertTrue(sac.uses("use_sde"))
        for config in (ppo, td3, sac):
            self.assertTrue(config.uses("learning_rate"))

    def test_network_architecture_uses_the_value_key_of_the_algorithm(self):
        self.assertEqual(set(default_config("PPO").policy_kwargs()["net_arch"]), {"pi", "vf"})
        self.assertEqual(set(default_config("SAC").policy_kwargs()["net_arch"]), {"pi", "qf"})

    def test_sac_entropy_settings_are_translated_for_stable_baselines3(self):
        automatic = default_config("SAC")
        self.assertEqual(automatic.sac_ent_coef(), "auto_1.0")
        self.assertEqual(automatic.sac_target_entropy(), "auto")
        fixed = replace(automatic, ent_coef_mode="fest", ent_coef_value=0.2, target_entropy="-2")
        self.assertEqual(fixed.sac_ent_coef(), 0.2)
        self.assertEqual(fixed.sac_target_entropy(), -2.0)

    def test_signature_ignores_the_budget_but_reacts_to_model_parameters(self):
        config = default_config("SAC")
        self.assertEqual(config.signature(), replace(config, total_timesteps=1).signature())
        self.assertNotEqual(config.signature(), replace(config, tau=0.02).signature())
        self.assertNotEqual(config.signature(), replace(config, algorithm="TD3").signature())

    def test_signature_ignores_parameters_the_algorithm_does_not_use(self):
        config = default_config("PPO")
        self.assertEqual(config.signature(), replace(config, buffer_size=123).signature())

    def test_td3_without_action_noise_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            replace(default_config("TD3"), action_noise="keins").validate()
        message = str(context.exception)
        self.assertIn("Action Noise", message)
        self.assertIn("deterministisch", message)

    def test_validation_messages_name_field_value_and_range(self):
        cases = (
            ("PPO", dict(batch_size=100), ("Batch-Größe B", "100")),
            ("PPO", dict(gamma=1.5), ("Diskontfaktor γ", "1.5")),
            ("PPO", dict(clip_range=0.0), ("Clip ε_clip", "0.0")),
            ("TD3", dict(tau=0.0), ("Soft-Update τ", "0.0")),
            ("TD3", dict(learning_starts=-1), ("Lernstart t₀", "-1")),
            ("SAC", dict(target_entropy="viel"), ("Zielentropie H*", "viel")),
            ("SAC", dict(batch_size=2_000_000), ("Batch-Größe B", "2000000")),
        )
        for algorithm, overrides, (field, value) in cases:
            with self.subTest(algorithm=algorithm, overrides=overrides):
                with self.assertRaises(ValueError) as context:
                    replace(default_config(algorithm), **overrides).validate()
                message = str(context.exception)
                self.assertIn(field, message)
                self.assertIn(value, message)
                self.assertIn("Gültig", message)

    def test_ppo_batch_size_must_divide_the_rollout(self):
        with self.assertRaises(ValueError) as context:
            replace(default_config("PPO"), n_steps=1000, batch_size=64).validate()
        self.assertIn("Teiler von 1000", str(context.exception))

    def test_action_noise_factory_matches_the_action_dimension(self):
        self.assertIsNone(make_action_noise("keins", 0.1))
        self.assertIsNone(make_action_noise("normal", 0.0))
        noise = make_action_noise("normal", 0.1)
        self.assertEqual(noise().shape, (ACTION_DIM,))


class DifferenceTests(unittest.TestCase):
    def test_same_algorithm_lists_exactly_the_changed_parameters(self):
        first = default_config("SAC")
        second = replace(first, learning_rate=1e-4, tau=0.02)
        names = [name for name, _, _ in config_differences(first, second)]
        self.assertEqual(names, ["Lernrate α", "Soft-Update τ"])

    def test_identical_configurations_have_no_differences(self):
        self.assertEqual(config_differences(default_config("TD3"), default_config("TD3")), [])

    def test_algorithm_exclusive_parameters_are_not_reported_as_differences(self):
        names = [name for name, _, _ in config_differences(default_config("PPO"), default_config("SAC"))]
        self.assertIn("Lernrate α", names)
        self.assertNotIn("Rollout n_steps", names)
        self.assertNotIn("Replay Buffer |D|", names)


class ReadoutTests(unittest.TestCase):
    def test_main_engine_is_off_for_non_positive_values(self):
        for value in (-1.0, -0.4, 0.0):
            readout = action_readout(np.array([value, 0.0]))
            self.assertEqual(readout["main_power"], 0.0)
            self.assertEqual(readout["main_text"], "aus")

    def test_main_engine_throttles_between_fifty_and_hundred_percent(self):
        self.assertAlmostEqual(action_readout(np.array([0.001, 0.0]))["main_power"], 0.5005, places=4)
        self.assertAlmostEqual(action_readout(np.array([1.0, 0.0]))["main_power"], 1.0)
        self.assertAlmostEqual(action_readout(np.array([0.5, 0.0]))["main_power"], 0.75)

    def test_lateral_engines_need_a_magnitude_above_one_half(self):
        self.assertEqual(action_readout(np.array([0.0, 0.5]))["lateral_side"], "aus")
        self.assertEqual(action_readout(np.array([0.0, -0.5]))["lateral_side"], "aus")
        self.assertEqual(action_readout(np.array([0.0, -0.8]))["lateral_side"], "links")
        self.assertEqual(action_readout(np.array([0.0, 0.8]))["lateral_side"], "rechts")
        self.assertAlmostEqual(action_readout(np.array([0.0, 1.0]))["lateral_power"], 1.0)

    def test_values_outside_the_box_are_clipped(self):
        readout = action_readout(np.array([5.0, -5.0]))
        self.assertEqual(readout["main_raw"], 1.0)
        self.assertEqual(readout["lateral_raw"], -1.0)

    def test_observation_readout_converts_angle_and_angular_velocity(self):
        observation = np.array([0.1, 0.2, 0.3, 0.4, math.pi / 2, 1.0, 1.0, 0.0], dtype=np.float32)
        values = observation_readout(observation)
        self.assertAlmostEqual(values["angle_degrees"], 90.0, places=4)
        self.assertAlmostEqual(values["angular_velocity"], 2.5, places=4)
        self.assertTrue(values["left_leg"])
        self.assertFalse(values["right_leg"])

    def test_landing_crash_and_time_limit_are_distinguished(self):
        self.assertEqual(episode_outcome(100.0, False, 250.0), (True, False, True))
        self.assertEqual(episode_outcome(-100.0, False, -150.0), (False, True, False))
        self.assertEqual(episode_outcome(-0.5, True, 50.0)[:2], (False, False))

    def test_solved_uses_the_official_threshold_of_200(self):
        self.assertTrue(episode_outcome(100.0, False, SOLVED_RETURN)[2])
        self.assertFalse(episode_outcome(100.0, False, SOLVED_RETURN - 0.1)[2])
        self.assertTrue(episode_outcome(100.0, False, 120.0)[0], "Landung ohne Lösung bleibt Landung")


class CallbackTests(unittest.TestCase):
    def _callback_with(self, reward: float, info: dict) -> LunarLanderPGCallback:
        callback = LunarLanderPGCallback(threading.Event())
        callback.model = Mock(num_timesteps=1)
        callback.num_timesteps = 1
        callback.locals = {"rewards": np.array([reward]), "dones": np.array([True]), "infos": [info]}
        callback._on_step()
        return callback

    def test_landing_and_crash_are_recorded(self):
        self.assertTrue(self._callback_with(100.0, {}).metrics[0].landed)
        self.assertTrue(self._callback_with(-100.0, {}).metrics[0].crashed)

    def test_truncated_episode_is_neither_landed_nor_crashed(self):
        metric = self._callback_with(-0.4, {"TimeLimit.truncated": True}).metrics[0]
        self.assertFalse(metric.landed); self.assertFalse(metric.crashed)

    def test_stop_event_ends_the_run(self):
        stop = threading.Event(); stop.set()
        callback = LunarLanderPGCallback(stop)
        callback.model = Mock(num_timesteps=1); callback.num_timesteps = 1
        callback.locals = {"rewards": np.array([0.0]), "dones": np.array([False]), "infos": [{}]}
        self.assertFalse(callback._on_step())


class EnvironmentTests(unittest.TestCase):
    @patch("lunarlander_pg_logic.gymnasium.make")
    def test_factory_always_requests_the_continuous_variant(self, make_mock):
        make_mock.return_value = Mock()
        make_lunarlander_env()
        make_mock.assert_called_once_with(ENV_ID, continuous=True, render_mode="rgb_array")

    def test_continuous_spaces_reward_threshold_and_time_limit(self):
        env = make_lunarlander_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=3)
            self.assertEqual(observation.shape, (8,))
            self.assertEqual(env.action_space.shape, (ACTION_DIM,))
            self.assertAlmostEqual(float(env.action_space.low[0]), -1.0)
            self.assertAlmostEqual(float(env.action_space.high[0]), 1.0)
            self.assertEqual(env.spec.max_episode_steps, 1000)
            self.assertEqual(env.spec.reward_threshold, SOLVED_RETURN)
        finally:
            env.close()

    def test_default_environment_arguments_are_untouched(self):
        env = make_lunarlander_env(render_mode=None)
        try:
            unwrapped = env.unwrapped
            self.assertEqual(unwrapped.continuous, CONTINUOUS)
            self.assertEqual(unwrapped.gravity, -10.0)
            self.assertFalse(unwrapped.enable_wind)
            self.assertEqual(unwrapped.wind_power, 15.0)
            self.assertEqual(unwrapped.turbulence_power, 1.5)
        finally:
            env.close()

    def test_isolated_renderer_accepts_continuous_actions(self):
        renderer = LunarLanderRenderer()
        try:
            observation, _info, frame = renderer.reset(seed=3)
            self.assertEqual(observation.shape, (8,))
            self.assertEqual(frame.shape, (400, 600, 3))
            next_observation, _reward, _terminated, _truncated, _info, next_frame = renderer.step(
                np.array([0.7, -0.9], dtype=np.float32))
            self.assertEqual(next_observation.shape, (8,))
            self.assertEqual(next_frame.shape, frame.shape)
        finally:
            renderer.close()


class ModelTests(unittest.TestCase):
    def test_only_off_policy_algorithms_own_a_replay_buffer(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                workbench = LunarLanderPGWorkbench(_small(algorithm))
                try:
                    model = workbench.create_model()
                    expected = algorithm in OFF_POLICY_ALGORITHMS
                    self.assertEqual(expected, workbench.uses_replay_buffer)
                    self.assertEqual(expected, getattr(model, "replay_buffer", None) is not None)
                finally:
                    workbench.close()

    def test_network_and_optimizer_settings_reach_the_policy(self):
        workbench = LunarLanderPGWorkbench(_small("SAC", actor_arch=(24, 12), critic_arch=(16,),
                                                  optimizer_eps=1e-6))
        try:
            model = workbench.create_model()
            self.assertEqual(model.policy.net_arch, {"pi": [24, 12], "qf": [16]})
            self.assertAlmostEqual(model.actor.optimizer.defaults["eps"], 1e-6)
        finally:
            workbench.close()

    def test_td3_parameters_reach_the_model(self):
        workbench = LunarLanderPGWorkbench(_small("TD3", policy_delay=3, target_policy_noise=0.3,
                                                  target_noise_clip=0.4, action_noise_sigma=0.25))
        try:
            model = workbench.create_model()
            self.assertEqual(model.policy_delay, 3)
            self.assertAlmostEqual(model.target_policy_noise, 0.3)
            self.assertAlmostEqual(model.target_noise_clip, 0.4)
            self.assertAlmostEqual(float(model.action_noise._sigma[0]), 0.25)
        finally:
            workbench.close()

    def test_sac_automatic_target_entropy_is_minus_action_dimension(self):
        workbench = LunarLanderPGWorkbench(_small("SAC"))
        try:
            model = workbench.create_model()
            self.assertAlmostEqual(model.target_entropy, -float(ACTION_DIM))
            self.assertIsNotNone(model.ent_coef_optimizer)
        finally:
            workbench.close()

    def test_sac_fixed_entropy_coefficient_disables_the_tuning_optimizer(self):
        workbench = LunarLanderPGWorkbench(_small("SAC", ent_coef_mode="fest", ent_coef_value=0.2))
        try:
            model = workbench.create_model()
            self.assertIsNone(model.ent_coef_optimizer)
            self.assertAlmostEqual(float(model.ent_coef_tensor), 0.2)
        finally:
            workbench.close()

    def test_ppo_objective_parameters_reach_the_model(self):
        workbench = LunarLanderPGWorkbench(_small("PPO", clip_range=0.15, gae_lambda=0.9,
                                                  n_epochs=7, ent_coef=0.02))
        try:
            model = workbench.create_model()
            self.assertAlmostEqual(model.clip_range(1.0), 0.15)
            self.assertAlmostEqual(model.gae_lambda, 0.9)
            self.assertEqual(model.n_epochs, 7)
            self.assertAlmostEqual(model.ent_coef, 0.02)
        finally:
            workbench.close()

    def test_td3_action_noise_does_not_reach_the_deterministic_evaluation(self):
        workbench = LunarLanderPGWorkbench(_small("TD3", action_noise_sigma=0.9))
        try:
            model = workbench.create_model()
            observation = np.zeros(8, dtype=np.float32)
            first, _ = model.predict(observation, deterministic=True)
            second, _ = model.predict(observation, deterministic=True)
            np.testing.assert_allclose(first, second)
            self.assertAlmostEqual(float(model.action_noise._sigma[0]), 0.9)
        finally:
            workbench.close()


class BehaviourTests(unittest.TestCase):
    def test_td3_updates_the_actor_only_every_policy_delay_steps(self):
        workbench = LunarLanderPGWorkbench(_small(
            "TD3", total_timesteps=64, learning_starts=8, train_freq=64, gradient_steps=4,
            batch_size=8, policy_delay=2,
        ))
        try:
            model = workbench.create_model()
            calls = []
            original = model.actor.optimizer.step

            def counting_step(*args, **kwargs):
                calls.append(1)
                return original(*args, **kwargs)

            model.actor.optimizer.step = counting_step
            workbench.train()
            self.assertEqual(len(calls), 2, "4 Gradientenschritte bei Delay 2 = 2 Actor-Updates")
        finally:
            workbench.close()

    def test_sac_tunes_the_entropy_temperature_during_training(self):
        # Konstante Lernrate: Das SAC-Profil fährt die Lernrate linear auf 0,
        # und am Ende eines so kurzen Budgets wäre sie bereits 0.
        workbench = LunarLanderPGWorkbench(_small(
            "SAC", total_timesteps=64, learning_starts=8, train_freq=32, gradient_steps=4,
            batch_size=8, learning_rate_schedule="konstant",
        ))
        try:
            model = workbench.create_model()
            before = model.log_ent_coef.detach().clone()
            workbench.train()
            self.assertFalse(torch.equal(before, model.log_ent_coef))
        finally:
            workbench.close()


class IntegrationTests(unittest.TestCase):
    def test_all_three_algorithms_train_and_report_episodes(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                workbench = LunarLanderPGWorkbench(_small(algorithm))
                try:
                    metrics = workbench.train()
                    self.assertGreater(workbench.model.num_timesteps, 0)
                    self.assertGreater(len(metrics), 0)
                    for parameter in workbench.model.policy.parameters():
                        self.assertFalse(torch.isnan(parameter).any(), f"NaN nach {algorithm}")
                finally:
                    workbench.close()

    def test_evaluation_is_deterministic_and_changes_no_learning_state(self):
        workbench = LunarLanderPGWorkbench(_small("SAC"))
        try:
            workbench.train()
            before = {name: value.detach().clone()
                      for name, value in workbench.model.policy.state_dict().items()}
            size = workbench.model.replay_buffer.size()
            observation = np.zeros(8, dtype=np.float32)
            first, _ = workbench.model.predict(observation, deterministic=True)
            second, _ = workbench.model.predict(observation, deterministic=True)
            np.testing.assert_allclose(first, second)
            result = workbench.evaluate(2, seed=10)
            self.assertEqual(result.episodes, 2)
            self.assertEqual(workbench.model.replay_buffer.size(), size)
            for name, value in before.items():
                torch.testing.assert_close(value, workbench.model.policy.state_dict()[name])
        finally:
            workbench.close()

    def test_evaluation_reports_all_lunarlander_rates(self):
        workbench = LunarLanderPGWorkbench(_small("TD3"))
        try:
            workbench.train()
            result = workbench.evaluate(2, seed=5)
            for rate in (result.landing_rate, result.solved_rate, result.crash_rate):
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 1.0)
            self.assertLessEqual(result.landing_rate + result.crash_rate, 1.0)
            self.assertGreater(result.mean_length, 0)
        finally:
            workbench.close()

    def test_automatic_evaluation_runs_at_the_configured_interval(self):
        workbench = LunarLanderPGWorkbench(_small("PPO", total_timesteps=192, n_steps=64))
        seen: list = []
        try:
            workbench.train(evaluation_interval=64, evaluation_episodes=1,
                            best_callback=lambda episode, steps, result: seen.append(result))
            self.assertGreaterEqual(len(seen), 2)
        finally:
            workbench.close()

    def test_save_load_and_continue_for_every_algorithm(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                workbench = LunarLanderPGWorkbench(_small(algorithm))
                try:
                    workbench.train()
                    with tempfile.TemporaryDirectory() as directory:
                        base = Path(directory) / "lunarlander_pg"
                        _model_path, replay_path, _metadata = workbench.save(base)
                        expected = algorithm in OFF_POLICY_ALGORITHMS
                        self.assertEqual(expected, replay_path is not None)
                        loaded = LunarLanderPGWorkbench.load(base, algorithm)
                        try:
                            self.assertEqual(loaded.config, workbench.config)
                            if expected:
                                self.assertEqual(loaded.model.replay_buffer.size(),
                                                 workbench.model.replay_buffer.size())
                            old = loaded.model.num_timesteps
                            loaded.train()
                            self.assertGreater(loaded.model.num_timesteps, old)
                        finally:
                            loaded.close()
                finally:
                    workbench.close()

    def test_loading_into_a_slot_with_another_algorithm_is_rejected(self):
        workbench = LunarLanderPGWorkbench(_small("PPO"))
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "lunarlander_pg"
                workbench.save(base)
                with self.assertRaises(ValueError) as context:
                    LunarLanderPGWorkbench.load(base, "SAC")
                message = str(context.exception)
                self.assertIn("PPO", message)
                self.assertIn("SAC", message)
        finally:
            workbench.close()

    def test_missing_replay_buffer_of_an_off_policy_model_is_rejected(self):
        workbench = LunarLanderPGWorkbench(_small("TD3"))
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "lunarlander_pg"
                _model, replay_path, _metadata = workbench.save(base)
                replay_path.unlink()
                with self.assertRaises(ValueError) as context:
                    LunarLanderPGWorkbench.load(base, "TD3")
                self.assertIn(replay_path.name, str(context.exception))
        finally:
            workbench.close()

    def test_incompatible_checkpoint_is_rejected_with_a_clear_message(self):
        workbench = LunarLanderPGWorkbench(_small("PPO"))
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "lunarlander_pg"
                workbench.save(base)
                metadata = base.with_name(base.name + "_metadata.json")
                metadata.write_text(
                    '{"format": 1, "environment": "LunarLander-v3", "continuous": false, '
                    '"algorithm": "PPO", "config": {}}', encoding="utf-8")
                with self.assertRaises(ValueError) as context:
                    LunarLanderPGWorkbench.load(base)
                self.assertIn("continuous", str(context.exception))
        finally:
            workbench.close()

    def test_two_slots_with_the_same_algorithm_stay_independent(self):
        """Kern des Zwei-Slot-Vergleichs: gleicher Algorithmus, andere Parameter."""
        first = LunarLanderPGWorkbench(_small("SAC", learning_rate=1e-4, seed=1))
        second = LunarLanderPGWorkbench(_small("SAC", learning_rate=1e-3, seed=2))
        try:
            first.train(); second.train()
            self.assertIsNot(first.model, second.model)
            self.assertNotEqual(first.config.signature(), second.config.signature())
            self.assertGreater(len(first.history), 0)
            self.assertGreater(len(second.history), 0)
            self.assertNotEqual(
                [item.reward for item in first.history], [item.reward for item in second.history],
                "Zwei unabhängige Läufe dürfen nicht dieselbe Historie liefern.",
            )
            differences = config_differences(first.config, second.config)
            self.assertIn("Lernrate α", [name for name, _, _ in differences])
        finally:
            first.close(); second.close()


if __name__ == "__main__":
    unittest.main()
