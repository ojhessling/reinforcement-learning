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

from bipedalwalker_logic import (
    ACTION_DIM,
    ALGORITHM_CLASSES,
    ALGORITHMS,
    DEFAULT_NET_ARCH,
    DEFAULT_TOTAL_TIMESTEPS,
    ENV_ID,
    FALL_REWARD_THRESHOLD,
    HARDCORE,
    JOINT_NAMES,
    LIDAR_COUNT,
    OBSERVATION_DIM,
    OFF_POLICY_ALGORITHMS,
    SOLVED_RETURN,
    BipedalWalkerCallback,
    BipedalWalkerConfig,
    BipedalWalkerWorkbench,
    LinearSchedule,
    action_readout,
    config_differences,
    default_config,
    episode_outcome,
    make_action_noise,
    make_bipedalwalker_env,
    observation_readout,
)
from bipedalwalker_render import BipedalWalkerRenderer


def _small(algorithm: str, **overrides) -> BipedalWalkerConfig:
    """Kleine, schnell laufende Variante des jeweiligen Profils.

    PPO bekommt ein größeres Budget: Seine Anfangs-Policy lässt den Roboter
    stehen statt stürzen, sodass die erste Episode erst am Zeitlimit von 1600
    Schritten endet.
    """
    base = dict(
        total_timesteps=1700 if algorithm == "PPO" else 600,
        batch_size=16, actor_arch=(32,), critic_arch=(32,), seed=7,
    )
    if algorithm == "PPO":
        base.update(n_steps=64)
    else:
        base.update(buffer_size=1_000, learning_starts=32, train_freq=16, gradient_steps=1)
    base.update(overrides)
    return replace(default_config(algorithm), **base)


class ProfileTests(unittest.TestCase):
    """Die Defaults stammen aus den `BipedalWalker-v3`-Profilen des Zoo."""

    def test_ppo_profile_matches_rl_zoo(self):
        config = default_config("PPO")
        self.assertEqual(config.n_steps, 2048)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.gae_lambda, 0.95)
        self.assertEqual(config.gamma, 0.999)
        self.assertEqual(config.n_epochs, 10)
        self.assertEqual(config.clip_range, 0.18)
        self.assertEqual(config.ent_coef, 0.0)
        self.assertEqual(config.activation, "Tanh")

    def test_ppo_profile_normalises_because_the_zoo_profile_demands_it(self):
        config = default_config("PPO")
        self.assertTrue(config.normalize_obs)
        self.assertTrue(config.normalize_reward)
        self.assertTrue(config.normalizes)

    def test_off_policy_profiles_do_not_normalise_by_default(self):
        """Ein Replay Buffer speichert Beobachtungen, deren Statistik driftet."""
        for algorithm in OFF_POLICY_ALGORITHMS:
            config = default_config(algorithm)
            self.assertFalse(config.normalizes, algorithm)

    def test_td3_profile_matches_rl_zoo(self):
        config = default_config("TD3")
        self.assertEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.gamma, 0.98)
        self.assertEqual(config.buffer_size, 200_000)
        self.assertEqual(config.learning_starts, 10_000)
        self.assertEqual(config.action_noise, "normal")
        self.assertEqual(config.action_noise_sigma, 0.1)

    def test_sac_profile_matches_rl_zoo_including_gsde(self):
        config = default_config("SAC")
        self.assertEqual(config.learning_rate, 7.3e-4)
        self.assertEqual(config.buffer_size, 300_000)
        self.assertEqual(config.gamma, 0.98)
        self.assertEqual(config.tau, 0.02)
        self.assertEqual(config.train_freq, 64)
        self.assertEqual(config.gradient_steps, 64)
        self.assertEqual(config.ent_coef_mode, "auto")
        self.assertTrue(config.use_sde)
        self.assertEqual(config.log_std_init, -3.0)

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
        self.assertEqual(default_config("SAC").learning_rate_value(), 7.3e-4)


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
        self.assertNotEqual(config.signature(), replace(config, tau=0.05).signature())
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
        second = replace(first, learning_rate=1e-4, tau=0.05)
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
    def test_every_joint_gets_direction_and_torque_share(self):
        readout = action_readout(np.array([0.5, -1.0, 0.0, 0.25]))
        self.assertEqual([joint["joint"] for joint in readout["joints"]], list(JOINT_NAMES))
        self.assertEqual(readout["joints"][0]["direction"], "vor")
        self.assertEqual(readout["joints"][1]["direction"], "zurück")
        self.assertEqual(readout["joints"][2]["text"], "aus")
        self.assertAlmostEqual(readout["joints"][1]["torque"], 1.0)
        self.assertAlmostEqual(readout["joints"][3]["torque"], 0.25)

    def test_torque_values_outside_the_box_are_clipped(self):
        readout = action_readout(np.array([5.0, -5.0, 0.0, 0.0]))
        self.assertEqual(readout["raw"][:2], [1.0, -1.0])

    def test_zero_torque_means_no_moment_not_holding_the_joint(self):
        readout = action_readout(np.zeros(ACTION_DIM))
        for joint in readout["joints"]:
            self.assertEqual(joint["torque"], 0.0)
            self.assertEqual(joint["text"], "aus")

    def test_observation_readout_splits_all_twenty_four_values(self):
        observation = np.arange(OBSERVATION_DIM, dtype=np.float32) / 10
        values = observation_readout(observation)
        self.assertAlmostEqual(values["hull_angle"], 0.0)
        self.assertAlmostEqual(values["hull_angular_velocity"], 0.1)
        self.assertAlmostEqual(values["vx"], 0.2)
        self.assertAlmostEqual(values["vy"], 0.3)
        self.assertEqual(len(values["legs"]), 2)
        self.assertEqual(len(values["lidar"]), LIDAR_COUNT)

    def test_observation_readout_reports_ground_contact_per_leg(self):
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        observation[8] = 1.0   # Bein 1 hat Bodenkontakt
        observation[13] = 0.0  # Bein 2 nicht
        legs = observation_readout(observation)["legs"]
        self.assertTrue(legs[0]["contact"])
        self.assertFalse(legs[1]["contact"])

    def test_hull_angle_is_also_reported_in_degrees(self):
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        observation[0] = math.pi / 4
        self.assertAlmostEqual(observation_readout(observation)["hull_angle_degrees"], 45.0, places=4)

    def test_lidar_values_are_passed_through_unchanged(self):
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        expected = np.linspace(0.1, 1.0, LIDAR_COUNT)
        observation[-LIDAR_COUNT:] = expected
        np.testing.assert_allclose(observation_readout(observation)["lidar"], expected, rtol=1e-6)

    def test_fall_goal_and_time_limit_are_distinguished(self):
        finished, fell, timeout, _ = episode_outcome(-100.0, False, -95.0)
        self.assertEqual((finished, fell, timeout), (False, True, False))
        finished, fell, timeout, _ = episode_outcome(0.4, False, 305.0)
        self.assertEqual((finished, fell, timeout), (True, False, False))
        finished, fell, timeout, _ = episode_outcome(0.2, True, 40.0)
        self.assertEqual((finished, fell, timeout), (False, False, True),
                         "Ein Zeitlimit ist weder Ziel noch Sturz.")

    def test_solved_uses_the_official_threshold_of_300(self):
        self.assertEqual(SOLVED_RETURN, 300.0)
        self.assertTrue(episode_outcome(0.4, False, SOLVED_RETURN)[3])
        self.assertFalse(episode_outcome(0.4, False, SOLVED_RETURN - 0.1)[3])

    def test_reaching_the_goal_below_the_threshold_still_counts_as_goal(self):
        finished, _, _, solved = episode_outcome(0.3, False, 250.0)
        self.assertTrue(finished)
        self.assertFalse(solved)


class CallbackTests(unittest.TestCase):
    def _callback_with(self, reward: float, info: dict) -> BipedalWalkerCallback:
        callback = BipedalWalkerCallback(threading.Event())
        callback.model = Mock(num_timesteps=1)
        callback.model.get_vec_normalize_env.return_value = None
        callback.num_timesteps = 1
        callback.locals = {"rewards": np.array([reward]), "dones": np.array([True]), "infos": [info]}
        callback._on_step()
        return callback

    def test_fall_and_goal_are_recorded(self):
        self.assertTrue(self._callback_with(-100.0, {}).metrics[0].fell)
        self.assertTrue(self._callback_with(0.4, {}).metrics[0].finished)

    def test_truncated_episode_is_recorded_as_timeout(self):
        metric = self._callback_with(0.2, {"TimeLimit.truncated": True}).metrics[0]
        self.assertTrue(metric.timeout)
        self.assertFalse(metric.fell)
        self.assertFalse(metric.finished)

    def test_monitor_return_has_priority_over_the_accumulated_reward(self):
        """Der Monitor führt den Return in Originaleinheiten – auch bei
        aktiver Reward-Normalisierung."""
        metric = self._callback_with(-100.0, {"episode": {"r": -123.5, "l": 88}}).metrics[0]
        self.assertAlmostEqual(metric.reward, -123.5)
        self.assertEqual(metric.length, 88)

    def test_raw_reward_is_taken_from_vecnormalize_when_rewards_are_scaled(self):
        callback = BipedalWalkerCallback(threading.Event())
        vec_normalize = Mock(norm_reward=True)
        vec_normalize.get_original_reward.return_value = np.array([-100.0])
        callback.model = Mock(num_timesteps=1)
        callback.model.get_vec_normalize_env.return_value = vec_normalize
        callback.num_timesteps = 1
        callback.locals = {"rewards": np.array([-3.7]), "dones": np.array([True]), "infos": [{}]}
        callback._on_step()
        self.assertTrue(callback.metrics[0].fell,
                        "Der skalierte Reward -3.7 würde den Sturz verdecken.")

    def test_stop_event_ends_the_run(self):
        stop = threading.Event(); stop.set()
        callback = BipedalWalkerCallback(stop)
        callback.model = Mock(num_timesteps=1)
        callback.model.get_vec_normalize_env.return_value = None
        callback.num_timesteps = 1
        callback.locals = {"rewards": np.array([0.0]), "dones": np.array([False]), "infos": [{}]}
        self.assertFalse(callback._on_step())


class EnvironmentTests(unittest.TestCase):
    @patch("bipedalwalker_logic.gymnasium.make")
    def test_factory_always_requests_the_normal_variant(self, make_mock):
        make_mock.return_value = Mock()
        make_bipedalwalker_env()
        make_mock.assert_called_once_with(ENV_ID, hardcore=False, render_mode="rgb_array")

    def test_spaces_reward_threshold_and_time_limit(self):
        env = make_bipedalwalker_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=3)
            self.assertEqual(observation.shape, (OBSERVATION_DIM,))
            self.assertEqual(env.action_space.shape, (ACTION_DIM,))
            self.assertAlmostEqual(float(env.action_space.low[0]), -1.0)
            self.assertAlmostEqual(float(env.action_space.high[0]), 1.0)
            self.assertEqual(env.spec.max_episode_steps, 1600)
            self.assertEqual(env.spec.reward_threshold, SOLVED_RETURN)
        finally:
            env.close()

    def test_hardcore_variant_stays_switched_off(self):
        env = make_bipedalwalker_env(render_mode=None)
        try:
            self.assertEqual(env.unwrapped.hardcore, HARDCORE)
            self.assertFalse(HARDCORE)
        finally:
            env.close()

    def test_a_fall_ends_the_episode_with_the_terminal_reward(self):
        """Belegt die Schwelle, an der Sturz und Zielankunft unterschieden
        werden: Ein Sturz setzt den Reward auf exakt -100."""
        env = make_bipedalwalker_env(render_mode=None)
        try:
            env.reset(seed=1)
            reward, terminated, truncated = 0.0, False, False
            for _ in range(1600):
                _obs, reward, terminated, truncated, _ = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertAlmostEqual(reward, -100.0)
            self.assertLess(reward, FALL_REWARD_THRESHOLD)
        finally:
            env.close()

    def test_isolated_renderer_accepts_four_torques(self):
        renderer = BipedalWalkerRenderer()
        try:
            observation, _info, frame = renderer.reset(seed=3)
            self.assertEqual(observation.shape, (OBSERVATION_DIM,))
            self.assertEqual(frame.shape, (400, 600, 3))
            next_observation, _reward, _terminated, _truncated, _info, next_frame = renderer.step(
                np.array([0.7, -0.9, 0.2, -0.1], dtype=np.float32))
            self.assertEqual(next_observation.shape, (OBSERVATION_DIM,))
            self.assertEqual(next_frame.shape, frame.shape)
        finally:
            renderer.close()


class ModelTests(unittest.TestCase):
    def test_only_off_policy_algorithms_own_a_replay_buffer(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                workbench = BipedalWalkerWorkbench(_small(algorithm))
                try:
                    model = workbench.create_model()
                    expected = algorithm in OFF_POLICY_ALGORITHMS
                    self.assertEqual(expected, workbench.uses_replay_buffer)
                    self.assertEqual(expected, getattr(model, "replay_buffer", None) is not None)
                finally:
                    workbench.close()

    def test_network_and_optimizer_settings_reach_the_policy(self):
        workbench = BipedalWalkerWorkbench(_small("SAC", actor_arch=(24, 12), critic_arch=(16,),
                                                  optimizer_eps=1e-6))
        try:
            model = workbench.create_model()
            self.assertEqual(model.policy.net_arch, {"pi": [24, 12], "qf": [16]})
            self.assertAlmostEqual(model.actor.optimizer.defaults["eps"], 1e-6)
        finally:
            workbench.close()

    def test_td3_parameters_reach_the_model(self):
        workbench = BipedalWalkerWorkbench(_small("TD3", policy_delay=3, target_policy_noise=0.3,
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
        workbench = BipedalWalkerWorkbench(_small("SAC"))
        try:
            model = workbench.create_model()
            self.assertEqual(ACTION_DIM, 4)
            self.assertAlmostEqual(model.target_entropy, -4.0)
            self.assertIsNotNone(model.ent_coef_optimizer)
        finally:
            workbench.close()

    def test_sac_fixed_entropy_coefficient_disables_the_tuning_optimizer(self):
        workbench = BipedalWalkerWorkbench(_small("SAC", ent_coef_mode="fest", ent_coef_value=0.2))
        try:
            model = workbench.create_model()
            self.assertIsNone(model.ent_coef_optimizer)
            self.assertAlmostEqual(float(model.ent_coef_tensor), 0.2)
        finally:
            workbench.close()

    def test_ppo_objective_parameters_reach_the_model(self):
        workbench = BipedalWalkerWorkbench(_small("PPO", clip_range=0.15, gae_lambda=0.9,
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
        workbench = BipedalWalkerWorkbench(_small("TD3", action_noise_sigma=0.9))
        try:
            model = workbench.create_model()
            observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
            first, _ = model.predict(observation, deterministic=True)
            second, _ = model.predict(observation, deterministic=True)
            np.testing.assert_allclose(first, second)
            self.assertAlmostEqual(float(model.action_noise._sigma[0]), 0.9)
        finally:
            workbench.close()


class NormalizationTests(unittest.TestCase):
    """Regeln der Workbench: Statistiken wachsen nur im Training, der
    ausgewiesene Return bleibt unnormalisiert, ein Roundtrip stellt sie her."""

    def test_training_updates_the_statistics(self):
        workbench = BipedalWalkerWorkbench(_small("PPO", total_timesteps=192, n_steps=64))
        try:
            workbench.create_model()
            self.assertIsNotNone(workbench.vec_normalize)
            before = workbench.vec_normalize.obs_rms.mean.copy()
            workbench.train()
            self.assertFalse(np.allclose(before, workbench.vec_normalize.obs_rms.mean))
        finally:
            workbench.close()

    def test_evaluation_leaves_the_statistics_untouched(self):
        workbench = BipedalWalkerWorkbench(_small("PPO", total_timesteps=192, n_steps=64))
        try:
            workbench.train()
            mean = workbench.vec_normalize.obs_rms.mean.copy()
            variance = workbench.vec_normalize.obs_rms.var.copy()
            workbench.evaluate(1, seed=2)
            np.testing.assert_allclose(mean, workbench.vec_normalize.obs_rms.mean)
            np.testing.assert_allclose(variance, workbench.vec_normalize.obs_rms.var)
        finally:
            workbench.close()

    def test_policy_observation_normalises_only_when_configured(self):
        raw = np.full(OBSERVATION_DIM, 5.0, dtype=np.float32)
        with_normalization = BipedalWalkerWorkbench(_small("PPO", total_timesteps=192, n_steps=64))
        without = BipedalWalkerWorkbench(_small("PPO", total_timesteps=192, n_steps=64,
                                                normalize_obs=False, normalize_reward=False))
        try:
            with_normalization.train()
            without.create_model()
            self.assertFalse(np.allclose(raw, with_normalization.policy_observation(raw)))
            np.testing.assert_allclose(raw, without.policy_observation(raw))
        finally:
            with_normalization.close(); without.close()

    def test_reported_returns_stay_unnormalised(self):
        """Sonst wäre die Referenzlinie bei +300 bedeutungslos."""
        workbench = BipedalWalkerWorkbench(_small("PPO", normalize_obs=True, normalize_reward=True))
        try:
            metrics = workbench.train()
            self.assertTrue(metrics)
            # Normalisierte Returns lägen im einstelligen Bereich; echte
            # BipedalWalker-Returns einer gestürzten oder abgelaufenen Episode
            # nicht.
            self.assertTrue(any(abs(metric.reward) > 10 for metric in metrics),
                            [metric.reward for metric in metrics])
        finally:
            workbench.close()

    def test_statistics_survive_a_save_load_roundtrip(self):
        workbench = BipedalWalkerWorkbench(_small("PPO", total_timesteps=192, n_steps=64))
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "state"
                workbench.save(base)
                self.assertTrue(base.with_name(base.name + "_vecnormalize.pkl").is_file())
                loaded = BipedalWalkerWorkbench.load(base, "PPO")
                try:
                    np.testing.assert_allclose(workbench.vec_normalize.obs_rms.mean,
                                               loaded.vec_normalize.obs_rms.mean)
                    self.assertTrue(loaded.vec_normalize.training,
                                    "Nach dem Laden darf weiter trainiert werden.")
                finally:
                    loaded.close()
        finally:
            workbench.close()

    def test_missing_statistics_are_rejected(self):
        workbench = BipedalWalkerWorkbench(_small("PPO", total_timesteps=192, n_steps=64))
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "state"
                workbench.save(base)
                base.with_name(base.name + "_vecnormalize.pkl").unlink()
                with self.assertRaises(ValueError) as context:
                    BipedalWalkerWorkbench.load(base, "PPO")
                self.assertIn("vecnormalize", str(context.exception))
        finally:
            workbench.close()


class BehaviourTests(unittest.TestCase):
    def test_td3_updates_the_actor_only_every_policy_delay_steps(self):
        workbench = BipedalWalkerWorkbench(_small(
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
        workbench = BipedalWalkerWorkbench(_small(
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
                workbench = BipedalWalkerWorkbench(_small(algorithm))
                try:
                    metrics = workbench.train()
                    self.assertGreater(workbench.model.num_timesteps, 0)
                    self.assertGreater(len(metrics), 0)
                    for parameter in workbench.model.policy.parameters():
                        self.assertFalse(torch.isnan(parameter).any(), f"NaN nach {algorithm}")
                finally:
                    workbench.close()

    def test_evaluation_is_deterministic_and_changes_no_learning_state(self):
        workbench = BipedalWalkerWorkbench(_small("SAC"))
        try:
            workbench.train()
            before = {name: value.detach().clone()
                      for name, value in workbench.model.policy.state_dict().items()}
            size = workbench.model.replay_buffer.size()
            observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
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

    def test_evaluation_reports_all_four_outcome_rates(self):
        workbench = BipedalWalkerWorkbench(_small("TD3"))
        try:
            workbench.train()
            result = workbench.evaluate(2, seed=5)
            rates = (result.goal_rate, result.solved_rate, result.fall_rate, result.timeout_rate)
            for rate in rates:
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 1.0)
            self.assertAlmostEqual(result.goal_rate + result.fall_rate + result.timeout_rate, 1.0,
                                   msg="Jede Episode endet mit genau einem der drei Ausgänge.")
            self.assertGreater(result.mean_length, 0)
            self.assertLessEqual(result.mean_length, 1600)
        finally:
            workbench.close()

    def test_automatic_evaluation_runs_at_the_configured_interval(self):
        workbench = BipedalWalkerWorkbench(_small("PPO", total_timesteps=192, n_steps=64))
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
                workbench = BipedalWalkerWorkbench(_small(algorithm))
                try:
                    workbench.train()
                    with tempfile.TemporaryDirectory() as directory:
                        base = Path(directory) / "bipedalwalker"
                        _model_path, replay_path, _metadata = workbench.save(base)
                        expected = algorithm in OFF_POLICY_ALGORITHMS
                        self.assertEqual(expected, replay_path is not None)
                        loaded = BipedalWalkerWorkbench.load(base, algorithm)
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
        workbench = BipedalWalkerWorkbench(_small("PPO"))
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "bipedalwalker"
                workbench.save(base)
                with self.assertRaises(ValueError) as context:
                    BipedalWalkerWorkbench.load(base, "SAC")
                message = str(context.exception)
                self.assertIn("PPO", message)
                self.assertIn("SAC", message)
        finally:
            workbench.close()

    def test_missing_replay_buffer_of_an_off_policy_model_is_rejected(self):
        workbench = BipedalWalkerWorkbench(_small("TD3"))
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "bipedalwalker"
                _model, replay_path, _metadata = workbench.save(base)
                replay_path.unlink()
                with self.assertRaises(ValueError) as context:
                    BipedalWalkerWorkbench.load(base, "TD3")
                self.assertIn(replay_path.name, str(context.exception))
        finally:
            workbench.close()

    def test_incompatible_checkpoint_is_rejected_with_a_clear_message(self):
        workbench = BipedalWalkerWorkbench(_small("PPO"))
        try:
            workbench.train()
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory) / "bipedalwalker"
                workbench.save(base)
                metadata = base.with_name(base.name + "_metadata.json")
                metadata.write_text(
                    '{"format": 1, "environment": "BipedalWalker-v3", "hardcore": true, '
                    '"algorithm": "PPO", "config": {}}', encoding="utf-8")
                with self.assertRaises(ValueError) as context:
                    BipedalWalkerWorkbench.load(base)
                self.assertIn("hardcore", str(context.exception))
        finally:
            workbench.close()

    def test_two_slots_with_the_same_algorithm_stay_independent(self):
        """Kern des Zwei-Slot-Vergleichs: gleicher Algorithmus, andere Parameter."""
        first = BipedalWalkerWorkbench(_small("SAC", learning_rate=1e-4, seed=1))
        second = BipedalWalkerWorkbench(_small("SAC", learning_rate=1e-3, seed=2))
        try:
            first.train(); second.train()
            self.assertIsNot(first.model, second.model)
            self.assertIsNot(first.history, second.history)
            self.assertNotEqual(first.config.signature(), second.config.signature())
            self.assertGreater(first.model.num_timesteps, 0)
            self.assertGreater(second.model.num_timesteps, 0)
            # Zwei unabhängige Läufe führen zu unterschiedlichen Gewichten.
            left = first.model.policy.state_dict()
            right = second.model.policy.state_dict()
            self.assertTrue(any(not torch.equal(left[name], right[name]) for name in left))
            differences = config_differences(first.config, second.config)
            self.assertIn("Lernrate α", [name for name, _, _ in differences])
        finally:
            first.close(); second.close()


if __name__ == "__main__":
    unittest.main()
