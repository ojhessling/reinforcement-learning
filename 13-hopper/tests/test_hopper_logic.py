"""Tests für Environment, Verfahren, Metriken und Vergleich des Hopper-Projekts."""

import dataclasses
import importlib.util
import json
import math
import queue
import tempfile
import threading
import unittest
from pathlib import Path

import gymnasium
import numpy as np
import torch

from hopper_logic import (
    ACTION_DIM,
    ACTION_NOISES,
    ALGORITHMS,
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_EVALUATION_INTERVAL,
    DEFAULT_NET_ARCH,
    DEFAULT_TOTAL_TIMESTEPS,
    ENV_ID,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GEAR,
    HEALTHY_ANGLE,
    HEALTHY_MIN_HEIGHT,
    JOINT_NAMES,
    MAX_EPISODE_STEPS,
    OBSERVATION_DIM,
    OFF_POLICY_ALGORITHMS,
    SOLVED_RETURN,
    VELOCITY_CLIP,
    EpisodeMetric,
    HopperConfig,
    HopperWorkbench,
    LinearSchedule,
    action_readout,
    config_differences,
    default_config,
    episode_outcome,
    make_action_noise,
    make_hopper_env,
    observation_readout,
    reward_readout,
)

#: MuJoCo und imageio gehören zu `gymnasium[mujoco]`. Ohne sie lässt sich kein
#: Environment erzeugen; die reine Konfigurations- und Metrikenlogik schon.
HAS_MUJOCO = all(importlib.util.find_spec(name) is not None for name in ("mujoco", "imageio"))
requires_mujoco = unittest.skipUnless(HAS_MUJOCO, "MuJoCo ist nicht installiert")


def tiny_config(algorithm: str, **overrides) -> HopperConfig:
    """Klein genug für einen schnellen Test, groß genug für ganze Episoden.

    Eine Hopper-Episode dauert ohne Sturz 1000 Schritte; ein ungelernter Agent
    stürzt jedoch nach wenigen Dutzend Schritten, sodass schon kleine Budgets
    mehrere abgeschlossene Episoden liefern.
    """
    values = dict(total_timesteps=400, actor_arch=(16, 16), critic_arch=(16, 16), seed=7)
    if algorithm == "PPO":
        values.update(n_steps=200, batch_size=25)
    else:
        values.update(batch_size=32, buffer_size=1_000, learning_starts=50)
    values.update(overrides)
    config = dataclasses.replace(default_config(algorithm), **values)
    config.validate()
    return config


class EnvironmentTests(unittest.TestCase):
    def test_registry_entry_matches_the_official_values(self):
        spec = gymnasium.envs.registration.registry[ENV_ID]
        self.assertEqual(spec.max_episode_steps, MAX_EPISODE_STEPS)
        self.assertEqual(spec.reward_threshold, SOLVED_RETURN)

    @requires_mujoco
    def test_factory_requests_exactly_the_frame_size_and_no_physics_arguments(self):
        env = make_hopper_env(render_mode=None)
        try:
            self.assertEqual(env.spec.id, ENV_ID)
            # Physik, Reward und Abbruchregeln bleiben unverändert: Kein
            # einziges der Environment-Argumente wird übergeben.
            self.assertEqual(set(env.spec.kwargs), {"width", "height"})
            self.assertEqual(env.spec.kwargs["width"], FRAME_WIDTH)
            self.assertEqual(env.spec.kwargs["height"], FRAME_HEIGHT)
        finally:
            env.close()

    @requires_mujoco
    def test_render_mode_is_only_added_when_asked_for(self):
        env = make_hopper_env(render_mode="rgb_array")
        try:
            self.assertEqual(env.spec.kwargs["render_mode"], "rgb_array")
        finally:
            env.close()

    @requires_mujoco
    def test_spaces_and_frame_rate(self):
        env = make_hopper_env(render_mode=None)
        try:
            self.assertEqual(env.action_space.shape, (ACTION_DIM,))
            self.assertEqual(float(env.action_space.low.min()), -1.0)
            self.assertEqual(float(env.action_space.high.max()), 1.0)
            self.assertEqual(env.observation_space.shape, (OBSERVATION_DIM,))
            self.assertEqual(env.observation_space.dtype, np.float64)
            self.assertEqual(env.metadata["render_fps"], 125)
            self.assertAlmostEqual(env.unwrapped.dt, 0.008)
        finally:
            env.close()

    @requires_mujoco
    def test_all_three_motors_use_the_documented_gear_ratio(self):
        env = make_hopper_env(render_mode=None)
        try:
            gears = env.unwrapped.model.actuator_gear[:, 0]
            self.assertEqual(len(gears), ACTION_DIM)
            for gear in gears:
                self.assertAlmostEqual(float(gear), GEAR)
        finally:
            env.close()

    @requires_mujoco
    def test_rendered_frame_has_the_declared_size(self):
        env = make_hopper_env(render_mode="rgb_array")
        try:
            env.reset(seed=0)
            frame = np.asarray(env.render())
            self.assertEqual(frame.shape, (FRAME_HEIGHT, FRAME_WIDTH, 3))
        finally:
            env.close()

    @requires_mujoco
    def test_reward_is_exactly_the_sum_of_its_three_reported_parts(self):
        env = make_hopper_env(render_mode=None)
        try:
            env.reset(seed=3)
            for _ in range(20):
                _obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                parts = info["reward_survive"] + info["reward_forward"] + info["reward_ctrl"]
                self.assertAlmostEqual(float(reward), float(parts), places=9)
                # Die Steuerkosten stehen bereits negativ in `info`.
                self.assertLessEqual(info["reward_ctrl"], 0.0)
                if terminated or truncated:
                    break
        finally:
            env.close()

    @requires_mujoco
    def test_position_is_reported_in_info_and_not_in_the_observation(self):
        env = make_hopper_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=5)
            for _ in range(30):
                observation, _reward, terminated, truncated, info = env.step(
                    np.array([1.0, 1.0, 1.0], dtype=np.float32))
                self.assertIn("x_position", info)
                self.assertIn("x_velocity", info)
                if terminated or truncated:
                    break
            # Die x-Position ist wegen
            # `exclude_current_positions_from_observation=True` nicht Teil der
            # Beobachtung; sie taucht dort auch nicht zufällig auf.
            self.assertEqual(len(observation), OBSERVATION_DIM)
            self.assertNotIn(float(info["x_position"]), [float(v) for v in observation])
        finally:
            env.close()

    @requires_mujoco
    def test_velocities_in_the_observation_are_clipped_to_ten(self):
        env = make_hopper_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=1)
            for _ in range(50):
                observation, _r, terminated, truncated, _i = env.step(
                    np.array([1.0, -1.0, 1.0], dtype=np.float32))
                self.assertTrue(np.all(np.abs(observation[5:]) <= VELOCITY_CLIP + 1e-9))
                if terminated or truncated:
                    break
        finally:
            env.close()

    @requires_mujoco
    def test_a_fall_terminates_without_any_terminal_penalty(self):
        """`Hopper-v5` kennt weder Terminalbonus noch Terminalstrafe."""
        env = make_hopper_env(render_mode=None)
        try:
            env.reset(seed=11)
            reward = 0.0
            info: dict = {}
            terminated = truncated = False
            while not (terminated or truncated):
                _obs, reward, terminated, truncated, info = env.step(
                    np.array([1.0, 1.0, -1.0], dtype=np.float32))
            self.assertTrue(terminated, "Der Agent sollte in diesem Test stürzen.")
            self.assertFalse(truncated)
            # Auch der letzte Reward besteht ausschließlich aus den drei
            # regulären Anteilen: Es gibt keinen zusätzlichen Terminalterm. Der
            # Überlebensbonus fällt lediglich weg, weil der Roboter ungesund ist.
            self.assertAlmostEqual(float(info["reward_survive"]), 0.0)
            parts = info["reward_survive"] + info["reward_forward"] + info["reward_ctrl"]
            self.assertAlmostEqual(float(reward), float(parts), places=9)
            # Eine feste Strafe wie die -100 von BipedalWalker gibt es nicht.
            self.assertLess(abs(float(reward)), 20.0)
        finally:
            env.close()

    @requires_mujoco
    def test_the_time_limit_truncates_after_exactly_one_thousand_steps(self):
        env = make_hopper_env(render_mode=None)
        try:
            env.reset(seed=0)
            steps = 0
            terminated = truncated = False
            # Ohne Moment bleibt der Roboter zwar nicht ewig stehen; die
            # Grenze prüfen wir deshalb über den Wrapper selbst.
            self.assertEqual(env.spec.max_episode_steps, MAX_EPISODE_STEPS)
            while not (terminated or truncated) and steps < MAX_EPISODE_STEPS:
                _o, _r, terminated, truncated, _i = env.step(np.zeros(ACTION_DIM, dtype=np.float32))
                steps += 1
            # Das Environment selbst liefert nie `truncated`; das tut der
            # TimeLimit-Wrapper aus `gymnasium.make`.
            self.assertLessEqual(steps, MAX_EPISODE_STEPS)
        finally:
            env.close()


class OutcomeTests(unittest.TestCase):
    def test_termination_is_a_fall_and_truncation_is_the_good_outcome(self):
        survived, fell, solved = episode_outcome(True, False, 120.0)
        self.assertFalse(survived)
        self.assertTrue(fell)
        self.assertFalse(solved)
        survived, fell, solved = episode_outcome(False, True, 120.0)
        self.assertTrue(survived, "Zeitlimit heißt bei Hopper: durchgehalten.")
        self.assertFalse(fell)

    def test_solved_needs_the_official_threshold(self):
        self.assertFalse(episode_outcome(False, True, SOLVED_RETURN - 0.1)[2])
        self.assertTrue(episode_outcome(False, True, SOLVED_RETURN)[2])
        self.assertTrue(episode_outcome(True, False, SOLVED_RETURN + 100)[2])

    def test_survival_and_fall_are_complementary(self):
        for terminated, truncated in ((True, False), (False, True)):
            survived, fell, _solved = episode_outcome(terminated, truncated, 0.0)
            self.assertNotEqual(survived, fell)


class ReadoutTests(unittest.TestCase):
    def test_action_readout_translates_sign_and_torque(self):
        readout = action_readout(np.array([1.0, -0.5, 0.0]))
        self.assertEqual([joint["joint"] for joint in readout["joints"]], list(JOINT_NAMES))
        self.assertAlmostEqual(readout["joints"][0]["torque"], GEAR)
        self.assertEqual(readout["joints"][0]["direction"], "+")
        self.assertAlmostEqual(readout["joints"][1]["torque"], 0.5 * GEAR)
        self.assertEqual(readout["joints"][1]["direction"], "−")
        # 0 heißt „kein Moment", nicht „Gelenk hält die Position".
        self.assertEqual(readout["joints"][2]["torque"], 0.0)
        self.assertIn("kein Moment", readout["joints"][2]["text"])

    def test_action_readout_clips_like_the_environment(self):
        readout = action_readout(np.array([2.5, -7.0, 0.25]))
        self.assertEqual(readout["raw"], [1.0, -1.0, 0.25])
        self.assertAlmostEqual(readout["joints"][0]["torque"], GEAR)

    def test_observation_readout_names_every_value(self):
        observation = np.arange(OBSERVATION_DIM, dtype=float) / 10.0
        values = observation_readout(observation)
        self.assertAlmostEqual(values["height"], 0.0)
        self.assertAlmostEqual(values["torso_angle"], 0.1)
        self.assertAlmostEqual(values["torso_angle_degrees"], math.degrees(0.1))
        self.assertEqual(len(values["joint_angles"]), 3)
        self.assertEqual(len(values["joint_velocities"]), 3)
        self.assertEqual(len(values["velocities"]), 6)
        self.assertFalse(values["velocity_clipped"])

    def test_observation_readout_flags_the_healthy_range(self):
        healthy = observation_readout(np.array([1.25, 0.0] + [0.0] * 9))
        self.assertTrue(healthy["healthy_height"])
        self.assertTrue(healthy["healthy_angle"])
        low = observation_readout(np.array([HEALTHY_MIN_HEIGHT, 0.0] + [0.0] * 9))
        self.assertFalse(low["healthy_height"])
        tilted = observation_readout(np.array([1.25, HEALTHY_ANGLE] + [0.0] * 9))
        self.assertFalse(tilted["healthy_angle"])

    def test_observation_readout_marks_clipped_velocities(self):
        observation = np.array([1.25, 0.0, 0.0, 0.0, 0.0, VELOCITY_CLIP, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(observation_readout(observation)["velocity_clipped"])

    def test_reward_readout_takes_the_parts_from_info(self):
        info = {"reward_survive": 1.0, "reward_forward": 0.4, "reward_ctrl": -0.002,
                "x_position": 2.5, "x_velocity": 0.4}
        values = reward_readout(info)
        self.assertAlmostEqual(values["survive"], 1.0)
        self.assertAlmostEqual(values["forward"], 0.4)
        self.assertAlmostEqual(values["ctrl"], -0.002)
        self.assertAlmostEqual(values["x_position"], 2.5)


class ConfigurationTests(unittest.TestCase):
    def test_every_algorithm_starts_with_the_same_step_budget(self):
        budgets = {default_config(name).total_timesteps for name in ALGORITHMS}
        self.assertEqual(budgets, {DEFAULT_TOTAL_TIMESTEPS})
        # Das ist zugleich das Budget der drei Zoo-Profile.
        self.assertEqual(DEFAULT_TOTAL_TIMESTEPS, 1_000_000)

    def test_evaluation_interval_is_a_tenth_of_the_step_budget(self):
        self.assertEqual(DEFAULT_EVALUATION_INTERVAL * 10, DEFAULT_TOTAL_TIMESTEPS)
        self.assertGreaterEqual(DEFAULT_EVALUATION_EPISODES, 1)

    def test_ppo_profile_matches_the_zoo_values(self):
        config = default_config("PPO")
        self.assertAlmostEqual(config.learning_rate, 9.80828e-5)
        self.assertAlmostEqual(config.gamma, 0.999)
        self.assertEqual(config.n_steps, 512)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.n_epochs, 5)
        self.assertAlmostEqual(config.gae_lambda, 0.99)
        self.assertAlmostEqual(config.clip_range, 0.2)
        self.assertAlmostEqual(config.ent_coef, 0.00229519)
        self.assertAlmostEqual(config.vf_coef, 0.835671)
        self.assertAlmostEqual(config.max_grad_norm, 0.7)
        self.assertAlmostEqual(config.log_std_init, -2.0)
        self.assertFalse(config.ortho_init)
        self.assertEqual(config.activation, "ReLU")
        # Das Zoo-Profil verlangt `normalize: true`.
        self.assertTrue(config.normalize_obs)
        self.assertTrue(config.normalize_reward)

    def test_off_policy_buffers_hold_the_whole_budget(self):
        for name in OFF_POLICY_ALGORITHMS:
            config = default_config(name)
            self.assertEqual(config.buffer_size, DEFAULT_TOTAL_TIMESTEPS)
            # SB3-Default, von den Zoo-Profilen nicht überschrieben.
            self.assertEqual(config.buffer_size, HopperConfig().buffer_size)

    def test_sac_profile_only_overrides_the_learning_start(self):
        config = default_config("SAC")
        self.assertEqual(config.learning_starts, 10_000)
        self.assertAlmostEqual(config.learning_rate, 3e-4)
        self.assertAlmostEqual(config.gamma, 0.99)
        self.assertAlmostEqual(config.tau, 0.005)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.train_freq, 1)
        self.assertEqual(config.gradient_steps, 1)
        self.assertEqual(config.ent_coef_mode, "auto")
        self.assertFalse(config.normalize_obs)
        self.assertFalse(config.normalize_reward)

    def test_td3_profile_matches_the_zoo_values(self):
        config = default_config("TD3")
        self.assertAlmostEqual(config.learning_rate, 1e-3)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.learning_starts, 10_000)
        self.assertEqual(config.train_freq, 1)
        self.assertEqual(config.gradient_steps, 1)
        self.assertEqual(config.action_noise, "normal")
        self.assertAlmostEqual(config.action_noise_sigma, 0.1)

    def test_networks_are_unified_across_the_three_methods(self):
        for name in ALGORITHMS:
            config = default_config(name)
            self.assertEqual(config.actor_arch, DEFAULT_NET_ARCH)
            self.assertEqual(config.critic_arch, DEFAULT_NET_ARCH)

    def test_sac_target_entropy_auto_equals_minus_action_dimension(self):
        config = default_config("SAC")
        self.assertEqual(config.sac_target_entropy(), "auto")
        self.assertEqual(-ACTION_DIM, -3)
        numeric = dataclasses.replace(config, target_entropy="-3")
        self.assertEqual(numeric.sac_target_entropy(), -3.0)

    def test_constructor_arguments_are_known_to_each_algorithm(self):
        import inspect

        from stable_baselines3 import PPO, SAC, TD3

        for name, algorithm_class in (("PPO", PPO), ("SAC", SAC), ("TD3", TD3)):
            known = set(inspect.signature(algorithm_class.__init__).parameters)
            unknown = set(default_config(name).model_kwargs()) - known
            self.assertEqual(unknown, set(), f"{name} kennt {unknown} nicht")

    def test_only_ppo_offers_the_ppo_parameters(self):
        self.assertTrue(default_config("PPO").uses("ortho_init"))
        self.assertFalse(default_config("SAC").uses("ortho_init"))
        self.assertFalse(default_config("TD3").uses("n_steps"))
        for name in OFF_POLICY_ALGORITHMS:
            self.assertTrue(default_config(name).uses("buffer_size"))
        self.assertFalse(default_config("PPO").uses("buffer_size"))

    def test_ortho_init_reaches_the_policy_arguments(self):
        self.assertFalse(default_config("PPO").policy_kwargs()["ortho_init"])
        self.assertNotIn("ortho_init", default_config("SAC").policy_kwargs())

    def test_batch_size_must_divide_the_rollout(self):
        config = dataclasses.replace(default_config("PPO"), batch_size=30)
        with self.assertRaises(ValueError) as error:
            config.validate()
        self.assertIn("teilt den Rollout", str(error.exception))

    def test_td3_rejects_missing_action_noise(self):
        config = dataclasses.replace(default_config("TD3"), action_noise="keins")
        with self.assertRaises(ValueError) as error:
            config.validate()
        self.assertIn("deterministischen Actor", str(error.exception))

    def test_error_messages_name_field_value_and_range(self):
        config = dataclasses.replace(default_config("SAC"), gamma=1.5)
        with self.assertRaises(ValueError) as error:
            config.validate()
        message = str(error.exception)
        self.assertIn("Diskontfaktor γ", message)
        self.assertIn("1.5", message)
        self.assertIn("Gültig", message)

    def test_linear_schedule_falls_to_zero(self):
        schedule = LinearSchedule(1e-3)
        self.assertAlmostEqual(schedule(1.0), 1e-3)
        self.assertAlmostEqual(schedule(0.0), 0.0)
        config = dataclasses.replace(default_config("SAC"), learning_rate_schedule="linear fallend")
        self.assertEqual(config.learning_rate_value(), LinearSchedule(config.learning_rate))

    def test_action_noise_has_the_action_dimension(self):
        noise = make_action_noise("normal", 0.1)
        self.assertEqual(noise().shape, (ACTION_DIM,))
        self.assertIsNone(make_action_noise("keins", 0.1))

    def test_budget_change_alone_keeps_the_model_signature(self):
        config = default_config("SAC")
        longer = dataclasses.replace(config, total_timesteps=config.total_timesteps * 2)
        self.assertEqual(config.signature(), longer.signature())
        other = dataclasses.replace(config, tau=0.5)
        self.assertNotEqual(config.signature(), other.signature())


class DifferenceTests(unittest.TestCase):
    def test_differences_list_one_value_per_slot(self):
        configs = [
            default_config("SAC"),
            dataclasses.replace(default_config("SAC"), tau=0.05),
            dataclasses.replace(default_config("SAC"), tau=0.5, gamma=0.95),
        ]
        differences = dict(config_differences(configs))
        self.assertIn("Soft-Update τ", differences)
        self.assertEqual(len(differences["Soft-Update τ"]), 3)
        self.assertEqual(differences["Soft-Update τ"], ["0.005", "0.05", "0.5"])
        self.assertIn("Diskontfaktor γ", differences)

    def test_identical_configurations_have_no_differences(self):
        self.assertEqual(config_differences([default_config("TD3")] * 4), [])

    def test_only_shared_parameters_are_compared(self):
        differences = dict(config_differences([default_config("PPO"), default_config("SAC")]))
        # `n_steps` kennt nur PPO, `buffer_size` nur die Off-Policy-Verfahren.
        self.assertNotIn("Rollout n_steps", differences)
        self.assertNotIn("Replay Buffer |D|", differences)
        self.assertIn("Lernrate α", differences)

    def test_four_slots_are_supported(self):
        configs = [dataclasses.replace(default_config("TD3"), seed=index) for index in range(4)]
        differences = dict(config_differences(configs))
        self.assertEqual(differences["Zufallsstart s"], ["0", "1", "2", "3"])


@requires_mujoco
class TrainingTests(unittest.TestCase):
    def test_every_algorithm_trains_evaluates_and_reports_metrics(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                workbench = HopperWorkbench(tiny_config(name))
                try:
                    metrics = workbench.train()
                    self.assertTrue(metrics)
                    for metric in metrics:
                        self.assertIsInstance(metric, EpisodeMetric)
                        self.assertLessEqual(metric.length, MAX_EPISODE_STEPS)
                        self.assertNotEqual(metric.survived, metric.fell)
                    result = workbench.evaluate(2, seed=3)
                    self.assertEqual(result.episodes, 2)
                    self.assertLessEqual(result.mean_length, MAX_EPISODE_STEPS)
                    for rate in (result.survive_rate, result.fall_rate, result.solved_rate):
                        self.assertGreaterEqual(rate, 0.0)
                        self.assertLessEqual(rate, 1.0)
                    self.assertAlmostEqual(result.survive_rate + result.fall_rate, 1.0)
                finally:
                    workbench.close()

    def test_evaluation_does_not_change_the_learning_state(self):
        workbench = HopperWorkbench(tiny_config("TD3"))
        try:
            workbench.train()
            before = [parameter.detach().clone()
                      for parameter in workbench.model.policy.parameters()]
            steps_before = workbench.model.num_timesteps
            workbench.evaluate(2, seed=0)
            for old, new in zip(before, workbench.model.policy.parameters()):
                self.assertTrue(torch.equal(old, new))
            self.assertEqual(workbench.model.num_timesteps, steps_before)
        finally:
            workbench.close()

    def test_continuing_training_keeps_the_step_counter(self):
        workbench = HopperWorkbench(tiny_config("PPO"))
        try:
            workbench.train()
            first = workbench.model.num_timesteps
            workbench.train()
            self.assertGreater(workbench.model.num_timesteps, first)
            self.assertGreater(len(workbench.history), 0)
        finally:
            workbench.close()

    def test_ppo_has_no_replay_buffer(self):
        workbench = HopperWorkbench(tiny_config("PPO"))
        try:
            workbench.create_model()
            self.assertFalse(workbench.uses_replay_buffer)
            self.assertFalse(hasattr(workbench.model, "replay_buffer"))
            self.assertTrue(hasattr(workbench.model, "rollout_buffer"))
        finally:
            workbench.close()

    def test_td3_delays_the_actor_update(self):
        """Mit sehr großem `policy_delay` bleibt der Actor stehen, der Critic nicht."""
        config = tiny_config("TD3", total_timesteps=200, learning_starts=0, policy_delay=10_000)
        workbench = HopperWorkbench(config)
        try:
            workbench.create_model()
            actor = [p.detach().clone() for p in workbench.model.actor.parameters()]
            critic = [p.detach().clone() for p in workbench.model.critic.parameters()]
            workbench.train()
            self.assertTrue(all(torch.equal(old, new) for old, new
                                in zip(actor, workbench.model.actor.parameters())))
            self.assertFalse(all(torch.equal(old, new) for old, new
                                 in zip(critic, workbench.model.critic.parameters())))
        finally:
            workbench.close()

    def test_td3_passes_target_noise_and_uses_action_noise_only_while_training(self):
        config = tiny_config("TD3", target_policy_noise=0.3, target_noise_clip=0.4)
        workbench = HopperWorkbench(config)
        try:
            workbench.create_model()
            self.assertAlmostEqual(workbench.model.target_policy_noise, 0.3)
            self.assertAlmostEqual(workbench.model.target_noise_clip, 0.4)
            self.assertIsNotNone(workbench.model.action_noise)
            observation = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
            first, _ = workbench.model.predict(observation, deterministic=True)
            second, _ = workbench.model.predict(observation, deterministic=True)
            # Ohne Exploration liefert der deterministische Actor zweimal dasselbe.
            np.testing.assert_allclose(first, second)
        finally:
            workbench.close()

    def test_sac_learns_its_temperature(self):
        workbench = HopperWorkbench(tiny_config("SAC", learning_starts=0))
        try:
            workbench.create_model()
            self.assertEqual(float(workbench.model.target_entropy), -ACTION_DIM)
            before = workbench.model.log_ent_coef.detach().clone()
            workbench.train()
            self.assertFalse(torch.equal(before, workbench.model.log_ent_coef.detach()))
        finally:
            workbench.close()

    def test_ppo_clip_range_and_epochs_reach_the_model(self):
        config = tiny_config("PPO", clip_range=0.15, gae_lambda=0.8, n_epochs=3)
        workbench = HopperWorkbench(config)
        try:
            workbench.create_model()
            self.assertAlmostEqual(workbench.model.clip_range(1.0), 0.15)
            self.assertAlmostEqual(workbench.model.gae_lambda, 0.8)
            self.assertEqual(workbench.model.n_epochs, 3)
        finally:
            workbench.close()

    def test_training_is_reproducible_for_a_fixed_seed(self):
        rewards = []
        for _ in range(2):
            workbench = HopperWorkbench(tiny_config("TD3", total_timesteps=300, seed=123))
            try:
                metrics = workbench.train()
                rewards.append([round(metric.reward, 6) for metric in metrics])
            finally:
                workbench.close()
        self.assertEqual(rewards[0], rewards[1])
        self.assertTrue(rewards[0])

    def test_callback_reports_episodes_and_evaluations_through_the_queue(self):
        output: queue.Queue = queue.Queue()
        workbench = HopperWorkbench(tiny_config("SAC", total_timesteps=300))
        try:
            workbench.train(threading.Event(), output, ("compare", 2),
                            evaluation_interval=150, evaluation_episodes=1)
        finally:
            workbench.close()
        kinds: dict[str, list] = {}
        while not output.empty():
            kind, payload = output.get()
            kinds.setdefault(kind, []).append(payload)
        self.assertIn("episode", kinds)
        self.assertIn("evaluation", kinds)
        self.assertEqual(kinds["episode"][0][0], ("compare", 2))

    def test_the_best_episode_is_stored_with_its_policy(self):
        """Gespeichert wird genau ein zusätzlicher Lernstand je Slot."""
        workbench = HopperWorkbench(tiny_config("TD3"))
        try:
            self.assertIsNone(workbench.best_snapshot())
            metrics = workbench.train()
            snapshot = workbench.best_snapshot()
            self.assertIsNotNone(snapshot)
            best_metric, state = snapshot
            self.assertEqual(best_metric, max(metrics, key=lambda item: item.reward))
            self.assertEqual(set(state), set(workbench.model.policy.state_dict()))
            # Losgelöste Kopie: Der weiterlaufende Optimizer verändert sie nicht.
            for tensor in state.values():
                self.assertFalse(tensor.requires_grad)
            frozen = {name: tensor.clone() for name, tensor in state.items()}
            workbench.train()
            for name, tensor in workbench.best_snapshot()[1].items():
                if best_metric == workbench.best_snapshot()[0]:
                    self.assertTrue(torch.equal(frozen[name], tensor))
        finally:
            workbench.close()

    def test_a_new_model_forgets_the_best_episode(self):
        workbench = HopperWorkbench(tiny_config("SAC"))
        try:
            workbench.train()
            self.assertIsNotNone(workbench.best_snapshot())
            workbench.create_model()
            self.assertIsNone(workbench.best_snapshot())
            self.assertIsNone(workbench.best_episode)
        finally:
            workbench.close()

    def test_stop_event_ends_training_early(self):
        stop = threading.Event()
        stop.set()
        workbench = HopperWorkbench(tiny_config("SAC", total_timesteps=10_000))
        try:
            workbench.train(stop)
            self.assertLess(workbench.model.num_timesteps, 10_000)
        finally:
            workbench.close()


@requires_mujoco
class NormalizationTests(unittest.TestCase):
    def test_statistics_grow_while_training_and_stay_frozen_in_evaluation(self):
        workbench = HopperWorkbench(tiny_config("PPO"))
        try:
            workbench.train()
            statistics = workbench.vec_normalize
            self.assertIsNotNone(statistics)
            self.assertGreater(statistics.obs_rms.count, 1.0)
            count = statistics.obs_rms.count
            mean = statistics.obs_rms.mean.copy()
            workbench.evaluate(2, seed=0)
            self.assertEqual(statistics.obs_rms.count, count)
            np.testing.assert_array_equal(statistics.obs_rms.mean, mean)
            # Auch die Aufbereitung für die Animation schreibt nichts fort.
            workbench.policy_observation(np.zeros(OBSERVATION_DIM, dtype=np.float32))
            self.assertEqual(statistics.obs_rms.count, count)
        finally:
            workbench.close()

    def test_reported_returns_stay_unnormalised(self):
        workbench = HopperWorkbench(tiny_config("PPO"))
        try:
            metrics = workbench.train()
            self.assertTrue(workbench.config.normalize_reward)
            # Jeder gesunde Schritt bringt +1; ein normalisierter Return läge
            # um Größenordnungen daneben.
            for metric in metrics:
                self.assertGreater(metric.reward, 0.3 * metric.length)
                self.assertLess(metric.reward, 3.0 * metric.length)
        finally:
            workbench.close()

    def test_off_policy_methods_do_not_normalise_by_default(self):
        for name in OFF_POLICY_ALGORITHMS:
            workbench = HopperWorkbench(tiny_config(name))
            try:
                workbench.create_model()
                self.assertIsNone(workbench.vec_normalize)
            finally:
                workbench.close()


@requires_mujoco
class CheckpointTests(unittest.TestCase):
    def test_roundtrip_keeps_exactly_the_state_each_method_owns(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                workbench = HopperWorkbench(tiny_config(name))
                with tempfile.TemporaryDirectory() as folder:
                    base = Path(folder) / "best"
                    try:
                        workbench.train()
                        model_path, replay_path, metadata_path = workbench.save(base)
                        self.assertTrue(model_path.is_file())
                        self.assertTrue(metadata_path.is_file())
                        if name in OFF_POLICY_ALGORITHMS:
                            self.assertTrue(replay_path.is_file())
                        else:
                            self.assertIsNone(replay_path)
                        restored = HopperWorkbench.load(base, name)
                        try:
                            self.assertEqual(restored.config, workbench.config)
                            observation = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
                            expected, _ = workbench.model.predict(observation, deterministic=True)
                            actual, _ = restored.model.predict(observation, deterministic=True)
                            np.testing.assert_allclose(expected, actual, atol=1e-6)
                        finally:
                            restored.close()
                    finally:
                        workbench.close()

    def test_normalisation_statistics_survive_the_roundtrip(self):
        workbench = HopperWorkbench(tiny_config("PPO"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                mean = workbench.vec_normalize.obs_rms.mean.copy()
                workbench.save(base)
                restored = HopperWorkbench.load(base, "PPO")
                try:
                    self.assertIsNotNone(restored.vec_normalize)
                    np.testing.assert_allclose(restored.vec_normalize.obs_rms.mean, mean)
                    # Nach dem Laden darf das Training die Statistik weiter
                    # fortschreiben.
                    self.assertTrue(restored.vec_normalize.training)
                finally:
                    restored.close()
            finally:
                workbench.close()

    def test_a_foreign_environment_is_rejected(self):
        workbench = HopperWorkbench(tiny_config("SAC"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                _model, _replay, metadata_path = workbench.save(base)
            finally:
                workbench.close()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["environment"] = "Walker2d-v5"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError) as error:
                HopperWorkbench.load(base)
            self.assertIn("Walker2d-v5", str(error.exception))
            self.assertIn(ENV_ID, str(error.exception))

    def test_a_foreign_algorithm_is_rejected(self):
        workbench = HopperWorkbench(tiny_config("TD3"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                workbench.save(base)
            finally:
                workbench.close()
            with self.assertRaises(ValueError) as error:
                HopperWorkbench.load(base, "PPO")
            self.assertIn("TD3", str(error.exception))

    def test_missing_metadata_is_reported_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError) as error:
                HopperWorkbench.load(Path(folder) / "nothing")
            self.assertIn("Metadatendatei", str(error.exception))


@requires_mujoco
class ComparisonTests(unittest.TestCase):
    def _run(self, configs: list[HopperConfig]) -> list[HopperWorkbench]:
        """Alle Slots parallel, so wie es die GUI tut."""
        benches = [HopperWorkbench(config) for config in configs]
        output: queue.Queue = queue.Queue()
        stop = threading.Event()
        barrier = threading.Barrier(len(benches))

        def run(index: int) -> None:
            barrier.wait()
            benches[index].train(stop, output, ("compare", index))

        threads = [threading.Thread(target=run, args=(index,)) for index in range(len(benches))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return benches

    def test_two_slots_with_the_same_algorithm_stay_independent(self):
        configs = [tiny_config("TD3", seed=1), tiny_config("TD3", seed=2, tau=0.05)]
        benches = self._run(configs)
        try:
            self.assertIsNot(benches[0].model, benches[1].model)
            self.assertIsNot(benches[0].env, benches[1].env)
            self.assertAlmostEqual(benches[1].model.tau, 0.05)
            self.assertAlmostEqual(benches[0].model.tau, 0.005)
            self.assertTrue(benches[0].history)
            self.assertTrue(benches[1].history)
            self.assertNotEqual([m.reward for m in benches[0].history],
                                [m.reward for m in benches[1].history])
        finally:
            for bench in benches:
                bench.close()

    def test_four_slots_run_in_parallel_and_keep_separate_results(self):
        configs = [tiny_config(name, total_timesteps=250)
                   for name in ("PPO", "SAC", "TD3", "SAC")]
        benches = self._run(configs)
        try:
            for bench in benches:
                self.assertTrue(bench.history, f"{bench.config.algorithm} lieferte keine Episode")
            models = {id(bench.model) for bench in benches}
            self.assertEqual(len(models), 4)
            environments = {id(bench.env) for bench in benches}
            self.assertEqual(len(environments), 4)
        finally:
            for bench in benches:
                bench.close()
