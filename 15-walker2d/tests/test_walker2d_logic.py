"""Tests für Environment, Verfahren, Metriken und Vergleich des Walker2d-Projekts."""

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

from walker2d_logic import (
    ACTION_DIM,
    ACTION_NOISES,
    ALGORITHMS,
    CMA_NET_ARCH,
    CTRL_COST_WEIGHT,
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_EVALUATION_INTERVAL,
    DEFAULT_NET_ARCH,
    DEFAULT_TOTAL_TIMESTEPS,
    ENV_ID,
    EVOLUTIONARY_ALGORITHMS,
    FORWARD_REWARD_WEIGHT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GEAR,
    HEALTHY_ANGLE_RANGE,
    HEALTHY_REWARD,
    HEALTHY_Z_RANGE,
    JOINT_NAMES,
    MAX_EPISODE_STEPS,
    OBSERVATION_DIM,
    OFF_POLICY_ALGORITHMS,
    SB3_ALGORITHMS,
    STEP_DURATION,
    TARGET_RETURN,
    TD3_NET_ARCH,
    VELOCITY_CLIP,
    CmaEsWorkbench,
    EpisodeMetric,
    LinearSchedule,
    LinearTanhPolicy,
    RunningNormalizer,
    Walker2dConfig,
    Walker2dWorkbench,
    action_readout,
    config_differences,
    control_cost,
    default_config,
    episode_outcome,
    load_workbench,
    make_action_noise,
    make_walker2d_env,
    make_workbench,
    observation_readout,
    reward_readout,
)

#: MuJoCo und imageio gehören zu `gymnasium[mujoco]`. Ohne sie lässt sich kein
#: Environment erzeugen; die reine Konfigurations- und Metrikenlogik schon.
HAS_MUJOCO = all(importlib.util.find_spec(name) is not None for name in ("mujoco", "imageio"))
requires_mujoco = unittest.skipUnless(HAS_MUJOCO, "MuJoCo ist nicht installiert")


def tiny_config(algorithm: str, **overrides) -> Walker2dConfig:
    """Klein genug für einen schnellen Test, groß genug für ganze Episoden.

    Ein ungelernter Walker2d stürzt nach wenigen Dutzend Schritten, deshalb
    liefern schon kleine Budgets mehrere abgeschlossene Episoden.
    """
    values: dict = dict(total_timesteps=1_200, seed=7)
    if algorithm in EVOLUTIONARY_ALGORITHMS:
        # Kleine Population, damit mehrere Generationen ins Budget passen.
        values.update(popsize="4")
    else:
        values.update(actor_arch=(16, 16), critic_arch=(16, 16),
                      batch_size=32, buffer_size=2_000, learning_starts=50)
    values.update(overrides)
    config = dataclasses.replace(default_config(algorithm), **values)
    config.validate()
    return config


class EnvironmentTests(unittest.TestCase):
    def test_registry_has_no_official_threshold(self):
        """`Walker2d-v5` führt keinen `reward_threshold` – die Zielmarke ist
        eine Projektkonstante und wird nirgends „gelöst" genannt."""
        spec = gymnasium.envs.registration.registry[ENV_ID]
        self.assertEqual(spec.max_episode_steps, MAX_EPISODE_STEPS)
        self.assertIsNone(spec.reward_threshold)
        self.assertEqual(TARGET_RETURN, 4000.0)

    @requires_mujoco
    def test_factory_requests_exactly_the_frame_size_and_no_physics_arguments(self):
        env = make_walker2d_env(render_mode=None)
        try:
            self.assertEqual(env.spec.id, ENV_ID)
            self.assertEqual(set(env.spec.kwargs), {"width", "height"})
            self.assertEqual(env.spec.kwargs["width"], FRAME_WIDTH)
            self.assertEqual(env.spec.kwargs["height"], FRAME_HEIGHT)
        finally:
            env.close()

    @requires_mujoco
    def test_render_mode_is_only_added_when_asked_for(self):
        env = make_walker2d_env(render_mode="rgb_array")
        try:
            self.assertEqual(env.spec.kwargs["render_mode"], "rgb_array")
        finally:
            env.close()

    @requires_mujoco
    def test_spaces_and_frame_rate(self):
        env = make_walker2d_env(render_mode=None)
        try:
            self.assertEqual(env.action_space.shape, (ACTION_DIM,))
            self.assertEqual(float(env.action_space.low.min()), -1.0)
            self.assertEqual(float(env.action_space.high.max()), 1.0)
            self.assertEqual(env.observation_space.shape, (OBSERVATION_DIM,))
            self.assertEqual(env.observation_space.dtype, np.float64)
            self.assertEqual(env.metadata["render_fps"], 125)
            self.assertAlmostEqual(env.unwrapped.dt, STEP_DURATION)
        finally:
            env.close()

    @requires_mujoco
    def test_all_six_motors_share_one_gear_ratio(self):
        """Anders als bei HalfCheetah ist die Übersetzung hier einheitlich."""
        env = make_walker2d_env(render_mode=None)
        try:
            gears = [float(value) for value in env.unwrapped.model.actuator_gear[:, 0]]
            self.assertEqual(len(gears), ACTION_DIM)
            self.assertEqual(set(gears), {GEAR})
            self.assertEqual(GEAR, 100.0)
            self.assertEqual(len(JOINT_NAMES), ACTION_DIM)
        finally:
            env.close()

    @requires_mujoco
    def test_the_joint_order_matches_the_action_order(self):
        """Die Observation führt die Gelenke ab Index 2 in Aktuatorreihenfolge."""
        import mujoco

        env = make_walker2d_env(render_mode=None)
        try:
            model = env.unwrapped.model
            actuated = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                          model.actuator_trnid[index, 0])
                        for index in range(model.nu)]
            # qpos[0..2] sind rootx, rootz, rooty; ab qpos[3] folgen die Gelenke.
            joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
                      for index in range(3, model.njnt)]
            self.assertEqual(actuated, joints)
            self.assertEqual(actuated, ["thigh_joint", "leg_joint", "foot_joint",
                                        "thigh_left_joint", "leg_left_joint",
                                        "foot_left_joint"])
        finally:
            env.close()

    @requires_mujoco
    def test_rendered_frame_has_the_declared_size(self):
        env = make_walker2d_env(render_mode="rgb_array")
        try:
            env.reset(seed=0)
            frame = np.asarray(env.render())
            self.assertEqual(frame.shape, (FRAME_HEIGHT, FRAME_WIDTH, 3))
        finally:
            env.close()

    @requires_mujoco
    def test_reward_is_exactly_the_sum_of_its_three_reported_parts(self):
        env = make_walker2d_env(render_mode=None)
        try:
            env.reset(seed=3)
            for _ in range(20):
                action = env.action_space.sample()
                _obs, reward, terminated, truncated, info = env.step(action)
                parts = (info["reward_survive"] + info["reward_forward"]
                         + info["reward_ctrl"])
                self.assertAlmostEqual(float(reward), float(parts), places=9)
                self.assertLessEqual(info["reward_ctrl"], 0.0)
                self.assertAlmostEqual(-float(info["reward_ctrl"]), control_cost(action),
                                       places=6)
                self.assertIn(float(info["reward_survive"]), (0.0, HEALTHY_REWARD))
                if terminated or truncated:
                    break
        finally:
            env.close()

    def test_control_cost_is_negligible_compared_to_halfcheetah(self):
        self.assertAlmostEqual(CTRL_COST_WEIGHT, 1e-3)
        # Voller Ausschlag aller sechs Motoren kostet 0,006 je Schritt.
        self.assertAlmostEqual(control_cost(np.ones(ACTION_DIM)), 0.006)
        self.assertAlmostEqual(control_cost(np.zeros(ACTION_DIM)), 0.0)
        self.assertAlmostEqual(control_cost(np.full(ACTION_DIM, 5.0)), 0.006)

    @requires_mujoco
    def test_a_fall_terminates_without_any_terminal_penalty(self):
        env = make_walker2d_env(render_mode=None)
        try:
            env.reset(seed=11)
            reward, info = 0.0, {}
            terminated = truncated = False
            while not (terminated or truncated):
                _obs, reward, terminated, truncated, info = env.step(
                    np.array([1, 1, -1, 1, 1, -1], dtype=np.float32))
            self.assertTrue(terminated, "Der Agent sollte in diesem Test stürzen.")
            self.assertFalse(truncated)
            # Der Überlebensbonus fällt weg, mehr passiert nicht: keine Strafe.
            self.assertAlmostEqual(float(info["reward_survive"]), 0.0)
            parts = (info["reward_survive"] + info["reward_forward"] + info["reward_ctrl"])
            self.assertAlmostEqual(float(reward), float(parts), places=9)
            self.assertLess(abs(float(reward)), 20.0)
        finally:
            env.close()

    @requires_mujoco
    def test_the_healthy_height_is_bounded_above_and_below(self):
        """Ein Sprung über 2,0 m beendet die Episode ebenso wie ein Sturz."""
        env = make_walker2d_env(render_mode=None)
        try:
            self.assertEqual(env.unwrapped._healthy_z_range, HEALTHY_Z_RANGE)
            self.assertEqual(env.unwrapped._healthy_angle_range, HEALTHY_ANGLE_RANGE)
            self.assertGreater(HEALTHY_Z_RANGE[1], HEALTHY_Z_RANGE[0])
            self.assertLess(HEALTHY_Z_RANGE[1], float("inf"))
        finally:
            env.close()

    @requires_mujoco
    def test_velocities_in_the_observation_are_clipped_to_ten(self):
        env = make_walker2d_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=1)
            for _ in range(80):
                observation, _r, terminated, truncated, _i = env.step(
                    np.array([1, -1, 1, -1, 1, -1], dtype=np.float32))
                self.assertTrue(np.all(np.abs(observation[8:]) <= VELOCITY_CLIP + 1e-9))
                if terminated or truncated:
                    env.reset(seed=1)
        finally:
            env.close()

    @requires_mujoco
    def test_position_is_reported_in_info_and_not_in_the_observation(self):
        env = make_walker2d_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=5)
            info: dict = {}
            for _ in range(30):
                observation, _r, terminated, truncated, info = env.step(
                    np.full(ACTION_DIM, 0.5, dtype=np.float32))
                if terminated or truncated:
                    break
            self.assertIn("x_position", info)
            self.assertIn("x_velocity", info)
            self.assertEqual(len(observation), OBSERVATION_DIM)
            self.assertNotIn(float(info["x_position"]), [float(v) for v in observation])
        finally:
            env.close()


class OutcomeTests(unittest.TestCase):
    def test_termination_is_a_fall_and_truncation_is_endurance(self):
        survived, fell, reached = episode_outcome(True, False, 120.0)
        self.assertFalse(survived)
        self.assertTrue(fell)
        self.assertFalse(reached)
        survived, fell, _reached = episode_outcome(False, True, 120.0)
        self.assertTrue(survived)
        self.assertFalse(fell)

    def test_the_target_mark_is_a_project_constant(self):
        self.assertFalse(episode_outcome(False, True, TARGET_RETURN - 0.1)[2])
        self.assertTrue(episode_outcome(False, True, TARGET_RETURN)[2])
        # Zwischen den Zoo-Benchmarkwerten von SAC (3863) und TD3 (4718).
        self.assertGreater(TARGET_RETURN, 3863.0)
        self.assertLess(TARGET_RETURN, 4718.0)

    def test_survival_and_fall_are_complementary(self):
        for terminated, truncated in ((True, False), (False, True)):
            survived, fell, _reached = episode_outcome(terminated, truncated, 0.0)
            self.assertNotEqual(survived, fell)


class ReadoutTests(unittest.TestCase):
    def test_action_readout_uses_one_gear_for_every_joint(self):
        readout = action_readout(np.ones(ACTION_DIM))
        self.assertEqual([joint["joint"] for joint in readout["joints"]], list(JOINT_NAMES))
        for joint in readout["joints"]:
            self.assertAlmostEqual(joint["torque"], GEAR)
            self.assertEqual(joint["direction"], "+")
        # Anders als bei HalfCheetah ist das Moment überall gleich.
        self.assertEqual(len({joint["torque"] for joint in readout["joints"]}), 1)

    def test_action_readout_translates_sign_and_zero(self):
        readout = action_readout(np.array([-1.0, 0.0, 0.5, 0.0, 0.0, 0.0]))
        self.assertEqual(readout["joints"][0]["direction"], "−")
        self.assertAlmostEqual(readout["joints"][0]["torque"], GEAR)
        self.assertEqual(readout["joints"][1]["torque"], 0.0)
        self.assertIn("kein Moment", readout["joints"][1]["text"])
        self.assertAlmostEqual(readout["joints"][2]["torque"], 0.5 * GEAR)

    def test_action_readout_clips_like_the_environment(self):
        readout = action_readout(np.array([2.5, -7.0, 0.25, 0.0, 1.0, -1.0]))
        self.assertEqual(readout["raw"][:2], [1.0, -1.0])

    def test_observation_readout_names_every_value(self):
        observation = np.arange(OBSERVATION_DIM, dtype=float) / 100.0
        values = observation_readout(observation)
        self.assertAlmostEqual(values["height"], 0.0)
        self.assertAlmostEqual(values["torso_angle"], 0.01)
        self.assertAlmostEqual(values["torso_angle_degrees"], math.degrees(0.01))
        self.assertEqual(len(values["joint_angles"]), 6)
        self.assertEqual(len(values["joint_velocities"]), 6)
        self.assertEqual(len(values["velocities"]), 9)
        self.assertFalse(values["velocity_clipped"])

    def test_observation_readout_flags_the_healthy_range(self):
        healthy = observation_readout(np.array([1.25, 0.0] + [0.0] * 15))
        self.assertTrue(healthy["healthy_height"])
        self.assertTrue(healthy["healthy_angle"])
        for height in (HEALTHY_Z_RANGE[0], HEALTHY_Z_RANGE[1]):
            self.assertFalse(observation_readout(
                np.array([height, 0.0] + [0.0] * 15))["healthy_height"])
        tilted = observation_readout(np.array([1.25, HEALTHY_ANGLE_RANGE[1]] + [0.0] * 15))
        self.assertFalse(tilted["healthy_angle"])

    def test_observation_readout_marks_clipped_velocities(self):
        observation = np.zeros(OBSERVATION_DIM)
        observation[8] = VELOCITY_CLIP
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

    def test_cma_profile_follows_pycma_and_stays_small(self):
        config = default_config("CMA-ES")
        self.assertEqual(config.actor_arch, CMA_NET_ARCH)
        self.assertEqual(config.actor_arch, ())          # linear
        self.assertEqual(config.parameter_count(), ACTION_DIM * OBSERVATION_DIM + ACTION_DIM)
        self.assertEqual(config.parameter_count(), 108)
        self.assertAlmostEqual(config.sigma0, 0.5)
        self.assertEqual(config.popsize, "auto")
        self.assertEqual(config.resolved_popsize(), 18)  # 4 + floor(3*ln 108)
        self.assertEqual(config.episodes_per_candidate, 1)
        self.assertFalse(config.diagonal)
        self.assertEqual(config.activation, "Tanh")
        # Ohne Beobachtungsstatistik arbeitet eine lineare Policy kaum.
        self.assertTrue(config.normalize_obs)

    def test_no_profile_requires_reward_normalisation(self):
        """Das wäre das PPO-Profil gewesen – PPO ist hier nicht dabei."""
        for name in ALGORITHMS:
            config = default_config(name)
            if config.uses("normalize_reward"):
                self.assertFalse(config.normalize_reward, name)

    def test_off_policy_buffers_hold_the_whole_budget(self):
        for name in OFF_POLICY_ALGORITHMS:
            config = default_config(name)
            self.assertEqual(config.buffer_size, DEFAULT_TOTAL_TIMESTEPS)
            # SB3-Default, von den Zoo-Profilen nicht überschrieben.
            self.assertEqual(config.buffer_size, Walker2dConfig().buffer_size)

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

    def test_sac_target_entropy_auto_equals_minus_action_dimension(self):
        config = default_config("SAC")
        self.assertEqual(config.sac_target_entropy(), "auto")
        self.assertEqual(-ACTION_DIM, -6)
        numeric = dataclasses.replace(config, target_entropy="-6")
        self.assertEqual(numeric.sac_target_entropy(), -6.0)

    def test_constructor_arguments_are_known_to_each_algorithm(self):
        import inspect

        from stable_baselines3 import SAC, TD3

        for name, algorithm_class in (("SAC", SAC), ("TD3", TD3)):
            known = set(inspect.signature(algorithm_class.__init__).parameters)
            unknown = set(default_config(name).model_kwargs()) - known
            self.assertEqual(unknown, set(), f"{name} kennt {unknown} nicht")

    def test_cma_es_has_none_of_the_gradient_parameters(self):
        """Weggelassen, nicht deaktiviert mitgeschleppt."""
        config = default_config("CMA-ES")
        for absent in ("learning_rate", "learning_rate_schedule", "batch_size", "gamma",
                       "buffer_size", "learning_starts", "tau", "train_freq",
                       "gradient_steps", "action_noise", "action_noise_sigma",
                       "critic_arch", "optimizer", "optimizer_eps",
                       "normalize_reward", "clip_reward"):
            self.assertFalse(config.uses(absent), absent)
        # Was es besitzt:
        for present in ("total_timesteps", "seed", "actor_arch", "activation",
                        "sigma0", "popsize", "episodes_per_candidate", "diagonal",
                        "normalize_obs", "clip_obs"):
            self.assertTrue(config.uses(present), present)

    def test_only_total_timesteps_and_seed_are_universal(self):
        universal = {name for name in ("total_timesteps", "seed", "learning_rate",
                                       "batch_size", "gamma", "buffer_size")
                     if all(default_config(a).uses(name) for a in ALGORITHMS)}
        self.assertEqual(universal, {"total_timesteps", "seed"})

    def test_cma_es_is_not_a_stable_baselines_algorithm(self):
        with self.assertRaises(ValueError):
            default_config("CMA-ES").model_kwargs()
        self.assertEqual(set(SB3_ALGORITHMS), {"SAC", "TD3"})
        self.assertEqual(set(EVOLUTIONARY_ALGORITHMS), {"CMA-ES"})

    def test_cma_es_rejects_a_population_that_cannot_be_ranked(self):
        for bad in ("1", "0", "abc"):
            config = dataclasses.replace(default_config("CMA-ES"), popsize=bad)
            with self.assertRaises(ValueError) as error:
                config.validate()
            self.assertIn("Population λ", str(error.exception))
        config = dataclasses.replace(default_config("CMA-ES"), sigma0=0.0)
        with self.assertRaises(ValueError) as error:
            config.validate()
        self.assertIn("Schrittweite σ₀", str(error.exception))

    def test_cma_es_needs_a_budget_for_one_full_generation(self):
        """Ohne vollständige Population gibt es keinen Update-Schritt."""
        config = dataclasses.replace(default_config("CMA-ES"),
                                     popsize="10", total_timesteps=5)
        with self.assertRaises(ValueError) as error:
            config.validate()
        self.assertIn("vollständige Generation", str(error.exception))

    def test_networks_are_not_unified_between_the_sb3_methods(self):
        """Mit CMA-ES ist eine gemeinsame Architektur ohnehin ausgeschlossen."""
        self.assertEqual(default_config("SAC").actor_arch, DEFAULT_NET_ARCH)
        self.assertEqual(default_config("TD3").actor_arch, TD3_NET_ARCH)
        self.assertNotEqual(DEFAULT_NET_ARCH, TD3_NET_ARCH)

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
        differences = dict(config_differences([default_config("CMA-ES"),
                                               default_config("SAC")]))
        # `buffer_size` kennt nur SAC, `sigma0` nur CMA-ES.
        self.assertNotIn("Replay Buffer |D|", differences)
        self.assertNotIn("Schrittweite σ₀", differences)
        self.assertNotIn("Lernrate α", differences)
        # Gemeinsam sind nur Budget, Seed, Policy und Normalisierung.
        self.assertIn("Actor-Hidden h_π", differences)

    def test_four_slots_are_supported(self):
        configs = [dataclasses.replace(default_config("TD3"), seed=index) for index in range(4)]
        differences = dict(config_differences(configs))
        self.assertEqual(differences["Zufallsstart s"], ["0", "1", "2", "3"])


@requires_mujoco
class TrainingTests(unittest.TestCase):
    def test_every_algorithm_trains_evaluates_and_reports_metrics(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                workbench = make_workbench(tiny_config(name))
                try:
                    metrics = workbench.train()
                    self.assertTrue(metrics)
                    for metric in metrics:
                        self.assertIsInstance(metric, EpisodeMetric)
                        self.assertLessEqual(metric.length, MAX_EPISODE_STEPS)
                        # Sturz und Durchhalten schließen sich aus.
                        self.assertNotEqual(metric.survived, metric.fell)
                    result = workbench.evaluate(1, seed=3)
                    self.assertEqual(result.episodes, 1)
                    self.assertLessEqual(result.mean_length, MAX_EPISODE_STEPS)
                    for rate in (result.survive_rate, result.fall_rate,
                                 result.target_rate):
                        self.assertGreaterEqual(rate, 0.0)
                        self.assertLessEqual(rate, 1.0)
                    self.assertAlmostEqual(result.survive_rate + result.fall_rate, 1.0)
                finally:
                    workbench.close()

    def test_evaluation_does_not_change_the_learning_state(self):
        workbench = Walker2dWorkbench(tiny_config("TD3"))
        try:
            workbench.train()
            before = [parameter.detach().clone()
                      for parameter in workbench.model.policy.parameters()]
            steps_before = workbench.model.num_timesteps
            workbench.evaluate(1, seed=0)
            for old, new in zip(before, workbench.model.policy.parameters()):
                self.assertTrue(torch.equal(old, new))
            self.assertEqual(workbench.model.num_timesteps, steps_before)
        finally:
            workbench.close()

    def test_continuing_training_keeps_the_step_counter(self):
        workbench = Walker2dWorkbench(tiny_config("SAC"))
        try:
            workbench.train()
            first = workbench.model.num_timesteps
            workbench.train()
            self.assertGreater(workbench.model.num_timesteps, first)
            self.assertGreater(len(workbench.history), 0)
        finally:
            workbench.close()

    def test_cma_es_has_no_replay_buffer_and_no_normaliser_object(self):
        workbench = make_workbench(tiny_config("CMA-ES"))
        try:
            workbench.create_model()
            self.assertFalse(workbench.uses_replay_buffer)
            self.assertIsNone(workbench.vec_normalize)
            self.assertIsNotNone(workbench.normalizer)
        finally:
            workbench.close()

    def test_td3_delays_the_actor_update(self):
        """Mit sehr großem `policy_delay` bleibt der Actor stehen, der Critic nicht."""
        config = tiny_config("TD3", total_timesteps=200, learning_starts=0, policy_delay=10_000)
        workbench = Walker2dWorkbench(config)
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
        workbench = Walker2dWorkbench(config)
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
        workbench = Walker2dWorkbench(tiny_config("SAC", learning_starts=0))
        try:
            workbench.create_model()
            self.assertEqual(float(workbench.model.target_entropy), -ACTION_DIM)
            before = workbench.model.log_ent_coef.detach().clone()
            workbench.train()
            self.assertFalse(torch.equal(before, workbench.model.log_ent_coef.detach()))
        finally:
            workbench.close()

    def test_sac_parameters_reach_the_model(self):
        config = tiny_config("SAC", tau=0.05, gamma=0.95)
        workbench = Walker2dWorkbench(config)
        try:
            workbench.create_model()
            self.assertAlmostEqual(workbench.model.tau, 0.05)
            self.assertAlmostEqual(workbench.model.gamma, 0.95)
        finally:
            workbench.close()

    def test_training_is_reproducible_for_a_fixed_seed(self):
        rewards = []
        for _ in range(2):
            workbench = Walker2dWorkbench(tiny_config("TD3", seed=123))
            try:
                metrics = workbench.train()
                rewards.append([round(metric.reward, 6) for metric in metrics])
            finally:
                workbench.close()
        self.assertEqual(rewards[0], rewards[1])
        self.assertTrue(rewards[0])

    def test_callback_reports_episodes_and_evaluations_through_the_queue(self):
        output: queue.Queue = queue.Queue()
        workbench = Walker2dWorkbench(tiny_config("SAC"))
        try:
            workbench.train(threading.Event(), output, ("compare", 2),
                            evaluation_interval=1_100, evaluation_episodes=1)
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
        workbench = Walker2dWorkbench(tiny_config("TD3"))
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
        workbench = Walker2dWorkbench(tiny_config("SAC"))
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
        workbench = Walker2dWorkbench(tiny_config("SAC", total_timesteps=20_000))
        try:
            workbench.train(stop)
            self.assertLess(workbench.model.num_timesteps, 20_000)
        finally:
            workbench.close()


@requires_mujoco
class NormalizationTests(unittest.TestCase):
    def test_statistics_grow_while_training_and_stay_frozen_in_evaluation(self):
        # Kein Profil dieses Projekts normalisiert per Voreinstellung.
        workbench = Walker2dWorkbench(tiny_config("SAC", normalize_obs=True))
        try:
            workbench.train()
            statistics = workbench.vec_normalize
            self.assertIsNotNone(statistics)
            self.assertGreater(statistics.obs_rms.count, 1.0)
            count = statistics.obs_rms.count
            mean = statistics.obs_rms.mean.copy()
            workbench.evaluate(1, seed=0)
            self.assertEqual(statistics.obs_rms.count, count)
            np.testing.assert_array_equal(statistics.obs_rms.mean, mean)
            # Auch die Aufbereitung für die Animation schreibt nichts fort.
            workbench.policy_observation(np.zeros(OBSERVATION_DIM, dtype=np.float32))
            self.assertEqual(statistics.obs_rms.count, count)
        finally:
            workbench.close()

    def test_reported_returns_stay_unnormalised(self):
        """Der Return muss exakt `Σvₓ + Σreward_ctrl` sein.

        Der Überlebensbonus liefert genau einen Punkt je gesundem Schritt, der
        Vorwärtsanteil `Σvₓ = mean_speed · length`, die Steuerkosten höchstens
        `0,006` je Schritt. Käme der ausgewiesene Return aus skalierten
        Rewards, träfe diese Schranke nicht zu.
        """
        workbench = Walker2dWorkbench(tiny_config("SAC", normalize_reward=True))
        try:
            metrics = workbench.train()
            self.assertTrue(workbench.config.normalize_reward)
            self.assertTrue(metrics)
            for metric in metrics:
                # Beim Sturz fällt der Bonus des letzten Schritts weg.
                expected = (metric.length - 1) + metric.mean_speed * metric.length
                ceiling = metric.length * CTRL_COST_WEIGHT * ACTION_DIM + 1.5
                self.assertLessEqual(abs(metric.reward - expected), ceiling)
        finally:
            workbench.close()

    def test_off_policy_methods_do_not_normalise_by_default(self):
        for name in OFF_POLICY_ALGORITHMS:
            workbench = Walker2dWorkbench(tiny_config(name))
            try:
                workbench.create_model()
                self.assertIsNone(workbench.vec_normalize)
            finally:
                workbench.close()


class CmaEsUnitTests(unittest.TestCase):
    """Bausteine von CMA-ES, die ohne Environment prüfbar sind."""

    def test_linear_policy_has_the_expected_parameter_count(self):
        policy = LinearTanhPolicy((), "Tanh")
        self.assertEqual(policy.size, ACTION_DIM * OBSERVATION_DIM + ACTION_DIM)
        self.assertEqual(policy.size, 108)
        # Ein Hidden Layer wächst wie erwartet.
        self.assertEqual(LinearTanhPolicy((16,), "ReLU").size,
                         OBSERVATION_DIM * 16 + 16 + 16 * ACTION_DIM + ACTION_DIM)

    def test_the_policy_output_always_fits_the_action_space(self):
        """Das abschließende `tanh` bildet exakt auf Box(-1, 1) ab."""
        policy = LinearTanhPolicy((8,), "ReLU")
        rng = np.random.default_rng(0)
        for _ in range(20):
            parameters = rng.normal(0, 50, policy.size)
            action = policy.act(parameters, rng.normal(0, 10, OBSERVATION_DIM))
            self.assertEqual(action.shape, (ACTION_DIM,))
            self.assertTrue(np.all(np.abs(action) <= 1.0))

    def test_the_policy_is_deterministic(self):
        policy = LinearTanhPolicy((), "Tanh")
        parameters = np.random.default_rng(1).normal(size=policy.size)
        observation = np.arange(OBSERVATION_DIM, dtype=float)
        np.testing.assert_array_equal(policy.act(parameters, observation),
                                      policy.act(parameters, observation))

    def test_normaliser_only_changes_on_commit(self):
        """Innerhalb einer Generation müssen alle Kandidaten dieselbe
        Normalisierung sehen, sonst wäre ihre Rangfolge verfälscht."""
        normalizer = RunningNormalizer(3, clip=10.0)
        before = normalizer.normalize(np.array([1.0, 2.0, 3.0])).copy()
        for _ in range(50):
            normalizer.observe(np.array([5.0, 5.0, 5.0]))
        # Noch nichts übernommen:
        np.testing.assert_array_equal(normalizer.normalize(np.array([1.0, 2.0, 3.0])), before)
        normalizer.commit()
        self.assertFalse(np.allclose(normalizer.normalize(np.array([1.0, 2.0, 3.0])), before))

    def test_normaliser_clips_and_survives_a_roundtrip(self):
        normalizer = RunningNormalizer(2, clip=3.0)
        for value in (0.0, 1.0, 2.0, 3.0):
            normalizer.observe(np.array([value, -value]))
        normalizer.commit()
        self.assertTrue(np.all(np.abs(normalizer.normalize(np.array([1e6, -1e6]))) <= 3.0))
        restored = RunningNormalizer.from_state(normalizer.state())
        np.testing.assert_allclose(restored.mean, normalizer.mean)
        np.testing.assert_allclose(restored.var, normalizer.var)
        self.assertEqual(restored.count, normalizer.count)


@requires_mujoco
class CmaEsTrainingTests(unittest.TestCase):
    def test_a_generation_moves_mean_and_step_size(self):
        workbench = make_workbench(tiny_config("CMA-ES"))
        try:
            workbench.create_model()
            mean = np.asarray(workbench.model.result.xfavorite).copy()
            sigma = workbench.model.sigma
            workbench.train()
            self.assertGreater(workbench.generations, 0)
            self.assertFalse(np.allclose(mean, workbench.model.result.xfavorite))
            self.assertNotAlmostEqual(sigma, workbench.model.sigma)
        finally:
            workbench.close()

    def test_the_step_budget_is_counted_in_environment_steps(self):
        budget = 900
        workbench = make_workbench(tiny_config("CMA-ES", total_timesteps=budget))
        try:
            workbench.train()
            self.assertGreaterEqual(workbench.num_timesteps, budget)
            # Der Überhang ist höchstens eine angefangene Episode.
            self.assertLess(workbench.num_timesteps - budget, MAX_EPISODE_STEPS)
            self.assertEqual(sum(item.length for item in workbench.history),
                             workbench.num_timesteps)
        finally:
            workbench.close()

    def test_evaluation_uses_the_distribution_mean_and_repeats_itself(self):
        """Nicht ein gezogener Kandidat: Sonst wäre die Evaluation zufällig."""
        workbench = make_workbench(tiny_config("CMA-ES"))
        try:
            workbench.train()
            snapshot, _ = workbench.current_snapshot()
            np.testing.assert_allclose(snapshot, workbench.model.result.xfavorite)
            first = workbench.evaluate(1, seed=5)
            second = workbench.evaluate(1, seed=5)
            self.assertAlmostEqual(first.mean_reward, second.mean_reward)
            self.assertEqual(first.mean_length, second.mean_length)
        finally:
            workbench.close()

    def test_a_candidate_evaluation_is_one_episode_and_one_curve_point(self):
        config = tiny_config("CMA-ES", popsize="4", total_timesteps=600)
        workbench = make_workbench(config)
        try:
            metrics = workbench.train()
            self.assertTrue(metrics)
            self.assertEqual([item.episode for item in metrics],
                             list(range(1, len(metrics) + 1)))
            # Jede Episode trägt ihre eigenen Schritte bei.
            self.assertEqual(metrics[-1].timesteps, workbench.num_timesteps)
        finally:
            workbench.close()

    def test_the_observation_statistic_grows_only_between_generations(self):
        workbench = make_workbench(tiny_config("CMA-ES"))
        try:
            workbench.create_model()
            self.assertIsNotNone(workbench.normalizer)
            start = workbench.normalizer.count
            workbench.train()
            self.assertGreater(workbench.normalizer.count, start)
            # In der Evaluation bleibt sie eingefroren.
            frozen = workbench.normalizer.count
            workbench.evaluate(1, seed=0)
            self.assertEqual(workbench.normalizer.count, frozen)
        finally:
            workbench.close()

    def test_the_best_episode_keeps_its_own_parameter_vector(self):
        workbench = make_workbench(tiny_config("CMA-ES"))
        try:
            metrics = workbench.train()
            snapshot = workbench.best_snapshot()
            self.assertIsNotNone(snapshot)
            best_metric, (parameters, _normalizer) = snapshot
            self.assertEqual(best_metric, max(metrics, key=lambda item: item.reward))
            self.assertEqual(parameters.shape, (workbench.config.parameter_count(),))
            # Der Mittelwert der Verteilung ist ein anderer Vektor.
            self.assertFalse(np.allclose(parameters, workbench.current_snapshot()[0]))
        finally:
            workbench.close()

    def test_a_new_model_forgets_everything(self):
        workbench = make_workbench(tiny_config("CMA-ES"))
        try:
            workbench.train()
            workbench.create_model()
            self.assertIsNone(workbench.best_snapshot())
            self.assertEqual(workbench.num_timesteps, 0)
            self.assertEqual(workbench.generations, 0)
            self.assertEqual(workbench.history, [])
        finally:
            workbench.close()

    def test_a_roundtrip_continues_the_same_search(self):
        workbench = make_workbench(tiny_config("CMA-ES"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                workbench.save(base)
                restored = load_workbench(base, "CMA-ES")
                try:
                    self.assertAlmostEqual(restored.model.sigma, workbench.model.sigma)
                    np.testing.assert_allclose(restored.model.result.xfavorite,
                                               workbench.model.result.xfavorite)
                    self.assertEqual(restored.generations, workbench.generations)
                    self.assertEqual(restored.num_timesteps, workbench.num_timesteps)
                    np.testing.assert_allclose(restored.normalizer.mean,
                                               workbench.normalizer.mean)
                finally:
                    restored.close()
            finally:
                workbench.close()


@requires_mujoco
class CheckpointTests(unittest.TestCase):
    def test_roundtrip_keeps_exactly_the_state_each_method_owns(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                workbench = make_workbench(tiny_config(name))
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
                        restored = load_workbench(base, name)
                        try:
                            self.assertEqual(restored.config, workbench.config)
                            observation = np.zeros((1, OBSERVATION_DIM), dtype=np.float32)
                            # Über die Slot-Abstraktion, damit derselbe Test
                            # für SB3 und CMA-ES gilt.
                            expected = workbench.act(workbench.current_snapshot(),
                                                     observation.reshape(-1))
                            actual = restored.act(restored.current_snapshot(),
                                                  observation.reshape(-1))
                            np.testing.assert_allclose(expected, actual, atol=1e-6)
                        finally:
                            restored.close()
                    finally:
                        workbench.close()

    def test_normalisation_statistics_survive_the_roundtrip(self):
        workbench = Walker2dWorkbench(tiny_config("SAC", normalize_obs=True))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                mean = workbench.vec_normalize.obs_rms.mean.copy()
                workbench.save(base)
                restored = Walker2dWorkbench.load(base, "SAC")
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
        workbench = make_workbench(tiny_config("SAC"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                _model, _replay, metadata_path = workbench.save(base)
            finally:
                workbench.close()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["environment"] = "Hopper-v5"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError) as error:
                load_workbench(base)
            self.assertIn("Hopper-v5", str(error.exception))
            self.assertIn(ENV_ID, str(error.exception))

    def test_a_foreign_algorithm_is_rejected(self):
        workbench = make_workbench(tiny_config("TD3"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                workbench.save(base)
            finally:
                workbench.close()
            with self.assertRaises(ValueError) as error:
                load_workbench(base, "CMA-ES")
            self.assertIn("TD3", str(error.exception))

    def test_missing_metadata_is_reported_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError) as error:
                load_workbench(Path(folder) / "nothing")
            self.assertIn("Metadatendatei", str(error.exception))


@requires_mujoco
class ComparisonTests(unittest.TestCase):
    def _run(self, configs: list[Walker2dConfig]) -> list[Walker2dWorkbench]:
        """Alle Slots parallel, so wie es die GUI tut."""
        benches = [make_workbench(config) for config in configs]
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

    def test_all_three_methods_run_in_parallel_and_keep_separate_results(self):
        configs = [tiny_config(name) for name in ("TD3", "SAC", "CMA-ES")]
        benches = self._run(configs)
        try:
            for bench in benches:
                self.assertTrue(bench.history, f"{bench.config.algorithm} lieferte keine Episode")
            self.assertEqual(len({id(bench.model) for bench in benches}), 3)
            # Nur CMA-ES zählt Generationen; die beiden anderen liefern None.
            generations = [bench.generations for bench in benches]
            self.assertEqual(sum(value is None for value in generations), 2)
            self.assertGreater(max(v for v in generations if v is not None), 0)
        finally:
            for bench in benches:
                bench.close()
