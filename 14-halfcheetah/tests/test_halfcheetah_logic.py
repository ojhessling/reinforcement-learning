"""Tests für Environment, Verfahren, Metriken und Vergleich des HalfCheetah-Projekts."""

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

from halfcheetah_logic import (
    ACTION_DIM,
    ACTION_NOISES,
    ALGORITHMS,
    CTRL_COST_WEIGHT,
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_EVALUATION_INTERVAL,
    DEFAULT_NET_ARCH,
    DEFAULT_TOTAL_TIMESTEPS,
    ENV_ID,
    FORWARD_REWARD_WEIGHT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GEARS,
    JOINT_NAMES,
    MAX_EPISODE_STEPS,
    OBSERVATION_DIM,
    OFF_POLICY_ALGORITHMS,
    SOLVED_RETURN,
    STEP_DURATION,
    EpisodeMetric,
    HalfCheetahConfig,
    HalfCheetahWorkbench,
    LinearSchedule,
    action_readout,
    config_differences,
    control_cost,
    default_config,
    episode_outcome,
    make_action_noise,
    make_halfcheetah_env,
    observation_readout,
    reward_readout,
)

#: MuJoCo und imageio gehören zu `gymnasium[mujoco]`. Ohne sie lässt sich kein
#: Environment erzeugen; die reine Konfigurations- und Metrikenlogik schon.
HAS_MUJOCO = all(importlib.util.find_spec(name) is not None for name in ("mujoco", "imageio"))
requires_mujoco = unittest.skipUnless(HAS_MUJOCO, "MuJoCo ist nicht installiert")


def tiny_config(algorithm: str, **overrides) -> HalfCheetahConfig:
    """Klein genug für einen schnellen Test, groß genug für ganze Episoden.

    Weil `HalfCheetah-v5` nie vorzeitig endet, dauert **jede** Episode exakt
    1000 Schritte. Ein Budget darunter liefert gar keine abgeschlossene
    Episode; deshalb 1200 Schritte und dafür sehr kleine Netze.
    """
    values = dict(total_timesteps=1_200, actor_arch=(16, 16), critic_arch=(16, 16), seed=7)
    if algorithm == "PPO":
        values.update(n_steps=400, batch_size=25)
    else:
        values.update(batch_size=32, buffer_size=2_000, learning_starts=50)
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
        env = make_halfcheetah_env(render_mode=None)
        try:
            self.assertEqual(env.spec.id, ENV_ID)
            # Physik und Reward bleiben unverändert: Kein einziger der
            # Environment-Parameter wird übergeben.
            self.assertEqual(set(env.spec.kwargs), {"width", "height"})
            self.assertEqual(env.spec.kwargs["width"], FRAME_WIDTH)
            self.assertEqual(env.spec.kwargs["height"], FRAME_HEIGHT)
        finally:
            env.close()

    @requires_mujoco
    def test_render_mode_is_only_added_when_asked_for(self):
        env = make_halfcheetah_env(render_mode="rgb_array")
        try:
            self.assertEqual(env.spec.kwargs["render_mode"], "rgb_array")
        finally:
            env.close()

    @requires_mujoco
    def test_spaces_and_frame_rate(self):
        env = make_halfcheetah_env(render_mode=None)
        try:
            self.assertEqual(env.action_space.shape, (ACTION_DIM,))
            self.assertEqual(float(env.action_space.low.min()), -1.0)
            self.assertEqual(float(env.action_space.high.max()), 1.0)
            self.assertEqual(env.observation_space.shape, (OBSERVATION_DIM,))
            self.assertEqual(env.observation_space.dtype, np.float64)
            self.assertEqual(env.metadata["render_fps"], 20)
            self.assertAlmostEqual(env.unwrapped.dt, STEP_DURATION)
        finally:
            env.close()

    @requires_mujoco
    def test_every_motor_has_its_own_gear_ratio(self):
        """Anders als beim Hopper unterscheiden sich die sechs Übersetzungen."""
        env = make_halfcheetah_env(render_mode=None)
        try:
            gears = tuple(float(value) for value in env.unwrapped.model.actuator_gear[:, 0])
            self.assertEqual(gears, GEARS)
            self.assertEqual(gears, (120.0, 90.0, 60.0, 120.0, 60.0, 30.0))
            self.assertEqual(len(GEARS), ACTION_DIM)
            self.assertEqual(len(JOINT_NAMES), ACTION_DIM)
            # Ein gemeinsamer Faktor wäre hier schlicht falsch.
            self.assertGreater(len(set(GEARS)), 1)
            names = [env.unwrapped.model.actuator(index).name for index in range(ACTION_DIM)]
            self.assertEqual(names, ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"])
        finally:
            env.close()

    @requires_mujoco
    def test_rendered_frame_has_the_declared_size(self):
        env = make_halfcheetah_env(render_mode="rgb_array")
        try:
            env.reset(seed=0)
            frame = np.asarray(env.render())
            self.assertEqual(frame.shape, (FRAME_HEIGHT, FRAME_WIDTH, 3))
        finally:
            env.close()

    @requires_mujoco
    def test_reward_is_exactly_forward_minus_control_cost(self):
        env = make_halfcheetah_env(render_mode=None)
        try:
            env.reset(seed=3)
            for _ in range(20):
                action = env.action_space.sample()
                _obs, reward, _t, _tr, info = env.step(action)
                parts = info["reward_forward"] + info["reward_ctrl"]
                self.assertAlmostEqual(float(reward), float(parts), places=9)
                # Es gibt genau zwei Anteile – keinen Überlebensbonus.
                self.assertNotIn("reward_survive", info)
                self.assertLessEqual(info["reward_ctrl"], 0.0)
                self.assertAlmostEqual(
                    float(info["reward_forward"]),
                    FORWARD_REWARD_WEIGHT * float(info["x_velocity"]), places=9)
                self.assertAlmostEqual(-float(info["reward_ctrl"]), control_cost(action),
                                       places=6)
        finally:
            env.close()

    def test_control_cost_weight_is_a_hundred_times_the_hopper_value(self):
        self.assertAlmostEqual(CTRL_COST_WEIGHT, 0.1)
        # Voller Ausschlag aller sechs Motoren kostet 0,6 je Schritt.
        self.assertAlmostEqual(control_cost(np.ones(ACTION_DIM)), 0.6)
        self.assertAlmostEqual(control_cost(np.zeros(ACTION_DIM)), 0.0)
        # Auch außerhalb des Wertebereichs wird geclippt gerechnet.
        self.assertAlmostEqual(control_cost(np.full(ACTION_DIM, 5.0)), 0.6)

    @requires_mujoco
    def test_the_environment_never_terminates(self):
        """Der wichtigste Test des Projekts: An ihm hängt die Metrikwahl.

        Ohne terminalen Zustand gibt es weder Sturz- noch Durchhaltequote, und
        jede Episode ist exakt gleich lang.
        """
        env = make_halfcheetah_env(render_mode=None)
        try:
            for seed, action in ((0, None), (1, np.ones(ACTION_DIM, dtype=np.float32)),
                                 (2, -np.ones(ACTION_DIM, dtype=np.float32))):
                env.reset(seed=seed)
                steps = 0
                terminated = truncated = False
                while not (terminated or truncated):
                    chosen = env.action_space.sample() if action is None else action
                    _obs, _r, terminated, truncated, _info = env.step(chosen)
                    steps += 1
                    self.assertFalse(terminated,
                                     f"HalfCheetah terminierte in Schritt {steps}")
                self.assertTrue(truncated)
                self.assertEqual(steps, MAX_EPISODE_STEPS)
        finally:
            env.close()

    @requires_mujoco
    def test_position_is_reported_in_info_and_not_in_the_observation(self):
        env = make_halfcheetah_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=5)
            info: dict = {}
            for _ in range(60):
                observation, _r, _t, _tr, info = env.step(
                    np.full(ACTION_DIM, 0.8, dtype=np.float32))
            self.assertIn("x_position", info)
            self.assertIn("x_velocity", info)
            self.assertEqual(len(observation), OBSERVATION_DIM)
            self.assertNotIn(float(info["x_position"]), [float(v) for v in observation])
        finally:
            env.close()

    @requires_mujoco
    def test_velocities_are_not_clipped(self):
        """Beim Hopper sind sie auf ±10 geclippt – hier ausdrücklich nicht."""
        env = make_halfcheetah_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=1)
            extremes = []
            for _ in range(400):
                observation, _r, _t, truncated, _i = env.step(
                    np.array([1, -1, 1, -1, 1, -1], dtype=np.float32))
                extremes.append(float(np.max(np.abs(observation[8:]))))
                if truncated:
                    break
            # Der Hopper würde hier exakt bei 10 abschneiden; hier nicht.
            self.assertGreater(max(extremes), 10.0)
        finally:
            env.close()


class OutcomeTests(unittest.TestCase):
    def test_forward_is_decided_by_the_mean_velocity(self):
        forward, _solved = episode_outcome(1000.0, 2.5)
        self.assertTrue(forward)
        forward, _solved = episode_outcome(-120.0, -0.4)
        self.assertFalse(forward)
        # Genau null gilt als nicht vorangekommen.
        self.assertFalse(episode_outcome(0.0, 0.0)[0])

    def test_solved_needs_the_official_threshold(self):
        self.assertFalse(episode_outcome(SOLVED_RETURN - 0.1, 5.0)[1])
        self.assertTrue(episode_outcome(SOLVED_RETURN, 5.0)[1])
        self.assertEqual(SOLVED_RETURN, 4800.0)

    def test_backward_episodes_are_negative(self):
        """Rückwärtslaufen liefert negativen Vorwärtsanteil und damit Return."""
        forward, solved = episode_outcome(-300.0, -1.2)
        self.assertFalse(forward)
        self.assertFalse(solved)


class ReadoutTests(unittest.TestCase):
    def test_action_readout_uses_the_gear_of_each_joint(self):
        readout = action_readout(np.ones(ACTION_DIM))
        self.assertEqual([joint["joint"] for joint in readout["joints"]], list(JOINT_NAMES))
        for joint, gear in zip(readout["joints"], GEARS):
            self.assertAlmostEqual(joint["gear"], gear)
            self.assertAlmostEqual(joint["torque"], gear)
            self.assertEqual(joint["direction"], "+")
        # Ein gemeinsamer Faktor ergäbe überall dasselbe Moment.
        torques = {joint["torque"] for joint in readout["joints"]}
        self.assertGreater(len(torques), 1)

    def test_action_readout_translates_sign_and_zero(self):
        readout = action_readout(np.array([-1.0, 0.0, 0.5, 0.0, 0.0, 0.0]))
        self.assertEqual(readout["joints"][0]["direction"], "−")
        self.assertAlmostEqual(readout["joints"][0]["torque"], GEARS[0])
        self.assertEqual(readout["joints"][1]["torque"], 0.0)
        self.assertIn("kein Moment", readout["joints"][1]["text"])
        self.assertAlmostEqual(readout["joints"][2]["torque"], 0.5 * GEARS[2])

    def test_action_readout_clips_like_the_environment(self):
        readout = action_readout(np.array([2.5, -7.0, 0.25, 0.0, 1.0, -1.0]))
        self.assertEqual(readout["raw"][:2], [1.0, -1.0])
        self.assertAlmostEqual(readout["joints"][0]["torque"], GEARS[0])

    def test_observation_readout_names_every_value(self):
        observation = np.arange(OBSERVATION_DIM, dtype=float) / 10.0
        values = observation_readout(observation)
        self.assertAlmostEqual(values["height"], 0.0)
        self.assertAlmostEqual(values["torso_angle"], 0.1)
        self.assertAlmostEqual(values["torso_angle_degrees"], math.degrees(0.1))
        self.assertEqual(len(values["joint_angles"]), 6)
        self.assertEqual(len(values["joint_velocities"]), 6)
        self.assertAlmostEqual(values["vx"], 0.8)
        self.assertAlmostEqual(values["vz"], 0.9)
        self.assertAlmostEqual(values["torso_angular_velocity"], 1.0)

    def test_observation_readout_has_no_clipping_or_health_flags(self):
        """Beides gibt es in diesem Environment nicht."""
        values = observation_readout(np.zeros(OBSERVATION_DIM))
        for absent in ("velocity_clipped", "healthy_height", "healthy_angle"):
            self.assertNotIn(absent, values)

    def test_reward_readout_takes_the_parts_from_info(self):
        info = {"reward_forward": 2.5, "reward_ctrl": -0.42,
                "x_position": 12.5, "x_velocity": 2.5}
        values = reward_readout(info)
        self.assertAlmostEqual(values["forward"], 2.5)
        self.assertAlmostEqual(values["ctrl"], -0.42)
        self.assertAlmostEqual(values["x_position"], 12.5)
        self.assertNotIn("survive", values)


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
        self.assertAlmostEqual(config.learning_rate, 2.0633e-5)
        self.assertAlmostEqual(config.gamma, 0.98)
        self.assertEqual(config.n_steps, 512)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.n_epochs, 20)
        self.assertAlmostEqual(config.gae_lambda, 0.92)
        self.assertAlmostEqual(config.clip_range, 0.1)
        self.assertAlmostEqual(config.ent_coef, 0.000401762)
        self.assertAlmostEqual(config.vf_coef, 0.58096)
        self.assertAlmostEqual(config.max_grad_norm, 0.8)
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
            self.assertEqual(config.buffer_size, HalfCheetahConfig().buffer_size)

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
        self.assertEqual(-ACTION_DIM, -6)
        numeric = dataclasses.replace(config, target_entropy="-6")
        self.assertEqual(numeric.sac_target_entropy(), -6.0)

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
                workbench = HalfCheetahWorkbench(tiny_config(name))
                try:
                    metrics = workbench.train()
                    self.assertTrue(metrics)
                    for metric in metrics:
                        self.assertIsInstance(metric, EpisodeMetric)
                        # Ohne terminalen Zustand ist jede Episode gleich lang.
                        self.assertEqual(metric.length, MAX_EPISODE_STEPS)
                        self.assertEqual(metric.forward, metric.mean_speed > 0)
                    result = workbench.evaluate(1, seed=3)
                    self.assertEqual(result.episodes, 1)
                    # Konstante Länge – deshalb keine Summary-Zeile dafür.
                    self.assertEqual(result.mean_length, MAX_EPISODE_STEPS)
                    for rate in (result.forward_rate, result.backward_rate,
                                 result.solved_rate):
                        self.assertGreaterEqual(rate, 0.0)
                        self.assertLessEqual(rate, 1.0)
                    self.assertAlmostEqual(result.forward_rate + result.backward_rate, 1.0)
                finally:
                    workbench.close()

    def test_evaluation_does_not_change_the_learning_state(self):
        workbench = HalfCheetahWorkbench(tiny_config("TD3"))
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
        workbench = HalfCheetahWorkbench(tiny_config("PPO"))
        try:
            workbench.train()
            first = workbench.model.num_timesteps
            workbench.train()
            self.assertGreater(workbench.model.num_timesteps, first)
            self.assertGreater(len(workbench.history), 0)
        finally:
            workbench.close()

    def test_ppo_has_no_replay_buffer(self):
        workbench = HalfCheetahWorkbench(tiny_config("PPO"))
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
        workbench = HalfCheetahWorkbench(config)
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
        workbench = HalfCheetahWorkbench(config)
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
        workbench = HalfCheetahWorkbench(tiny_config("SAC", learning_starts=0))
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
        workbench = HalfCheetahWorkbench(config)
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
            workbench = HalfCheetahWorkbench(tiny_config("TD3", seed=123))
            try:
                metrics = workbench.train()
                rewards.append([round(metric.reward, 6) for metric in metrics])
            finally:
                workbench.close()
        self.assertEqual(rewards[0], rewards[1])
        self.assertTrue(rewards[0])

    def test_callback_reports_episodes_and_evaluations_through_the_queue(self):
        output: queue.Queue = queue.Queue()
        workbench = HalfCheetahWorkbench(tiny_config("SAC"))
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
        workbench = HalfCheetahWorkbench(tiny_config("TD3"))
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
        workbench = HalfCheetahWorkbench(tiny_config("SAC"))
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
        workbench = HalfCheetahWorkbench(tiny_config("SAC", total_timesteps=20_000))
        try:
            workbench.train(stop)
            self.assertLess(workbench.model.num_timesteps, 20_000)
        finally:
            workbench.close()


@requires_mujoco
class NormalizationTests(unittest.TestCase):
    def test_statistics_grow_while_training_and_stay_frozen_in_evaluation(self):
        workbench = HalfCheetahWorkbench(tiny_config("PPO"))
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

        Beide Summanden stammen aus `info` und sind nie normalisiert. Käme der
        ausgewiesene Return aus den skalierten Rewards, stimmte die Identität
        nicht mehr.
        """
        workbench = HalfCheetahWorkbench(tiny_config("PPO"))
        try:
            metrics = workbench.train()
            self.assertTrue(workbench.config.normalize_reward)
            self.assertTrue(metrics)
            for metric in metrics:
                expected = metric.mean_speed * metric.length + metric.ctrl_cost
                self.assertAlmostEqual(metric.reward, expected, places=3)
        finally:
            workbench.close()

    def test_off_policy_methods_do_not_normalise_by_default(self):
        for name in OFF_POLICY_ALGORITHMS:
            workbench = HalfCheetahWorkbench(tiny_config(name))
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
                workbench = HalfCheetahWorkbench(tiny_config(name))
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
                        restored = HalfCheetahWorkbench.load(base, name)
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
        workbench = HalfCheetahWorkbench(tiny_config("PPO"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                mean = workbench.vec_normalize.obs_rms.mean.copy()
                workbench.save(base)
                restored = HalfCheetahWorkbench.load(base, "PPO")
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
        workbench = HalfCheetahWorkbench(tiny_config("SAC"))
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
                HalfCheetahWorkbench.load(base)
            self.assertIn("Walker2d-v5", str(error.exception))
            self.assertIn(ENV_ID, str(error.exception))

    def test_a_foreign_algorithm_is_rejected(self):
        workbench = HalfCheetahWorkbench(tiny_config("TD3"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                workbench.save(base)
            finally:
                workbench.close()
            with self.assertRaises(ValueError) as error:
                HalfCheetahWorkbench.load(base, "PPO")
            self.assertIn("TD3", str(error.exception))

    def test_missing_metadata_is_reported_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError) as error:
                HalfCheetahWorkbench.load(Path(folder) / "nothing")
            self.assertIn("Metadatendatei", str(error.exception))


@requires_mujoco
class ComparisonTests(unittest.TestCase):
    def _run(self, configs: list[HalfCheetahConfig]) -> list[HalfCheetahWorkbench]:
        """Alle Slots parallel, so wie es die GUI tut."""
        benches = [HalfCheetahWorkbench(config) for config in configs]
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
        configs = [tiny_config(name) for name in ("PPO", "SAC", "TD3", "SAC")]
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
