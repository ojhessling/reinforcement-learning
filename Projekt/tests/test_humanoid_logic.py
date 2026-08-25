"""Tests für Environment, Verfahren, Metriken und Vergleich des Humanoid-Projekts."""

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

from humanoid_logic import (
    ACTION_DIM,
    ACTION_LIMIT,
    ACTION_NOISES,
    ACTUATOR_GROUPS,
    ACTUATOR_LABELS,
    ACTUATOR_NAMES,
    ACTUATOR_TO_JOINT,
    ALGORITHMS,
    CONTACT_COST_MAX,
    CTRL_COST_WEIGHT,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_EPISODES,
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_NET_ARCH,
    DEFAULT_PROFILES,
    DEFAULT_TORCH_THREADS,
    DEFAULT_TOTAL_TIMESTEPS,
    ENV_ID,
    FORWARD_REWARD_WEIGHT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GEARS,
    HEALTHY_REWARD,
    HEALTHY_Z_RANGE,
    JOINT_NAMES,
    MAX_EPISODE_STEPS,
    OBS_CFRC_EXT,
    OBS_CINERT,
    OBS_CVEL,
    OBS_HEIGHT,
    OBS_JOINT_ANGLES,
    OBS_QFRC_ACTUATOR,
    OBS_QUATERNION,
    OBS_QVEL,
    OBSERVATION_DIM,
    OFF_POLICY_ALGORITHMS,
    PROFILE_BUDGET,
    STEP_DURATION,
    REPORT_TIMESTEPS,
    STEPS_PER_THOUSAND_EPISODES,
    STOP_REASONS,
    TARGET_RETURN,
    VELOCITY_CLIP,
    EpisodeMetric,
    Float32Observation,
    HumanoidCallback,
    HumanoidConfig,
    HumanoidWorkbench,
    LinearSchedule,
    action_readout,
    config_differences,
    contact_cost,
    control_cost,
    default_config,
    episode_outcome,
    make_action_noise,
    make_humanoid_env,
    observation_readout,
    reward_readout,
    set_torch_threads,
    torso_tilt_degrees,
)

#: MuJoCo und imageio gehören zu `gymnasium[mujoco]`. Ohne sie lässt sich kein
#: Environment erzeugen; die reine Konfigurations- und Metrikenlogik schon.
HAS_MUJOCO = all(importlib.util.find_spec(name) is not None for name in ("mujoco", "imageio"))
requires_mujoco = unittest.skipUnless(HAS_MUJOCO, "MuJoCo ist nicht installiert")


def tiny_config(algorithm: str, **overrides) -> HumanoidConfig:
    """Klein genug für einen schnellen Test, groß genug für ganze Episoden.

    `Humanoid-v5` terminiert beim Sturz. Eine untrainierte Policy fällt nach
    im Mittel 24,5 Schritten, sodass 1200 Schritte rund 50 abgeschlossene
    Episoden ergeben. Die Netze sind bewusst winzig: Mit 348 Beobachtungswerten
    kostet die Standardgröße auch bei kurzen Läufen spürbar Zeit.
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
        # Humanoid-v5 fuehrt KEINEN reward_threshold. Die Zielmarke 5000 ist
        # eine Projektkonstante und darf nicht als offizielle Schwelle gelten.
        self.assertIsNone(spec.reward_threshold)
        self.assertEqual(TARGET_RETURN, 5000.0)

    def test_target_return_equals_a_full_episode_of_survival(self):
        """Die Zielmarke ist begruendet, nicht gegriffen."""
        self.assertAlmostEqual(TARGET_RETURN, MAX_EPISODE_STEPS * HEALTHY_REWARD)

    @requires_mujoco
    def test_factory_requests_exactly_the_frame_size_and_no_physics_arguments(self):
        env = make_humanoid_env(render_mode=None)
        try:
            self.assertEqual(env.spec.id, ENV_ID)
            # Physik, Reward, Abbruch und Observationsschalter bleiben
            # unveraendert: Kein einziger dieser Parameter wird uebergeben.
            self.assertEqual(set(env.spec.kwargs), {"width", "height"})
            self.assertEqual(env.spec.kwargs["width"], FRAME_WIDTH)
            self.assertEqual(env.spec.kwargs["height"], FRAME_HEIGHT)
        finally:
            env.close()

    @requires_mujoco
    def test_render_mode_is_only_added_when_asked_for(self):
        env = make_humanoid_env(render_mode="rgb_array")
        try:
            self.assertEqual(env.spec.kwargs["render_mode"], "rgb_array")
        finally:
            env.close()

    @requires_mujoco
    def test_spaces_and_frame_rate(self):
        env = make_humanoid_env(render_mode=None)
        try:
            self.assertEqual(env.action_space.shape, (ACTION_DIM,))
            # Der Wertebereich ist +-0,4, NICHT +-1 wie bei Hopper,
            # HalfCheetah und Walker2d.
            self.assertAlmostEqual(float(env.action_space.low.min()), -ACTION_LIMIT, places=6)
            self.assertAlmostEqual(float(env.action_space.high.max()), ACTION_LIMIT, places=6)
            self.assertNotAlmostEqual(float(env.action_space.high.max()), 1.0)
            self.assertEqual(env.observation_space.shape, (OBSERVATION_DIM,))
            self.assertEqual(env.observation_space.dtype, np.float64)
            self.assertEqual(env.metadata["render_fps"], 67)
            self.assertAlmostEqual(env.unwrapped.dt, STEP_DURATION)
            self.assertEqual(env.unwrapped.frame_skip, 5)
        finally:
            env.close()

    @requires_mujoco
    def test_observation_splits_into_the_eight_documented_blocks(self):
        """Die Aufteilung wird gegen das Modell geprueft, nicht hergeleitet."""
        env = make_humanoid_env(render_mode=None)
        try:
            structure = env.unwrapped.observation_structure
            self.assertEqual(structure["skipped_qpos"], 2)
            self.assertEqual(structure["qpos"], 22)
            self.assertEqual(structure["qvel"], 23)
            self.assertEqual(structure["cinert"], 130)
            self.assertEqual(structure["cvel"], 78)
            self.assertEqual(structure["qfrc_actuator"], 17)
            self.assertEqual(structure["cfrc_ext"], 78)
            self.assertEqual(sum(structure[key] for key in
                                 ("qpos", "qvel", "cinert", "cvel",
                                  "qfrc_actuator", "cfrc_ext")), OBSERVATION_DIM)
            observation, _ = env.reset(seed=3)
            for _ in range(5):
                observation, *_ = env.step(env.action_space.sample())
            data = env.unwrapped.data
            self.assertAlmostEqual(observation[OBS_HEIGHT], data.qpos[2])
            np.testing.assert_allclose(observation[OBS_QUATERNION], data.qpos[3:7])
            np.testing.assert_allclose(observation[OBS_JOINT_ANGLES], data.qpos[7:24])
            np.testing.assert_allclose(observation[OBS_QVEL], data.qvel)
            np.testing.assert_allclose(observation[OBS_CINERT], data.cinert[1:].ravel())
            np.testing.assert_allclose(observation[OBS_CVEL], data.cvel[1:].ravel())
            np.testing.assert_allclose(observation[OBS_QFRC_ACTUATOR], data.qfrc_actuator[6:])
            np.testing.assert_allclose(observation[OBS_CFRC_EXT], data.cfrc_ext[1:].ravel())
        finally:
            env.close()

    @requires_mujoco
    def test_every_motor_has_its_own_gear_ratio(self):
        """Die Uebersetzungen reichen von 25 (Arme) bis 300 (Huefte)."""
        env = make_humanoid_env(render_mode=None)
        try:
            gears = tuple(float(value) for value in env.unwrapped.model.actuator_gear[:, 0])
            self.assertEqual(gears, GEARS)
            self.assertEqual(len(GEARS), ACTION_DIM)
            self.assertEqual(len(ACTUATOR_NAMES), ACTION_DIM)
            self.assertEqual(len(ACTUATOR_LABELS), ACTION_DIM)
            self.assertEqual(set(GEARS), {25.0, 100.0, 200.0, 300.0})
            names = [env.unwrapped.model.actuator(index).name for index in range(ACTION_DIM)]
            self.assertEqual(names, list(ACTUATOR_NAMES))
        finally:
            env.close()

    @requires_mujoco
    def test_joint_order_in_the_observation_differs_from_the_actuator_order(self):
        """Die Stolperstelle des Projekts.

        `abdomen_y` und `abdomen_z` sind vertauscht: Wer `obs[5 + i]` dem
        Aktuator `i` zuordnet, beschriftet die ersten beiden Werte falsch.
        """
        import mujoco
        env = make_humanoid_env(render_mode=None)
        try:
            model = env.unwrapped.model
            hinges = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
                      for index in range(model.njnt)
                      if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE]
            self.assertEqual(hinges, list(JOINT_NAMES))
            driven = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        model.actuator_trnid[index, 0])
                      for index in range(model.nu)]
            self.assertEqual(driven, list(ACTUATOR_NAMES))
            # Genau hier weichen die beiden Reihenfolgen voneinander ab.
            self.assertNotEqual(list(JOINT_NAMES), list(ACTUATOR_NAMES))
            self.assertEqual(ACTUATOR_TO_JOINT[:3], (1, 0, 2))
            for index, name in enumerate(ACTUATOR_NAMES):
                self.assertEqual(JOINT_NAMES[ACTUATOR_TO_JOINT[index]], name)
        finally:
            env.close()

    @requires_mujoco
    def test_rendered_frame_has_the_declared_size(self):
        env = make_humanoid_env(render_mode="rgb_array")
        try:
            env.reset(seed=0)
            frame = np.asarray(env.render())
            self.assertEqual(frame.shape, (FRAME_HEIGHT, FRAME_WIDTH, 3))
        finally:
            env.close()

    @requires_mujoco
    def test_reward_is_the_sum_of_exactly_four_parts(self):
        env = make_humanoid_env(render_mode=None)
        try:
            env.reset(seed=3)
            for _ in range(20):
                action = env.action_space.sample()
                _obs, reward, terminated, _tr, info = env.step(action)
                parts = (info["reward_survive"] + info["reward_forward"]
                         + info["reward_ctrl"] + info["reward_contact"])
                self.assertAlmostEqual(float(reward), float(parts), places=9)
                # Der Ueberlebensbonus ist der dominante Anteil - aber nur,
                # solange die Figur gesund ist. Im Schritt, der die Episode
                # beendet, faellt er auf null.
                self.assertAlmostEqual(float(info["reward_survive"]),
                                       0.0 if terminated else HEALTHY_REWARD)
                self.assertLessEqual(info["reward_ctrl"], 0.0)
                self.assertLessEqual(info["reward_contact"], 0.0)
                self.assertAlmostEqual(
                    float(info["reward_forward"]),
                    FORWARD_REWARD_WEIGHT * float(info["x_velocity"]), places=9)
                self.assertAlmostEqual(-float(info["reward_ctrl"]), control_cost(action),
                                       places=6)
                if terminated:
                    break
        finally:
            env.close()

    def test_control_and_contact_cost_weights(self):
        self.assertAlmostEqual(CTRL_COST_WEIGHT, 0.1)
        self.assertAlmostEqual(HEALTHY_REWARD, 5.0)
        self.assertAlmostEqual(FORWARD_REWARD_WEIGHT, 1.25)
        # Voller Ausschlag aller 17 Motoren kostet 0,1 * 17 * 0,4^2 = 0,272.
        self.assertAlmostEqual(control_cost(np.full(ACTION_DIM, ACTION_LIMIT)), 0.272)
        self.assertAlmostEqual(control_cost(np.zeros(ACTION_DIM)), 0.0)
        # Auch ausserhalb des Wertebereichs wird geclippt gerechnet.
        self.assertAlmostEqual(control_cost(np.full(ACTION_DIM, 5.0)), 0.272)
        # Kontaktkosten sind nach oben gedeckelt.
        self.assertAlmostEqual(contact_cost(np.zeros(78)), 0.0)
        self.assertAlmostEqual(contact_cost(np.full(78, 1e9)), CONTACT_COST_MAX)

    @requires_mujoco
    def test_the_environment_terminates_on_height_alone(self):
        """Der wichtigste Test des Projekts: An ihm haengt die Metrikwahl.

        Anders als HalfCheetah besitzt Humanoid einen terminalen Zustand.
        Ausgeloest wird er ALLEIN durch die Rumpfhoehe; eine Winkelbedingung
        gibt es nicht.
        """
        env = make_humanoid_env(render_mode=None)
        try:
            self.assertEqual(env.unwrapped._healthy_z_range, HEALTHY_Z_RANGE)
            self.assertTrue(env.unwrapped._terminate_when_unhealthy)
            observation, _ = env.reset(seed=0)
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                observation, _r, terminated, truncated, _info = env.step(
                    env.action_space.sample())
                steps += 1
            self.assertTrue(terminated, "Eine Zufallspolicy muss stuerzen")
            self.assertLess(steps, MAX_EPISODE_STEPS)
            low, high = HEALTHY_Z_RANGE
            self.assertFalse(low <= float(observation[OBS_HEIGHT]) <= high)
        finally:
            env.close()

    @requires_mujoco
    def test_no_terminal_bonus_or_penalty(self):
        """Ein Sturz kostet nur die Rewards der ausbleibenden Schritte."""
        env = make_humanoid_env(render_mode=None)
        try:
            env.reset(seed=7)
            last = None
            terminated = truncated = False
            while not (terminated or truncated):
                action = env.action_space.sample()
                _obs, reward, terminated, truncated, info = env.step(action)
                last = (float(reward), dict(info))
            reward, info = last
            parts = (info["reward_survive"] + info["reward_forward"]
                     + info["reward_ctrl"] + info["reward_contact"])
            # Kein zusaetzlicher Term im Schritt des Sturzes: Der Reward ist
            # exakt die Summe derselben vier Anteile wie sonst auch. Der
            # Ueberlebensbonus faellt weg, mehr passiert nicht.
            self.assertAlmostEqual(reward, float(parts), places=9)
            self.assertAlmostEqual(float(info["reward_survive"]), 0.0)
            self.assertNotIn("terminal_reward", info)
            self.assertNotIn("reward_terminal", info)
        finally:
            env.close()

    @requires_mujoco
    def test_position_is_reported_in_info_and_not_in_the_observation(self):
        env = make_humanoid_env(render_mode=None)
        try:
            observation, _ = env.reset(seed=5)
            info: dict = {}
            for _ in range(10):
                observation, _r, terminated, truncated, info = env.step(
                    np.full(ACTION_DIM, 0.3, dtype=np.float32))
                if terminated or truncated:
                    break
            for key in ("x_position", "y_position", "x_velocity", "y_velocity",
                        "distance_from_origin"):
                self.assertIn(key, info)
            self.assertEqual(len(observation), OBSERVATION_DIM)
            self.assertNotIn(float(info["x_position"]), [float(v) for v in observation])
        finally:
            env.close()

    @requires_mujoco
    def test_velocities_are_not_clipped(self):
        """Bei Walker2d sind sie auf +-10 geclippt - hier ausdruecklich nicht."""
        env = make_humanoid_env(render_mode=None)
        try:
            self.assertIsNone(VELOCITY_CLIP)
            extremes = []
            for seed in range(6):
                observation, _ = env.reset(seed=seed)
                terminated = truncated = False
                while not (terminated or truncated):
                    observation, _r, terminated, truncated, _i = env.step(
                        env.action_space.sample())
                    extremes.append(float(np.max(np.abs(observation[OBS_QVEL]))))
            # Walker2d wuerde hier exakt bei 10 abschneiden; hier nicht.
            self.assertGreater(max(extremes), 10.0)
        finally:
            env.close()


class OutcomeTests(unittest.TestCase):
    def test_survived_needs_truncation_without_termination(self):
        survived, fell, reached = episode_outcome(5200.0, terminated=False, truncated=True)
        self.assertTrue(survived)
        self.assertFalse(fell)
        self.assertTrue(reached)

    def test_a_fall_is_a_termination(self):
        survived, fell, reached = episode_outcome(120.0, terminated=True, truncated=False)
        self.assertFalse(survived)
        self.assertTrue(fell)
        self.assertFalse(reached)

    def test_target_mark_is_project_internal_and_not_solved(self):
        self.assertFalse(episode_outcome(TARGET_RETURN - 0.1, False, True)[2])
        self.assertTrue(episode_outcome(TARGET_RETURN, False, True)[2])
        self.assertEqual(TARGET_RETURN, 5000.0)
        spec = gymnasium.envs.registration.registry[ENV_ID]
        self.assertIsNone(spec.reward_threshold)

    def test_survival_and_fall_are_complementary(self):
        for terminated, truncated in ((True, False), (False, True)):
            survived, fell, _ = episode_outcome(1000.0, terminated, truncated)
            self.assertNotEqual(survived, fell)


class ReadoutTests(unittest.TestCase):
    def test_action_readout_uses_the_gear_of_each_joint(self):
        readout = action_readout(np.full(ACTION_DIM, ACTION_LIMIT))
        self.assertEqual([joint["actuator"] for joint in readout["joints"]],
                         list(ACTUATOR_NAMES))
        self.assertEqual([joint["joint"] for joint in readout["joints"]],
                         list(ACTUATOR_LABELS))
        for joint, gear in zip(readout["joints"], GEARS):
            self.assertAlmostEqual(joint["gear"], gear)
            self.assertAlmostEqual(joint["torque"], ACTION_LIMIT * gear)
            self.assertEqual(joint["direction"], "+")
        # Ein gemeinsamer Faktor ergaebe ueberall dasselbe Moment.
        self.assertGreater(len({joint["torque"] for joint in readout["joints"]}), 1)
        # Maximales Moment: Huefte vor/zurueck mit gear 300.
        self.assertAlmostEqual(max(joint["torque"] for joint in readout["joints"]), 120.0)

    def test_action_readout_groups_all_seventeen_joints(self):
        readout = action_readout(np.zeros(ACTION_DIM))
        self.assertEqual([group["name"] for group in readout["groups"]],
                         [name for name, _ in ACTUATOR_GROUPS])
        indices = [joint["index"] for group in readout["groups"]
                   for joint in group["joints"]]
        self.assertEqual(sorted(indices), list(range(ACTION_DIM)))
        self.assertAlmostEqual(readout["limit"], ACTION_LIMIT)

    def test_action_readout_translates_sign_and_zero(self):
        values = np.zeros(ACTION_DIM)
        values[0], values[2] = -ACTION_LIMIT, 0.2
        readout = action_readout(values)
        self.assertEqual(readout["joints"][0]["direction"], "−")
        self.assertAlmostEqual(readout["joints"][0]["torque"], ACTION_LIMIT * GEARS[0])
        self.assertEqual(readout["joints"][1]["torque"], 0.0)
        self.assertIn("kein Moment", readout["joints"][1]["text"])
        self.assertAlmostEqual(readout["joints"][2]["torque"], 0.2 * GEARS[2])

    def test_action_readout_clips_at_the_environment_limit_not_at_one(self):
        readout = action_readout(np.full(ACTION_DIM, 2.5))
        # Geclippt wird bei 0,4 - nicht bei 1,0.
        self.assertAlmostEqual(readout["raw"][0], ACTION_LIMIT)
        self.assertAlmostEqual(readout["joints"][0]["torque"], ACTION_LIMIT * GEARS[0])
        readout = action_readout(np.full(ACTION_DIM, -7.0))
        self.assertAlmostEqual(readout["raw"][0], -ACTION_LIMIT)

    def test_observation_readout_condenses_the_three_large_blocks(self):
        observation = np.arange(OBSERVATION_DIM, dtype=float) / 100.0
        values = observation_readout(observation)
        self.assertAlmostEqual(values["height"], 0.0)
        self.assertEqual(len(values["quaternion"]), 4)
        self.assertEqual(len(values["joint_angles"]), 17)
        self.assertEqual(len(values["joint_velocities"]), 17)
        self.assertEqual(len(values["angular_velocity"]), 3)
        self.assertAlmostEqual(values["vx"], 0.22)
        self.assertAlmostEqual(values["vy"], 0.23)
        self.assertAlmostEqual(values["vz"], 0.24)
        # cinert, cvel und cfrc_ext erscheinen NICHT einzeln.
        for absent in ("cinert", "cvel", "cfrc_ext"):
            self.assertNotIn(absent, values)
        expected = float(np.sum(np.square(observation[OBS_CFRC_EXT])))
        self.assertAlmostEqual(values["contact_force_squared"], expected)
        self.assertLessEqual(values["contact_cost"], CONTACT_COST_MAX)

    def test_observation_readout_reports_the_healthy_range(self):
        observation = np.zeros(OBSERVATION_DIM)
        observation[OBS_HEIGHT] = 1.4
        self.assertTrue(observation_readout(observation)["healthy"])
        observation[OBS_HEIGHT] = 0.7
        self.assertFalse(observation_readout(observation)["healthy"])
        observation[OBS_HEIGHT] = 2.5
        values = observation_readout(observation)
        self.assertFalse(values["healthy"])
        self.assertEqual(values["healthy_z_range"], HEALTHY_Z_RANGE)

    def test_tilt_is_zero_for_an_upright_torso(self):
        self.assertAlmostEqual(torso_tilt_degrees([1.0, 0.0, 0.0, 0.0]), 0.0)
        # 180 Grad um die x-Achse: der Rumpf steht auf dem Kopf.
        self.assertAlmostEqual(torso_tilt_degrees([0.0, 1.0, 0.0, 0.0]), 180.0)
        self.assertAlmostEqual(torso_tilt_degrees([0.0, 0.0, 0.0, 0.0]), 0.0)

    def test_observation_readout_has_no_clipping_flag(self):
        """Geschwindigkeiten sind hier nicht geclippt."""
        values = observation_readout(np.zeros(OBSERVATION_DIM))
        self.assertNotIn("velocity_clipped", values)

    def test_reward_readout_takes_all_four_parts_from_info(self):
        info = {"reward_survive": 5.0, "reward_forward": 2.5, "reward_ctrl": -0.42,
                "reward_contact": -0.01, "x_position": 12.5, "y_position": -1.5,
                "x_velocity": 2.0, "y_velocity": -0.3, "distance_from_origin": 12.6}
        values = reward_readout(info)
        self.assertAlmostEqual(values["survive"], 5.0)
        self.assertAlmostEqual(values["forward"], 2.5)
        self.assertAlmostEqual(values["ctrl"], -0.42)
        self.assertAlmostEqual(values["contact"], -0.01)
        self.assertAlmostEqual(values["y_position"], -1.5)
        self.assertAlmostEqual(values["distance_from_origin"], 12.6)


class ConfigurationTests(unittest.TestCase):
    def test_every_algorithm_starts_with_the_same_step_budget(self):
        budgets = {default_config(name).total_timesteps for name in ALGORITHMS}
        self.assertEqual(budgets, {DEFAULT_TOTAL_TIMESTEPS})
        self.assertEqual(DEFAULT_TOTAL_TIMESTEPS, 100_000)
        # Die Voreinstellung passt zur Episodenvorgabe: 1000 Episoden sind
        # gemessen rund 80.000 Schritte, beide Grenzen also gleicher
        # Groessenordnung. Stuende hier das Budget der Messlaeufe, griffe immer
        # die Episodengrenze und die Schrittzahl waere Dekoration.
        self.assertLess(DEFAULT_TOTAL_TIMESTEPS, 2 * STEPS_PER_THOUSAND_EPISODES)
        self.assertGreater(DEFAULT_TOTAL_TIMESTEPS, STEPS_PER_THOUSAND_EPISODES // 2)
        # Das Budget der Messlaeufe liegt bewusst UNTER allen drei Profilwerten.
        self.assertEqual(REPORT_TIMESTEPS, 500_000)
        for name in ALGORITHMS:
            self.assertLess(REPORT_TIMESTEPS, PROFILE_BUDGET[name])
        # PPO steht dabei relativ am weitesten von seinem Arbeitspunkt weg.
        self.assertAlmostEqual(REPORT_TIMESTEPS / PROFILE_BUDGET["PPO"], 0.05)
        self.assertAlmostEqual(REPORT_TIMESTEPS / PROFILE_BUDGET["SAC"], 0.25)

    def test_episode_budget_defaults_to_one_thousand(self):
        """Von der Aufgabenstellung vorgegeben."""
        self.assertEqual(DEFAULT_EPISODES, 1000)
        for name in ALGORITHMS:
            self.assertEqual(default_config(name).episodes, DEFAULT_EPISODES)

    def test_replay_buffer_covers_the_longest_intended_run(self):
        """Verhaltensneutral, spart aber 3,9 GB auf einer 8-GB-Maschine.

        Der Buffer folgt dem Budget der **Messlaeufe**, nicht der
        Voreinstellung: Wer das Budget auf den Berichtswert hochsetzt, soll
        nicht unbemerkt in eine Verdraengung laufen.
        """
        self.assertEqual(DEFAULT_BUFFER_SIZE, REPORT_TIMESTEPS)
        for name in OFF_POLICY_ALGORITHMS:
            config = default_config(name)
            self.assertEqual(config.buffer_size, REPORT_TIMESTEPS)
            self.assertGreaterEqual(config.buffer_size, config.total_timesteps)

    def test_evaluation_stays_available_but_is_no_longer_periodic(self):
        """`evaluate()` bleibt für eine explorationsfreie Messung von Hand.
        Eine automatische Zwischenevaluation gibt es nicht mehr – die Summary
        mittelt stattdessen über die letzten Episoden des Trainings."""
        self.assertGreaterEqual(DEFAULT_EVALUATION_EPISODES, 1)
        self.assertTrue(hasattr(HumanoidWorkbench, "evaluate"))
        import inspect
        parameters = inspect.signature(HumanoidWorkbench.train).parameters
        for absent in ("evaluation_interval", "evaluation_episodes", "best_callback"):
            self.assertNotIn(absent, parameters)

    def test_ppo_profile_matches_the_zoo_values(self):
        config = default_config("PPO")
        self.assertAlmostEqual(config.learning_rate, 3.56987e-05)
        # Auffaellig niedrig, stammt so aus dem Profil und wird nicht
        # "korrigiert".
        self.assertAlmostEqual(config.gamma, 0.95)
        self.assertEqual(config.n_steps, 512)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.n_epochs, 5)
        self.assertAlmostEqual(config.gae_lambda, 0.9)
        self.assertAlmostEqual(config.clip_range, 0.3)
        self.assertAlmostEqual(config.ent_coef, 0.00238306)
        self.assertAlmostEqual(config.vf_coef, 0.431892)
        self.assertAlmostEqual(config.max_grad_norm, 2.0)
        self.assertAlmostEqual(config.log_std_init, -2.0)
        self.assertFalse(config.ortho_init)
        self.assertEqual(config.activation, "ReLU")
        # Das Zoo-Profil verlangt `normalize: true`.
        self.assertTrue(config.normalize_obs)
        self.assertTrue(config.normalize_reward)

    def test_off_policy_buffers_hold_the_whole_report_budget(self):
        for name in OFF_POLICY_ALGORITHMS:
            config = default_config(name)
            self.assertEqual(config.buffer_size, REPORT_TIMESTEPS)
            self.assertEqual(config.buffer_size, HumanoidConfig().buffer_size)

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
        self.assertEqual(-ACTION_DIM, -17)
        numeric = dataclasses.replace(config, target_entropy="-17")
        self.assertEqual(numeric.sac_target_entropy(), -17.0)

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
        config = dataclasses.replace(default_config("SAC"), learning_rate_schedule="linear")
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
                workbench = HumanoidWorkbench(tiny_config(name))
                try:
                    metrics = workbench.train()
                    self.assertTrue(metrics)
                    for metric in metrics:
                        self.assertIsInstance(metric, EpisodeMetric)
                        # Humanoid terminiert beim Sturz: Die Laenge schwankt
                        # und ist damit selbst eine Kennzahl.
                        self.assertGreater(metric.length, 0)
                        self.assertLessEqual(metric.length, MAX_EPISODE_STEPS)
                        # Durchhalten und Sturz schliessen einander aus.
                        self.assertNotEqual(metric.survived, metric.fell)
                        self.assertEqual(metric.survived,
                                         metric.length == MAX_EPISODE_STEPS)
                        self.assertEqual(metric.reached_target,
                                         metric.reward >= TARGET_RETURN)
                        self.assertGreaterEqual(metric.lateral, 0.0)
                    result = workbench.evaluate(1, seed=3)
                    self.assertEqual(result.episodes, 1)
                    self.assertGreater(result.mean_length, 0)
                    self.assertLessEqual(result.mean_length, MAX_EPISODE_STEPS)
                    for rate in (result.survival_rate, result.fall_rate,
                                 result.target_rate):
                        self.assertGreaterEqual(rate, 0.0)
                        self.assertLessEqual(rate, 1.0)
                    self.assertAlmostEqual(result.survival_rate + result.fall_rate, 1.0)
                finally:
                    workbench.close()

    def test_evaluation_does_not_change_the_learning_state(self):
        workbench = HumanoidWorkbench(tiny_config("TD3"))
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
        workbench = HumanoidWorkbench(tiny_config("PPO"))
        try:
            workbench.train()
            first = workbench.model.num_timesteps
            workbench.train()
            self.assertGreater(workbench.model.num_timesteps, first)
            self.assertGreater(len(workbench.history), 0)
        finally:
            workbench.close()

    def test_ppo_has_no_replay_buffer(self):
        workbench = HumanoidWorkbench(tiny_config("PPO"))
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
        workbench = HumanoidWorkbench(config)
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
        workbench = HumanoidWorkbench(config)
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
        workbench = HumanoidWorkbench(tiny_config("SAC", learning_starts=0))
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
        workbench = HumanoidWorkbench(config)
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
            workbench = HumanoidWorkbench(tiny_config("TD3", seed=123))
            try:
                metrics = workbench.train()
                rewards.append([round(metric.reward, 6) for metric in metrics])
            finally:
                workbench.close()
        self.assertEqual(rewards[0], rewards[1])
        self.assertTrue(rewards[0])

    def test_callback_reports_episodes_and_progress_through_the_queue(self):
        output: queue.Queue = queue.Queue()
        workbench = HumanoidWorkbench(tiny_config("SAC"))
        try:
            workbench.train(threading.Event(), output, ("compare", 2))
        finally:
            workbench.close()
        kinds: dict[str, list] = {}
        while not output.empty():
            kind, payload = output.get()
            kinds.setdefault(kind, []).append(payload)
        self.assertIn("episode", kinds)
        self.assertIn("progress", kinds)
        # Es gibt keine Zwischenevaluation mehr.
        self.assertNotIn("evaluation", kinds)
        self.assertEqual(kinds["episode"][0][0], ("compare", 2))

    def test_the_best_episode_is_stored_with_its_policy(self):
        """Gespeichert wird genau ein zusätzlicher Lernstand je Slot."""
        workbench = HumanoidWorkbench(tiny_config("TD3"))
        try:
            self.assertIsNone(workbench.best_snapshot())
            metrics = workbench.train()
            snapshot = workbench.best_snapshot()
            self.assertIsNotNone(snapshot)
            best_metric, state, statistics = snapshot
            # TD3 normalisiert nicht: keine Statistiken zu sichern.
            self.assertIsNone(statistics)
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
        workbench = HumanoidWorkbench(tiny_config("SAC"))
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
        workbench = HumanoidWorkbench(tiny_config("SAC", total_timesteps=20_000))
        try:
            workbench.train(stop)
            self.assertLess(workbench.model.num_timesteps, 20_000)
        finally:
            workbench.close()


@requires_mujoco
class NormalizationTests(unittest.TestCase):
    def test_statistics_grow_while_training_and_stay_frozen_in_evaluation(self):
        workbench = HumanoidWorkbench(tiny_config("PPO"))
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
        workbench = HumanoidWorkbench(tiny_config("PPO"))
        try:
            metrics = workbench.train()
            self.assertTrue(workbench.config.normalize_reward)
            self.assertTrue(metrics)
            for metric in metrics:
                # Der Return setzt sich aus vier Anteilen zusammen; der
                # Ueberlebensbonus dominiert. Ein normalisierter Return laege
                # in der Groessenordnung 1, ein echter bei rund 5 je Schritt.
                self.assertGreater(metric.reward, 2.0 * metric.length)
                self.assertLess(metric.reward, 8.0 * metric.length)
        finally:
            workbench.close()

    def test_off_policy_methods_do_not_normalise_by_default(self):
        for name in OFF_POLICY_ALGORITHMS:
            workbench = HumanoidWorkbench(tiny_config(name))
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
                workbench = HumanoidWorkbench(tiny_config(name))
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
                        restored = HumanoidWorkbench.load(base, name)
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
        workbench = HumanoidWorkbench(tiny_config("PPO"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                mean = workbench.vec_normalize.obs_rms.mean.copy()
                workbench.save(base)
                restored = HumanoidWorkbench.load(base, "PPO")
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
        workbench = HumanoidWorkbench(tiny_config("SAC"))
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
                HumanoidWorkbench.load(base)
            self.assertIn("Walker2d-v5", str(error.exception))
            self.assertIn(ENV_ID, str(error.exception))

    def test_a_foreign_algorithm_is_rejected(self):
        workbench = HumanoidWorkbench(tiny_config("TD3"))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "best"
            try:
                workbench.train()
                workbench.save(base)
            finally:
                workbench.close()
            with self.assertRaises(ValueError) as error:
                HumanoidWorkbench.load(base, "PPO")
            self.assertIn("TD3", str(error.exception))

    def test_missing_metadata_is_reported_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError) as error:
                HumanoidWorkbench.load(Path(folder) / "nothing")
            self.assertIn("Metadatendatei", str(error.exception))


@requires_mujoco
class ComparisonTests(unittest.TestCase):
    def _run(self, configs: list[HumanoidConfig]) -> list[HumanoidWorkbench]:
        """Alle Slots parallel, so wie es die GUI tut."""
        benches = [HumanoidWorkbench(config) for config in configs]
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


class BudgetTests(unittest.TestCase):
    """Zwei Budgetgrenzen: Der Lauf endet an der zuerst erreichten."""

    def test_episode_limit_ends_the_run_and_is_reported(self):
        config = tiny_config("SAC")
        config.total_timesteps, config.episodes = 50_000, 3
        workbench = HumanoidWorkbench(config)
        try:
            metrics = workbench.train()
            self.assertEqual(len(metrics), 3)
            self.assertEqual(workbench.stop_reason, "episodes")
            self.assertIn(workbench.stop_reason, STOP_REASONS)
            # Das Schrittbudget wurde dabei bei Weitem nicht ausgeschöpft.
            self.assertLess(workbench.num_timesteps, config.total_timesteps)
        finally:
            workbench.close()

    def test_step_limit_ends_the_run_when_episodes_are_unlimited(self):
        config = tiny_config("SAC")
        config.total_timesteps, config.episodes = 300, 0
        workbench = HumanoidWorkbench(config)
        try:
            workbench.train()
            self.assertEqual(workbench.stop_reason, "steps")
            self.assertGreaterEqual(workbench.num_timesteps, config.total_timesteps)
        finally:
            workbench.close()

    def test_zero_episodes_means_unlimited(self):
        config = dataclasses.replace(default_config("SAC"), episodes=0)
        config.validate()
        self.assertEqual(config.episodes, 0)

    def test_negative_episode_budget_is_rejected(self):
        config = dataclasses.replace(default_config("SAC"), episodes=-1)
        with self.assertRaises(ValueError) as error:
            config.validate()
        self.assertIn("Episoden", str(error.exception))

    def test_changing_either_budget_keeps_the_learning_state(self):
        """Ein geändertes Budget baut das Modell nicht neu auf."""
        base = default_config("TD3")
        for changed in (dataclasses.replace(base, total_timesteps=123_456),
                        dataclasses.replace(base, episodes=17)):
            self.assertEqual(base.signature(), changed.signature())
        # Jede andere Änderung schon.
        self.assertNotEqual(base.signature(), dataclasses.replace(base, gamma=0.5).signature())


class MemoryTests(unittest.TestCase):
    """Der Replay Buffer ist auf einer 8-GB-Maschine der Engpass."""

    def test_observations_are_buffered_as_float32(self):
        config = tiny_config("SAC")
        workbench = HumanoidWorkbench(config)
        try:
            workbench.create_model()
            buffer = workbench.model.replay_buffer
            self.assertEqual(buffer.observations.dtype, np.float32)
            self.assertEqual(buffer.next_observations.dtype, np.float32)
        finally:
            workbench.close()

    def test_float32_wrapper_preserves_the_values(self):
        env = Float32Observation(make_humanoid_env(render_mode=None))
        try:
            self.assertEqual(env.observation_space.dtype, np.float32)
            self.assertEqual(env.observation_space.shape, (OBSERVATION_DIM,))
            observation, _ = env.reset(seed=11)
            self.assertEqual(observation.dtype, np.float32)
            raw = env.unwrapped._get_obs()
            np.testing.assert_allclose(observation, raw, rtol=1e-6, atol=1e-6)
        finally:
            env.close()

    def test_buffer_holds_the_longest_intended_run_so_nothing_is_evicted(self):
        for name in OFF_POLICY_ALGORITHMS:
            config = default_config(name)
            self.assertEqual(config.buffer_size, REPORT_TIMESTEPS)
            self.assertGreaterEqual(config.buffer_size, config.total_timesteps)

    def test_memory_optimisation_stays_off_because_humanoid_truncates(self):
        """`optimize_memory_usage=True` verträgt sich nicht mit
        `handle_timeout_termination` – und Humanoid trunkiert nach 1000
        Schritten. Ohne korrekte Timeout-Behandlung würde das Abschneiden als
        echtes Episodenende gelernt."""
        config = tiny_config("TD3")
        workbench = HumanoidWorkbench(config)
        try:
            workbench.create_model()
            buffer = workbench.model.replay_buffer
            self.assertFalse(buffer.optimize_memory_usage)
            self.assertTrue(buffer.handle_timeout_termination)
        finally:
            workbench.close()


class ThreadTests(unittest.TestCase):
    def test_set_torch_threads_applies_and_reports_the_value(self):
        before = torch.get_num_threads()
        try:
            self.assertEqual(set_torch_threads(2), 2)
            self.assertEqual(torch.get_num_threads(), 2)
            # Ungültige Werte fallen auf mindestens einen Thread zurück.
            self.assertEqual(set_torch_threads(0), 1)
            self.assertEqual(set_torch_threads(DEFAULT_TORCH_THREADS),
                             DEFAULT_TORCH_THREADS)
        finally:
            torch.set_num_threads(before)

    def test_default_thread_count_is_the_measured_optimum(self):
        self.assertEqual(DEFAULT_TORCH_THREADS, 4)


class BestEpisodeSnapshotTests(unittest.TestCase):
    """Der gesicherte Lernstand muss der sein, der die Episode erzeugt hat.

    Off-Policy-Verfahren aktualisieren die Policy bei `train_freq = 1` nach
    JEDEM Schritt. Wird am Episodenende gesichert, gehört der Stand zu einer
    Policy, die es zu Beginn der Episode noch gar nicht gab – die Wiederholung
    zeigt dann etwas anderes als die Episode, deren Nummer sie trägt.
    """

    def test_the_recorded_episode_replays_exactly(self):
        """Die entscheidende Zusicherung: Die beste Episode wird aufgezeichnet
        und laesst sich Schritt fuer Schritt identisch nachspielen.

        Die Policy allein genuegt dafuer nicht. Die beste Episode ist das
        Maximum ueber Hunderte Episoden und verdankt ihren Wert zum Teil
        gluecklichen Explorationszuegen und ihrem Startzustand; eine
        deterministische Wiederholung erreicht gemessen nur 80 bis 90 %.
        """
        workbench = HumanoidWorkbench(tiny_config("SAC"))
        try:
            workbench.train()
            recording = workbench.best_recording()
            self.assertIsNotNone(recording)
            metric, (qpos, qvel), actions = recording
            self.assertEqual(len(actions), metric.length)
            env = make_humanoid_env(render_mode=None)
            try:
                env.reset(seed=0)
                env.unwrapped.set_state(qpos, qvel)
                total, steps = 0.0, 0
                for action in actions:
                    _obs, reward, terminated, truncated, _info = env.step(action)
                    total += float(reward)
                    steps += 1
                    if terminated or truncated:
                        break
            finally:
                env.close()
            self.assertEqual(steps, metric.length)
            self.assertAlmostEqual(total, metric.reward, places=6)
        finally:
            workbench.close()

    def test_the_recording_belongs_to_the_best_episode_not_the_last(self):
        workbench = HumanoidWorkbench(tiny_config("TD3"))
        try:
            metrics = workbench.train()
            metric, _state, _actions = workbench.best_recording()
            self.assertEqual(metric, max(metrics, key=lambda item: item.reward))
        finally:
            workbench.close()

    def test_the_stored_policy_is_detached_from_further_training(self):
        workbench = HumanoidWorkbench(tiny_config("SAC"))
        try:
            workbench.train()
            best, state, _ = workbench.best_snapshot()
            frozen = {name: tensor.clone() for name, tensor in state.items()}
            workbench.train()
            if workbench.best_snapshot()[0] == best:
                for name, tensor in workbench.best_snapshot()[1].items():
                    self.assertTrue(torch.equal(frozen[name], tensor))
        finally:
            workbench.close()

    def test_normalisation_statistics_travel_with_the_snapshot(self):
        """Eine alte Policy mit heutigen Statistiken saehe die Beobachtungen
        anders als im Training. PPO normalisiert laut Zoo-Profil."""
        workbench = HumanoidWorkbench(tiny_config("PPO"))
        try:
            workbench.train()
            _, _, statistics = workbench.best_snapshot()
            self.assertIsNotNone(statistics)
            mean, var = statistics
            self.assertEqual(mean.shape, (OBSERVATION_DIM,))
            self.assertEqual(var.shape, (OBSERVATION_DIM,))
            # Eingefrorene Statistiken werden vom Weitertrainieren nicht beruehrt.
            frozen = (mean.copy(), var.copy())
            workbench.train()
            best = workbench.best_snapshot()
            if best[0].episode <= len(workbench.history):
                np.testing.assert_array_equal(statistics[0], frozen[0])
        finally:
            workbench.close()

    def test_off_policy_snapshot_carries_no_statistics(self):
        workbench = HumanoidWorkbench(tiny_config("TD3"))
        try:
            workbench.train()
            self.assertIsNone(workbench.best_snapshot()[2])
        finally:
            workbench.close()

    def test_frozen_statistics_change_what_the_policy_sees(self):
        """Belegt, dass der Parameter ueberhaupt wirkt."""
        workbench = HumanoidWorkbench(tiny_config("PPO"))
        try:
            workbench.train()
            observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
            aktuell = workbench.policy_observation(observation)
            verschoben = workbench.policy_observation(
                observation, (np.full(OBSERVATION_DIM, 5.0), np.ones(OBSERVATION_DIM)))
            self.assertFalse(np.allclose(aktuell, verschoben))
        finally:
            workbench.close()
