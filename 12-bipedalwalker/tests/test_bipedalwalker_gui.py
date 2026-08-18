import tempfile
import time
import tkinter as tk
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import math

import numpy as np

from bipedalwalker_gui import (
    EXPORT_DIR,
    MAX_ANIMATION_FPS,
    MIN_ANIMATION_FPS,
    PARAMETER_GROUPS,
    RENDER_FPS,
    SLOT_LABELS,
    BipedalWalkerGUI,
    downsample_minmax,
    rolling_average,
    slugify,
)
from bipedalwalker_logic import (
    ALGORITHMS,
    JOINT_NAMES,
    LIDAR_COUNT,
    OBSERVATION_DIM,
    OFF_POLICY_ALGORITHMS,
    default_config,
)


class PlotHelperTests(unittest.TestCase):
    def test_rolling_average_uses_up_to_twenty_values(self):
        values = list(range(1, 22))
        result = rolling_average(values)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[-1], np.mean(values[-20:]))

    def test_downsampling_keeps_extrema_and_limit(self):
        x = list(range(10_000))
        y = np.sin(np.linspace(0, 100, 10_000)).tolist()
        sampled_x, sampled_y = downsample_minmax(x, y, limit=2_000)
        self.assertLessEqual(len(sampled_x), 2_000)
        self.assertEqual(len(sampled_x), len(sampled_y))
        self.assertEqual(sampled_x[0], 0)
        self.assertEqual(sampled_x[-1], 9_999)

    def test_default_frame_rate_is_the_environment_frame_rate(self):
        self.assertEqual(RENDER_FPS, 50)
        self.assertLessEqual(MIN_ANIMATION_FPS, RENDER_FPS)
        self.assertGreaterEqual(MAX_ANIMATION_FPS, RENDER_FPS)

    def test_entry_point_exposes_main_without_starting_a_window(self):
        import bipedalwalker_app

        self.assertTrue(callable(bipedalwalker_app.main))
        self.assertIs(bipedalwalker_app.BipedalWalkerGUI, BipedalWalkerGUI)


class GUITestCase(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Kein grafisches Display verfügbar: {error}")
        self.app = BipedalWalkerGUI(self.root)
        self.root.update()

    def tearDown(self):
        self.app.close()

    def _select(self, slot: int) -> None:
        self.app.parameter_tabs.select(slot)
        self.root.update()

    def _set_algorithm(self, slot: int, algorithm: str) -> None:
        self.app.algorithm_vars[slot].set(algorithm)
        self.app._algorithm_changed(slot)
        self.root.update()


class LayoutSmokeTest(GUITestCase):
    def test_start_layout_has_no_clipped_controls(self):
        self.app._initialize_layout()
        self.root.update()
        self.assertEqual(self.app.layout_visibility_issues(), [])

    def test_layout_stays_clean_for_every_algorithm_in_both_slots(self):
        self.app._initialize_layout()
        for algorithm in ALGORITHMS:
            for slot in range(len(SLOT_LABELS)):
                with self.subTest(algorithm=algorithm, slot=slot):
                    self._set_algorithm(slot, algorithm)
                    self._select(slot)
                    self.assertEqual(self.app.layout_visibility_issues(), [])

    def test_layout_stays_clean_with_both_animation_panels(self):
        self.app._initialize_layout()
        self.app.animation_live = [True, True]
        self.app._sync_panels()
        self.root.update()
        self.assertEqual(self.app.layout_visibility_issues(), [])

    def test_layout_stays_clean_on_every_chart_tab(self):
        self.app._initialize_layout()
        for index in range(len(self.app.charts.tabs())):
            with self.subTest(tab=index):
                self.app.charts.select(index)
                self.root.update()
                self.assertEqual(self.app.layout_visibility_issues(), [])

    def test_both_chart_tabs_exist(self):
        self.assertEqual([self.app.charts.tab(index, "text") for index in self.app.charts.tabs()],
                         ["Training", "Vergleich"])

    def test_summary_is_scrollable_and_read_only(self):
        self.assertEqual(str(self.app.summary_text.cget("state")), "disabled")
        self.assertTrue(self.app.summary_text.cget("yscrollcommand"))
        self.assertTrue(self.app.summary_text.cget("xscrollcommand"))


class SlotTests(GUITestCase):
    def test_two_dropdowns_offer_all_three_algorithms(self):
        self.assertEqual(len(self.app.algorithm_combos), 2)
        for combo in self.app.algorithm_combos:
            self.assertEqual(tuple(combo.cget("values")), ALGORITHMS)

    def test_two_parameter_tabs_are_named_after_the_slots(self):
        names = [self.app.parameter_tabs.tab(index, "text") for index in self.app.parameter_tabs.tabs()]
        self.assertEqual(names, list(SLOT_LABELS))

    def test_active_slot_follows_the_selected_tab(self):
        for slot in range(len(SLOT_LABELS)):
            self._select(slot)
            self.assertEqual(self.app.active_slot, slot)
            self.assertIn(SLOT_LABELS[slot], self.app.active_label.get())
            self.assertIn(self.app.algorithm_vars[slot].get(), self.app.active_label.get())

    def test_running_slot_stays_pinned_while_busy(self):
        self._select(0)
        self.app._set_busy(True, "Läuft – Test")
        self.app.running_slot = 0
        self._select(1)
        self.assertEqual(self.app.active_slot, 0, "Ein Tabwechsel ändert den laufenden Lauf nicht.")
        self.app._set_busy(False, "Bereit")
        self.assertEqual(self.app.active_slot, 1)

    def test_both_slots_may_use_the_same_algorithm(self):
        for slot in range(len(SLOT_LABELS)):
            self._set_algorithm(slot, "TD3")
        self.assertEqual([runtime.algorithm for runtime in self.app.slots], ["TD3", "TD3"])
        self.assertEqual(self.app._config(0).algorithm, self.app._config(1).algorithm)

    def test_switching_the_algorithm_loads_that_zoo_profile(self):
        self._set_algorithm(0, "TD3")
        config = self.app._config(0)
        self.assertEqual(config.algorithm, "TD3")
        self.assertEqual(config.total_timesteps, default_config("TD3").total_timesteps)
        self.assertEqual(config.action_noise_sigma, default_config("TD3").action_noise_sigma)

    def test_switching_the_algorithm_resets_only_that_slot(self):
        self.app.slots[1].history.append("marker")
        self._set_algorithm(0, "TD3")
        self.assertEqual(self.app.slots[0].history, [])
        self.assertEqual(self.app.slots[1].history, ["marker"])

    def test_a_tab_shows_only_the_parameters_of_its_algorithm(self):
        for slot in range(len(SLOT_LABELS)):
            for algorithm in ALGORITHMS:
                with self.subTest(slot=slot, algorithm=algorithm):
                    self._set_algorithm(slot, algorithm)
                    config = default_config(algorithm)
                    expected = {name for _column, _title, names in PARAMETER_GROUPS[algorithm]
                                for name in names}
                    self.assertEqual(set(self.app.entries[slot]), expected)
                    for name in expected:
                        self.assertTrue(config.uses(name), f"{algorithm} kennt '{name}' nicht")

    def test_ppo_tab_has_no_replay_fields_and_off_policy_tabs_no_rollout_fields(self):
        self._set_algorithm(0, "PPO")
        for name in ("buffer_size", "learning_starts", "tau", "action_noise"):
            self.assertNotIn(name, self.app.entries[0])
        for algorithm in OFF_POLICY_ALGORITHMS:
            self._set_algorithm(1, algorithm)
            for name in ("n_steps", "n_epochs", "clip_range", "max_grad_norm"):
                self.assertNotIn(name, self.app.entries[1])

    def test_no_epsilon_greedy_parameters_exist_anywhere(self):
        """Keines der drei Verfahren exploriert ε-greedy."""
        for slot in range(len(SLOT_LABELS)):
            for algorithm in ALGORITHMS:
                self._set_algorithm(slot, algorithm)
                for name in self.app.entries[slot]:
                    self.assertNotIn("exploration", name)
                    self.assertNotIn("eps", name.split("_")[0])

    def test_parameters_are_independent_between_the_slots(self):
        self.app.values[0]["batch_size"].set("32")
        self.assertNotEqual(self.app.values[1]["batch_size"].get(), "32")
        self.assertEqual(self.app._config(0).batch_size, 32)

    def test_invalid_input_names_slot_field_value_and_range(self):
        self.app.values[1]["gamma"].set("2.5")
        with self.assertRaises(ValueError) as context:
            self.app._config(1)
        message = str(context.exception)
        self.assertIn(SLOT_LABELS[1], message)
        self.assertIn("2.5", message)
        self.assertIn("Gültig", message)


class BipedalWalkerViewTests(GUITestCase):
    def test_every_tab_offers_the_normalisation_group(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                self._set_algorithm(0, algorithm)
                for name in ("normalize_obs", "normalize_reward", "clip_obs", "clip_reward"):
                    self.assertIn(name, self.app.entries[0])

    def test_ppo_normalises_by_default_and_off_policy_does_not(self):
        self._set_algorithm(0, "PPO")
        self.assertTrue(self.app._config(0).normalizes)
        for algorithm in OFF_POLICY_ALGORITHMS:
            self._set_algorithm(1, algorithm)
            self.assertFalse(self.app._config(1).normalizes, algorithm)

    def test_readout_shows_all_torques_lidar_values_and_both_legs(self):
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        observation[8] = 1.0
        observation[-LIDAR_COUNT:] = np.linspace(0.1, 1.0, LIDAR_COUNT)
        self.app._show_readout(0, observation, np.array([0.5, -1.0, 0.0, 0.25]))
        text = self.app.observations[0].get()
        self.assertIn("a0=+0.50", text)
        self.assertIn("a3=+0.25", text)
        for joint in JOINT_NAMES:
            self.assertIn(joint, text)
        self.assertIn("Bein 1", text); self.assertIn("Bein 2", text)
        lidar_line = [line for line in text.splitlines() if line.startswith("Lidar")][0]
        self.assertEqual(len(lidar_line.split()) - 1, LIDAR_COUNT,
                         "Alle zehn Lidar-Werte werden angezeigt.")

    def test_readout_marks_normalised_values_and_reports_the_hull_angle_in_degrees(self):
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        observation[0] = math.pi / 4
        self.app._show_readout(0, observation)
        text = self.app.observations[0].get()
        self.assertIn("+45.0°", text)
        self.assertIn("normierte Werte", text)

    def test_episode_outcome_text_names_the_three_endings(self):
        self.assertIn("Zeitlimit", self.app._episode_outcome_text(0.2, True))
        self.assertIn("gestürzt", self.app._episode_outcome_text(-100.0, False))
        self.assertIn("Strecke", self.app._episode_outcome_text(0.4, False))


class ControlTests(GUITestCase):
    EXPECTED = (
        "Training starten / fortsetzen", "Stoppen", "Deterministisch evaluieren",
        "Sichtbare Episode abspielen", "Vergleich starten / fortsetzen",
        "Bestes Modell wiederherstellen", "Neues Modell",
    )

    def test_control_buttons_match_the_workbench_specification(self):
        self.assertEqual(tuple(button.cget("text") for button in self.app.buttons), self.EXPECTED)

    def test_button_labels_promise_no_replay_buffer(self):
        """PPO besitzt keinen Replay Buffer; kein Label darf einen versprechen."""
        for button in self.app.buttons:
            self.assertNotIn("replay", str(button.cget("text")).lower())

    def test_control_buttons_reflect_busy_state(self):
        self.app._set_busy(True, "Läuft – Test")
        self.assertEqual(str(self.app.stop_button.cget("state")), "normal")
        self.assertTrue(all(str(button.cget("state")) == "disabled"
                            for button in self.app.buttons if button is not self.app.stop_button))
        self.assertTrue(all(str(combo.cget("state")) == "disabled"
                            for combo in self.app.algorithm_combos))
        self.app._set_busy(False, "Bereit")
        self.assertEqual(str(self.app.stop_button.cget("state")), "disabled")
        self.assertTrue(all(str(combo.cget("state")) == "readonly"
                            for combo in self.app.algorithm_combos))

    def test_restore_best_stays_disabled_without_a_checkpoint(self):
        self.assertEqual(str(self.app.best_button.cget("state")), "disabled")

    def test_animation_controls_are_global_and_complete(self):
        """Ein- und Ausschalten, Zuschalten während Läufen, Bildrate."""
        labels = []
        stack = list(self.root.winfo_children())
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            try:
                text = str(widget.cget("text"))
            except tk.TclError:
                continue
            if text:
                labels.append(text)
        self.assertIn("Animation zeigen", labels)
        self.assertIn("Bildrate (FPS)", labels)
        self.assertEqual(sum(1 for text in labels if "Animation" in text), 1,
                         "Genau ein Schalter steuert die Animation.")
        for variables in self.app.values:
            self.assertNotIn("fps", variables, "Die Bildrate gehört zu keinem Verfahrensslot.")

    def test_animation_can_be_switched_off(self):
        self.assertTrue(self.app.animation_enabled.get())
        self.app.animation_enabled.set(False)
        self.assertFalse(self.app.animation_enabled.get())

    def test_frame_rate_is_configurable_and_validated(self):
        self.assertEqual(self.app.fps.get(), str(RENDER_FPS))
        self.assertEqual(self.app._animation_settings(), RENDER_FPS)
        self.assertEqual(self.app.animation_interval, 20)
        self.app.fps.set("25")
        self.assertEqual(self.app._animation_settings(), 25)
        self.assertEqual(self.app.animation_interval, 40)
        for value in ("0", str(MAX_ANIMATION_FPS + 1), "schnell"):
            self.app.fps.set(value)
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as context:
                    self.app._animation_settings()
                message = str(context.exception)
                self.assertIn("Bildrate (FPS)", message)
                self.assertIn("Gültig", message)

    def test_invalid_frame_rate_blocks_a_run_with_a_clear_message(self):
        self.app.fps.set("0")
        with patch("bipedalwalker_gui.messagebox.showerror") as dialog:
            self.app.start_training()
        self.assertTrue(dialog.called)
        self.assertIn("Bildrate (FPS)", dialog.call_args.args[1])
        self.assertFalse(self.app.busy)

    def test_comparison_shows_both_slots_side_by_side(self):
        self.app.animation_live = [True, True]
        self.app._sync_panels()
        self.root.update()
        self.assertEqual([frame.winfo_ismapped() for frame in self.app.env_frames], [1, 1])
        self.app.animation_live = [False, False]
        self.app._sync_panels()
        self.root.update()
        self.assertEqual([frame.winfo_ismapped() for frame in self.app.env_frames],
                         [1, 0], "Ohne Lauf ist nur das aktive Verfahren zu sehen.")

    def test_each_slot_has_its_own_readout(self):
        self.assertEqual(len(self.app.observations), len(SLOT_LABELS))
        self.assertIsNot(self.app.observations[0], self.app.observations[1])

    def test_evaluation_settings_are_global_and_validated(self):
        self.assertEqual(self.app._evaluation_settings(), (5, 10000))
        self.app.evaluation_interval.set("0")
        self.assertEqual(self.app._evaluation_settings()[1], 0)
        self.app.evaluation_interval.set("-5")
        with self.assertRaises(ValueError) as context:
            self.app._evaluation_settings()
        self.assertIn("Eval-Intervall", str(context.exception))
        self.assertIn("Gültig", str(context.exception))

    def test_equal_default_budgets_start_without_a_warning(self):
        """Alle Verfahren starten mit demselben Budget, der Vergleich ist von
        Haus aus budgetgleich."""
        self._set_algorithm(0, "PPO")
        self._set_algorithm(1, "SAC")
        self.assertEqual(self.app._config(0).total_timesteps,
                         self.app._config(1).total_timesteps)

    def test_differing_budgets_are_flagged_before_the_comparison_starts(self):
        self._set_algorithm(0, "PPO")
        self._set_algorithm(1, "SAC")
        self.app.values[1]["total_timesteps"].set("50000")
        with patch("bipedalwalker_gui.messagebox.askyesno", return_value=False) as dialog:
            self.app.start_comparison()
        self.assertTrue(dialog.called)
        self.assertIn("Budgets", dialog.call_args.args[0])
        self.assertFalse(self.app.busy, "Bei Abbruch startet kein Vergleich.")


class SummaryTests(GUITestCase):
    def test_training_summary_names_the_active_slot_and_its_algorithm(self):
        self._select(1)
        self.app._training_summary()
        text = self.app._summary_text()
        self.assertIn(SLOT_LABELS[1], text)
        self.assertIn(self.app.slots[1].algorithm, text)
        self.assertIn("Zielquote", text)
        self.assertIn("Gelöst-Quote", text)
        self.assertIn("Sturzquote", text)
        self.assertIn("Zeitlimitquote", text)

    def test_comparison_summary_lists_the_configuration_differences(self):
        self._set_algorithm(0, "SAC")
        self._set_algorithm(1, "SAC")
        self.app.values[1]["learning_rate"].set("0.0001")
        self.app.slots[1].workbench.config = self.app._config(1)
        self.app._comparison_summary()
        text = self.app._summary_text()
        self.assertIn("Unterschiede", text)
        self.assertIn("Lernrate α", text)
        self.assertIn("0.0001", text)

    def test_identical_slots_report_no_differences(self):
        self._set_algorithm(0, "TD3")
        self._set_algorithm(1, "TD3")
        self.app._comparison_summary()
        self.assertIn("Unterschiede: keine", self.app._summary_text())

    def test_comparison_summary_has_exactly_two_result_columns(self):
        self.app.slots[0].comparison_history.append(_metric())
        self.app._comparison_summary()
        header = self.app._summary_text().splitlines()[2]
        self.assertIn(SLOT_LABELS[0], header)
        self.assertIn(SLOT_LABELS[1], header)


class ExportTests(GUITestCase):
    def test_export_chart_writes_png_of_the_selected_tab(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "chart.png")
            with patch("bipedalwalker_gui.filedialog.asksaveasfilename", return_value=path):
                self.app.export_chart()
            self.assertGreater(Path(path).stat().st_size, 0)

    def test_export_summary_includes_summary_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "summary.txt")
            with patch("bipedalwalker_gui.filedialog.asksaveasfilename", return_value=path):
                self.app.export_summary()
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("Training", text)
            self.assertIn("Konfiguration:", text)
            self.assertIn("algorithm: PPO", text)
            self.assertIn("evaluation_interval:", text)

    def test_comparison_export_documents_both_slots(self):
        self.app.charts.select(1)
        self.root.update()
        lines = "\n".join(self.app._config_snapshot_lines())
        for label in SLOT_LABELS:
            self.assertIn(f"[{label}]", lines)

    def test_single_export_documents_only_the_active_slot(self):
        self.app.charts.select(0)
        self._select(0)
        lines = "\n".join(self.app._config_snapshot_lines())
        self.assertIn(f"[{SLOT_LABELS[0]}]", lines)
        self.assertNotIn(f"[{SLOT_LABELS[1]}]", lines)

    def test_exports_default_into_the_project_exports_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("bipedalwalker_gui.filedialog.asksaveasfilename",
                       return_value=str(Path(directory) / "chart.png")) as dialog:
                self.app.export_chart()
            self.assertEqual(dialog.call_args.kwargs["initialdir"], EXPORT_DIR)
            self.assertTrue(EXPORT_DIR.is_dir())

    def test_chart_and_summary_share_a_matching_base_name(self):
        suggested = []

        def fake_dialog(**kwargs):
            suggested.append(kwargs["initialfile"])
            return str(Path(tempfile.mkdtemp()) / kwargs["initialfile"])

        with patch("bipedalwalker_gui.filedialog.asksaveasfilename", side_effect=fake_dialog):
            self.app.export_chart()
            self.app.export_summary()
        png_name, txt_name = suggested
        self.assertEqual(png_name[:-len("_training.png")], txt_name[:-len("_config.txt")])

    def test_export_name_reflects_slot_and_algorithm(self):
        self._select(0)
        self.app.charts.select(0)
        self.assertIn("v1-ppo", self.app._export_base_name())
        self.assertEqual(slugify("Ornstein-Uhlenbeck"), "ornstein-uhlenbeck")


class ComparisonRunTests(GUITestCase):
    """Kurzer echter Vergleichslauf über Worker-Threads und Ereignis-Queue."""

    def _configure(self, slot: int, **values) -> None:
        self._set_algorithm(slot, "TD3")
        # Die ersten `learning_starts` Schritte handeln zufällig; damit stürzt
        # der Roboter nach rund 40 bis 130 Schritten und liefert zuverlässig
        # abgeschlossene Episoden. Eine gelernte Policy kann dagegen stehen
        # bleiben, bis das Zeitlimit von 1600 Schritten greift.
        defaults = dict(total_timesteps=900, learning_starts=300, train_freq=32, batch_size=16,
                        buffer_size=1000, actor_arch="32", critic_arch="32")
        defaults.update(values)
        for name, value in defaults.items():
            self.app.values[slot][name].set(str(value))

    def _pump(self, limit: float = 240.0) -> None:
        start = time.monotonic()
        while self.app.busy and time.monotonic() - start < limit:
            self.root.update()
            time.sleep(0.01)
        self.root.update()

    def test_short_comparison_fills_both_slots_without_touching_single_training(self):
        self._configure(0, action_noise_sigma=0.1)
        self._configure(1, action_noise_sigma=0.4)
        self.app.animation_enabled.set(False)
        self.app.evaluation_interval.set("100")
        self.app.evaluation_episodes.set("1")
        self.app.start_comparison()
        self.assertTrue(self.app.busy)
        self._pump()
        self.assertFalse(self.app.busy)
        self.assertEqual(self.app.status.get(), "Abgeschlossen – Vergleich")
        for slot in range(len(SLOT_LABELS)):
            self.assertGreater(len(self.app.slots[slot].comparison_history), 0)
            self.assertGreater(len(self.app.slots[slot].comparison_evaluations), 0)
            self.assertIsNone(self.app.slots[slot].workbench.model,
                              "Der Vergleich verändert das sichtbare Einzelexperiment nicht.")
        self.assertIsNot(self.app.slots[0].comparison.model, self.app.slots[1].comparison.model)
        self.assertAlmostEqual(self.app.progress.get(), 100.0, delta=5.0)
        self.assertIn("Noise σ", self.app._summary_text())

    def test_live_animation_shows_both_verfahren_during_a_comparison(self):
        self._configure(0, total_timesteps=1800)
        self._configure(1, total_timesteps=1800)
        self.app.evaluation_interval.set("0")
        self.app.fps.set(str(MAX_ANIMATION_FPS))
        frames, snapshots = [0, 0], []
        show_frame, refresh = self.app._show_frame, self.app._refresh_snapshot

        def counting_show(slot, frame):
            frames[slot] += 1
            return show_frame(slot, frame)

        def traced_refresh(slot, model):
            refresh(slot, model)
            snapshots.append((slot, self.app.animation_policy[slot] is not model.policy))

        self.app._show_frame, self.app._refresh_snapshot = counting_show, traced_refresh
        self.app.start_comparison()
        self.assertEqual(self.app.animation_live, [True, True],
                         "Im Vergleich werden beide Verfahren gleichzeitig gezeigt.")
        self._pump()
        for slot in range(len(SLOT_LABELS)):
            self.assertGreater(frames[slot], 0, f"{SLOT_LABELS[slot]} wurde nie angezeigt")
            self.assertIn(f"({self.app.slots[slot].algorithm})", self.app.observations[slot].get())
        self.assertEqual({slot for slot, _ in snapshots}, {0, 1})
        self.assertTrue(all(distinct for _, distinct in snapshots),
                        "Die Animation spielt auf einer Kopie, nie auf dem lernenden Netz.")
        self.assertEqual([frame.winfo_ismapped() for frame in self.app.env_frames], [1, 0],
                         "Nach dem Lauf bleibt nur das aktive Verfahren sichtbar.")

    def test_animation_can_be_switched_on_and_off_during_a_run(self):
        self._configure(0, total_timesteps=1800)
        self._configure(1, total_timesteps=1800)
        self.app.evaluation_interval.set("0")
        self.app.animation_enabled.set(False)
        self.app.start_comparison()
        self.root.update()
        self.assertEqual(self.app.animation_live, [False, False])
        self.assertTrue(self.app.busy, "Der Lauf muss für diesen Test noch laufen.")
        self.app.animation_enabled.set(True)
        self.app._animation_toggled()
        self.root.update()
        self.assertEqual(self.app.animation_live, [True, True],
                         "Einschalten wirkt sofort im laufenden Vergleich.")
        self.app.animation_enabled.set(False)
        self.app._animation_toggled()
        self.root.update()
        self.assertEqual(self.app.animation_live, [False, False],
                         "Ausschalten beendet die Animation mitten im Lauf.")
        self.assertEqual(self.app.animation_after, [None, None])
        self._pump()
        self.assertFalse(self.app.busy)

    def test_live_animation_stays_off_when_the_checkbox_is_unchecked(self):
        self._configure(0)
        self._configure(1)
        self.app.animation_enabled.set(False)
        self.app.start_comparison()
        self.root.update()
        self.assertEqual(self.app.animation_live, [False, False])
        self.assertIsNone(self.app.renderers[1],
                          "Ohne Live-Animation entsteht kein zweiter Renderprozess.")
        self._pump()


def _metric():
    from bipedalwalker_logic import EpisodeMetric
    return EpisodeMetric(1, -120.0, 80, False, True, False, False, 80)


if __name__ == "__main__":
    unittest.main()
