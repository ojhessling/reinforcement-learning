"""Tests für Layout, Slotverwaltung, Farbgebung und Beschriftung der HalfCheetah-GUI."""

import dataclasses
import importlib.util
import tkinter as tk
import unittest
from unittest.mock import patch

import numpy as np

from halfcheetah_gui import (
    BEST_EPISODE,
    CURRENT_EPISODE,
    DEFAULT_SLOT_ALGORITHMS,
    DEFAULT_SMOOTHING,
    MAX_SMOOTHING,
    MIN_SMOOTHING,
    DEFAULT_SLOT_COUNT,
    MAX_ANIMATION_FPS,
    MAX_SLOTS,
    EPISODE_CHOICES,
    MIN_ANIMATION_FPS,
    PARAMETER_GROUPS,
    REFERENCE_COLOR,
    REFERENCE_STYLE,
    RENDER_FPS,
    SLOT_COLORS,
    SLOT_COUNTS,
    SLOT_LABELS,
    SLOT_LINESTYLE,
    SLOT_SHORT,
    HalfCheetahGUI,
    downsample_minmax,
    german,
    panel_grid_positions,
    rolling_average,
    slugify,
)
from halfcheetah_logic import (
    ACTION_DIM,
    ALGORITHMS,
    GEARS,
    OBSERVATION_DIM,
    config_differences,
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_EVALUATION_INTERVAL,
    DEFAULT_TOTAL_TIMESTEPS,
    MAX_EPISODE_STEPS,
    OBSERVATION_DIM,
    SOLVED_RETURN,
    EpisodeMetric,
    EvaluationResult,
    default_config,
)
from halfcheetah_render import MUJOCO_BACKENDS, mujoco_gl_backend, register_backend


def metric(episode: int, reward: float, speed: float = 2.5) -> EpisodeMetric:
    """Jede Episode ist 1000 Schritte lang – das Environment terminiert nie."""
    return EpisodeMetric(episode, reward, MAX_EPISODE_STEPS, speed > 0,
                         reward >= SOLVED_RETURN, episode * MAX_EPISODE_STEPS,
                         speed, speed * MAX_EPISODE_STEPS * 0.05, -300.0)


class HelperTests(unittest.TestCase):
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

    def test_default_frame_rate_is_valid_and_matches_the_environment(self):
        self.assertEqual(RENDER_FPS, 20)
        # Der Standardwert muss selbst im gültigen Bereich liegen.
        self.assertLessEqual(MIN_ANIMATION_FPS, RENDER_FPS)
        self.assertGreaterEqual(MAX_ANIMATION_FPS, RENDER_FPS)

    def test_german_numbers_use_comma_and_dot(self):
        self.assertEqual(german(412.7), "412,7")
        self.assertEqual(german(3800.0, 0), "3.800")
        self.assertEqual(slugify("SAC Profil"), "sac-profil")

    def test_entry_point_exposes_main_without_starting_a_window(self):
        import halfcheetah_app

        self.assertTrue(callable(halfcheetah_app.main))
        self.assertIs(halfcheetah_app.HalfCheetahGUI, HalfCheetahGUI)


class PanelGridTests(unittest.TestCase):
    """Höchstens zwei Spalten und zwei Zeilen, alle Zellen gleich groß."""

    def test_single_panel_spans_the_whole_area(self):
        self.assertEqual(panel_grid_positions(1), [(0, 0, 2)])

    def test_two_panels_stand_side_by_side(self):
        self.assertEqual(panel_grid_positions(2), [(0, 0, 1), (0, 1, 1)])

    def test_three_panels_fill_the_first_row_first(self):
        self.assertEqual(panel_grid_positions(3), [(0, 0, 1), (0, 1, 1), (1, 0, 1)])

    def test_four_panels_form_two_by_two(self):
        positions = panel_grid_positions(4)
        self.assertEqual(positions, [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)])
        self.assertEqual(len({row for row, _, _ in positions}), 2)
        self.assertEqual(len({column for _, column, _ in positions}), 2)

    def test_never_more_than_two_rows_or_columns(self):
        for count in range(1, MAX_SLOTS + 1):
            positions = panel_grid_positions(count)
            self.assertEqual(len(positions), count)
            self.assertLessEqual(max(row for row, _, _ in positions), 1)
            self.assertLessEqual(max(column for _, column, _ in positions), 1)


class PaletteTests(unittest.TestCase):
    def test_each_slot_has_its_own_colour_in_the_prescribed_order(self):
        self.assertEqual(len(SLOT_COLORS), MAX_SLOTS)
        self.assertEqual(len(set(SLOT_COLORS)), MAX_SLOTS)
        blue, red, yellow, green = SLOT_COLORS
        for colour in SLOT_COLORS:
            self.assertRegex(colour, r"^#[0-9a-f]{6}$")

        def channels(value: str) -> tuple[int, int, int]:
            return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))

        r, g, b = channels(blue)
        self.assertGreater(b, max(r, g), "Verfahren 1 muss blau sein")
        r, g, b = channels(red)
        self.assertGreater(r, max(g, b), "Verfahren 2 muss rot sein")
        r, g, b = channels(yellow)
        self.assertGreater(min(r, g), b, "Verfahren 3 muss gelb sein")
        r, g, b = channels(green)
        self.assertGreater(g, max(r, b), "Verfahren 4 muss grün sein")

    def test_slot_curves_are_solid_and_the_reference_line_is_white_and_dashed(self):
        self.assertEqual(SLOT_LINESTYLE, "-")
        self.assertEqual(REFERENCE_COLOR, "#ffffff")
        self.assertEqual(REFERENCE_STYLE, "--")
        self.assertNotIn(REFERENCE_COLOR, SLOT_COLORS)

    def test_line_widths_and_raw_transparency(self):
        self.assertEqual(HalfCheetahGUI.LINE_WIDTH, 1.2)
        self.assertAlmostEqual(HalfCheetahGUI.RAW_ALPHA, 0.10)


class RendererBackendTests(unittest.TestCase):
    def test_macos_uses_cgl_and_linux_uses_egl(self):
        # GLFW meldet seinen Prozess als Vordergrund-App an und erzeugt damit
        # je Renderprozess einen eigenen Dock-Eintrag; CGL rendert offscreen.
        self.assertEqual(mujoco_gl_backend("darwin"), "cgl")
        self.assertEqual(mujoco_gl_backend("linux"), "egl")
        self.assertIn(mujoco_gl_backend("darwin"), MUJOCO_BACKENDS)
        self.assertIn(mujoco_gl_backend("linux"), MUJOCO_BACKENDS)

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco") is not None
        and importlib.util.find_spec("imageio") is not None,
        "MuJoCo ist nicht installiert")
    def test_cgl_is_added_to_the_gymnasium_backend_table(self):
        from gymnasium.envs.mujoco import mujoco_rendering

        register_backend("cgl")
        self.assertIn("cgl", mujoco_rendering._ALL_RENDERERS)
        # Die mitgelieferten Backends bleiben unangetastet.
        for name in ("glfw", "egl", "osmesa"):
            self.assertIn(name, mujoco_rendering._ALL_RENDERERS)
        register_backend("glfw")  # ergänzt nichts


class ParameterGroupTests(unittest.TestCase):
    def test_each_tab_only_offers_parameters_of_its_algorithm(self):
        for algorithm in ALGORITHMS:
            config = default_config(algorithm)
            names = [name for _column, _title, group in PARAMETER_GROUPS[algorithm]
                     for name in group]
            self.assertEqual(len(names), len(set(names)))
            for name in names:
                self.assertTrue(config.uses(name),
                                f"{algorithm} kennt {name} nicht, zeigt es aber an")

    def test_normalisation_is_offered_for_every_method(self):
        for algorithm in ALGORITHMS:
            titles = [title for _column, title, _group in PARAMETER_GROUPS[algorithm]]
            self.assertIn("Normalisierung", titles)


class GUITestCase(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Kein grafisches Display verfügbar: {error}")
        # Der Renderprozess startet MuJoCo; die Layouttests brauchen ihn nicht.
        patcher = patch.object(HalfCheetahGUI, "_show_initial_frame", lambda self, slot=0: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = HalfCheetahGUI(self.root)
        self.root.update()

    def tearDown(self):
        self.app.close()

    def _set_slot_count(self, count: int) -> None:
        self.app.slot_count_var.set(str(count))
        self.app._slot_count_changed()
        self.root.update()

    def _select(self, slot: int) -> None:
        self.app.parameter_tabs.select(self.app.tab_frames[slot])
        self.root.update()

    def _show(self, slots: list[int]) -> None:
        self.app.show_panels(slots)
        self.root.update()

    def _fill_frames(self, slots: list[int]) -> None:
        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        for slot in slots:
            self.app._show_frame(slot, frame)
        self.root.update()


class LayoutSmokeTest(GUITestCase):
    def test_start_layout_has_no_clipped_controls(self):
        self.app._initialize_layout()
        self.root.update()
        self.assertEqual(self.app.layout_visibility_issues(), [])

    def test_every_tab_of_every_slot_count_is_fully_visible(self):
        for count in SLOT_COUNTS:
            self._set_slot_count(count)
            for slot in range(count):
                self._select(slot)
                with self.subTest(count=count, slot=slot):
                    self.assertEqual(self.app.layout_visibility_issues(), [])

    def test_every_algorithm_fits_into_a_tab(self):
        for algorithm in ALGORITHMS:
            self.app.algorithm_vars[0].set(algorithm)
            self.app._algorithm_changed(0)
            self._select(0)
            with self.subTest(algorithm=algorithm):
                self.assertEqual(self.app.layout_visibility_issues(), [])

    def test_animation_grid_stays_visible_for_one_to_four_panels(self):
        self._set_slot_count(MAX_SLOTS)
        for count in range(1, MAX_SLOTS + 1):
            slots = list(range(count))
            self._show(slots)
            self._fill_frames(slots)
            with self.subTest(panels=count):
                self.assertEqual(self.app.layout_visibility_issues(), [])
                sizes = {(self.app.env_frames[slot].winfo_width(),
                          self.app.env_frames[slot].winfo_height()) for slot in slots}
                # Alle belegten Zellen sind gleich groß (auf ein Pixel genau).
                widths = {width for width, _ in sizes}
                heights = {height for _, height in sizes}
                self.assertLessEqual(max(widths) - min(widths), 1)
                self.assertLessEqual(max(heights) - min(heights), 1)
                for slot in slots:
                    photo_width, photo_height = self.app.photo_sizes[slot]
                    self.assertGreater(photo_width, 1)
                    # Das Seitenverhältnis des 480x480-Frames bleibt erhalten.
                    self.assertEqual(photo_width, photo_height)

    def test_the_caption_stays_visible_in_every_panel_count(self):
        """Das expandierende Bild darf die Zeile nicht aus der Zelle drücken."""
        self._set_slot_count(MAX_SLOTS)
        for count in range(1, MAX_SLOTS + 1):
            slots = list(range(count))
            self._show(slots)
            self._fill_frames(slots)
            for slot in slots:
                caption, frame = self.app.caption_labels[slot], self.app.env_frames[slot]
                with self.subTest(panels=count, slot=slot):
                    self.assertTrue(caption.winfo_ismapped())
                    self.assertGreater(caption.winfo_height(), 1)
                    self.assertLessEqual(
                        caption.winfo_rooty() + caption.winfo_height(),
                        frame.winfo_rooty() + frame.winfo_height())

    def test_export_buttons_are_compact_and_cost_no_extra_row(self):
        for button in (self.app.export_button, self.app.summary_button):
            self.assertEqual(str(button.cget("style")), "Compact.TButton")
            self.assertTrue(button.winfo_ismapped())
        self.assertEqual(self.app.export_button.cget("text"), "PNG exportieren")
        self.assertEqual(self.app.summary_button.cget("text"), "TXT exportieren")
        # Sie liegen als Overlay in Tableiste beziehungsweise Titelzeile und
        # damit nicht im Fluss über dem Diagramm.
        canvas = self.app.canvas.get_tk_widget()
        self.assertLessEqual(self.app.export_button.winfo_rooty() + 2,
                             canvas.winfo_rooty())

    def test_the_legend_stays_inside_the_figure(self):
        """Lange Labels wie `V3 – SAC (Lernrate α 0.0003)` dürfen nicht abreißen."""
        self._set_slot_count(MAX_SLOTS)
        for slot in range(MAX_SLOTS):
            self.app.algorithm_vars[slot].set("SAC")
            self.app._algorithm_changed(slot)
            self.app.slots[slot].workbench.config = dataclasses.replace(
                self.app.slots[slot].workbench.config, learning_rate=3e-4 + slot * 1e-4)
            self.app.slots[slot].comparison_history.extend(
                metric(index, 100.0 + index) for index in range(1, 30))
        self.app.charts.select(1)
        self.root.update()
        for _ in range(2):
            self.app._refresh_comparison_plot()
            self.app.comparison_canvas.draw()
        legend = self.app.comparison_axes.get_legend()
        width = self.app.comparison_canvas.get_tk_widget().winfo_width()
        self.assertLessEqual(legend.get_window_extent().x1, width + 1)
        self.assertGreaterEqual(legend.get_window_extent().x0, 0)

    def test_a_single_panel_gets_more_room_than_one_of_four(self):
        self._set_slot_count(MAX_SLOTS)
        self._show([0, 1, 2, 3])
        self._fill_frames([0, 1, 2, 3])
        small = self.app.photo_sizes[0]
        self._show([0])
        self._fill_frames([0])
        large = self.app.photo_sizes[0]
        self.assertGreater(large[0], small[0])


class SlotCountTests(GUITestCase):
    def test_four_slots_are_the_default(self):
        self.assertEqual(DEFAULT_SLOT_COUNT, MAX_SLOTS)
        self.assertEqual(self.app.slot_count, DEFAULT_SLOT_COUNT)
        self.assertEqual(len(self.app.parameter_tabs.tabs()), DEFAULT_SLOT_COUNT)
        algorithms = [self.app.algorithm_vars[slot].get() for slot in range(DEFAULT_SLOT_COUNT)]
        self.assertEqual(algorithms, list(DEFAULT_SLOT_ALGORITHMS))
        # Alle drei Verfahren des Projekts kommen vor, eines doppelt.
        self.assertEqual(set(algorithms), set(ALGORITHMS))
        self.assertEqual(len(algorithms) - len(set(algorithms)), 1)

    def test_the_repeated_algorithm_starts_with_a_visible_difference(self):
        """Zwei identisch konfigurierte Slots zeigten sonst keinen Unterschied."""
        repeated = [slot for slot in range(MAX_SLOTS)
                    if DEFAULT_SLOT_ALGORITHMS.count(DEFAULT_SLOT_ALGORITHMS[slot]) > 1]
        self.assertTrue(repeated)
        configs = [HalfCheetahGUI.initial_config(slot) for slot in repeated]
        self.assertTrue(config_differences(configs))

    def test_evaluation_defaults_follow_the_step_budget(self):
        self.assertEqual(self.app.evaluation_interval.get(), str(DEFAULT_EVALUATION_INTERVAL))
        self.assertEqual(self.app.evaluation_episodes.get(), str(DEFAULT_EVALUATION_EPISODES))
        self.assertEqual(int(self.app.evaluation_interval.get()) * 10, DEFAULT_TOTAL_TIMESTEPS)

    def test_growing_adds_tabs_and_keeps_the_existing_slots(self):
        self.app.values[0]["seed"].set("99")
        self._set_slot_count(MAX_SLOTS)
        self.assertEqual(self.app.slot_count, MAX_SLOTS)
        self.assertEqual(len(self.app.parameter_tabs.tabs()), MAX_SLOTS)
        self.assertEqual(self.app.values[0]["seed"].get(), "99")
        # Der neue Slot startet mit den Standardwerten seines Algorithmus.
        expected = default_config(self.app.algorithm_vars[3].get())
        self.assertEqual(self.app.values[3]["seed"].get(), str(expected.seed))
        self.assertEqual(self.app.values[3]["total_timesteps"].get(),
                         str(expected.total_timesteps))

    def test_shrinking_without_data_needs_no_confirmation(self):
        with patch("halfcheetah_gui.messagebox.askyesno") as ask:
            self._set_slot_count(2)
            ask.assert_not_called()
        self.assertEqual(self.app.slot_count, 2)
        self.assertEqual(len(self.app.parameter_tabs.tabs()), 2)

    def test_shrinking_asks_before_discarding_a_learning_state(self):
        self.app.slots[2].history.append(metric(1, 100.0))
        with patch("halfcheetah_gui.messagebox.askyesno", return_value=False) as ask:
            self._set_slot_count(2)
            ask.assert_called_once()
        self.assertEqual(self.app.slot_count, DEFAULT_SLOT_COUNT)
        self.assertTrue(self.app.slots[2].history)
        with patch("halfcheetah_gui.messagebox.askyesno", return_value=True):
            self._set_slot_count(2)
        self.assertEqual(self.app.slot_count, 2)
        self.assertFalse(self.app.slots[2].history)

    def test_a_dropped_active_slot_falls_back_to_the_first(self):
        self._set_slot_count(MAX_SLOTS)
        self._select(3)
        self.assertEqual(self.app.active_slot, 3)
        self._set_slot_count(2)
        self.assertLess(self.app.active_slot, 2)

    def test_the_slot_count_is_locked_during_a_run(self):
        self.app.busy = True
        self.app.slot_count_var.set("4")
        self.app._slot_count_changed()
        self.assertEqual(self.app.slot_count, DEFAULT_SLOT_COUNT)
        self.assertEqual(self.app.slot_count_var.get(), str(DEFAULT_SLOT_COUNT))
        self.app.busy = False

    def test_switching_an_algorithm_only_resets_its_own_slot(self):
        self.app.values[1]["seed"].set("77")
        self.app.algorithm_vars[0].set("TD3")
        self.app._algorithm_changed(0)
        self.root.update()
        self.assertEqual(self.app.slots[0].algorithm, "TD3")
        self.assertEqual(self.app.values[1]["seed"].get(), "77")
        self.assertEqual(self.app.slots[1].algorithm, DEFAULT_SLOT_ALGORITHMS[1])


class PlotTests(GUITestCase):
    def test_comparison_plot_draws_one_solid_line_per_slot_in_its_colour(self):
        self._set_slot_count(MAX_SLOTS)
        for slot in range(MAX_SLOTS):
            self.app.slots[slot].comparison_history.extend(
                metric(index, 100.0 + 10 * slot + index) for index in range(1, 30))
        self.app._refresh_comparison_plot()
        highlighted = [line for line in self.app.comparison_axes.get_lines()
                       if line.get_linewidth() == HalfCheetahGUI.LINE_WIDTH
                       and str(line.get_label()).startswith("V")]
        self.assertEqual(len(highlighted), MAX_SLOTS)
        for slot, line in enumerate(highlighted):
            self.assertEqual(line.get_color(), SLOT_COLORS[slot])
            self.assertEqual(line.get_linestyle(), "-")
            self.assertTrue(line.get_label().startswith(SLOT_SHORT[slot]))

    def test_neither_graph_shows_the_deterministic_evaluation(self):
        """Die Zwischenevaluation steht ausschließlich in der Summary."""
        self.app.slots[0].evaluations.append(
            (5, EvaluationResult(3, 900.0, 10.0, 1000.0, 1.0, 0.0, 0.0, 2.0, 100.0, -250.0)))
        self.app.slots[0].comparison_evaluations.append(
            (5, EvaluationResult(3, 900.0, 10.0, 1000.0, 1.0, 0.0, 0.0, 2.0, 100.0, -250.0)))
        self.app.slots[0].history.extend(metric(index, 100.0 + index) for index in range(1, 30))
        self.app._refresh_training_plot()
        self.app._refresh_comparison_plot()
        for axes in (self.app.axes, self.app.comparison_axes):
            labels = [str(line.get_label()) for line in axes.get_lines()]
            self.assertFalse([label for label in labels if "Evaluation" in label], labels)

    def test_graphs_have_no_title_and_a_plain_return_axis(self):
        self.app._refresh_training_plot()
        self.app._refresh_comparison_plot()
        for axes in (self.app.axes, self.app.comparison_axes):
            self.assertEqual(axes.get_title(), "")
            self.assertEqual(axes.get_ylabel(), "Return")
            self.assertEqual(axes.get_xlabel(), "Episode")

    def test_raw_episodes_use_the_configured_transparency(self):
        self.app.slots[0].comparison_history.extend(
            metric(index, 100.0 + index) for index in range(1, 30))
        self.app._refresh_comparison_plot()
        faint = [line for line in self.app.comparison_axes.get_lines()
                 if line.get_alpha() is not None]
        self.assertTrue(faint)
        for line in faint:
            self.assertAlmostEqual(line.get_alpha(), HalfCheetahGUI.RAW_ALPHA)

    def test_the_solved_line_is_white_and_dashed(self):
        self.app._refresh_comparison_plot()
        for axes in (self.app.axes, self.app.comparison_axes):
            reference = [line for line in axes.get_lines()
                         if "Gelöst" in str(line.get_label())]
            self.assertEqual(len(reference), 1)
            self.assertEqual(reference[0].get_color(), REFERENCE_COLOR)
            self.assertEqual(reference[0].get_linestyle(), "--")
            self.assertAlmostEqual(reference[0].get_ydata()[0], SOLVED_RETURN)

    def test_the_axis_follows_the_data_instead_of_the_far_away_threshold(self):
        self.app.slots[0].comparison_history.extend(
            metric(index, 100.0 + index) for index in range(1, 40))
        self.app._refresh_comparison_plot()
        low, high = self.app.comparison_axes.get_ylim()
        self.assertLess(high, SOLVED_RETURN,
                        "Die Gelöst-Marke darf die Achse nicht aufspannen")
        self.assertGreater(high, 130)
        self.assertLess(low, 101)

    def test_labels_name_the_differing_parameter_for_equal_algorithms(self):
        self._set_slot_count(2)
        for slot in (0, 1):
            self.app.algorithm_vars[slot].set("SAC")
            self.app._algorithm_changed(slot)
        self.app.slots[1].workbench.config = dataclasses.replace(
            self.app.slots[1].workbench.config, tau=0.05)
        configs = [self.app.slots[slot].workbench.config for slot in range(2)]
        self.assertIn("Soft-Update τ", self.app._series_label(0, configs))
        self.assertIn("0.005", self.app._series_label(0, configs))
        self.assertIn("0.05", self.app._series_label(1, configs))


class SummaryTests(GUITestCase):
    def test_comparison_summary_has_no_heading(self):
        self._set_slot_count(2)
        for slot in range(2):
            self.app.slots[slot].comparison_history.append(metric(1, 250.0))
        self.app._comparison_summary()
        first = self.app._summary_text().splitlines()[0]
        # Die Spaltenköpfe sagen bereits, was verglichen wird.
        self.assertTrue(first.startswith("Statistik"), first)
        self.assertNotIn("Vergleich der Verfahrensslots", self.app._summary_text())

    def test_training_summary_keeps_its_heading(self):
        self._select(0)
        self.app.slots[0].history.append(metric(1, 100.0))
        self.app._training_summary()
        self.assertIn(f"Training – {SLOT_LABELS[0]}", self.app._summary_text())

    def test_comparison_summary_has_one_column_per_active_slot(self):
        self._set_slot_count(MAX_SLOTS)
        for slot in range(MAX_SLOTS):
            self.app.slots[slot].comparison_history.append(metric(1, 250.0 + slot))
        self.app._comparison_summary()
        text = self.app._summary_text()
        for slot in range(MAX_SLOTS):
            self.assertIn(SLOT_LABELS[slot], text)
        for label in ("Vorwärtsquote", "Gelöst-Quote", "Ø Tempo m/s", "Ø Strecke m",
                      "Ø Steuerkosten"):
            self.assertIn(label, text)
        # Konstante Kennzahl: Die Episodenlänge gehört nicht in die Summary.
        self.assertNotIn("Ø Länge", text)
        for absent in ("Durchhaltequote", "Sturzquote", "Zielquote"):
            self.assertNotIn(absent, text)

    def test_difference_block_lists_a_value_per_slot(self):
        self._set_slot_count(3)
        for slot in range(3):
            self.app.algorithm_vars[slot].set("TD3")
            self.app._algorithm_changed(slot)
            self.app.slots[slot].workbench.config = dataclasses.replace(
                self.app.slots[slot].workbench.config, seed=slot)
        lines = self.app._difference_lines(
            [self.app.slots[slot].workbench.config for slot in range(3)])
        text = "\n".join(lines)
        self.assertIn("Unterschiede", text)
        self.assertIn("Zufallsstart s", text)
        for short in SLOT_SHORT[:3]:
            self.assertIn(short, text)

    def test_summary_lists_the_best_episode_with_number_and_return(self):
        history = [metric(1, 100.0), metric(2, 980.5), metric(3, 340.0)]
        self.app.slots[0].history.extend(history)
        self._select(0)
        self.app._training_summary()
        text = self.app._summary_text()
        self.assertIn("Beste Episode", text)
        self.assertIn("#2: 980,5", text)
        self.assertEqual(HalfCheetahGUI.best_episode(history).episode, 2)
        self.assertIsNone(HalfCheetahGUI.best_episode([]))

    def test_best_episode_appears_for_every_comparison_slot(self):
        self._set_slot_count(3)
        for slot in range(3):
            self.app.slots[slot].comparison_history.extend(
                [metric(1, 10.0 * slot), metric(2, 500.0 + slot)])
        self.app._comparison_summary()
        text = self.app._summary_text()
        self.assertIn("Beste Episode", text)
        for slot in range(3):
            self.assertIn(f"#2: {german(500.0 + slot)}", text)

    def test_training_summary_reports_the_evaluation_metrics(self):
        result = EvaluationResult(5, 1234.5, 20.0, 1000.0, 0.8, 0.2, 0.0, 1.5, 75.0, -310.0)
        self.app.slots[0].evaluations.append((3, result))
        self._select(0)
        self.app._training_summary()
        text = self.app._summary_text()
        self.assertIn("1.234,5", text)
        self.assertIn("Letzte Evaluation", text)
        self.assertIn("m/s", text)


class CaptionAndHoverTests(GUITestCase):
    def _prepare(self, slot: int = 0) -> None:
        self.app.animation_observation[slot] = np.array(
            [-0.05, 0.02, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6,
             12.5, -0.3, 0.2, 1.0, -2.0, 3.0, -1.5, 2.5, -3.5])
        self.app.animation_action[slot] = np.array([1.0, -0.25, 0.0, 0.5, -1.0, 0.75])
        self.app.animation_info[slot] = {
            "reward_forward": 12.5, "reward_ctrl": -0.28,
            "x_position": 62.5, "x_velocity": 12.5,
        }
        self.app.animation_episode[slot] = 12
        self.app.animation_step[slot] = 348
        self.app.animation_reward[slot] = 412.7
        self.app._update_caption(slot)
        self._fill_frames([slot])

    def test_caption_holds_exactly_episode_step_and_return(self):
        self._prepare(0)
        caption = self.app.captions[0].get()
        self.assertEqual(len(caption.splitlines()), 1)
        self.assertIn("E:   12", caption)
        self.assertIn(f"S:  348/{MAX_EPISODE_STEPS}", caption)
        self.assertIn("R:", caption)
        self.assertIn("412,7", caption)
        # Keine Actionwerte und keine Observationswerte unter dem Bild.
        for forbidden in ("a0", "N·m", "Höhe", "Rumpfwinkel", "rad", "Reward des"):
            self.assertNotIn(forbidden, caption)

    def test_the_method_is_named_in_the_panel_title(self):
        title = self.app.env_frames[0].cget("text")
        self.assertIn(SLOT_LABELS[0], title)
        self.assertIn(self.app.slots[0].algorithm, title)
        self.assertNotIn("Gymnasium", title)

    def test_caption_keeps_a_constant_shape_while_counting(self):
        self._prepare(0)
        first = self.app.captions[0].get()
        self.app.animation_step[0] = 9
        self.app.animation_reward[0] = 5.0
        self.app._update_caption(0)
        second = self.app.captions[0].get()
        self.assertEqual(first.count("·"), second.count("·"))
        self.assertEqual(len(first), len(second))

    def test_hover_shows_actions_and_state_next_to_the_image(self):
        self._prepare(0)
        text = self.app._hover_lines(0)
        for index in range(ACTION_DIM):
            self.assertIn(f"a{index}", text)
        self.assertIn("N·m", text)
        self.assertIn("Höhe", text)
        self.assertIn("Rumpfwinkel", text)
        self.assertIn("vorwärts", text)
        self.assertIn("aus info", text)
        # Jedes Gelenk zeigt seine eigene Übersetzung, kein gemeinsamer Faktor.
        for gear in {int(value) for value in GEARS}:
            self.assertIn(str(gear), text)
        # Kein Clipping-Hinweis und kein Überlebensbonus.
        self.assertNotIn("±10", text)
        self.assertNotIn("Überleben", text)
        self.assertIn("unbeschränkt", text)

    def test_hover_appears_only_while_pointing_at_an_image(self):
        self._prepare(0)
        self.app._hide_hover()
        self.root.update_idletasks()
        self.assertIsNone(self.app.hover_slot)
        self.assertFalse(self.app.hover_panel.winfo_ismapped())
        self.app._hover_enter(0)
        self.root.update_idletasks()
        self.assertTrue(self.app.hover_panel.winfo_ismapped())
        self.assertEqual(self.app.hover_slot, 0)
        self.app._hover_leave(0)
        self.root.update_idletasks()
        self.assertFalse(self.app.hover_panel.winfo_ismapped())
        self.assertIsNone(self.app.hover_slot)

    def test_hover_never_covers_its_own_image(self):
        self._set_slot_count(MAX_SLOTS)
        for count in range(1, MAX_SLOTS + 1):
            slots = list(range(count))
            self._show(slots)
            for slot in slots:
                self._prepare(slot)
            self.root.update_idletasks()
            for slot in slots:
                self.app._hover_enter(slot)
                self.root.update_idletasks()
                with self.subTest(panels=count, slot=slot):
                    label = self.app.image_labels[slot]
                    photo_width, _ = self.app.photo_sizes[slot]
                    origin = self.root.winfo_rootx()
                    image_left = label.winfo_rootx() - origin \
                        + max(0, (label.winfo_width() - photo_width) // 2)
                    image_right = image_left + photo_width
                    panel_left = self.app.hover_panel.winfo_x()
                    panel_right = panel_left + self.app.hover_panel.winfo_width()
                    self.assertTrue(panel_right <= image_left or panel_left >= image_right,
                                    "Die Einblendung überdeckt ihr eigenes Bild")
                self.app._hover_leave(slot)
                self.root.update_idletasks()

    def test_hover_does_not_move_or_resize_the_images(self):
        self._set_slot_count(MAX_SLOTS)
        self._show([0, 1, 2, 3])
        for slot in range(MAX_SLOTS):
            self._prepare(slot)
        self.root.update()
        before = [(self.app.image_labels[slot].winfo_rootx(),
                   self.app.image_labels[slot].winfo_rooty(),
                   self.app.image_labels[slot].winfo_width(),
                   self.app.image_labels[slot].winfo_height()) for slot in range(MAX_SLOTS)]
        self.app._hover_enter(1)
        self.root.update()
        during = [(self.app.image_labels[slot].winfo_rootx(),
                   self.app.image_labels[slot].winfo_rooty(),
                   self.app.image_labels[slot].winfo_width(),
                   self.app.image_labels[slot].winfo_height()) for slot in range(MAX_SLOTS)]
        self.assertEqual(before, during)
        self.app._hover_leave(1)
        self.root.update()

    def test_hover_follows_the_current_frame(self):
        self._prepare(0)
        self.app._hover_enter(0)
        self.root.update()
        first = self.app.hover_body.cget("text")
        self.app.animation_observation[0] = np.full(OBSERVATION_DIM, 0.75)
        self.app._refresh_hover()
        second = self.app.hover_body.cget("text")
        self.assertNotEqual(first, second)
        self.assertIn("+0.750", second)
        self.app._hover_leave(0)


class PanelTitleTests(GUITestCase):
    def test_title_names_slot_and_algorithm(self):
        for slot in range(self.app.slot_count):
            title = self.app.env_frames[slot].cget("text")
            self.assertTrue(
                title.startswith(f"{SLOT_LABELS[slot]} – {self.app.slots[slot].algorithm}"), title)

    def test_a_unique_algorithm_needs_no_parameter_suffix(self):
        self._set_slot_count(2)  # PPO und TD3, beide einmalig
        for slot in range(2):
            self.assertEqual(self.app.env_frames[slot].cget("text"),
                             f"{SLOT_LABELS[slot]} – {self.app.slots[slot].algorithm}")

    def test_title_adds_the_differing_parameter_for_equal_algorithms(self):
        self._set_slot_count(MAX_SLOTS)
        for slot in range(MAX_SLOTS):
            self.app.algorithm_vars[slot].set("SAC")
            self.app._algorithm_changed(slot)
        for slot in range(MAX_SLOTS):
            self.app.slots[slot].workbench.config = dataclasses.replace(
                self.app.slots[slot].workbench.config, learning_rate=3e-4 + slot * 1e-4)
        self.app._refresh_comparison_plot()
        self.root.update()
        configs = self.app._panel_configs()
        for slot in range(MAX_SLOTS):
            title = self.app.env_frames[slot].cget("text")
            self.assertIn("Lernrate", title)
            # Titel und Legende benennen denselben Unterschied.
            legend = self.app._series_label(slot, configs)
            self.assertTrue(title.endswith(legend[legend.index("(") :]))
            self.assertTrue(title.startswith(f"{SLOT_LABELS[slot]} – SAC"))

    def test_title_follows_an_algorithm_change(self):
        self.app.algorithm_vars[0].set("TD3")
        self.app._algorithm_changed(0)
        self.root.update()
        title = self.app.env_frames[0].cget("text")
        # Slot 3 ist ebenfalls TD3, deshalb nennt der Titel zusätzlich den
        # Unterschied – der Algorithmus steht in jedem Fall vorn.
        self.assertTrue(title.startswith(f"{SLOT_LABELS[0]} – TD3"), title)
        self.assertNotIn("PPO", title)


class SmoothingTests(GUITestCase):
    """Der gleitende Durchschnitt ist global einstellbar und wirkt sofort."""

    def _highlighted(self):
        return [line for line in self.app.comparison_axes.get_lines()
                if line.get_linewidth() == HalfCheetahGUI.LINE_WIDTH
                and str(line.get_label()).startswith("V")]

    def _fill(self):
        self._set_slot_count(MAX_SLOTS)
        rng = np.random.default_rng(0)
        for slot in range(MAX_SLOTS):
            self.app.slots[slot].comparison_history.extend(
                metric(index, 500.0 + 10 * index + float(rng.normal(0, 250)))
                for index in range(1, 80))

    def test_default_window_is_twenty(self):
        self.assertEqual(DEFAULT_SMOOTHING, 20)
        self.assertEqual(self.app.smoothing_window, DEFAULT_SMOOTHING)
        self.assertEqual(self.app.smoothing.get(), str(DEFAULT_SMOOTHING))
        self.assertEqual((MIN_SMOOTHING, MAX_SMOOTHING), (1, 500))

    def test_a_wider_window_smooths_more(self):
        self._fill()
        self.app.smoothing.set("1")
        self.app._smoothing_changed()
        narrow = [np.ptp(np.asarray(line.get_ydata())) for line in self._highlighted()]
        self.app.smoothing.set("50")
        self.app._smoothing_changed()
        wide = [np.ptp(np.asarray(line.get_ydata())) for line in self._highlighted()]
        self.assertEqual(len(narrow), MAX_SLOTS)
        # Die Einstellung wirkt auf alle Slots gleichzeitig.
        for tight, loose in zip(narrow, wide):
            self.assertLess(loose, tight)

    def test_window_one_equals_the_raw_values(self):
        self._fill()
        self.app.smoothing.set("1")
        self.app._smoothing_changed()
        for slot, line in enumerate(self._highlighted()):
            raw = [item.reward for item in self.app.slots[slot].comparison_history]
            np.testing.assert_allclose(np.asarray(line.get_ydata()), raw)

    def test_a_change_redraws_immediately_even_while_running(self):
        self._fill()
        self.app.busy = True
        try:
            self.app.smoothing.set("40")
            self.app._smoothing_changed()
            self.assertEqual(self.app.smoothing_window, 40)
            # Die Messdaten bleiben unangetastet.
            self.assertEqual(len(self.app.slots[0].comparison_history), 79)
        finally:
            self.app.busy = False

    def test_invalid_input_keeps_the_last_window_and_explains_itself(self):
        self.app.smoothing.set("35")
        self.app._smoothing_changed()
        for bad in ("abc", "0", str(MAX_SMOOTHING + 1), ""):
            with self.subTest(eingabe=bad):
                self.app.smoothing.set(bad)
                self.app._smoothing_changed()
                self.assertEqual(self.app.smoothing_window, 35)
                self.assertIn("Glättung", self.app.status.get())
                self.assertIn(str(MAX_SMOOTHING), self.app.status.get())

    def test_the_control_sits_at_the_chart_and_is_visible(self):
        self.assertTrue(self.app.smoothing_box.winfo_ismapped())
        self.assertEqual(self.app.layout_visibility_issues(), [])


class EpisodeChoiceTests(GUITestCase):
    def test_every_animation_defaults_to_the_current_learning_state(self):
        self.assertEqual(EPISODE_CHOICES, (CURRENT_EPISODE, BEST_EPISODE))
        for slot in range(MAX_SLOTS):
            self.assertEqual(self.app.episode_choice[slot].get(), CURRENT_EPISODE)

    def test_each_animation_chooses_independently(self):
        self._set_slot_count(MAX_SLOTS)
        self.app.episode_choice[2].set(BEST_EPISODE)
        self.app._episode_choice_changed(2)
        self.assertEqual(self.app.episode_choice[2].get(), BEST_EPISODE)
        for slot in (0, 1, 3):
            self.assertEqual(self.app.episode_choice[slot].get(), CURRENT_EPISODE)

    def test_every_episode_has_the_same_length(self):
        """Ohne terminalen Zustand ist die Länge konstant – deshalb steht sie
        auch nicht in der Summary."""
        self.assertEqual(metric(1, 100.0).length, MAX_EPISODE_STEPS)
        self.app.slots[0].history.extend(metric(index, 100.0) for index in range(1, 5))
        self._select(0)
        self.app._training_summary()
        self.assertNotIn("Ø Länge", self.app._summary_text())

    def test_the_caption_uses_the_trained_episode_number(self):
        self.app.slots[0].history.extend(metric(index, 10.0) for index in range(1, 43))
        self.assertEqual(self.app.trained_episodes(0), 42)
        # Im Vergleich zählt die Vergleichshistorie desselben Slots.
        self.app.comparison_running = True
        self.app.slots[0].comparison_history.extend(metric(index, 10.0) for index in range(1, 8))
        self.assertEqual(self.app.trained_episodes(0), 7)
        self.app.comparison_running = False

    def test_choosing_best_without_a_snapshot_says_so(self):
        self.app.episode_choice[0].set(BEST_EPISODE)
        self.app._episode_choice_changed(0)
        self.assertIn("noch keine beste Episode", self.app.status.get())

    def test_the_marker_distinguishes_the_best_episode_replay(self):
        self.app.animation_episode[0] = 17
        self.app.animation_step[0] = 5
        self.app.animation_reward[0] = 20.0
        self.app.animation_from_best[0] = False
        self.app._update_caption(0)
        current = self.app.captions[0].get()
        self.app.animation_from_best[0] = True
        self.app._update_caption(0)
        best = self.app.captions[0].get()
        self.assertNotEqual(current, best)
        self.assertIn("*", best)
        self.assertNotIn("*", current)
        # Die Zeile behält ihre Länge, damit die Bilder nicht springen.
        self.assertEqual(len(current), len(best))


class ValidationTests(GUITestCase):
    def test_frame_rate_outside_the_range_is_reported_with_field_and_bounds(self):
        self.app.fps.set("500")
        with self.assertRaises(ValueError) as error:
            self.app._animation_settings()
        message = str(error.exception)
        self.assertIn("Bildrate (FPS)", message)
        self.assertIn("500", message)
        self.assertIn(str(MAX_ANIMATION_FPS), message)

    def test_the_default_frame_rate_passes_validation(self):
        self.assertEqual(self.app.fps.get(), str(RENDER_FPS))
        self.assertEqual(self.app._animation_settings(), RENDER_FPS)

    def test_evaluation_settings_reject_negative_intervals(self):
        self.app.evaluation_interval.set("-5")
        with self.assertRaises(ValueError) as error:
            self.app._evaluation_settings()
        self.assertIn("Eval-Intervall", str(error.exception))

    def test_invalid_slot_input_names_the_slot(self):
        self.app.values[1]["gamma"].set("abc")
        with self.assertRaises(ValueError) as error:
            self.app._config(1)
        self.assertIn(SLOT_LABELS[1], str(error.exception))

    def test_configuration_round_trips_through_the_entry_fields(self):
        for slot in range(self.app.slot_count):
            config = self.app._config(slot)
            self.assertEqual(config.algorithm, self.app.algorithm_vars[slot].get())
            # Die Startbelegung entspricht dem Zoo-Profil plus den bewussten
            # Startabweichungen der Slots.
            self.assertEqual(config, HalfCheetahGUI.initial_config(slot))
