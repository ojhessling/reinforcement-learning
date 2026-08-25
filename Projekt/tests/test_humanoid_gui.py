"""Tests für Layout, Slotverwaltung, Farbgebung und Beschriftung der Humanoid-GUI."""

import dataclasses
import importlib.util
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from tkinter import filedialog, font as tkfont, ttk
from unittest.mock import patch

import numpy as np
import torch

from humanoid_gui import (
    BEST_EPISODE,
    CURRENT_EPISODE,
    DEFAULT_ANIMATION_FPS,
    DEFAULT_SLOT_ALGORITHMS,
    DEFAULT_SMOOTHING,
    MAX_SMOOTHING,
    MIN_SMOOTHING,
    DEFAULT_SLOT_COUNT,
    MAX_ANIMATION_FPS,
    MAX_SLOTS,
    EPISODE_CHOICES,
    BEST_POLICY,
    INACTIVE_EPISODE,
    MAX_PLOT_POINTS,
    SELECTION_ROW_OFFSET,
    best_columns,
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
    HumanoidGUI,
    downsample_minmax,
    german,
    panel_grid_positions,
    rolling_average,
    slugify,
)
from humanoid_logic import (
    ACTION_DIM,
    ACTION_LIMIT,
    ACTUATOR_GROUPS,
    ALGORITHMS,
    DEFAULT_EPISODES,
    DEFAULT_EVALUATION_EPISODES,
    DEFAULT_TORCH_THREADS,
    DEFAULT_TOTAL_TIMESTEPS,
    GEARS,
    MAX_EPISODE_STEPS,
    OBSERVATION_DIM,
    STOP_REASONS,
    TARGET_RETURN,
    EpisodeMetric,
    EvaluationResult,
    config_differences,
    default_config,
)
from humanoid_render import MUJOCO_BACKENDS, mujoco_gl_backend, register_backend


def metric(episode: int, reward: float, speed: float = 2.5,
           length: int = MAX_EPISODE_STEPS) -> EpisodeMetric:
    """Eine abgeschlossene Episode.

    Humanoid terminiert beim Sturz: `length` ist deshalb variabel, und
    „durchgehalten" heißt genau, dass die vollen 1000 Schritte erreicht wurden.
    """
    survived = length >= MAX_EPISODE_STEPS
    return EpisodeMetric(episode, reward, length, survived, not survived,
                         reward >= TARGET_RETURN, episode * length,
                         speed, speed * length * 0.015, 0.4, 0.5, -3.0, -0.1)


def evaluation(mean_reward: float = 1234.5, length: float = 1000.0) -> EvaluationResult:
    return EvaluationResult(5, mean_reward, 20.0, length, 0.8, 0.2, 0.0,
                            1.5, 75.0, 0.4, -3.0, -0.1)


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
        self.assertEqual(RENDER_FPS, 67)
        # Der Standardwert muss selbst im gültigen Bereich liegen.
        self.assertLessEqual(MIN_ANIMATION_FPS, RENDER_FPS)
        self.assertGreaterEqual(MAX_ANIMATION_FPS, RENDER_FPS)

    def test_german_numbers_use_comma_and_dot(self):
        self.assertEqual(german(412.7), "412,7")
        self.assertEqual(german(3800.0, 0), "3.800")
        self.assertEqual(slugify("SAC Profil"), "sac-profil")

    def test_entry_point_exposes_main_without_starting_a_window(self):
        import humanoid_app

        self.assertTrue(callable(humanoid_app.main))
        self.assertIs(humanoid_app.HumanoidGUI, HumanoidGUI)


class PanelGridTests(unittest.TestCase):
    """Höchstens zwei Spalten und zwei Zeilen, alle Zellen gleich groß."""

    def test_single_panel_fills_the_whole_area(self):
        """Eine Anzeige bekommt eine Spalte und eine Zeile - und damit die ganze
        Flaeche. Eine Spannweite ueber Nachbarzellen braucht es nicht mehr, seit
        die Spaltenzahl der Flaeche folgt."""
        self.assertEqual(panel_grid_positions(1, best_columns(1, 800, 400, 72)),
                         [(0, 0, 1)])

    def test_two_panels_stand_side_by_side(self):
        self.assertEqual(panel_grid_positions(2), [(0, 0, 1), (0, 1, 1)])

    def test_three_panels_keep_equal_cells(self):
        self.assertEqual(panel_grid_positions(3, 2), [(0, 0, 1), (0, 1, 1), (1, 0, 1)])
        self.assertEqual(panel_grid_positions(3, 3), [(0, 0, 1), (0, 1, 1), (0, 2, 1)])
        # Gleich grosse Zellen: Eine Spannweite braechte bei quadratischen
        # Frames nichts, weil sie hoehenbegrenzt sind.
        self.assertTrue(all(span == 1 for _, _, span in panel_grid_positions(3, 2)))

    def test_the_arrangement_follows_the_shape_of_the_area(self):
        """Feste zwei mal zwei verschenkt Platz, sobald der Bereich nicht
        zufaellig dasselbe Seitenverhaeltnis hat. Quadratische Frames sind in
        einem breiten, flachen Bereich hoehenbegrenzt: Mehr Zeilen kosten
        unmittelbar, mehr Spalten nicht."""
        # Breit und flach: nebeneinander.
        self.assertEqual(best_columns(3, 1200, 400, 72), 3)
        self.assertEqual(best_columns(4, 1200, 400, 72), 4)
        # Schmal und hoch: uebereinander.
        self.assertEqual(best_columns(3, 400, 1200, 72), 1)
        # Eine Anzeige bekommt immer die ganze Flaeche.
        self.assertEqual(best_columns(1, 800, 400, 72), 1)

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
        self.assertEqual(HumanoidGUI.LINE_WIDTH, 1.2)
        self.assertAlmostEqual(HumanoidGUI.RAW_ALPHA, 0.10)


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
        patcher = patch.object(HumanoidGUI, "_show_initial_frame", lambda self, slot=0: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = HumanoidGUI(self.root)
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
    def test_three_slots_are_the_default_one_per_algorithm(self):
        """Die Aufgabe teilt genau drei Verfahren zu; jedes kommt einmal vor."""
        self.assertEqual(DEFAULT_SLOT_COUNT, 3)
        self.assertEqual(self.app.slot_count, DEFAULT_SLOT_COUNT)
        self.assertEqual(len(self.app.parameter_tabs.tabs()), DEFAULT_SLOT_COUNT)
        algorithms = [self.app.algorithm_vars[slot].get() for slot in range(DEFAULT_SLOT_COUNT)]
        self.assertEqual(algorithms, list(DEFAULT_SLOT_ALGORITHMS[:DEFAULT_SLOT_COUNT]))
        self.assertEqual(set(algorithms), set(ALGORITHMS))
        # Kein Algorithmus doppelt: Die Regel zum abweichenden Startwert bei
        # doppelter Belegung greift erst beim vierten Slot.
        self.assertEqual(len(algorithms), len(set(algorithms)))

    def test_the_repeated_algorithm_starts_with_a_visible_difference(self):
        """Zwei identisch konfigurierte Slots zeigten sonst keinen Unterschied."""
        repeated = [slot for slot in range(MAX_SLOTS)
                    if DEFAULT_SLOT_ALGORITHMS.count(DEFAULT_SLOT_ALGORITHMS[slot]) > 1]
        self.assertTrue(repeated)
        configs = [HumanoidGUI.initial_config(slot) for slot in repeated]
        self.assertTrue(config_differences(configs))

    def test_the_configurator_offers_no_evaluation_settings(self):
        """Die Summary mittelt über die letzten Episoden – eine gesonderte
        deterministische Evaluation braucht die Oberfläche nicht mehr."""
        for absent in ("evaluation_interval", "evaluation_episodes"):
            self.assertFalse(hasattr(self.app, absent))
        labels = [button.cget("text") for button in self.app.buttons]
        self.assertEqual(labels, ["Training starten / fortsetzen",
                                  "Vergleich starten / fortsetzen",
                                  "Stoppen", "Zurücksetzen"])

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
        with patch("humanoid_gui.messagebox.askyesno") as ask:
            self._set_slot_count(2)
            ask.assert_not_called()
        self.assertEqual(self.app.slot_count, 2)
        self.assertEqual(len(self.app.parameter_tabs.tabs()), 2)

    def test_shrinking_asks_before_discarding_a_learning_state(self):
        self.app.slots[2].history.append(metric(1, 100.0))
        with patch("humanoid_gui.messagebox.askyesno", return_value=False) as ask:
            self._set_slot_count(2)
            ask.assert_called_once()
        self.assertEqual(self.app.slot_count, DEFAULT_SLOT_COUNT)
        self.assertTrue(self.app.slots[2].history)
        with patch("humanoid_gui.messagebox.askyesno", return_value=True):
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
                       if line.get_linewidth() == HumanoidGUI.LINE_WIDTH
                       and str(line.get_label()).startswith("V")]
        self.assertEqual(len(highlighted), MAX_SLOTS)
        for slot, line in enumerate(highlighted):
            self.assertEqual(line.get_color(), SLOT_COLORS[slot])
            self.assertEqual(line.get_linestyle(), "-")
            self.assertTrue(line.get_label().startswith(SLOT_SHORT[slot]))

    def test_no_graph_shows_an_evaluation_series(self):
        """Es gibt keine Zwischenevaluation mehr – nur den Episoden-Return."""
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
            self.assertAlmostEqual(line.get_alpha(), HumanoidGUI.RAW_ALPHA)

    def test_the_target_line_is_white_dashed_and_not_called_solved(self):
        self.app._refresh_comparison_plot()
        self.app._refresh_single_plot()
        for axes in (self.app.axes, self.app.comparison_axes, self.app.single_axes):
            labels = [str(line.get_label()) for line in axes.get_lines()]
            # Humanoid-v5 fuehrt keinen reward_threshold: nirgends "geloest".
            self.assertNotIn("Gelöst", " ".join(labels))
            reference = [line for line in axes.get_lines()
                         if "Zielmarke" in str(line.get_label())]
            self.assertEqual(len(reference), 1)
            self.assertEqual(reference[0].get_color(), REFERENCE_COLOR)
            self.assertEqual(reference[0].get_linestyle(), "--")
            self.assertAlmostEqual(reference[0].get_ydata()[0], TARGET_RETURN)

    def test_the_axis_follows_the_data_instead_of_the_far_away_threshold(self):
        self.app.slots[0].comparison_history.extend(
            metric(index, 100.0 + index) for index in range(1, 40))
        self.app._refresh_comparison_plot()
        low, high = self.app.comparison_axes.get_ylim()
        self.assertLess(high, TARGET_RETURN,
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

    def test_training_summary_names_slot_and_algorithm_without_a_heading(self):
        """Ohne Titelzeile: Die Spaltenueberschrift nennt den Slot, die Zeile
        'Algorithmus' das Verfahren. Zwei gesparte Zeilen entscheiden darueber,
        ob die Tabelle ohne Scrollen ins Feld passt."""
        self._select(0)
        self.app.slots[0].history.append(metric(1, 100.0))
        self.app._training_summary()
        text = self.app._summary_text()
        self.assertNotIn("Training –", text)
        self.assertIn(SLOT_LABELS[0], text)
        self.assertIn("Algorithmus", text)
        self.assertIn(self.app.slots[0].algorithm, text)

    def test_comparison_summary_has_one_column_per_active_slot(self):
        self._set_slot_count(MAX_SLOTS)
        for slot in range(MAX_SLOTS):
            self.app.slots[slot].comparison_history.append(metric(1, 250.0 + slot))
        self.app._comparison_summary()
        text = self.app._summary_text()
        for slot in range(MAX_SLOTS):
            self.assertIn(SLOT_LABELS[slot], text)
        for label in ("Durchhaltequote", "Sturzquote", "Zielquote", "Tempo m/s",
                      "Strecke x m", "seitlich m", "Länge",
                      "Budget N/E", "Ende durch"):
            self.assertIn(label, text)
        # Unterhalb der Zwischenueberschrift steht kein Mittelwertzeichen mehr.
        for absent in ("Ø Tempo", "Ø Länge", "Ø Return", "Vorwärtsquote", "Gelöst-Quote"):
            self.assertNotIn(absent, text)

    def test_differing_parameters_become_table_rows(self):
        """Als Zeile statt als eigener Block: Die Tabelle ist ohnehin
        'Bezeichnung + ein Wert je Slot' - genau das, was ein Block darunter
        nur wiederholen wuerde. Das spart vier Zeilen."""
        self._set_slot_count(3)
        for slot in range(3):
            self.app.algorithm_vars[slot].set("TD3")
            self.app._algorithm_changed(slot)
            self.app.slots[slot].workbench.config = dataclasses.replace(
                self.app.slots[slot].workbench.config, seed=slot)
            self.app.slots[slot].comparison_history.extend(
                metric(index, 100.0) for index in range(1, 4))
        rows = self.app._difference_rows(
            [self.app.slots[slot].workbench.config for slot in range(3)])
        self.assertEqual([name for name, _ in rows], ["Zufallsstart s"])
        self.assertEqual(rows[0][1], ["0", "1", "2"])
        self.app._comparison_summary()
        text = self.app._summary_text()
        # Die Zeile steht direkt unter dem Algorithmus, in denselben Spalten.
        zeilen = [z for z in text.splitlines() if z.strip()]
        namen = [z.split()[0] for z in zeilen]
        self.assertIn("Zufallsstart", " ".join(namen))
        self.assertLess(namen.index("Zufallsstart"), namen.index("Episoden"))
        # Kein gesonderter Block mehr.
        self.assertNotIn("Unterschiede", text)

    def test_summary_lists_the_best_episode_with_number_and_return(self):
        history = [metric(1, 100.0), metric(2, 980.5), metric(3, 340.0)]
        self.app.slots[0].history.extend(history)
        self._select(0)
        self.app._training_summary()
        text = self.app._summary_text()
        self.assertIn("Beste Episode", text)
        self.assertIn("#2: 980,5", text)
        self.assertEqual(HumanoidGUI.best_episode(history).episode, 2)
        self.assertIsNone(HumanoidGUI.best_episode([]))

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

    def test_training_summary_averages_over_the_smoothing_window(self):
        """Gemittelt wird über die letzten X Episoden, X = Glättung."""
        self.app.slots[0].history.extend(metric(index, float(index)) for index in range(1, 101))
        self.app.smoothing.set("10")
        self.app._smoothing_changed()
        self._select(0)
        self.app._training_summary()
        text = self.app._summary_text()
        self.assertIn("Ø der letzten 10 Episoden", text)
        # Mittel der Episoden 91..100 ist 95,5 - nicht 50,5 wie über alle.
        self.assertIn("95,5", text)
        self.assertNotIn("50,5", text)
        # Die beste Episode zaehlt weiter ueber ALLE Episoden.
        self.assertIn("#100: 100,0", text)
        self.assertNotIn("Evaluation", text)

    def test_summary_window_shrinks_to_the_available_episodes(self):
        self.app.slots[0].history.extend(metric(index, 10.0) for index in range(1, 4))
        self.app.smoothing.set("20")
        self.app._smoothing_changed()
        self._select(0)
        self.app._training_summary()
        self.assertIn("Ø der letzten 3 Episoden", self.app._summary_text())

    def test_difference_rows_only_when_an_algorithm_is_used_twice(self):
        """Bei lauter verschiedenen Verfahren entfaellt er - sonst muesste man
        scrollen, um die Tabelle zu sehen."""
        self._set_slot_count(3)
        for slot in range(3):
            self.app.slots[slot].comparison_history.extend(
                metric(index, 100.0) for index in range(1, 4))
        self.app._comparison_summary()
        # Verschiedene Verfahren: keine zusaetzliche Unterschiedszeile noetig.
        self.assertEqual(self.app._difference_rows(
            [self.app.slots[s].workbench.config for s in range(3)]), [])
        # Derselbe Algorithmus mehrfach: Block erscheint und nennt den Grund.
        for slot in range(3):
            self.app.algorithm_vars[slot].set("SAC")
            self.app._algorithm_changed(slot)   # setzt den Slot zurueck
        for slot in range(3):
            self.app.values[slot]["learning_rate"].set(str(1e-4 * (slot + 1)))
            # Die Summary liest die Slot-Konfiguration, nicht die Eingabefelder.
            self.app.slots[slot].workbench.config = self.app._config(slot)
            self.app.slots[slot].comparison_history.extend(
                metric(index, 100.0) for index in range(1, 4))
        self.app._comparison_summary()
        text = self.app._summary_text()
        self.assertIn("Lernrate α", text)
        self.assertIn("0.0003", text)


class CaptionAndHoverTests(GUITestCase):
    def _prepare(self, slot: int = 0) -> None:
        observation = np.linspace(-1.0, 1.0, OBSERVATION_DIM)
        observation[0] = 1.35            # Rumpfhöhe, im gesunden Bereich
        observation[1:5] = [1.0, 0.0, 0.0, 0.0]   # aufrecht
        self.app.animation_observation[slot] = observation
        action = np.linspace(-0.4, 0.4, ACTION_DIM)
        action[2] = 0.0
        self.app.animation_action[slot] = action
        self.app.animation_info[slot] = {
            "reward_survive": 5.0, "reward_forward": 1.5, "reward_ctrl": -0.28,
            "reward_contact": -0.01, "x_position": 6.25, "y_position": -0.4,
            "x_velocity": 1.2, "y_velocity": -0.1, "distance_from_origin": 6.26,
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
        self.assertIn("Rumpfhöhe", text)
        self.assertIn("Neigung", text)
        self.assertIn("Quaternion", text)
        # Alle vier Rewardanteile, der Überlebensbonus eingeschlossen.
        for part in ("Überleben", "vorwärts", "Steuerung", "Kontakt"):
            self.assertIn(part, text)
        # Jedes Gelenk zeigt seine eigene Übersetzung, kein gemeinsamer Faktor.
        for gear in {int(value) for value in GEARS}:
            self.assertIn(str(gear), text)
        # Alle fünf Gruppen, damit 17 Gelenke lesbar bleiben.
        for name, _ in ACTUATOR_GROUPS:
            self.assertIn(name, text)
        # Der Wertebereich ist ±0,4 – kein Clipping-Hinweis auf ±10.
        self.assertIn("±0.4", text)
        self.assertNotIn("±10", text)
        self.assertIn("kein Clipping", text)
        # Die 286 Werte aus cinert/cvel/cfrc_ext erscheinen nur verdichtet.
        self.assertIn("Σcfrc²", text)

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
                if line.get_linewidth() == HumanoidGUI.LINE_WIDTH
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
        self.assertEqual(EPISODE_CHOICES,
                         (CURRENT_EPISODE, BEST_EPISODE, BEST_POLICY, INACTIVE_EPISODE))
        for slot in range(MAX_SLOTS):
            self.assertEqual(self.app.episode_choice[slot].get(), CURRENT_EPISODE)
            self.assertTrue(self.app._slot_animated(slot))

    def test_inactive_hides_only_that_animation(self):
        """`inaktiv` blendet genau eine Anzeige aus; die uebrigen bekommen den
        Platz. Die Wahl steht neben dem Verfahren, nicht ueber dem Bild."""
        self._set_slot_count(3)
        self._show([0, 1, 2])
        self.assertEqual(self.app._panel_slots(), [0, 1, 2])
        self.app.episode_choice[1].set(INACTIVE_EPISODE)
        self.app._episode_choice_changed(1)
        self.root.update_idletasks()
        self.assertFalse(self.app._slot_animated(1))
        self.assertTrue(self.app._slot_animated(0))
        self.assertTrue(self.app._slot_animated(2))
        self.assertEqual(self.app._panel_slots(), [0, 2])
        self.assertFalse(self.app.env_frames[1].winfo_ismapped())
        self.assertTrue(self.app.env_frames[0].winfo_ismapped())
        self.app.busy = True
        self.app.comparison_running = True
        try:
            self.assertEqual(self.app._live_slots(), [0, 2])
        finally:
            self.app.busy = False
            self.app.comparison_running = False

    def test_the_choice_sits_next_to_the_method_not_above_the_image(self):
        for slot in range(self.app.slot_count):
            combo = self.app.episode_combos[slot]
            self.assertEqual(str(combo.master), str(self.app.slot_count_combo.master))
            # Die Slotzeilen stehen UNTER den globalen Einstellungen.
            self.assertEqual(combo.grid_info()["row"], SELECTION_ROW_OFFSET + slot)
            self.assertEqual(combo.grid_info()["column"], 2)
        # Unter dem Bild bleibt nur die eine Zeile mit E, S und R.
        kinder = self.app.env_frames[0].winfo_children()
        self.assertEqual(len(kinder), 2, [str(k) for k in kinder])

    def test_each_animation_chooses_independently(self):
        self._set_slot_count(MAX_SLOTS)
        self.app.episode_choice[2].set(BEST_EPISODE)
        self.app._episode_choice_changed(2)
        self.assertEqual(self.app.episode_choice[2].get(), BEST_EPISODE)
        for slot in (0, 1, 3):
            self.assertEqual(self.app.episode_choice[slot].get(), CURRENT_EPISODE)

    def test_every_episode_has_the_same_length(self):
        """Humanoid terminiert beim Sturz: Die Länge schwankt und ist damit
        selbst eine Kennzahl – anders als bei HalfCheetah."""
        short, full = metric(1, 100.0, length=37), metric(2, 5100.0)
        self.assertEqual(short.length, 37)
        self.assertTrue(short.fell)
        self.assertFalse(short.survived)
        self.assertTrue(full.survived)
        self.assertFalse(full.fell)
        self.app.slots[0].history.extend([short, full])
        self._select(0)
        self.app._training_summary()
        self.assertIn("Länge", self.app._summary_text())

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
        self.assertEqual(self.app.fps.get(), str(DEFAULT_ANIMATION_FPS))
        self.assertEqual(self.app._animation_settings(), DEFAULT_ANIMATION_FPS)

    def test_episode_budget_zero_is_accepted_as_unlimited(self):
        self.app.values[0]["episodes"].set("0")
        self.assertEqual(self.app._config(0).episodes, 0)

    def test_negative_episode_budget_names_the_field(self):
        self.app.values[0]["episodes"].set("-3")
        with self.assertRaises(ValueError) as error:
            self.app._config(0)
        self.assertIn("Episoden", str(error.exception))

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
            self.assertEqual(config, HumanoidGUI.initial_config(slot))


class SingleChartTests(GUITestCase):
    """Einzelgraph je Verfahren – Workbench 8.9."""

    def _fill_comparison(self, slots: list[int]) -> None:
        for slot in slots:
            self.app.slots[slot].comparison_history.extend(
                metric(index, 100.0 * (slot + 1) + index, length=40 + index)
                for index in range(1, 6))

    def test_the_third_tab_shows_exactly_one_method(self):
        labels = [self.app.charts.tab(index, "text")
                  for index in range(self.app.charts.index("end"))]
        self.assertEqual(labels, ["Training", "Vergleich", "Einzelverfahren"])
        self._fill_comparison([0, 1, 2])
        self.app._sync_single_combo()
        self.app.single_slot_var.set(SLOT_LABELS[1])
        self.app._refresh_single_plot()
        curves = [line for line in self.app.single_axes.get_lines()
                  if "Zielmarke" not in str(line.get_label())]
        # Genau ein Slot: Rohkurve plus gleitender Durchschnitt.
        self.assertEqual(len(curves), 2)
        labelled = [line for line in curves if str(line.get_label()).startswith("Vergleich")]
        self.assertEqual(len(labelled), 1)
        self.assertIn(SLOT_SHORT[1], str(labelled[0].get_label()))
        self.assertEqual(labelled[0].get_color(), SLOT_COLORS[1])

    def test_single_chart_is_available_after_a_comparison_without_retraining(self):
        """Die Messdaten liegen bereits vor – es ist eine Frage der Darstellung."""
        self._fill_comparison([0, 1, 2])
        for slot in range(3):
            history, origin = self.app._slot_curve_source(slot)
            self.assertEqual(origin, "Vergleich")
            self.assertEqual(len(history), 5)
        self.assertIsNone(self.app.slots[0].workbench.model)

    def test_single_chart_falls_back_to_the_training_history(self):
        self.app.slots[0].history.extend(metric(index, 50.0) for index in range(1, 4))
        history, origin = self.app._slot_curve_source(0)
        self.assertEqual(origin, "Training")
        self.assertEqual(len(history), 3)

    def test_single_and_comparison_chart_share_axes_and_smoothing(self):
        self._fill_comparison([0, 1, 2])
        self.app.smoothing.set("7")
        self.app._smoothing_changed()
        self.app._refresh_comparison_plot()
        self.app._refresh_single_plot()
        self.assertEqual(self.app.smoothing_window, 7)
        self.assertEqual(self.app.single_axes.get_xlabel(),
                         self.app.comparison_axes.get_xlabel())
        self.assertEqual(self.app.single_axes.get_ylabel(),
                         self.app.comparison_axes.get_ylabel())
        self.assertEqual(self.app.single_axes.get_ylabel(), "Return")

    def test_the_selection_follows_the_active_slot_count(self):
        self._set_slot_count(2)
        self.app._sync_single_combo()
        self.assertEqual(list(self.app.single_combo.cget("values")),
                         list(SLOT_LABELS[:2]))
        self.assertIn(self.app.single_slot_var.get(), SLOT_LABELS[:2])

    def test_bulk_export_writes_one_file_per_active_slot(self):
        self._fill_comparison([0, 1, 2])
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(filedialog, "askdirectory", return_value=folder):
                self.app.export_each_slot()
            written = sorted(path.name for path in Path(folder).glob("*.png"))
        self.assertEqual(len(written), 3)
        for slot in range(3):
            short = SLOT_SHORT[slot].lower()
            self.assertTrue(any(f"_{short}-" in name for name in written),
                            f"Keine Datei für {SLOT_SHORT[slot]}: {written}")
        # Die Auswahl steht danach wieder, wo sie war.
        self.assertIn(self.app.single_slot_var.get(), SLOT_LABELS[:3])

    def test_bulk_export_skips_slots_without_data(self):
        self._fill_comparison([0])
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(filedialog, "askdirectory", return_value=folder):
                self.app.export_each_slot()
            written = list(Path(folder).glob("*.png"))
        self.assertEqual(len(written), 1)


class GlobalSettingTests(GUITestCase):
    def test_every_slot_starts_with_its_animation_on(self):
        """Die Animation ist der sichtbare Zweck der Anwendung. Einen globalen
        Schalter gibt es nicht mehr - die Wahl je Verfahren kennt `inaktiv`,
        und alle darauf zu stellen ist dasselbe."""
        self.assertFalse(hasattr(self.app, "animation_enabled"))
        for slot in range(MAX_SLOTS):
            self.assertEqual(self.app.episode_choice[slot].get(), CURRENT_EPISODE)
            self.assertTrue(self.app._slot_animated(slot))

    def test_frame_rate_default_is_slower_than_the_environment_rate(self):
        """20 FPS gegen 67 FPS des Environments: rund dreifache Zeitlupe."""
        self.assertEqual(self.app.fps.get(), str(DEFAULT_ANIMATION_FPS))
        self.assertLess(DEFAULT_ANIMATION_FPS, RENDER_FPS)
        self.assertLessEqual(MIN_ANIMATION_FPS, DEFAULT_ANIMATION_FPS)
        self.assertGreaterEqual(MAX_ANIMATION_FPS, DEFAULT_ANIMATION_FPS)

    def test_thread_setting_is_applied_and_falls_back_when_invalid(self):
        self.assertEqual(self.app.torch_threads.get(), str(DEFAULT_TORCH_THREADS))
        before = torch.get_num_threads()
        try:
            self.app.torch_threads.set("2")
            self.assertEqual(self.app._apply_threads(), 2)
            self.assertEqual(torch.get_num_threads(), 2)
            self.app.torch_threads.set("völlig falsch")
            self.assertEqual(self.app._apply_threads(), DEFAULT_TORCH_THREADS)
            self.assertEqual(self.app.torch_threads.get(), str(DEFAULT_TORCH_THREADS))
            self.assertIn("ungültig", self.app.status.get())
        finally:
            torch.set_num_threads(before)

    def test_both_budget_fields_are_offered_per_slot(self):
        for slot in range(self.app.slot_count):
            self.assertIn("total_timesteps", self.app.values[slot])
            self.assertIn("episodes", self.app.values[slot])
            self.assertEqual(self.app.values[slot]["episodes"].get(), str(DEFAULT_EPISODES))
            self.assertEqual(self.app.values[slot]["total_timesteps"].get(),
                             str(DEFAULT_TOTAL_TIMESTEPS))

    def test_summary_names_the_limit_that_ended_the_run(self):
        self.app.slots[0].history.extend(metric(index, 100.0) for index in range(1, 4))
        self.app.slots[0].workbench.stop_reason = "episodes"
        self._select(0)
        self.app._training_summary()
        text = self.app._summary_text()
        self.assertIn("Ende durch", text)
        self.assertIn(STOP_REASONS["episodes"], text)


class LiveAnimationTests(GUITestCase):
    """Regression: `_start_live_animation` griff auf eine Variable zu, die es
    in seinem Gueltigkeitsbereich nicht gab. Jeder Aufruf endete mit einem
    Fehler, die Animation startete also weder beim Training noch beim
    Vergleich - und nichts hat es bemerkt."""

    def test_a_comparison_starts_every_visible_animation(self):
        self._set_slot_count(3)
        self.app.busy = True
        self.app.comparison_running = True
        try:
            with patch.object(HumanoidGUI, "_live_episode", lambda self, slot: None):
                self.app._start_live_animation(self.app._live_slots())
            self.assertEqual([slot for slot in range(MAX_SLOTS)
                              if self.app.animation_live[slot]], [0, 1, 2])
            self.assertEqual(self.app._panel_slots(), [0, 1, 2])
        finally:
            self.app.busy = False
            self.app.comparison_running = False

    def test_inactive_slots_are_skipped_but_the_rest_starts(self):
        self._set_slot_count(3)
        self.app.episode_choice[1].set(INACTIVE_EPISODE)
        self.app.busy = True
        self.app.comparison_running = True
        try:
            with patch.object(HumanoidGUI, "_live_episode", lambda self, slot: None):
                self.app._start_live_animation(self.app._live_slots())
            self.assertEqual([slot for slot in range(MAX_SLOTS)
                              if self.app.animation_live[slot]], [0, 2])
        finally:
            self.app.busy = False
            self.app.comparison_running = False

    def test_animation_notes_never_overwrite_the_running_status(self):
        """Waehrend eines Laufs gehoert die Statuszeile dem Lauf: Sie nennt die
        beteiligten Verfahren und ihr Budget."""
        laeuft = "Läuft – Vergleich: V1 PPO gegen V2 TD3 über je 300.000 Schritte"
        self.app.status.set(laeuft)
        self.app.busy = True
        try:
            self.app._animation_note("Animation gestoppt – irgendein Grund")
            self.assertEqual(self.app.status.get(), laeuft)
        finally:
            self.app.busy = False
        self.app._animation_note("jetzt schon")
        self.assertEqual(self.app.status.get(), "jetzt schon")

    def test_switching_a_slot_to_inactive_keeps_the_status(self):
        self._set_slot_count(3)
        laeuft = "Läuft – Vergleich über je 300.000 Schritte"
        self.app.status.set(laeuft)
        self.app.busy = True
        try:
            self.app.episode_choice[1].set(INACTIVE_EPISODE)
            self.app._episode_choice_changed(1)
            self.assertEqual(self.app.status.get(), laeuft)
        finally:
            self.app.busy = False


class ManualWindowTests(GUITestCase):
    @staticmethod
    def _text_of(window: tk.Toplevel) -> tk.Text:
        for frame in window.winfo_children():
            for widget in frame.winfo_children():
                if isinstance(widget, tk.Text):
                    return widget
        raise AssertionError("kein Textfeld gefunden")

    @staticmethod
    def _scrollbars_of(window: tk.Toplevel) -> list:
        return [widget for frame in window.winfo_children()
                for widget in frame.winfo_children()
                if isinstance(widget, ttk.Scrollbar)]

    def _open(self) -> tk.Toplevel:
        self.app.instructions()
        # `update()` statt `update_idletasks()`: Letzteres verarbeitet die
        # Ereigniswarteschlange nicht, und das Fenster waere noch nicht
        # vollstaendig aufgebaut, wenn der Test Tasten schickt.
        self.root.update()
        windows = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(windows), 1)
        return windows[0]

    def test_the_manual_fits_on_the_screen(self):
        """Ein Meldungsdialog waechst mit dem Text, bis seine Schaltflaeche unter
        den Bildschirmrand rutscht - dann laesst er sich nicht mehr schliessen."""
        window = self._open()
        try:
            self.assertLessEqual(window.winfo_width(), self.root.winfo_screenwidth())
            self.assertLessEqual(window.winfo_height(),
                                 int(self.root.winfo_screenheight() * .85))
        finally:
            window.destroy()

    def test_the_lines_are_not_cut_off(self):
        window = self._open()
        try:
            view = self._text_of(window)
            schrift = tkfont.Font(font=view.cget("font"))
            zeilen = view.get("1.0", "end").split("\n")
            self.assertGreaterEqual(view.winfo_width(),
                                    max(schrift.measure(zeile) for zeile in zeilen))
        finally:
            window.destroy()

    def test_escape_closes_the_manual(self):
        """Das Fenster holt sich den Tastaturfokus selbst - ohne ihn kaeme kein
        Tastendruck an, und Escape bliebe wirkungslos, bis jemand hineinklickt."""
        window = self._open()
        self.assertTrue(window.bind("<Escape>"))
        window.event_generate("<Escape>", when="now")
        self.root.update()
        self.assertFalse([w for w in self.root.winfo_children()
                          if isinstance(w, tk.Toplevel) and w.winfo_exists()])

    def test_the_close_button_keeps_its_place(self):
        """Der expandierende Text darf die Schaltflaeche nicht aus einem knappen
        Fenster druecken - genau daran scheitert ein zu grosser Meldungsdialog."""
        window = self._open()
        try:
            window.geometry("420x240")
            self.root.update_idletasks()
            button = [w for w in window.winfo_children()
                      if isinstance(w, ttk.Button)][0]
            self.assertTrue(button.winfo_ismapped())
            self.assertLessEqual(button.winfo_y() + button.winfo_height(),
                                 window.winfo_height())
        finally:
            window.destroy()

    def test_the_manual_scrolls_instead_of_growing(self):
        window = self._open()
        try:
            self.assertTrue(self._scrollbars_of(window))
        finally:
            window.destroy()


class SelectionHeadingTests(GUITestCase):
    def test_every_choice_is_offered(self):
        """Vier Moeglichkeiten: laufender Stand, die beste Episode exakt
        nachgespielt, ihre Policy deterministisch, und aus."""
        self.assertEqual(EPISODE_CHOICES,
                         (CURRENT_EPISODE, BEST_EPISODE, BEST_POLICY, INACTIVE_EPISODE))
        for slot in range(self.app.slot_count):
            self.assertEqual(tuple(self.app.episode_combos[slot].cget("values")),
                             EPISODE_CHOICES)

    def test_the_animation_column_is_labelled(self):
        """Ohne Ueberschrift ist nicht erkennbar, worauf sich
        `aktuell`/`beste`/`inaktiv` bezieht."""
        selection = self.app.slot_count_combo.master
        beschriftungen = [w.cget("text") for w in selection.winfo_children()
                          if isinstance(w, ttk.Label)]
        self.assertIn("Animation", beschriftungen)
        kopf = [w for w in selection.winfo_children()
                if isinstance(w, ttk.Label) and w.cget("text") == "Animation"][0]
        self.assertEqual(kopf.grid_info()["column"], 2)
        self.assertEqual(kopf.grid_info()["row"], SELECTION_ROW_OFFSET - 1)
