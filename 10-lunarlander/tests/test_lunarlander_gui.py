import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lunarlander_gui import (
    ANIMATION_INTERVAL_MS,
    EXPORT_DIR,
    RENDER_FPS,
    LunarLanderGUI,
    downsample_minmax,
    rolling_average,
    slugify,
)
from lunarlander_logic import ALGORITHMS, DISTRIBUTIONAL_ALGORITHMS, MULTISTEP_ALGORITHMS, NOISY_ALGORITHMS


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

    def test_animation_interval_follows_the_environment_frame_rate(self):
        self.assertEqual(RENDER_FPS, 50)
        self.assertEqual(ANIMATION_INTERVAL_MS, 20)


class GUITestCase(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Kein grafisches Display verfügbar: {error}")
        self.app = LunarLanderGUI(self.root)
        self.root.update()

    def tearDown(self):
        self.app.close()


class LayoutSmokeTest(GUITestCase):
    def test_start_layout_has_no_clipped_controls(self):
        self.app._initialize_layout()
        self.root.update()
        self.assertEqual(self.app.layout_visibility_issues(), [])

    def test_both_chart_tabs_exist(self):
        self.assertEqual([self.app.charts.tab(index, "text") for index in self.app.charts.tabs()],
                         ["Training", "Vergleich"])

    def test_layout_stays_clean_on_every_chart_tab(self):
        self.app._initialize_layout()
        for index in range(len(self.app.charts.tabs())):
            with self.subTest(tab=index):
                self.app.charts.select(index)
                self.root.update()
                self.assertEqual(self.app.layout_visibility_issues(), [])

    def test_summary_is_scrollable_and_read_only(self):
        self.assertEqual(str(self.app.summary_text.cget("state")), "disabled")
        self.assertTrue(self.app.summary_text.cget("yscrollcommand"))
        self.assertTrue(self.app.summary_text.cget("xscrollcommand"))


class WorkbenchComplianceTests(GUITestCase):
    def test_no_animation_speed_input_exists(self):
        """Die Workbench verbietet ein Feld für die Animationsgeschwindigkeit."""
        forbidden = ("δt", "Δt", "intervall (ms)", "geschwindigkeit", "speed", "fps ")
        stack, labels = list(self.root.winfo_children()), []
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            try:
                text = str(widget.cget("text"))
            except tk.TclError:
                continue
            if text:
                labels.append(text.lower())
        offenders = [text for text in labels
                     if any(token in text for token in forbidden) and "animation zeigen" not in text]
        self.assertEqual(offenders, [])
        self.assertNotIn("animation_delay", self.app.values)

    def test_animation_can_be_switched_off(self):
        self.assertTrue(self.app.animation_enabled.get())
        self.app.animation_enabled.set(False)
        self.assertFalse(self.app.animation_enabled.get())

    def test_automatic_evaluation_interval_is_configurable(self):
        self.assertEqual(self.app.evaluation_interval.get(), "10000")
        episodes, interval = self.app._evaluation_settings()
        self.assertEqual((episodes, interval), (5, 10000))
        self.app.evaluation_interval.set("0")
        self.assertEqual(self.app._evaluation_settings()[1], 0)

    def test_invalid_evaluation_settings_name_field_and_range(self):
        self.app.evaluation_interval.set("-5")
        with self.assertRaises(ValueError) as context:
            self.app._evaluation_settings()
        self.assertIn("Eval-Intervall", str(context.exception))
        self.assertIn("Gültig", str(context.exception))

    def test_control_buttons_reflect_busy_state(self):
        self.app._set_busy(True, "Läuft – Test")
        self.assertEqual(str(self.app.stop_button.cget("state")), "normal")
        self.assertTrue(all(str(button.cget("state")) == "disabled"
                            for button in self.app.buttons if button is not self.app.stop_button))
        self.app._set_busy(False, "Bereit")
        self.assertEqual(str(self.app.stop_button.cget("state")), "disabled")


class VariantFieldTests(GUITestCase):
    def test_every_algorithm_enables_exactly_its_own_fields(self):
        from lunarlander_gui import FIELD_ALGORITHMS

        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                self.app.algorithm.set(algorithm)
                self.app._algorithm_fields()
                for name, entry in self.app.variant_entries.items():
                    expected = "normal" if algorithm in FIELD_ALGORITHMS[name] else "disabled"
                    self.assertEqual(str(entry.cget("state")), expected, f"{algorithm}/{name}")
                epsilon_state = "disabled" if algorithm in NOISY_ALGORITHMS else "normal"
                for entry in self.app.exploration_entries:
                    self.assertEqual(str(entry.cget("state")), epsilon_state)

    def test_rainbow_enables_multistep_and_c51_fields(self):
        self.app.algorithm.set("Rainbow DDQN")
        self.app._algorithm_fields()
        for name in ("multistep_n", "c51_atoms", "c51_v_min", "c51_v_max"):
            self.assertEqual(str(self.app.variant_entries[name].cget("state")), "normal")
        self.assertIn("Rainbow DDQN", MULTISTEP_ALGORITHMS & DISTRIBUTIONAL_ALGORITHMS)


class ExportTests(GUITestCase):
    def test_export_chart_writes_png_of_the_selected_tab(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "chart.png")
            with patch("lunarlander_gui.filedialog.asksaveasfilename", return_value=path):
                self.app.export_chart()
            self.assertGreater(Path(path).stat().st_size, 0)

    def test_export_summary_includes_summary_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "summary.txt")
            with patch("lunarlander_gui.filedialog.asksaveasfilename", return_value=path):
                self.app.export_summary()
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("Training", text)
            self.assertIn("Konfiguration:", text)
            self.assertIn("algorithm: DDQN", text)
            self.assertIn("evaluation_interval:", text)

    def test_exports_default_into_the_project_exports_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("lunarlander_gui.filedialog.asksaveasfilename",
                       return_value=str(Path(directory) / "chart.png")) as dialog:
                self.app.export_chart()
            self.assertEqual(dialog.call_args.kwargs["initialdir"], EXPORT_DIR)
            self.assertTrue(EXPORT_DIR.is_dir())

    def test_chart_and_summary_share_a_matching_base_name(self):
        suggested = []

        def fake_dialog(**kwargs):
            suggested.append(kwargs["initialfile"])
            return str(Path(tempfile.mkdtemp()) / kwargs["initialfile"])

        with patch("lunarlander_gui.filedialog.asksaveasfilename", side_effect=fake_dialog):
            self.app.export_chart()
            self.app.export_summary()
        png_name, txt_name = suggested
        self.assertEqual(png_name[:-len("_training.png")], txt_name[:-len("_config.txt")])

    def test_slug_reflects_the_selected_algorithm(self):
        self.assertEqual(slugify("Multi-Step DDQN"), "multi-step-ddqn")
        self.assertIn("ddqn", self.app._export_base_name())


if __name__ == "__main__":
    unittest.main()
