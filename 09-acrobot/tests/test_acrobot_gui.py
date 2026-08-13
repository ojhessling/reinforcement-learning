import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from acrobot_gui import EXPORT_DIR, AcrobotGUI, downsample_minmax, rolling_average, slugify


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


class LayoutSmokeTest(unittest.TestCase):
    def test_start_layout_has_no_clipped_controls(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Kein grafisches Display verfügbar: {error}")
        app = None
        try:
            app = AcrobotGUI(root)
            root.update()
            app._initialize_layout()
            root.update()
            self.assertEqual(app.layout_visibility_issues(), [])
        finally:
            if app is not None:
                app.close()
            else:
                root.destroy()


class ExportTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Kein grafisches Display verfügbar: {error}")
        self.app = AcrobotGUI(self.root)
        self.root.update()

    def tearDown(self):
        self.app.close()

    def test_export_chart_writes_png_with_current_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "chart.png")
            with patch("acrobot_gui.filedialog.asksaveasfilename", return_value=path):
                self.app.export_chart()
            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 0)

    def test_export_summary_includes_live_summary_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "summary.txt")
            with patch("acrobot_gui.filedialog.asksaveasfilename", return_value=path):
                self.app.export_summary()
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn(self.app.summary.get(), text)
            self.assertIn("Konfiguration:", text)
            self.assertIn("algorithm: DDQN", text)

    def test_export_cancelled_by_user_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("acrobot_gui.filedialog.asksaveasfilename", return_value=""):
                self.app.export_chart()
                self.app.export_summary()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_exports_default_into_project_exports_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "chart.png")
            with patch("acrobot_gui.filedialog.asksaveasfilename", return_value=path) as mock_dialog:
                self.app.export_chart()
            self.assertEqual(mock_dialog.call_args.kwargs["initialdir"], EXPORT_DIR)
            self.assertTrue(EXPORT_DIR.is_dir())

    def test_chart_and_summary_share_matching_base_name_for_same_snapshot(self):
        suggested = []

        def fake_dialog(**kwargs):
            suggested.append(kwargs["initialfile"])
            return str(Path(tempfile.mkdtemp()) / kwargs["initialfile"])

        with patch("acrobot_gui.filedialog.asksaveasfilename", side_effect=fake_dialog):
            self.app.export_chart()
            self.app.export_summary()
        png_name, txt_name = suggested
        self.assertTrue(png_name.endswith(".png"))
        self.assertTrue(txt_name.endswith("_config.txt"))
        self.assertEqual(png_name[:-len(".png")], txt_name[:-len("_config.txt")])

    def test_slug_reflects_selected_algorithm(self):
        self.assertEqual(slugify("Rainbow DDQN"), "rainbow-ddqn")
        base = self.app._export_base_name()
        self.assertIn(slugify(self.app.workbench.config.algorithm), base)


if __name__ == "__main__":
    unittest.main()
