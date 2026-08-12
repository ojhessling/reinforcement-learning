import tkinter as tk
import unittest

import numpy as np

from mountaincar_gui import MountainCarGUI, downsample_minmax, rolling_average


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
            app = MountainCarGUI(root)
            root.update()
            app._initialize_layout()
            root.update()
            self.assertEqual(app.layout_visibility_issues(), [])
        finally:
            if app is not None:
                app.close()
            else:
                root.destroy()


if __name__ == "__main__":
    unittest.main()
