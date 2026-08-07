"""Entry point for the Taxi workbench."""

import tkinter as tk

from taxi_gui import TaxiGUI
from taxi_logic import TaxiEnvironment


def main() -> None:
    environment = TaxiEnvironment()
    environment.render_rgb()
    root = tk.Tk()
    TaxiGUI(root, environment=environment)
    root.mainloop()


if __name__ == "__main__":
    main()
