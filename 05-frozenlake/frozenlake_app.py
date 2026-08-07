"""Entry point for the FrozenLake workbench."""

import tkinter as tk

from frozenlake_gui import FrozenLakeGUI
from frozenlake_logic import FrozenLakeEnvironment


def main() -> None:
    environment = FrozenLakeEnvironment()
    environment.render_rgb()
    root = tk.Tk()
    FrozenLakeGUI(root, environment=environment)
    root.mainloop()


if __name__ == "__main__":
    main()
