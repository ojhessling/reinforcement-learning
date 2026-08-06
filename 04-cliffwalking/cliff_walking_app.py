"""Entry point for the minimal CliffWalking workbench."""

import tkinter as tk

from cliff_walking_gui import CliffWalkingGUI
from cliff_walking_logic import CliffWalkingEnvironment


def main() -> None:
    environment = CliffWalkingEnvironment()
    # On macOS/Python 3.13 Gymnasium's Pygame renderer must initialize before
    # Tk creates its native application context.
    environment.render_rgb()
    root = tk.Tk()
    CliffWalkingGUI(root, environment=environment)
    root.mainloop()


if __name__ == "__main__":
    main()
