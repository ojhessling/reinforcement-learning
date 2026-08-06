"""Entry point for the minimal CliffWalking workbench."""

import tkinter as tk

from cliff_walking_gui import CliffWalkingGUI


def main() -> None:
    root = tk.Tk()
    CliffWalkingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
