"""Entry point for the Hopper workbench."""

import tkinter as tk

from hopper_gui import HopperGUI


def main() -> None:
    root = tk.Tk()
    HopperGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
