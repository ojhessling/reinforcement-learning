"""Entry point for the model-based Gridworld application."""

import tkinter as tk

from gridworld_gui import GridGUI


def main() -> None:
    root = tk.Tk()
    GridGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
