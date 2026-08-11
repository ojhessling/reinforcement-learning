"""Entry point for the CartPole RL workbench."""

import multiprocessing
import tkinter as tk

from cartpole_gui import CartPoleGUI


def main() -> None:
    root = tk.Tk()
    CartPoleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
