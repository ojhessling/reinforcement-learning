"""Entry point for the MountainCar workbench."""

import tkinter as tk

from mountaincar_gui import MountainCarGUI


def main() -> None:
    root = tk.Tk()
    MountainCarGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
