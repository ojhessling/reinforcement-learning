"""Entry point for the HalfCheetah workbench."""

import tkinter as tk

from halfcheetah_gui import HalfCheetahGUI


def main() -> None:
    root = tk.Tk()
    HalfCheetahGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
