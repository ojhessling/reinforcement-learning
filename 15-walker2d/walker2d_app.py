"""Entry point for the Walker2d workbench."""

import tkinter as tk

from walker2d_gui import Walker2dGUI


def main() -> None:
    root = tk.Tk()
    Walker2dGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
