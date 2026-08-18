"""Entry point for the BipedalWalker policy-gradient workbench."""

import tkinter as tk

from bipedalwalker_gui import BipedalWalkerGUI


def main() -> None:
    root = tk.Tk()
    BipedalWalkerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
