"""Entry point for the LunarLander policy-gradient workbench."""

import tkinter as tk

from lunarlander_pg_gui import LunarLanderPGGUI


def main() -> None:
    root = tk.Tk()
    LunarLanderPGGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
