"""Entry point for the Acrobot workbench."""

import tkinter as tk

from acrobot_gui import AcrobotGUI


def main() -> None:
    root = tk.Tk()
    AcrobotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
