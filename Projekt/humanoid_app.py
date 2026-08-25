"""Einstiegspunkt der Humanoid-Workbench (Abschlussprojekt)."""

import tkinter as tk

from humanoid_gui import HumanoidGUI
from humanoid_logic import set_torch_threads


def main() -> None:
    # Ohne diesen Aufruf startet PyTorch so viele Threads wie Kerne und
    # blockiert sich auf Effizienzkernen selbst; die GUI kann den Wert ändern.
    set_torch_threads()
    root = tk.Tk()
    HumanoidGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
