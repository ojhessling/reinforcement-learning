"""Entry point for the Multi-Armed Bandit demo."""

import tkinter as tk

from bandit_gui import BanditGUI


def main() -> None:
    root = tk.Tk()
    BanditGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
