"""Application entry point for the Model-Free Gridworld RL Lab."""

import tkinter as tk

from gridworld_gui import GridGUI
from gridworld_logic import Agent, GridWorld, POLICY_CLASSES


def main() -> None:
    root = tk.Tk()
    environment = GridWorld()
    policies = {name: policy_class(seed=42) for name, policy_class in POLICY_CLASSES.items()}
    agent = Agent(environment, policies["Monte Carlo"])
    GridGUI(root, environment, agent, policies)
    root.mainloop()


if __name__ == "__main__":
    main()
