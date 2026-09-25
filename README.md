# Reinforcement Learning — Workbenches

Sixteen interactive learning applications, from the multi-armed bandit to
Walker2d, plus a final project on `Humanoid-v5`. Each workbench is a Tkinter
interface for starting methods side by side, tuning their parameters and
watching them learn.

Built during course D21195UYS (AlfaTraining, August 2026), instructor Manfred
Messing.

## Final project: Humanoid-v5

Keep a three-dimensional humanoid figure of 42 kg upright and make it walk
forward. **PPO**, **TD3** and **SAC** compared head to head, 300,000 steps
each, every configuration run twice.

| Method | Mean return |
|---|---|
| **SAC** | **2,712** |
| PPO | 492 |
| TD3 | 407 |

SAC wins by a factor of 5.5 and is the only method to reach the 5,000 mark —
one episode upright for a full 1,000 steps. Between PPO and TD3 there is **no**
measurable difference worth trusting.

Two findings would have come out wrong with only a single run per
configuration: TD3 either learns decently or not at all depending on the random
seed, and in the parameter study the order of the two smaller learning rates
flips with the seed. The recommended value `3e−4` therefore wins not on the
highest return but on the most reliable one — its runs vary six times less.

Two outlooks beyond the graded comparison: given **7 million steps**, the same
SAC configuration walks at 1.95 m/s, survives 92 % of episodes and beats the
official benchmark of 6,232 with 6,718. And **CrossQ** reaches 4,435 at the
same 300,000 steps — at three times the compute.

→ [`Projekt/`](Projekt/) with report, slides, metrics and videos
· [`Projekt-Erweiterung/`](Projekt-Erweiterung/) — SAC against CrossQ and TQC,
three methods that differ in the critic alone

## The workbenches

| | | |
|---|---|---|
| [`01-bandit`](01-bandit/) | Multi-armed bandit | ε-greedy, UCB |
| [`02`](02-gridworld_model_based/)–[`03-gridworld`](03-gridworld_model_free/) | Gridworld | model-based and model-free |
| [`04-cliffwalking`](04-cliffwalking/) · [`05-frozenlake`](05-frozenlake/) · [`06-taxi`](06-taxi/) | Tabular methods | Q-learning, SARSA |
| [`07-cartpole`](07-cartpole/) · [`08-mountaincar`](08-mountaincar/) | First networks | DQN, DDQN |
| [`09-acrobot`](09-acrobot/) · [`10-lunarlander`](10-lunarlander/) | Rainbow DDQN | |
| [`11-lunarlander_policy_gradient`](11-lunarlander_policy_gradient/) | Policy gradient | |
| [`12-bipedalwalker`](12-bipedalwalker/) … [`15-walker2d`](15-walker2d/) | Continuous control | up to CMA-ES as a second class of method |

[`workbench.md`](workbench.md) is the shared specification behind all of them:
binding rules for structure, interface, tests and acceptance. Each folder adds
its own `prompt.md` with the environment-specific decisions.

Everything in this repository is written in German.

## Getting started

```bash
conda env create -f environment.yml
conda activate rl-26-08
cd Projekt && python humanoid_app.py
```

Tests per project with `python -m pytest tests -q`.

## License

[MIT](LICENSE). The environments used come from
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and
[MuJoCo](https://github.com/google-deepmind/mujoco) and carry their own
licenses.
