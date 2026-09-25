# Reinforcement Learning — Workbenches

Sechzehn interaktive Lernanwendungen, vom Multi-Armed Bandit bis Walker2d, und
ein Abschlussprojekt zu `Humanoid-v5`. Jede Workbench ist eine Tkinter-Oberfläche,
in der sich Verfahren nebeneinander starten, parametrieren und beim Lernen
zusehen lassen.

Entstanden im Kurs D21195UYS (AlfaTraining, August 2026), Dozent Manfred Messing.

## Abschlussprojekt: Humanoid-v5

Eine dreidimensionale humanoide Figur von 42 kg aufrecht halten und vorwärts
laufen lassen. **PPO**, **TD3** und **SAC** im direkten Vergleich, je 300 000
Schritte, jede Konfiguration zweimal gefahren.

| Verfahren | Ø Return |
|---|---|
| **SAC** | **2 712** |
| PPO | 492 |
| TD3 | 407 |

SAC gewinnt mit Faktor 5,5 und erreicht als einziges die Zielmarke von 5 000 —
eine Episode über 1 000 Schritte aufrecht. Zwischen PPO und TD3 ist **kein**
belastbarer Unterschied messbar.

Zwei Befunde wären mit nur einem Durchgang je Konfiguration falsch geworden:
TD3 lernt je nach Zufallsstart ordentlich oder gar nicht, und in der
Parameterstudie dreht sich die Reihenfolge der beiden kleineren Lernraten mit
dem Seed. Der empfohlene Wert `3e−4` gewinnt deshalb nicht über den höchsten,
sondern über den verlässlichsten Return — seine Läufe streuen sechsmal weniger.

Zwei Ausblicke außerhalb des bewerteten Vergleichs: mit **7 Mio. Schritten**
geht dieselbe SAC-Konfiguration mit 1,95 m/s, hält in 92 % der Episoden durch
und übertrifft mit 6 718 den offiziellen Benchmark von 6 232. Und **CrossQ**
erreicht bei denselben 300 000 Schritten 4 435 — kostet dafür die dreifache
Rechenzeit.

→ [`Projekt/`](Projekt/) mit Bericht, Präsentation, Kennzahlen und Videos
· [`Projekt-Erweiterung/`](Projekt-Erweiterung/) — SAC gegen CrossQ und TQC,
drei Verfahren, die sich allein im Critic unterscheiden

## Die Workbenches

| | | |
|---|---|---|
| [`01-bandit`](01-bandit/) | Multi-Armed Bandit | ε-greedy, UCB |
| [`02`](02-gridworld_model_based/)–[`03-gridworld`](03-gridworld_model_free/) | Gridworld | modellbasiert und modellfrei |
| [`04-cliffwalking`](04-cliffwalking/) · [`05-frozenlake`](05-frozenlake/) · [`06-taxi`](06-taxi/) | Tabellarische Verfahren | Q-Learning, SARSA |
| [`07-cartpole`](07-cartpole/) · [`08-mountaincar`](08-mountaincar/) | Erste Netze | DQN, DDQN |
| [`09-acrobot`](09-acrobot/) · [`10-lunarlander`](10-lunarlander/) | Rainbow-DDQN | |
| [`11-lunarlander_policy_gradient`](11-lunarlander_policy_gradient/) | Policy Gradient | |
| [`12-bipedalwalker`](12-bipedalwalker/) … [`15-walker2d`](15-walker2d/) | Kontinuierliche Steuerung | bis CMA-ES als zweite Verfahrensklasse |

[`workbench.md`](workbench.md) ist die gemeinsame Spezifikation hinter allen
Projekten: verbindliche Regeln zu Aufbau, Oberfläche, Tests und Abnahme. Jeder
Ordner hat zusätzlich einen eigenen `prompt.md` mit den environmentspezifischen
Festlegungen.

## Start

```bash
conda env create -f environment.yml
conda activate rl-26-08
cd Projekt && python humanoid_app.py
```

Tests je Projekt mit `python -m pytest tests -q`.
