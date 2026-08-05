# Model-Free Gridworld RL Lab

Eine lokale Tkinter-Anwendung zum interaktiven Lernen und Vergleichen von
tabellarischen Reinforcement-Learning-Verfahren.

## Zentrales Conda-Environment

Das Repository verwendet das gemeinsame Environment `rl-26-08`. Einmalig im
Repository-Root erstellen:

```bash
cd /Users/oliver/git/RL-26-08
conda env create -f environment.yml
```

Aktivieren und Anwendung starten:

```bash
conda activate rl-26-08
cd Oliver/03-gridworld_model_free
python gridworld_app.py
```

## Standard-Grid

- Größe: `5 x 3`
- Start: `(0, 2)`
- Ziel: `(4, 2)`
- blockiert: `(2, 1)` und `(2, 2)`
- Reward: `-1` pro normalem Schritt, `0` beim Ziel
- Ende: Ziel oder maximale Schrittzahl

Bewegungen sind deterministisch. Eine Bewegung gegen Rand oder Blockade lässt
den Agenten stehen, kostet aber weiterhin einen Schritt und Reward `-1`.

## Algorithmen

- **Every-Visit Monte Carlo Control:** aktualisiert `Q(s,a)` nach der Episode
  anhand diskontierter Returns und eines Sample Average.
- **SARSA:** On-Policy-TD-Verfahren; die für das Update ausgewählte nächste
  Aktion wird tatsächlich ausgeführt.
- **Expected SARSA:** verwendet den Erwartungswert der epsilon-greedy Policy.
- **Q-Learning:** Off-Policy-TD-Verfahren mit dem maximalen zukünftigen Q-Wert.

Jede Methode besitzt ihren eigenen standardmäßigen Epsilon-Verlauf. Alternativ
können gemeinsame benutzerdefinierte Epsilon-Parameter verwendet werden.

## Bedienung

- Die Pfeilbuttons oder Pfeiltasten steuern eine reine manuelle Demo, die keine
  Lernwerte verändert.
- `Einzelschritt` führt genau einen Trainingsschritt aus.
- `Eine Episode ausführen` animiert eine Trainingsepisode.
- `N Episoden trainieren` führt einen responsiven Trainingslauf aus.
- `Gelernte Policy ausführen` zeigt eine greedy Episode ohne weitere Updates.
- Value- und Q-Dialog erklären `V(s)` beziehungsweise alle vier `Q(s,a)`.
- Trajektorien lassen sich nach `exports/` als CSV exportieren.
- Return-Plots lassen sich nach `plots/` speichern.

## Methodenvergleich

Im Tab `Methoden vergleichen` können mehrere Verfahren über reproduzierbare,
unabhängige Wiederholungen verglichen werden. Angezeigt werden mittlere
Episode-Returns, ein gleitender Durchschnitt, optionale
95-%-Konfidenzintervalle, Episodenlänge und Erfolgsquote. Der Vergleich
verändert den interaktiven Agenten nicht.

## Tests

```bash
python -m unittest discover -s tests -v
```

Die Logiktests öffnen keine GUI.
