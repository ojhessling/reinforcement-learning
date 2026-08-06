# CliffWalking RL Workbench – Minimalversion

Die lokale Tkinter-Anwendung demonstriert drei tabellarische
Reinforcement-Learning-Verfahren im Gymnasium-Environment
`CliffWalking-v1`:

- Q-Learning
- SARSA
- Expected SARSA

Ein normaler Schritt liefert Reward `-1`. Ein Schritt in das Cliff liefert
`-100` und setzt den Agenten zurück zum Start. Erst das Ziel beendet die
Episode. `Max. Schritte` verhindert unendliche Episoden.

## Installation

Vom Repository-Root:

```bash
conda env update --name rl-26-08 --file Oliver/environment.yml
conda activate rl-26-08
```

## Start

```bash
cd Oliver/04-cliffwalking
python cliff_walking_app.py
```

## Bedienung

1. Methode und Parameter auswählen.
2. `N Episoden trainieren` starten. Geänderte Einstellungen werden automatisch
   übernommen; strukturelle Änderungen setzen das Training zurück.
3. Return-Kurve und Policy-Pfeile beobachten.
4. Mit `Gelernte Policy ausführen` eine greedy Episode ohne Lernupdates
   animieren.
5. Über `Q-Tabelle öffnen` gelernte und unbesuchte Zustände untersuchen.

Die Checkbox `Slippery` aktiviert die stochastische Gymnasium-Variante.

## Methoden

Q-Learning ist Off-Policy und verwendet für sein Update die beste bekannte
nächste Action. SARSA ist On-Policy und berücksichtigt die tatsächlich unter
Exploration gewählte nächste Action. Expected SARSA verwendet stattdessen den
Erwartungswert über die aktuelle Epsilon-greedy Policy.

## Tests

Vom Projektordner:

```bash
conda run --name rl-26-08 python -m unittest discover -s tests -v
```

Die Minimalversion enthält bewusst noch kein DQN, Double DQN, Speichern/Laden
oder einen automatischen Methodenvergleich. Diese Funktionen sind als zweite
Ausbaustufe im `prompt.md` beschrieben.
