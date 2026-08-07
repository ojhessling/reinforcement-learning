# FrozenLake RL Workbench

Lokale Tkinter-Lernanwendung für Gymnasiums `FrozenLake-v1` mit Q-Learning,
SARSA und Expected SARSA.

## Environment

Wählbar sind die Karten `4x4` und `8x8` sowie der deterministische oder
`Slippery`-Modus. Gymnasiums Regeln bleiben unverändert: Das Ziel liefert
Reward `1`, alle anderen Übergänge `0`; Ziel und Löcher beenden die Episode.
Bei Erreichen von `Max. Schritte` wird die Episode ohne Bootstrap abgebrochen.

Das Spielfeld ist Gymnasiums offizieller RGB-Frame. Die App zeichnet kein
eigenes Grid.

## Methoden und Exploration

- Q-Learning verwendet den höchsten nächsten Q-Value.
- SARSA verwendet die tatsächlich ausgewählte nächste Action.
- Expected SARSA verwendet den Erwartungswert der Epsilon-greedy Policy.
- `Epsilon konstant` behält Epsilon bei.
- `Epsilon-Decay` reduziert Epsilon nach jeder Trainingsepisode bis zum
  Minimum.

## Installation und Start

Vom Repository-Root:

```bash
conda env update --name rl-26-08 --file Oliver/environment.yml
conda activate rl-26-08
cd Oliver/05-frozenlake
python frozenlake_app.py
```

## Bedienung

1. Karte, Slippery, Methode und Exploration wählen.
2. Episoden trainieren.
3. Erfolgsrate und Q-Tabelle untersuchen.
4. Die gelernte Policy ohne Exploration und Lernupdates ausführen.

Geänderte Lernparameter werden beim Start einer Aktion validiert und setzen
den Lernzustand zurück. Die Animation kann deaktiviert oder über das Intervall
in Millisekunden gesteuert werden. Massentraining wird nicht animiert.

Bei Slippery sind einzelne Ergebnisse zufällig; feste Seeds machen Läufe
weitgehend reproduzierbar.

## Tests

```bash
conda run --name rl-26-08 python -m unittest discover -s tests -v
```

Die Anwendung enthält bewusst keinen Reward Schedule, keine einstellbare
Success Rate und keine zusätzlichen Deep-Learning-Verfahren.
