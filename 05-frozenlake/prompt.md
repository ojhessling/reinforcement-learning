# FrozenLake – RL-Workbench

Projektordner: `Oliver/05-frozenlake`

## Grundlage

Berücksichtige alle verbindlichen Regeln aus `../workbench.md`. Dieser Prompt
enthält nur die FrozenLake-spezifischen Anforderungen.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung für Gymnasiums `FrozenLake-v1`.

Auswahlmöglichkeiten:

- Lernmethode: `Q-Learning`, `SARSA` oder `Expected SARSA`
- Exploration: konstantes Epsilon oder Epsilon-Decay
- Karte: `4x4` oder `8x8`
- Checkbox `Slippery`

## Environment

Erzeuge das Environment entsprechend der Auswahl:

```python
gymnasium.make(
    "FrozenLake-v1",
    map_name=selected_map,
    is_slippery=is_slippery,
    render_mode="rgb_array",
)
```

Verwende die Reward- und Erfolgsregeln von Gymnasium unverändert. Reward
Schedule und Success Rate sind nicht einstellbar.

## Darstellung

Zeige ausschließlich den offiziellen Gymnasium-RGB-Frame aus:

`https://gymnasium.farama.org/environments/toy_text/frozen_lake/`

Bette den Frame in Tkinter ein. Erstelle keine eigene Spielfeldgrafik und
öffne kein separates Pygame-Fenster. Beachte die allgemeinen Animationsregeln
aus `../workbench.md`.

## Abnahme

Fertig, wenn beide Karten, Slippery-Modus, alle drei Lernmethoden und beide
Explorationsvarianten funktionieren, Gymnasiums Grafik verwendet wird und die
allgemeinen Tests und Abnahmekriterien aus `../workbench.md` erfüllt sind.
