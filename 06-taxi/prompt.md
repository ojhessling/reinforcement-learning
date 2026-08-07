# Taxi – RL-Workbench

Projektordner: `Oliver/06-taxi`

## Grundlage

Berücksichtige alle verbindlichen Regeln aus `../workbench.md`. Dieser Prompt
enthält nur die Taxi-spezifischen Anforderungen.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung für Gymnasiums `Taxi-v3`.

Auswahlmöglichkeiten:

- Lernmethode: `Q-Learning`, `SARSA` oder `Expected SARSA`
- Exploration: konstantes Epsilon oder Epsilon-Decay
- Checkbox `Rainy`, Standard `False`
- Checkbox `Fickle Passenger`, Standard `False`
- Checkbox `Action Mask verwenden`, Standard `False`

## Environment

```python
gymnasium.make(
    "Taxi-v3",
    is_rainy=is_rainy,
    fickle_passenger=fickle_passenger,
    render_mode="rgb_array",
)
```

Actions:

- `0`: South
- `1`: North
- `2`: East
- `3`: West
- `4`: Pickup
- `5`: Dropoff

Rewards und Episodenende bleiben unverändert:

- normaler Schritt: `-1`
- erfolgreiche Ablieferung: `+20` und `terminated=True`
- illegales Pickup oder Dropoff: `-10`
- Zeitlimit: `200` Schritte

Das Environment besitzt 500 codierte Zustände. Zeige Taxi-Position,
Fahrgastposition und Ziel zusätzlich in verständlicher Form an.

Bei deaktivierter Action Mask darf der Agent alle sechs Actions auswählen und
lernt ungültige Aktionen über deren Reward. Bei aktivierter Action Mask dürfen
Exploration, Greedy-Auswahl und Bootstrap nur die in `info["action_mask"]`
erlaubten Actions berücksichtigen.

## Darstellung

Zeige ausschließlich den offiziellen Gymnasium-RGB-Frame aus:

`https://gymnasium.farama.org/environments/toy_text/taxi/`

Bette den Frame in Tkinter ein. Erstelle keine eigene Spielfeldgrafik und
öffne kein separates Pygame-Fenster. Beachte die Animationsregeln aus
`../workbench.md`.

## Abnahme

Fertig, wenn alle drei Lernmethoden, beide Explorationsvarianten und alle drei
Checkboxen funktionieren, Pickup und Dropoff korrekt gelernt werden,
Evaluation keine Lernwerte verändert, Gymnasiums Grafik verwendet wird und die
allgemeinen Tests und Abnahmekriterien aus `../workbench.md` erfüllt sind.
