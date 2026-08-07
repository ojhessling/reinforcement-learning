# Taxi RL Workbench

Lokale Tkinter-Anwendung für Gymnasiums `Taxi-v3` mit Q-Learning, SARSA und
Expected SARSA.

## Environment

Der Agent steuert ein Taxi, holt einen Fahrgast ab und bringt ihn zum Ziel.
Actions sind South, North, East, West, Pickup und Dropoff. Jeder normale
Schritt liefert Reward `-1`, illegales Pickup oder Dropoff `-10` und eine
erfolgreiche Ablieferung `+20`. Sie beendet die Episode; nach spätestens 200
Schritten wird sie ohne Bootstrap abgebrochen.

Optionen:

- `Rainy`: Bewegungen können seitlich abweichen.
- `Fickle Passenger`: Der Fahrgast kann sein Ziel einmal ändern.
- `Action Mask verwenden`: Auswahl und Updates berücksichtigen nur gültige
  Actions. Ohne Maske lernt der Agent ungültige Actions über deren Reward.

Das Spielfeld ist ausschließlich Gymnasiums offizieller RGB-Frame.

## Methoden und Exploration

- Q-Learning verwendet den höchsten nächsten Q-Value.
- SARSA verwendet die tatsächlich gewählte nächste Action.
- Expected SARSA verwendet den Erwartungswert der Epsilon-greedy Policy.
- Epsilon bleibt konstant oder sinkt nach jeder Episode bis zum Minimum.

## Installation und Start

```bash
conda env update --name rl-26-08 --file Oliver/environment.yml
conda activate rl-26-08
cd Oliver/06-taxi
python taxi_app.py
```

## Bedienung

1. Methode, Exploration und Checkboxen wählen.
2. N Episoden trainieren.
3. Return-Kurve und Q-Tabelle untersuchen.
4. Gelernte Policy ohne Exploration und Lernupdates ausführen.

Parameter werden beim Start einer Aktion automatisch validiert. Änderungen an
Lern- oder Environment-Einstellungen setzen den Lernzustand zurück.
Massentraining wird nicht animiert.

## Tests

```bash
conda run --name rl-26-08 python -m unittest discover -s tests -v
```
