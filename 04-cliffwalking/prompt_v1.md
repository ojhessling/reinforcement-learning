# CliffWalking – Minimalversion V1

Projektordner: `Oliver/04-cliffwalking`

## Grundlage

Berücksichtige alle verbindlichen Regeln aus `../workbench.md`, insbesondere zu:

- Sprache und Benennung
- Projektstruktur und Architektur
- Trennung von Training und Evaluation
- automatischer und atomarer Parameterübernahme
- responsiver Tkinter-GUI
- Reproduzierbarkeit
- Tabellen und Visualisierungen
- Tests, README und Abnahmekriterien

Dieser Prompt enthält nur die projektspezifischen Anforderungen.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung für Gymnasiums
`CliffWalking-v1` mit drei tabellarischen Methoden:

- `QLearning`
- `Sarsa`
- `ExpectedSarsa`

V1 verwendet keine neuronalen Netze, Stable-Baselines3, DQN, Double DQN,
Replay Buffer, Modellpersistenz oder automatischen Methodenvergleich.

## Dateien

```text
cliff_walking_app.py
cliff_walking_logic.py
cliff_walking_gui.py
README.md
requirements.txt
tests/test_cliff_walking_logic.py
```

## Environment

```text
gymnasium.make(
    "CliffWalking-v1",
    is_slippery=False,
)
```

Standardwerte:

- Grid: `4 x 12`
- Start: `(3, 0)`
- Ziel: `(3, 11)`
- Cliff: `(3, 1)` bis `(3, 10)`
- `max_steps`: `500`
- `seed`: `42`
- `is_slippery`: `False`

Gymnasium codiert Positionen als:

```text
observation = row * 12 + column
```

Actions in der Gymnasium-Reihenfolge:

- `0`: Up
- `1`: Right
- `2`: Down
- `3`: Left

Rewards und Episodenende:

- normaler Schritt: `-1`
- Cliff-Fall: `-100` und Rückkehr zum Start
- Cliff-Fall beendet die Episode nicht
- Ziel: `terminated=True`
- Max Steps: `truncated=True`

`CliffWalkingEnvironment` kapselt Gymnasium, konvertiert Observation und
Position und liefert eindeutige `StepResult`-Objekte.

## Policies

Gemeinsame Q-Tabelle:

```text
q_values[48][4]
```

Standardparameter:

- `alpha`: `0.5`
- `gamma`: `0.99`
- `epsilon_start`: `1.0`
- `epsilon_min`: `0.05`
- `epsilon_decay`: `0.995`

Epsilon-Decay nach jeder abgeschlossenen Trainingsepisode:

```text
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

Greedy Evaluation verwendet `epsilon=0` und verändert keine Lernwerte oder
Trainingsstatistiken.

### Q-Learning

```text
target = reward                         if done
target = reward + gamma * max(Q(s', a')) otherwise
Q(s, a) += alpha * (target - Q(s, a))
```

### SARSA

```text
target = reward                              if done
target = reward + gamma * Q(s', next_action) otherwise
Q(s, a) += alpha * (target - Q(s, a))
```

Die für das Update gewählte `next_action` wird im folgenden Schritt tatsächlich
ausgeführt.

### Expected SARSA

```text
target = reward                                  if done
target = reward + gamma * expected_q(next_state) otherwise
```

Der Erwartungswert verwendet die vollständige aktuelle Epsilon-greedy
Verteilung und behandelt gleich gute Actions gleichmäßig.

## Agent

`TabularAgent` verwaltet:

- Einzelschritte und vollständige Episoden
- SARSA-`next_action`
- aktuelle und letzte Trajektorie
- Returns und Episodenlängen
- Cliff-Fälle und Erfolge
- Training und isolierte greedy Evaluation

## GUI

### Einstellungen

- Methode
- Episoden
- Max. Schritte
- Alpha
- Gamma
- Epsilon Start, Min und Decay
- Random Seed
- Slippery
- Policy-Pfeile anzeigen

### Steuerung

- `Einzelschritt trainieren`
- `Eine Episode animieren`
- `N Episoden trainieren`
- `Training stoppen`
- `Gelernte Policy ausführen`
- `Training zurücksetzen`
- `Q-Tabelle öffnen`
- `Bedienungsanleitung`

### CliffWalking-Anzeige

- Start grün, Ziel gelb, Cliff rot
- Agent als blauer Kreis
- Observation-Nummern
- Policy-Pfeile; unbesuchte Zustände als `?`
- ausgeführter Weg als gut sichtbare blaue Pfeillinien
- ungültige Bewegung als violette Schleife
- Cliff-Bewegung rot und gestrichelt
- Cliff-Fall mit weißem `×`
- aktuelle, sonst zuletzt abgeschlossene Trajektorie
- höchstens die letzten 100 Übergänge

### Summary und Plot

Summary:

- Methode und Episoden
- aktuelles Epsilon
- Erfolge
- durchschnittlicher Return der letzten 20 Episoden
- Cliff-Fälle
- letzter Return und Status

Plot:

- dünne Linie für Episode-Returns
- hervorgehobener gleitender Durchschnitt über 20 Episoden

### Q-Tabelle

Zeige für alle 48 Observations:

- Observation, Row und Column
- Q Up, Right, Down und Left
- beste Actions
- Besuchszahl
- unbesuchte Werte als `—`

## Tests

Zusätzlich zu den allgemeinen Regeln aus `workbench.md` prüfe:

- Observation-/Positionsumrechnung
- Gymnasium-Action-Reihenfolge
- normale Schritte, Cliff-Fall, Ziel und Max Steps
- alle drei Updateformeln
- SARSA-`next_action`
- Expected-SARSA-Ties
- Epsilon-Grenzen
- Evaluation verändert Training nicht
- Statistik eines Trainingslaufs
- Q-Learning lernt mit festem Seed eine erfolgreiche Policy

## Start

```bash
conda activate rl-26-08
cd Oliver/04-cliffwalking
python cliff_walking_app.py
```

## Abnahme

V1 ist fertig, wenn das echte Gymnasium-Environment und alle drei Methoden
korrekt funktionieren, Cliff-Fälle und Wege sichtbar sind, Training die GUI
nicht dauerhaft blockiert und alle Tests erfolgreich sind.
