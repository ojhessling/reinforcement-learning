# CliffWalking – Minimalversion V1

Projektordner: `Oliver/04-cliffwalking`

## Grundlage und Ziel

Berücksichtige alle verbindlichen Regeln aus `../workbench.md`. Dieser Prompt
enthält nur die CliffWalking-spezifischen Anforderungen.

Erstelle eine lokale Tkinter-Lernanwendung für Gymnasiums
`CliffWalking-v1` mit:

- `QLearning`
- `Sarsa`
- `ExpectedSarsa`

V1 enthält keine neuronalen Netze, DQN-Verfahren, Modellpersistenz oder
automatischen Methodenvergleich.

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

```python
gymnasium.make("CliffWalking-v1", render_mode="rgb_array")
```

- Grid `4 x 12`, Start `(3, 0)`, Ziel `(3, 11)`
- Cliff `(3, 1)` bis `(3, 10)`
- Observation: `row * 12 + column`
- Actions: `0 Up`, `1 Right`, `2 Down`, `3 Left`
- normaler Schritt: Reward `-1`
- Cliff: Reward `-100`, Rückkehr zum Start, nicht terminal
- Ziel: `terminated=True`
- `max_steps=500`: `truncated=True`
- Seed: `42`

`CliffWalkingEnvironment` kapselt Gymnasium, Positionsumrechnung,
Episodenlimit und RGB-Rendering. `StepResult` unterscheidet `terminated`,
`truncated` und `done`. Bei Termination und Truncation wird nicht gebootstrapt.

## Lernen

Gemeinsame Q-Tabelle: `q_values[48][4]`.

Standardwerte:

- Alpha `0.5`
- Gamma `0.99`
- Epsilon Start `1.0`
- Epsilon Minimum `0.05`
- Epsilon Decay `0.995`

Nach jeder Trainingsepisode:

```text
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

Updates:

```text
Q-Learning:
target = reward if done else reward + gamma * max(Q(next_state))

SARSA:
target = reward if done else reward + gamma * Q(next_state, next_action)

Expected SARSA:
target = reward if done else reward + gamma * expected_q(next_state)

Q(state, action) += alpha * (target - Q(state, action))
```

SARSA führt seine vorbereitete `next_action` tatsächlich aus. Expected SARSA
verwendet die vollständige Epsilon-greedy-Verteilung. Ties werden zufällig,
gleichmäßig und reproduzierbar behandelt. Greedy Evaluation nutzt Epsilon `0`
und verändert weder Lernwerte noch Trainingsstatistiken.

## GUI

Einstellungen: Methode, Episoden, Max. Schritte, Alpha, Gamma, Epsilon Start,
Minimum und Decay, Seed, `Animation anzeigen` und Animationsintervall mit
Standardwert `10 ms`.

Steuerung:

- Einzelschritt trainieren
- eine Episode animieren
- N Episoden trainieren und stoppen
- gelernte Policy ohne Lernupdate ausführen
- Training zurücksetzen
- Q-Tabelle und Bedienungsanleitung öffnen

Zeige ausschließlich Gymnasiums offiziellen RGB-Frame skaliert in Tkinter.
Gemeint ist die Darstellung aus
`https://gymnasium.farama.org/environments/toy_text/cliff_walking/`.
Erstelle keine eigene Spielfeldgrafik, kein Canvas-Grid, Weg-Overlay oder
Policy-Pfeile. Aktualisiere den Frame nach `reset()` und jedem `step()`, erhalte
das Seitenverhältnis und zeige Observation, Position, Action, Reward und
Cliff-Fall zusätzlich als Textstatus. Nutze dafür Pillow und `pygame`, ohne ein
separates Pygame-Fenster zu öffnen.

Bei aktivierter Animation wird der Frame nach jedem Agentenschritt für das
eingestellte Intervall angezeigt. Bei deaktivierter Animation läuft die
Episode ohne Einzelbilder und zeigt nur das Endergebnis. Massentraining wird
niemals animiert.

Eine zyklische oder ungelernte Policy endet sicher nach `max_steps` und darf
die Anwendung nicht abbrechen.

Summary: Methode, Episoden, Epsilon, Erfolge, Cliff-Fälle, letzter Return,
Status und mittlerer Return der letzten 20 Episoden.

Plot: Episode-Returns als dünne Linie und gleitender Mittelwert über 20
Episoden als hervorgehobene Linie.

Die Q-Tabelle zeigt alle 48 Zustände, Position, vier Q-Werte, beste Actions und
Besuchszahl. Unbesuchte Werte erscheinen als `—`.

## Tests

Zusätzlich zu `../workbench.md` prüfe:

- Observation-/Positionsumrechnung und Action-Reihenfolge
- normalen Schritt, Cliff-Rücksprung, Ziel und Max-Steps-Truncation
- alle drei Updateformeln, Expected-SARSA-Ties und SARSA-`next_action`
- Epsilon-Grenzen und reproduzierbares Tie-Breaking
- Evaluation verändert den Lernzustand nicht
- ungelernte oder zyklische Policy endet sicher
- Q-Learning lernt mit festem Seed mindestens eine erfolgreiche Policy

## Start und Abnahme

```bash
conda activate rl-26-08
cd Oliver/04-cliffwalking
python cliff_walking_app.py
```

V1 ist fertig, wenn das echte Gymnasium-Environment, dessen offizieller
RGB-Frame und alle drei Methoden funktionieren, die GUI responsiv bleibt und
alle Tests erfolgreich sind.
