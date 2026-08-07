# CliffWalking – vollständige Workbench

Projektordner: `Oliver/04-cliffwalking`

## Grundlage

Berücksichtige alle verbindlichen Regeln aus `../workbench.md`, insbesondere
Sprache und Benennung, Architektur, responsive GUI, atomare
Parameterübernahme, Training/Evaluation, Reproduzierbarkeit, Visualisierung,
Export, Tests, README und Abnahmekriterien.

Dieser Prompt beschreibt nur die CliffWalking-spezifischen Anforderungen. Er
ist die Ausbaustufe nach `prompt_v1.md`.

## Ziel

Erweitere die lokale Tkinter-Lernanwendung für Gymnasiums
`CliffWalking-v1`. Sie soll den Unterschied zwischen sicheren und riskanten
Policies sowie zwischen On-Policy-, Off-Policy- und Deep-Q-Verfahren zeigen.

Methoden:

- `QLearning`
- `Sarsa`
- `ExpectedSarsa`
- `VanillaDQN` mit `stable_baselines3.DQN`
- `DoubleDQN` als eigene PyTorch-Implementierung

Stable-Baselines3 enthält kein Double DQN. `DoubleDQN` darf daher nicht nur
ein umbenanntes Stable-Baselines3-Modell sein.

## Dateien und Abhängigkeiten

Mindestens:

```text
cliff_walking_app.py
cliff_walking_logic.py
cliff_walking_gui.py
README.md
requirements.txt
tests/test_cliff_walking_logic.py
tests/test_cliff_walking_smoke.py
```

Größere Deep-Q- oder Vergleichskomponenten dürfen gemäß `../workbench.md` in
eigene Module ausgelagert werden. Benötigt werden Gymnasium, NumPy,
Matplotlib, Tkinter, Pillow, `pygame`, PyTorch und Stable-Baselines3. Ergänze direkte
Abhängigkeiten in `requirements.txt` und in `../environment.yml`.

## Environment

```python
gymnasium.make("CliffWalking-v1", render_mode="rgb_array")
```

- Grid: `4 x 12`
- Start: `(3, 0)`
- Ziel: `(3, 11)`
- Cliff: `(3, 1)` bis `(3, 10)`
- Observation: `row * 12 + column`
- Actions: `0 Up`, `1 Right`, `2 Down`, `3 Left`
- normaler Schritt: Reward `-1`
- Cliff-Fall: Reward `-100`, Rückkehr zum Start, nicht terminal
- Ziel: `terminated=True`
- `max_steps`: erzeugt `truncated=True`, Standard `500`

`CliffWalkingEnvironment` kapselt Gymnasium, Positionsumrechnung,
Cliff-Erkennung, Episodenlimit und RGB-Rendering. Ein `StepResult` unterscheidet
`terminated`, `truncated` und `done` und enthält Reward, Positionen, Action,
Schrittzahl, Cliff-Fall und Abbruchgrund.

Bei `terminated` und `truncated` wird in allen fünf Verfahren nicht
gebootstrapt. Diese Entscheidung muss in Code, Tests und README übereinstimmen.

## Tabellarische Methoden

Gemeinsame Q-Tabelle:

```text
q_values[48][4]
```

Epsilon-greedy verwendet zufälliges, reproduzierbares Tie-Breaking. Epsilon
wird nach jeder Trainingsepisode aktualisiert:

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

SARSA führt die für das Update ausgewählte `next_action` tatsächlich als
nächste Action aus. Expected SARSA verwendet die vollständige aktuelle
Epsilon-greedy-Verteilung und verteilt Greedy-Wahrscheinlichkeit bei Ties
gleichmäßig.

## Deep-Q-Methoden

Diskrete Observations werden als One-Hot-Vektoren mit 48 Elementen codiert.
Training und Evaluation verwenden denselben Wrapper.

### Vanilla DQN

Verwende `stable_baselines3.DQN("MlpPolicy", wrapped_environment, ...)` mit
Callback für Fortschritt, Evaluation und Abbruch.

### Double DQN

Implementiere mit PyTorch:

- `ReplayBuffer` und `QNetwork`
- Online- und Target-Netz
- Adam, Huber Loss und Gradient Clipping
- periodisches Target-Network-Update
- Speichern und Laden inklusive Metadaten

Double-DQN-Target:

```text
best_action = online_network(next_state).argmax()
next_q = target_network(next_state)[best_action]
target = reward if done else reward + gamma * next_q
```

## Standardparameter

Allgemein:

- Methode: `Sarsa`
- Episoden: `500`
- Max. Schritte: `500`
- Gamma: `0.99`
- Seed: `42`
- Evaluation alle `25` Episoden, jeweils `10` Episoden
- Animation anzeigen: `True`
- Animationsintervall: `10 ms`

Tabellarisch:

- Alpha: `0.5`
- Epsilon Start: `1.0`
- Epsilon Minimum: `0.05`
- Epsilon Decay: `0.995`

Deep Q:

- Learning Rate: `0.0005`
- Buffer Size: `50_000`
- Learning Starts: `1_000`
- Batch Size: `64`
- Train Frequency: `4`
- Gradient Steps: `1`
- Target Update Interval: `500`
- Exploration Start/End: `1.0 / 0.05`
- Exploration Fraction: `0.3`
- Max. Gradient Norm: `10.0`
- Netzwerk: `64, 64`, Aktivierung `ReLU`

Die GUI zeigt nur Parameter der gewählten Methode. Änderungen werden beim
Start einer Aktion automatisch und atomar übernommen.

## GUI und Bedienung

Ergänzend zur Workbench:

- Zeige ausschließlich Gymnasiums offiziellen RGB-Frame skaliert in Tkinter.
  Gemeint ist die Darstellung aus
  `https://gymnasium.farama.org/environments/toy_text/cliff_walking/`.
- Erstelle kein eigenes Grid, Canvas-Rendering, Weg-Overlay, Policy-Pfeile
  oder andere Spielfeldgrafiken.
- Aktualisiere den Frame nach `reset()` und jedem sichtbaren `step()`, erhalte
  das Seitenverhältnis und öffne kein separates Pygame-Fenster.
- Zeige Observation, Position, Action, Reward und Cliff-Fall zusätzlich als
  Textstatus. Pillow bettet den von Gymnasium erzeugten Frame in Tkinter ein.

Steuerung:

- Einzelschritt und eine Episode animieren
- N Episoden ohne Animation trainieren und sicher stoppen
- gelernte Policy ohne Lernupdates ausführen
- Training zurücksetzen
- Q-Tabelle oder Netzwerk-Informationen öffnen
- Modell speichern/laden
- Bedienungsanleitung öffnen

Das Animationsintervall bestimmt die Pause zwischen zwei sichtbaren
Agentenschritten. Bei deaktivierter Animation laufen Trainingsepisode und
Evaluation ohne Einzelbilder und zeigen nur das Endergebnis. Massentraining
bleibt immer ohne Animation.

Eine zyklische oder noch unbrauchbare Policy darf die Anwendung nicht
abbrechen oder blockieren. Sie endet nach `max_steps` mit einer verständlichen
Meldung.

Summary: Methode, Episoden, Environment-Schritte, Epsilon, letzter und
gleitender Return, Erfolgsrate, Cliff-Fälle, Episodenlänge, Evaluation,
Abbruchgrund und bei Deep Q der Loss.

Diagramm-Tabs:

- Training: Episoden-Returns, gleitender Mittelwert und Evaluationen
- Cliff-Fälle pro Episode
- Episodenlänge
- Loss nur für Deep-Q-Methoden
- Methodenvergleich

Die Q-Tabelle zeigt alle 48 Zustände, Position, vier Q-Werte, beste Actions und
Besuchszahl. Unbesuchte Werte erscheinen als `—`. Netzwerk-Informationen
zeigen Architektur, Parameterzahl, Buffer-Belegung, Updates, Loss, Epsilon und
Gerät.

## Methodenvergleich

Frei auswählbare Methoden werden isoliert mit gleicher Anzahl
Environment-Schritte verglichen.

Standardwerte:

- Wiederholungen: `10`
- Base Seed: `42`
- Evaluationsepisoden: `20`
- optionales 95-%-Konfidenzintervall

Jede Wiederholung verwendet einen neuen Agenten und abgeleitete, zwischen den
Methoden identische Seeds. Der Vergleich verändert den sichtbaren Agenten
nicht. Zeige Lernkurven sowie finalen Return, Erfolgsrate, Cliff-Fälle,
Episodenlänge und Laufzeit. Vor dem Start werden Gesamtumfang und Fortschritt
verständlich angezeigt; Teilresultate bleiben nach Abbruch erhalten.

## Evaluation, Export und Persistenz

Greedy Evaluation nutzt keine Exploration oder Lernupdates. Exportiere
Trainings-, Evaluations- und Vergleichsdaten sowie Q-Tabellen als CSV.

Speicherformate:

- tabellarisch: komprimierte NumPy-Datei oder JSON mit Metadaten
- Vanilla DQN: Stable-Baselines3-ZIP
- Double DQN: PyTorch-Checkpoint

Prüfe beim Laden Methode, Formatversion und Environment-Kompatibilität, bevor
der aktive Zustand verändert wird.

## CliffWalking-spezifische Tests

Zusätzlich zu `../workbench.md` prüfe:

- Observation-/Positionsumrechnung und Gymnasium-Action-Reihenfolge
- Reward, Cliff-Rücksprung, Ziel und Max-Steps-Truncation
- Updates aller drei tabellarischen Methoden einschließlich Ties
- SARSA verwendet seine vorbereitete `next_action`
- Evaluation verändert keinen Lernzustand
- Replay Buffer, Netz-Ausgabe und Target-Network-Update
- Double-DQN-Auswahl über Online- und Auswertung über Target-Netz
- Save-/Load-Roundtrips
- kurzer Trainingslauf aller Methoden
- Vergleich verändert den sichtbaren Agenten nicht
- Q-Learning und SARSA lernen mit festen Seeds mindestens eine erfolgreiche
  Policy

Deep-Learning-Tests verwenden kleine Netze und kurze Läufe; langsame Tests
werden markiert.

## Start

```bash
conda activate rl-26-08
cd Oliver/04-cliffwalking
python cliff_walking_app.py
```

Tests vom Repository-Root:

```bash
conda run --name rl-26-08 python -m unittest discover \
  -s Oliver/04-cliffwalking/tests -v
```

## Abnahme

Fertig, wenn alle fünf Methoden fachlich korrekt funktionieren, Animation und
Massentraining die GUI nicht blockieren, Evaluation den Lernzustand nicht
verändert, Gymnasiums offizielles Rendering verwendet wird, der Vergleich fair und
reproduzierbar ist und alle Tests erfolgreich laufen. README und
Bedienungsanleitung erklären den typischen sicheren SARSA-Weg gegenüber dem
riskanteren Q-Learning-Weg, ohne dieses Ergebnis für jeden Seed zu garantieren.
