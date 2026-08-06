# CliffWalking – Reinforcement-Learning-Workbench

Projektordner: `Oliver/04-cliffwalking`

Berücksichtige die verbindlichen allgemeinen Regeln aus `../workbench.md`.

## Rolle und Zielgruppe

Erstelle eine lokale Lernanwendung für einen Developer und
Reinforcement-Learning-Anfänger. Die Anwendung soll fachlich korrekt,
nachvollziehbar und eigenständig lauffähig sein.

GUI-Texte, Hilfetexte, Validierungsfehler, Dokumentation und Kommentare werden
auf Deutsch formuliert. Datei-, Klassen-, Methoden- und Variablennamen bleiben
auf Englisch.

## Ziel

Implementiere eine lokale Tkinter-Anwendung namens `cliff_walker`, die das
Gymnasium-Environment `CliffWalking-v1` visualisiert und unterschiedliche
modellfreie Reinforcement-Learning-Methoden trainiert und vergleicht.

Die Anwendung soll insbesondere zeigen:

- warum eine kürzere Route am Cliff riskanter sein kann
- warum On-Policy- und Off-Policy-Verfahren unterschiedliche Wege lernen
- wie Epsilon-Exploration die Sicherheit und den Return beeinflusst
- wie sich tabellarische Verfahren und Deep-Q-Verfahren unterscheiden
- wie stabil Ergebnisse über mehrere Seeds und Wiederholungen sind

Die Anwendung läuft lokal ohne Webserver.

## Verbindliche fachliche Entscheidung

Der ursprüngliche Entwurf nennt gleichzeitig `VDQN`, `DDQN`, Q-Learning,
SARSA, Expected SARSA und Stable-Baselines3. Daraus wird folgende konsistente
Methodenauswahl:

### Tabellarische Methoden

- `QLearning`
- `Sarsa`
- `ExpectedSarsa`

### Deep-Q-Methoden

- `VanillaDQN`: Verwendung von `stable_baselines3.DQN`
- `DoubleDQN`: eigene, klar abgegrenzte PyTorch-Implementierung auf Basis
  derselben Netzwerk-, Replay-Buffer- und Trainingsschnittstellen

`VDQN` wird im Code und in der GUI nicht als Abkürzung verwendet, sondern
eindeutig `Vanilla DQN` genannt.

Stable-Baselines3 enthält Vanilla DQN, aber kein Double DQN. Deshalb darf
`DoubleDQN` nicht lediglich ein zweites Stable-Baselines3-DQN mit anderem Namen
sein.

## Voraussetzungen

- zentrales Conda-Environment `rl-26-08`
- Python 3.13
- Gymnasium
- `CliffWalking-v1`
- Tkinter
- Matplotlib
- NumPy
- Pillow
- Stable-Baselines3
- PyTorch
- keine Webanwendung und kein Webserver
- responsive GUI während Training und Vergleich
- reproduzierbare Experimente über Random Seeds

Ergänze die benötigten Pakete in `../environment.yml`. Installiere keine
virtuelle Umgebung innerhalb des Projektordners.

## Zu erstellende Dateien

- `cliff_walking_logic.py`: Environment-Adapter, tabellarische Methoden,
  Deep-Q-Komponenten, Agenten, Training, Evaluation und Vergleich; keine
  Tkinter-Abhängigkeiten
- `cliff_walking_gui.py`: Tkinter-Oberfläche und Matplotlib-Diagramme
- `cliff_walking_app.py`: ausschließlich Entry Point
- `README.md`: Installation, Start, Bedienung und fachliche Erklärung
- `requirements.txt`: direkte Projektabhängigkeiten
- `tests/test_cliff_walking_logic.py`: Unit- und Regressionstests ohne GUI
- `tests/test_cliff_walking_smoke.py`: kurze Integrations- und Importtests

Die Anwendung ist von den Gridworld-Projekten in `../02-gridworld_model_based`
und `../03-gridworld_model_free` getrennt. Dateien oder veränderlicher Zustand
dürfen nicht projektübergreifend importiert werden.

## Architektur

### `CliffWalkingEnvironment`

Kapselt das Gymnasium-Environment:

```text
gymnasium.make(
    "CliffWalking-v1",
    is_slippery=False,
    render_mode="rgb_array",
)
```

Verantwortlichkeiten:

- Erzeugen und Schließen des Gymnasium-Environments
- Weiterreichen von `reset(seed=...)` und `step(action)`
- einheitliche Verarbeitung von `terminated` und `truncated`
- zusätzliches Episodenlimit `max_steps`
- Umrechnung zwischen Observation und `(row, column)`
- Erkennung eines Cliff-Falls
- Statistik für Schritte, Cliff-Fälle und Episoden-Return
- Bereitstellung eines RGB-Frames für die GUI

Schnittstelle:

```text
reset(seed=None) -> observation
step(action) -> StepResult
render_rgb() -> np.ndarray
observation_to_position(observation) -> Tuple[int, int]
position_to_observation(position) -> int
close() -> None
```

`StepResult` enthält:

```text
observation: int
action: int
next_observation: int
reward: float
terminated: bool
truncated: bool
done: bool
fell_into_cliff: bool
termination_reason: Optional[str]
step: int
```

### `BaseAgent`

Gemeinsame Schnittstelle aller Methoden:

```text
reset(seed=None) -> None
select_action(observation, training=True) -> int
observe(step_result) -> None
end_episode() -> None
train_episode(environment) -> EpisodeResult
evaluate_episode(environment) -> EpisodeResult
save(path) -> None
load(path) -> None
```

Gemeinsame Eigenschaften:

```text
name: str
gamma: float
seed: Optional[int]
training_steps: int
episodes_completed: int
```

Tabellarische und Deep-Q-Agenten dürfen intern unterschiedlich arbeiten,
liefern nach außen aber dieselben Ergebnisobjekte.

### Ergebnisobjekte

`EpisodeResult` enthält:

```text
method: str
episode: int
transitions: List[StepResult]
total_reward: float
steps: int
cliff_falls: int
success: bool
termination_reason: str
epsilon: float
seed: Optional[int]
```

`TrainingPoint` enthält:

```text
episode: int
environment_steps: int
training_return: float
moving_average_return: float
evaluation_return: Optional[float]
success: bool
cliff_falls: int
epsilon: float
loss: Optional[float]
```

`ComparisonResult` enthält:

```text
method: str
repetition: int
seed: int
training_points: List[TrainingPoint]
final_evaluation_returns: List[float]
mean_final_return: float
success_rate: float
mean_cliff_falls: float
mean_episode_length: float
runtime_seconds: float
completed: bool
```

### `TrainingRunner`

Verantwortlichkeiten:

- Training einer ausgewählten Methode
- Fortschrittsmeldungen an die GUI
- sichere Abbruchprüfung zwischen vollständigen Schritten oder Updates
- regelmäßige Evaluation ohne Exploration
- keine direkten Tkinter-Aufrufe

### `ComparisonRunner`

Verantwortlichkeiten:

- isoliertes Training aller ausgewählten Methoden
- identische Environment-Einstellungen
- reproduzierbar abgeleitete Seeds
- gleiche Anzahl Environment-Schritte als faire Vergleichsbasis
- Aggregation über Wiederholungen
- Konfidenzintervalle
- keine Veränderung des sichtbaren Agenten oder seiner Tabellen beziehungsweise
  Netze

## Definition von `CliffWalking-v1`

### Grid

- feste Größe: `4 x 12`
- Start: `(3, 0)`
- Ziel: `(3, 11)`
- Cliff: `(3, 1)` bis `(3, 10)`
- Observation Space: `Discrete(48)`
- Action Space: `Discrete(4)`

Die Position wird als `(row, column)` dargestellt. Gymnasium codiert eine
Position als:

```text
observation = row * 12 + column
```

### Aktionen

- Action `0`: Up
- Action `1`: Right
- Action `2`: Down
- Action `3`: Left

Verwende genau diese Gymnasium-Reihenfolge. Sie unterscheidet sich von der
Action-Reihenfolge des vorherigen Gridworld-Projekts.

### Rewards

- normaler Schritt: `-1`
- Schritt in das Cliff: `-100`
- beim Cliff-Fall wird der Agent auf den Start zurückgesetzt
- ein Cliff-Fall beendet die Episode nicht
- das Ziel beendet die Episode

### `is_slippery`

- Checkbox `Slippery`
- Standardwert `False`
- Änderung erzeugt ein neues Environment und setzt Training und Evaluation
  zurück
- bei `True` wird die von Gymnasium bereitgestellte stochastische Variante
  verwendet
- GUI zeigt die tatsächlich ausgeführte und nicht nur die gewählte Action

### Episodenlimit

Da eine schlechte Policy das Ziel möglicherweise nie erreicht, wird jede
Episode zusätzlich durch `max_steps` begrenzt.

- Standardwert: `500`
- positive Ganzzahl
- bei Erreichen: `truncated=True`
- `termination_reason="max_steps"`
- kein Programmabbruch und keine Endlosschleife

## Tabellarische Methoden

### Gemeinsame Q-Tabelle

```text
q_values[observation][action]
```

Initialisierung mit `0.0`. Form: `48 x 4`.

### Epsilon-greedy Action-Auswahl

```text
if random() < epsilon:
    action = random_action
else:
    action = random_choice(best_actions)
```

Bei mehreren gleich guten Actions erfolgt reproduzierbares zufälliges
Tie-Breaking. Kein Action-Index darf systematisch bevorzugt werden.

Epsilon-Verlauf pro abgeschlossener Trainingsepisode:

```text
epsilon = max(epsilon_min, epsilon * epsilon_decay)
```

Evaluation verwendet immer `epsilon = 0.0` und verändert weder Q-Werte noch
Epsilon oder Trainingsstatistiken.

### `QLearning`

Off-Policy-Update:

```text
target = reward                         if done
target = reward + gamma * max(Q(s', a')) otherwise
Q(s, a) += alpha * (target - Q(s, a))
```

### `Sarsa`

On-Policy-Update:

```text
target = reward                              if done
target = reward + gamma * Q(s', next_action) otherwise
Q(s, a) += alpha * (target - Q(s, a))
```

Die für das SARSA-Update ausgewählte `next_action` muss im nächsten Schritt
tatsächlich ausgeführt werden. Sie darf nicht erneut ausgewählt und verworfen
werden.

### `ExpectedSarsa`

```text
target = reward + gamma * expected_q(next_observation)
```

`expected_q` verwendet exakt die aktuelle Epsilon-greedy
Wahrscheinlichkeitsverteilung. Gleichstände der besten Actions werden
gleichmäßig berücksichtigt.

## Deep-Q-Methoden

### Beobachtungscodierung

Die diskrete Observation wird für das neuronale Netz eindeutig codiert.
Bevorzugt wird ein One-Hot-Vektor mit 48 Elementen. Eine reine unskalierte
Ganzzahl als einzelnes kontinuierliches Merkmal ist nicht zulässig.

Dokumentiere den notwendigen Gymnasium-Wrapper und stelle sicher, dass Training
und Evaluation dieselbe Codierung verwenden.

### `VanillaDQN`

Verwendet:

```text
stable_baselines3.DQN("MlpPolicy", wrapped_environment, ...)
```

Stable-Baselines3 verwaltet Replay Buffer, Target Network, Optimizer und
Gradient Clipping. Implementiere einen Callback für Fortschritt, Evaluation,
Abbruch und Messwerte.

### `DoubleDQN`

Eigene PyTorch-Implementierung. Sie verwendet dieselbe Grundstruktur wie
Vanilla DQN, aber das Target wird getrennt ausgewählt und ausgewertet:

```text
best_next_action = online_network(next_state).argmax()
next_q = target_network(next_state)[best_next_action]
target = reward + gamma * next_q
```

Bei terminalen oder abgeschnittenen Übergängen wird kein zukünftiger Wert
addiert.

Erforderliche Komponenten:

- `ReplayBuffer`
- `QNetwork`
- `DoubleDQNAgent`
- Online Network
- Target Network
- Adam-Optimizer
- Huber Loss
- Gradient Clipping
- periodische Target-Network-Aktualisierung
- Modell speichern und laden

## Parameter und Standardwerte

### Allgemein

- `method`: `Sarsa`
- `episodes`: `500`
- `max_steps`: `500`
- `gamma`: `0.99`
- `seed`: `42`
- `is_slippery`: `False`
- `evaluation_interval`: `25`
- `evaluation_episodes`: `10`
- `animation_delay_ms`: `100`

### Tabellarisch

- `alpha`: `0.5`
- `epsilon_start`: `1.0`
- `epsilon_min`: `0.05`
- `epsilon_decay`: `0.995`

### Vanilla DQN und Double DQN

- `learning_rate`: `0.0005`
- `buffer_size`: `50_000`
- `learning_starts`: `1_000`
- `batch_size`: `64`
- `tau`: `1.0`
- `train_frequency`: `4`
- `gradient_steps`: `1`
- `target_update_interval`: `500`
- `exploration_initial_epsilon`: `1.0`
- `exploration_final_epsilon`: `0.05`
- `exploration_fraction`: `0.3`
- `max_gradient_norm`: `10.0`
- `network_architecture`: `64,64`
- `activation_function`: `ReLU`

Alle numerischen Eingaben werden vor dem Start vollständig validiert.
Fehlerhafte Eingaben verändern das laufende Experiment nicht.

## Dynamische Parameteranzeige

Zeige nur Parameter, die für die ausgewählte Methode relevant sind:

- `QLearning`, `Sarsa`, `ExpectedSarsa`: Alpha und tabellarische
  Epsilon-Parameter
- `VanillaDQN`, `DoubleDQN`: Replay-Buffer-, Netzwerk-, Optimizer-, Target- und
  Deep-Epsilon-Parameter
- `DoubleDQN`: zusätzlich Erklärung des Double-DQN-Targets

Allgemeine Parameter bleiben immer sichtbar.

Parameteränderungen werden beim Start einer Aktion automatisch übernommen.
Ein separater Anwenden-Button darf zusätzlich existieren, ist aber nicht
erforderlich.

## GUI-Aufbau

### Kopfbereich

- Titel `CliffWalking RL Workbench`
- Untertitel `Sichere und riskante Policies vergleichen`
- Status für Methode, Episode, Schritt und Trainingszustand
- Button `Bedienungsanleitung`

### Linkes Bedienpanel

Environment:

- Checkbox `Slippery`
- `Max. Schritte`
- `Random Seed`
- Button `Environment zurücksetzen`

Methode und Training:

- Dropdown mit allen fünf Methoden
- dynamische Hyperparameter
- `Episoden`
- `Evaluation alle N Episoden`
- `Evaluationsepisoden`

Steuerung:

- `Einzelschritt`
- `Eine Episode trainieren`
- `N Episoden trainieren`
- `Training stoppen`
- `Gelernte Policy ausführen`
- `Training zurücksetzen`
- `Modell speichern`
- `Modell laden`
- `Q-Tabelle` nur für tabellarische Methoden
- `Netzwerk-Informationen` nur für Deep-Q-Methoden

### CliffWalking-Ansicht

Zeige das vollständige `4 x 12`-Grid:

- Start grün
- Ziel gelb
- Cliff rot
- normale Zellen hell
- Agent als blauer Kreis
- gewählte Action und tatsächlich ausgeführte Bewegung
- aktueller Episodenpfad
- Cliff-Fall als kurze rote Animation mit Rücksprung zum Start
- optional Gymnasium-RGB-Frame

Die eigene Grid-Darstellung ist die primäre Ansicht. Der Gymnasium-Frame kann
in einem separaten Bereich oder per Umschalter angezeigt werden. Die GUI darf
nicht von einem externen Pygame-Fenster abhängen.

### Policy-Anzeige

Für tabellarische Methoden:

- optimale Action-Pfeile pro Zustand
- alle gleich guten Actions anzeigen
- unbesuchte Zustände als `?`
- umschaltbare Anzeige von `V(s)`

Für Deep-Q-Methoden:

- Q-Netz für alle 48 One-Hot-Zustände auswerten
- daraus greedy Pfeile ableiten
- Auswertung ohne Gradienten

### Summary

- Methode
- Trainingsepisoden
- Environment-Schritte
- aktuelles Epsilon
- letzter Return
- gleitender Durchschnitt der letzten 20 Episoden
- Erfolgsrate
- Cliff-Fälle insgesamt und pro Episode
- durchschnittliche Episodenlänge
- letzter Evaluations-Return
- aktuelle Loss nur bei Deep-Q-Methoden
- Abbruchgrund der letzten Episode

## Diagramme

Verwende Matplotlib und Tabs:

### `Training`

- dünne Linie: Return jeder Trainingsepisode
- fette Linie: gleitender Durchschnitt über 20 Episoden
- Markierungen für regelmäßige greedy Evaluationen
- unterschiedliche Farben pro Methode

### `Cliff-Fälle`

- Cliff-Fälle je Episode
- gleitender Durchschnitt
- optional kumulative Cliff-Fälle

### `Episodenlänge`

- Schritte je Episode
- gleitender Durchschnitt

### `Loss`

- nur für Vanilla DQN und Double DQN sichtbar
- geglättete Trainings-Loss
- keine künstlichen Nullwerte vor `learning_starts`

### `Methodenvergleich`

- gemeinsame Lernkurven
- Mittelwert über Wiederholungen
- optionales 95-%-Konfidenzintervall
- Tabelle mit finalem Return, Erfolgsrate, Cliff-Fällen, Episodenlänge und
  Laufzeit

## Methodenvergleich

Einstellungen:

- frei auswählbare Methoden
- `training_steps` als gemeinsame Trainingsbasis
- `repetitions`: Standard `10`
- `base_seed`: Standard `42`
- `evaluation_episodes`: Standard `20`
- Checkbox `95-%-Konfidenzintervall`

Gesamtumfang vor dem Start anzeigen:

```text
5 Methoden × 100.000 Schritte × 10 Wiederholungen
```

Fairnessregeln:

- jede Wiederholung beginnt mit neuen Agenten und leerem Lernzustand
- jede Methode erhält dieselben abgeleiteten Seeds
- gleiche Anzahl Environment-Schritte
- Evaluation immer ohne Exploration und ohne Lernupdates
- tabellarische und Deep-Q-Methoden werden anhand derselben Evaluationsmetriken
  verglichen
- Vergleich läuft getrennt vom sichtbaren Experiment

Fortschrittsanzeige:

```text
Methode 2 von 5: SARSA
Wiederholung 4 von 10
Schritte 35.000 von 100.000
```

`Stoppen` beendet nach dem aktuellen sicheren Update. Bereits vollständige
Ergebnisse bleiben sichtbar; unvollständige Ergebnisse werden entsprechend
markiert.

## Trainings- und Animationsverhalten

- Tkinter darf niemals aus einem Worker-Thread aktualisiert werden.
- Training läuft in kleinen `after()`-Chunks oder in einem Worker mit Queue.
- Deep-Q-Training blockiert die GUI nicht.
- Einzelschritte und sichtbare Episoden werden animiert.
- Massentraining läuft ohne Animation.
- nach Abschluss sind alle Controls wieder aktiv
- Exceptions werden abgefangen, angezeigt und setzen den Busy-Zustand zurück
- beim Schließen werden Worker gestoppt und Environments geschlossen

## Evaluation

`Gelernte Policy ausführen`:

- verwendet keine Exploration
- verändert keine Tabellen, Netze, Replay Buffer oder Trainingsstatistiken
- startet immer am Startzustand
- endet bei Ziel oder `max_steps`
- meldet verständlich, wenn noch keine brauchbare Policy gelernt wurde
- bleibt auch bei einer zyklischen Policy bedienbar

Bei `is_slippery=True` kann dieselbe Policy unterschiedliche Pfade erzeugen.
Bewerte deshalb mehrere Evaluationsepisoden und zeige Mittelwert und
Erfolgsrate.

## Tabellen und Inspektion

### Q-Tabelle

Nur für tabellarische Methoden. Modaler, scrollbar bedienbarer Dialog mit:

- Observation
- Row und Column
- Q Up, Q Right, Q Down, Q Left
- beste Actions
- Besuchszahl
- Start, Ziel oder normaler Zustand
- gelernt oder unbesucht

Unbesuchte Werte werden als `—` dargestellt, nicht als vermeintlich gelernte
Nullwerte.

### Netzwerk-Informationen

Für Deep-Q-Methoden:

- Architektur
- Anzahl trainierbarer Parameter
- Replay-Buffer-Belegung
- Training Steps
- Gradient Steps
- Target Updates
- aktueller Loss
- aktuelles Epsilon
- Gerät `cpu`, `mps` oder `cuda`

## Export und Persistenz

### CSV-Export

- Trainingshistorie
- Evaluationshistorie
- Vergleichsergebnisse
- Q-Tabelle für tabellarische Methoden
- UTF-8
- Komma als Trennzeichen
- Fließkommazahlen mit sechs Nachkommastellen
- eindeutige Dateinamen ohne Überschreiben

### Modell speichern und laden

Tabellarisch:

- Q-Tabelle als komprimierte NumPy-Datei oder JSON plus Metadaten

Deep-Q:

- Stable-Baselines3-ZIP für Vanilla DQN
- PyTorch-Checkpoint für Double DQN

Metadaten enthalten mindestens:

- Methode
- Hyperparameter
- Seed
- `is_slippery`
- Training Steps
- Episoden
- Softwareformat-Version

Beim Laden werden Methode und Environment-Kompatibilität geprüft. Fehlerhafte
oder inkompatible Dateien verändern den aktuellen Zustand nicht.

## Validierung und Fehlermeldungen

Validiere mindestens:

- positive Episoden-, Schritt- und Wiederholungszahlen
- `0 <= gamma <= 1`
- `0 < alpha <= 1`
- `0 <= epsilon_min <= epsilon_start <= 1`
- `0 < epsilon_decay <= 1`
- positive Learning Rate
- `buffer_size >= batch_size`
- `learning_starts < buffer_size`
- positive Netzwerkbreiten
- bekannte Aktivierungsfunktion
- gültiger optionaler Seed
- vorhandene Abhängigkeiten

Fehlermeldungen nennen das betroffene Feld und einen gültigen Wertebereich.
Einstellungen werden atomar übernommen: Entweder sind alle Werte gültig oder
das aktive Experiment bleibt vollständig unverändert.

## Reproduzierbarkeit

Setze und dokumentiere Seeds für:

- Python `random`
- NumPy
- Gymnasium `reset(seed=...)`
- Action Space
- PyTorch
- Stable-Baselines3

Für Deep Learning kann trotz Seeds plattformabhängige numerische Abweichung
auftreten. Die README weist darauf hin.

## Tests

### Environment

- Observation- und Positionsumrechnung
- Action-Reihenfolge
- normaler Reward `-1`
- Cliff-Reward `-100`
- Rücksprung zum Start nach Cliff-Fall
- Cliff-Fall ist nicht terminal
- Ziel ist terminal
- `max_steps` erzeugt Truncation
- `is_slippery=False` ist deterministisch
- Seeds reproduzieren `is_slippery=True`

### Tabellarische Methoden

- Q-Learning-Update
- SARSA verwendet tatsächlich `next_action`
- Expected-SARSA-Erwartungswert inklusive Ties
- terminales Update ohne zukünftigen Wert
- Epsilon-Grenzen und Decay
- reproduzierbares Tie-Breaking
- Evaluation verändert keine Lernwerte

### Replay Buffer und Netze

- Buffer-Kapazität und Sampling-Formen
- Sampling erst bei ausreichender Belegung
- Q-Network-Ausgabeform `[batch_size, 4]`
- Double-DQN-Target verwendet Online-Auswahl und Target-Auswertung
- terminale und abgeschnittene Targets ohne Bootstrap
- Target-Network-Update
- Save-/Load-Roundtrip

### Training und Vergleich

- Episoden enden sicher
- Abbruchcallback wird beachtet
- identische Seeds liefern reproduzierbare tabellarische Ergebnisse
- Vergleich verändert sichtbaren Agenten nicht
- alle Methoden liefern dieselbe Ergebnisstruktur
- Metriken enthalten keine Evaluation als Training
- Konfidenzintervall bei nur einer Wiederholung hat Breite `0`

### Smoke Tests

- alle Module importierbar
- `CliffWalking-v1` kann erzeugt, zurückgesetzt und geschlossen werden
- kurze Trainingsläufe aller Methoden ohne Exception
- GUI-Konstruktion, sofern ein Display verfügbar ist

Langsame Deep-Learning-Tests werden als solche markiert. Die normale Testsuite
muss mit kleinen Netzen und wenigen Schritten schnell ausführbar bleiben.

## README-Inhalt

Die README erklärt:

- Installation im zentralen Conda-Environment
- Startbefehl
- CliffWalking-Regeln
- Unterschied zwischen Cliff-Fall, Termination und Truncation
- Q-Learning gegenüber SARSA
- Expected SARSA
- Vanilla DQN gegenüber Double DQN
- Bedeutung aller wesentlichen Hyperparameter
- Methodenvergleich und Wiederholungen
- Interpretation der Diagramme
- Modell- und CSV-Export
- Testbefehle
- bekannte Grenzen und Reproduzierbarkeit

## Startbefehle

Vom Repository-Root:

```bash
conda activate rl-26-08
cd Oliver/04-cliffwalking
python cliff_walking_app.py
```

Tests:

```bash
conda run --name rl-26-08 python -m unittest discover \
  -s Oliver/04-cliffwalking/tests -v
```

## Abnahmekriterien

Die Aufgabe ist abgeschlossen, wenn:

- `CliffWalking-v1` korrekt verwendet wird
- deterministische und slippery Variante funktionieren
- alle fünf Methoden auswählbar sind
- Vanilla DQN tatsächlich Stable-Baselines3 verwendet
- Double DQN fachlich korrekt separat implementiert ist
- alle relevanten Parameter dynamisch einstellbar sind
- Training, Animation und Vergleich die GUI nicht blockieren
- Cliff-Fälle sichtbar und statistisch erfasst werden
- greedy Evaluation keine Lernwerte verändert
- Tabellen, Netzinformationen, Diagramme und Exporte funktionieren
- Vergleich über Seeds und Wiederholungen fair ist
- alle Tests erfolgreich sind
- README und GUI vollständig auf Deutsch sind
- Code-Bezeichner konsistent auf Englisch bleiben
