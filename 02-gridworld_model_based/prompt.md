# Modellbasiertes Gridworld mit Value Iteration

Projektordner: `Oliver/02-gridworld_model_based`

## Rolle und Zielgruppe

Erstelle die Anwendung für einen Developer und Reinforcement-Learning-Anfänger.
Der Code soll fachlich korrekt, klar strukturiert und an wichtigen Stellen kurz
kommentiert sein.

Klassen-, Methoden-, Datei- und Variablennamen bleiben Englisch. GUI-Texte,
Erklärungen und Fehlermeldungen werden auf Deutsch formuliert.

## Ziel

Erstelle eine zweite lokale Python-Anwendung namens `gridworld`, die zwei
modellbasierte Planungsmethoden demonstriert und vergleicht:

- `ValueIteration`
- `QValueIteration`

Die Anwendung zeigt:

- Entwicklung von `V(s)` und `Q(s,a)` über Bellman-Iterationen
- Konvergenz der Methoden
- daraus abgeleitete greedy Policy
- sichtbare Agenten-Rollouts durch das Grid
- Vergleich von Konvergenz, Startwert und Rollout-Reward

Verwende keine neuronalen Netze und keine modellfreien Lernverfahren.

## Fachliche Festlegungen

Value Iteration und Q-Value Iteration kennen das vollständige Modell des
Environments. Sie lernen nicht aus Episoden, sondern berechnen Werte anhand
aller Zustände, Aktionen, Übergänge und Rewards.

- `epsilon` beeinflusst niemals die Bellman-Updates.
- Epsilon-Parameter gelten ausschließlich für sichtbare Agenten-Rollouts.
- Ein `planner_sweep` entspricht einem vollständigen Bellman-Sweep über alle
  begehbaren, nicht terminalen Zustände.
- `Planner single sweep` führt genau einen Bellman-Sweep aus.
- `Planner bis Konvergenz` führt höchstens `planner_max_iterations` Sweeps aus
  und stoppt bei Konvergenz vorzeitig.
- `Agent single step` führt genau einen sichtbaren Agentenschritt aus.
- `Agent run n loops` führt höchstens `agent_loops` sichtbare Agentenschritte
  aus und stoppt früher, wenn das Ziel erreicht wird.
- `QValueIteration` iteriert direkt über `Q(s,a)`.

## Voraussetzungen

- lokale Python-Anwendung ohne Webserver
- zentrales Conda-Environment `rl-26-08`
- Python 3.13
- Tkinter
- Matplotlib
- ausschließlich tabellarische Werte
- deterministisches Environment
- responsive GUI

## Projektdateien

- `gridworld_logic.py`: Environment, Planner, Rollout-Agent und Vergleich;
  keine GUI-Abhängigkeiten
- `gridworld_gui.py`: Tkinter und Matplotlib
- `gridworld_app.py`: ausschließlich Entry Point
- `README_model_based.md`: Anleitung und fachliche Erklärung
- `tests/test_gridworld_logic.py`: Logiktests ohne GUI

Das Projekt ist vollständig von der modellfreien Variante im Ordner
`../03-gridworld_model_free` getrennt. Dateien und Imports dürfen nicht
projektübergreifend vermischt werden.

## Architektur

### `GridWorld`

Verwaltet:

- `rows`, `columns`
- `start`, `goal`, `obstacles`
- `current_state`, `step_count`, `max_steps`
- deterministische Transitions und Rewards
- Ende eines Agenten-Rollouts

Es stellt sowohl eine zustandslose Modellfunktion als auch eine zustandsbehaftete
Rollout-Funktion bereit:

```text
transition(state, action) -> (next_state, reward, done)
reset() -> state
step(action) -> (next_state, reward, done, info)
```

`transition()` verändert `current_state` nicht.

`reset()` setzt `current_state` und `step_count` zurück, startet den
Zufallsgenerator aber nicht neu. `reset(seed=...)` initialisiert den
Zufallsgenerator reproduzierbar neu.

### `BasePlanner`

Gemeinsame Schnittstelle:

```text
reset() -> None
single_sweep() -> SweepResult
run(max_sweeps, cancelled=None) -> List[SweepResult]
get_value(state) -> float
get_q_value(state, action) -> float
get_best_actions(state) -> List[int]
get_greedy_action(state) -> int
```

`SweepResult` enthält:

```text
iteration: int
delta: float
start_value: float
converged: bool
```

Implementiere:

- `ValueIteration`
- `QValueIteration`

Beide erhalten dasselbe `GridWorld`, `gamma`, `tolerance` und einen optionalen
`seed` für reproduzierbares Tie-Breaking. `tolerance` wird ausschließlich im
Konstruktor konfiguriert und nicht zusätzlich an `run()` übergeben.

### `RolloutAgent`

Führt eine berechnete Policy sichtbar aus und verändert keine Planner-Werte:

```text
reset() -> state
select_action(state, epsilon) -> action
single_step(epsilon) -> Transition
run_episode(epsilon, max_steps) -> RolloutResult
run_greedy_episode() -> RolloutResult
```

Verwende eindeutige Ergebnisobjekte:

```text
Transition:
    step: int
    state: State
    action: int
    next_state: State
    reward: float
    done: bool
    termination_reason: Optional[str]

RolloutResult:
    method: str
    transitions: List[Transition]
    total_reward: float
    steps: int
    success: bool
    epsilon: float

ComparisonResult:
    method: str
    sweep_results: List[SweepResult]
    converged: bool
    runtime_ms: float
    final_start_value: float
    rollout_results: List[RolloutResult]
```

### `ComparisonRunner`

Führt beide Methoden isoliert aus. Der Vergleich verändert den sichtbaren
Planner, Agenten, Pfad und dessen Statistiken nicht.

## Gridworld-Modell

### Koordinaten

Zustände werden überall als `(row, column)` dargestellt.

- oben links: `(0, 0)`
- unten rechts: `(rows - 1, columns - 1)`

Nur die GUI rechnet in Canvas-Koordinaten um:

```text
x = column * cell_size
y = row * cell_size
```

### Standardkonfiguration

- `rows`: `3`, maximal `8`
- `columns`: `5`, maximal `10`
- `start`: `(2, 0)`
- `goal`: `(2, 4)`
- `obstacles`: `(1, 2)` und `(2, 2)`
- `max_steps`: `50`

Mindestgröße: `2 x 2`.

### Actions

- Action `0`: Up `(-1, 0)`
- Action `1`: Down `(+1, 0)`
- Action `2`: Left `(0, -1)`
- Action `3`: Right `(0, +1)`

### Transitions und Rewards

- gültige Action: neuer Zustand wird betreten
- Rand oder Hindernis: Zustand bleibt unverändert
- normaler Schritt: Reward `-1`
- Eintritt in `goal`: Reward `0`
- `goal` ist terminal und besitzt keinen zukünftigen Wert

Für `state == goal` liefert `transition(state, action)` unabhängig von der
Action `(goal, 0, True)`. Nach `done=True` verweigert `step()` weitere Aktionen
mit einem verständlichen Fehler, bis `reset()` aufgerufen wurde.

Alle vier Actions bleiben in jedem nicht terminalen Zustand auswählbar. Der
negative Reward sorgt dafür, dass Bewegungen gegen Wand oder Hindernis
unattraktiv werden.

### Validierung

Einstellungen werden vollständig validiert und erst danach atomar übernommen:

- `2 <= rows <= 8`
- `2 <= columns <= 10`
- Start, Ziel und Hindernisse liegen im Grid
- Start und Ziel sind verschieden und nicht blockiert
- Hindernisse enthalten keine Duplikate
- mindestens ein Weg vom Start zum Ziel existiert
- `max_steps` ist mindestens so groß wie der kürzeste Weg

Verwende Breitensuche für Erreichbarkeit und Weglänge. Fehler werden auf
Deutsch angezeigt; das aktive Experiment bleibt unverändert.

## `ValueIteration`

Speichert:

```text
values[state]
```

Alle Werte starten bei `0.0`. Hindernisse und Ziel werden nicht aktualisiert.

Für jede Action:

```text
action_value = reward + gamma * values[next_state]
```

Bei `done=True` gilt `action_value = reward`.

Synchrones Update:

```text
new_values[state] = max(action_value for action in actions)
```

Alle neuen Werte werden aus einer unveränderten Kopie der alten Tabelle
berechnet. Innerhalb desselben Sweeps dürfen keine neuen Werte wiederverwendet
werden.

```text
delta = max(abs(new_values[state] - values[state]))
converged = delta < tolerance
```

Q-Werte werden für Anzeige und Policy abgeleitet:

```text
Q(s,a) = reward + gamma * V(next_state)
```

Bei terminalem `next_state` gilt `Q(s,a) = reward`.

## `QValueIteration`

Speichert direkt:

```text
q_values[state][action]
```

Alle Werte starten bei `0.0`.

Synchrones Update:

```text
new_q_values[state][action] =
    reward + gamma * max_a' q_values[next_state][a']
```

Bei terminalem Übergang gilt `new_q_values[state][action] = reward`.

```text
delta = max(
    abs(new_q_values[state][action] - q_values[state][action])
)
V(s) = max_a Q(s,a)
```

## Tie-Breaking

Bei mehreren gleich guten Actions:

- zeigt die Policy-Ansicht alle gleichwertigen Pfeile
- wählt `RolloutAgent` reproduzierbar zufällig zwischen ihnen
- wird kein Action-Index systematisch bevorzugt

Vor dem ersten Sweep zeigt die Policy-Ansicht `?`.

Verwende wegen möglicher Fließkommaabweichungen keine exakte Gleichheit zur
Bestimmung optimaler Actions:

```text
action_tolerance = max(tolerance, 1e-9)
best_value = max(action_values)
best_actions = [
    action
    for action, action_value in action_values
    if best_value - action_value <= action_tolerance
]
```

Dieselbe Regel gilt für Policy-Anzeige, Tie-Breaking und Methodenvergleich.

## Einstellungen

### Planung

- Dropdown `Methode`: Value Iteration, Q-Value Iteration
- Checkbox `Methodenvergleich`
- `planner_max_iterations`: `1000`, positive Ganzzahl
- `gamma`: `0.9`, Wertebereich `[0, 1]`
- `tolerance`: `0.0001`, größer als `0`

### Agenten-Rollout

Diese Felder müssen exakt wie in der Aufgabenstellung beschriftet sein und
beeinflussen ausschließlich Rollouts:

- `Agent-Loops`: Variable `agent_loops`, Standardwert `50`
- `Epsilon (greedy)`: Variable `epsilon_greedy`, Standardwert `0.1`
- `Epsilon Max (decay)`: Variable `epsilon_max`, Standardwert `0.995`
- `Epsilon Min (decay)`: Variable `epsilon_min`, Standardwert `0.01`
- `Decay (decay)`: Variable `epsilon_decay`, Standardwert `0.05`

```text
epsilon_decay_current = max(
    epsilon_min,
    epsilon_decay_current * (1 - epsilon_decay),
)
```

Verwende zwei getrennte Werte:

```text
epsilon_greedy = 0.1
epsilon_decay_current = epsilon_max
```

- `Agent single step` verwendet immer `epsilon_greedy`.
- `Agent run n loops` und `Agent run episode` verwenden
  `epsilon_decay_current`.
- Nach einem vollständig abgeschlossenen automatischen Rollout wird nur
  `epsilon_decay_current` reduziert.
- Manuelle Bewegungen, einzelne Agentenschritte und greedy Rollouts verändern
  keinen Epsilon-Wert.
- `Reset` setzt `epsilon_decay_current` auf `epsilon_max` zurück und lässt den
  eingegebenen Wert `epsilon_greedy` unverändert.

`Agent run n loops` führt bis zu `agent_loops` Schritte mit demselben
`epsilon_decay_current` aus. Falls das Ziel vorher erreicht wird, endet der
Lauf sofort.

Zeige direkt bei den Feldern:

```text
Epsilon steuert nur den Agenten-Rollout, nicht Value Iteration.
```

## GUI-Aufbau

Die Anwendung soll wie ein übersichtliches Lernlabor wirken.

### Kopfbereich

- Titel `Modellbasiertes Gridworld Lab`
- Untertitel `Value Iteration und Q-Value Iteration verstehen und vergleichen`
- Status für Methode, Iteration und Konvergenz

### Bedienpanel

Grid-Konfiguration:

- Rows und Columns
- Start Row/Column
- Goal Row/Column
- Obstacles als `(row,column), ...`
- `max_steps`
- optionaler Random Seed
- `Grid anwenden und zurücksetzen`

Einstellungen:

- Methode und Methodenvergleich
- Planner Max Iterations, Gamma und Tolerance
- Agent-Loops, Epsilon (greedy), Epsilon Max (decay), Epsilon Min (decay)
  und Decay (decay)
- Erklärung zur Rollout-Bedeutung von Epsilon

Steuerung:

- Up, Down, Left, Right für manuelle Demo
- `Planner single sweep`
- `Planner bis Konvergenz`
- `Cancel`
- `Reset`
- `Agent single step`
- `Agent run n loops`
- `Agent run episode`
- `Greedy Policy ausführen`
- `Value-Table`
- `Q-Table`

### Grid-Anzeige

- quadratische Zellen
- Start grün, Ziel gelb, Hindernisse grau
- Agent als blauer Kreis mit `A`
- Koordinaten in Hellgrau
- Pfad mit Richtungspfeilen
- ungültige Bewegung als Schleife oder Zähler
- optionale greedy Policy-Pfeile
- optionale Werte direkt in den Zellen
- lesbare Darstellung bis `8 x 10`

Steuerung der Zellinhalte:

- Checkbox `Policy anzeigen`
- Dropdown `Zellanzeige` mit:
  - Keine
  - `V(s)`
  - beste Actions
  - `V(s)` und beste Actions

### Auswertungs-Tabs

1. `Reward`
2. `Konvergenz`
3. `Methodenvergleich`

## Verhalten der Steuerung

### Manuelle Demo

- verändert keine Value- oder Q-Tabelle
- verändert keine Planner-Iteration
- verändert weder Rollout-Statistik noch Epsilon
- wird während Planung und Animation deaktiviert
- ist ein eigener Modus; beim Wechsel zu einem Agenten-Rollout wird das
  Environment auf `start` zurückgesetzt

### Planner

`Planner single sweep`:

- genau ein synchroner Bellman-Sweep
- aktualisiert Werte, Policy, `delta` und Iteration
- erzeugt keinen Agenten-Reward

`Planner bis Konvergenz`:

- höchstens `planner_max_iterations` Sweeps
- stoppt bei `delta < tolerance`
- aktualisiert Fortschritt und Konvergenzdiagramm
- blockiert die GUI nicht

`Cancel`:

- stoppt nach dem aktuellen Sweep
- behält vollständig berechnete Werte
- führt keinen Reset aus

`Reset`:

- setzt Planner-Werte, Iterationen, Agentenpfad, Summary und Diagramme zurück
- behält gültige Eingabewerte

Nach Konvergenz sind weitere Planner-Sweeps deaktiviert, bis ein Reset oder
eine relevante Parameteränderung erfolgt.

Jede Methode besitzt einen getrennten Planner-Zustand. Beim Wechsel der
Dropdown-Auswahl bleiben die Tabellen, Historien und Iterationszähler beider
Methoden erhalten. Änderungen an Grid, `gamma` oder `tolerance` setzen beide
Planner nach erfolgreicher Validierung zurück.

### Agenten-Rollout

- verwendet die aktuelle Policy
- verändert keine Planner-Werte
- beginnt bei `start`
- endet bei `goal` oder `max_steps`
- zeichnet den Pfad animiert
- speichert Step-Rewards und Episode-Reward
- `Greedy Policy ausführen` verwendet `epsilon = 0`

`Agent single step` führt exakt einen Schritt aus. Falls kein Rollout aktiv
ist, wird zuerst bei `start` begonnen. Nach Ziel oder `max_steps` startet der
nächste Einzelschritt einen neuen Rollout.

`Agent run n loops` führt höchstens `agent_loops` Agentenschritte aus. Der Lauf
stoppt sofort bei `goal` oder `max_steps`. Nicht verbrauchte Loops werden nicht
in eine neue Episode übertragen.

Falls `agent_loops` verbraucht ist, bevor `goal` oder `max_steps` erreicht
wurde:

- bleibt der Rollout aktiv
- setzt der nächste `Agent single step` oder `Agent run n loops` denselben
  Rollout fort
- wird noch kein Eintrag im Reward-Diagramm oder in der Summary erzeugt
- wird `epsilon_decay_current` noch nicht reduziert

Nur vollständig durch `goal` oder `max_steps` beendete Rollouts werden in
Reward-Statistik, Erfolgsquote und Epsilon Decay aufgenommen. `Reset`, eine
Grid-Änderung oder ein Methodenwechsel verwirft einen unvollständigen Rollout
mit einem verständlichen Statushinweis.

`Agent run episode` läuft unabhängig von `agent_loops` bis `goal` oder
`max_steps`.

## Summary

Zeige:

- Methode
- Bellman-Iterationen
- konvergiert: Ja/Nein
- aktuelles `delta`
- aktuelles `V(start)`
- Anzahl Rollouts und Erfolge
- durchschnittlicher Episode-Reward
- `epsilon_greedy`
- aktuelles `epsilon_decay_current`
- Länge und Reward des letzten Rollouts

Verwende nicht `Episoden` als Bezeichnung für Bellman-Iterationen.

## Reward-Diagramm

Titel: `Kumulativer Agent-Reward je Methode`

Es basiert ausschließlich auf Rollouts:

- x-Achse: Rollout
- y-Achse: kumulativer Reward des Rollouts
- dünne Linie: einzelner Rollout-Reward
- dicke Linie: gleitender Durchschnitt der letzten 20 Rollouts
- eigene farbenblind-freundliche Farbe je Methode
- Legende im Vergleich

Dabei gilt:

```text
rollout_reward = sum(transition.reward for transition in transitions)
```

Der dargestellte kumulative Reward ist die Summe innerhalb eines Rollouts,
nicht die fortlaufende Summe über mehrere Rollouts.

Vor dem ersten Rollout erscheint ein erklärender Hinweis.

## Konvergenzdiagramm

Titel: `Bellman-Konvergenz`

- x-Achse: Bellman-Iteration
- y-Achse: `delta`
- logarithmische y-Skala, sofern Werte positiv sind
- horizontale gestrichelte Linie für `tolerance`
- optional `V(start)` als zweite Kurve

Dieses Diagramm ist die zentrale fachliche Visualisierung der Planner.

## Methodenvergleich

Bei aktiviertem Vergleich:

- starten beide Methoden mit leeren Tabellen
- verwenden beide dasselbe Grid, `gamma`, `tolerance` und
  `planner_max_iterations`
- laufen bis Konvergenz oder zum Limit
- verändern den sichtbaren Planner nicht
- führen anschließend jeweils `comparison_rollouts = 20` Rollouts aus
- verwenden für Rollouts dieselbe Epsilon-Folge und reproduzierbar abgeleitete
  Seeds
- verwenden dieselbe Anzahl erlaubter Schritte je Rollout

Zeige gemeinsam:

- `delta` pro Iteration
- `V(start)` pro Iteration
- Iterationen bis Konvergenz
- finalen Startwert
- Laufzeit in Millisekunden
- greedy Pfadlänge und greedy Rollout-Reward
- durchschnittlichen Reward und Erfolgsquote der 20 Vergleichs-Rollouts

Beide Methoden erscheinen in denselben Achsen mit unterschiedlichen Farben.
Nach Konvergenz sollen sie innerhalb der Toleranz denselben optimalen Startwert
und dieselben Mengen optimaler Actions pro Zustand liefern. Einzelne sichtbare
Pfade dürfen sich durch reproduzierbares Tie-Breaking unterscheiden.

Wenn der Vergleich abgebrochen wird, darf eine vollständig berechnete Methode
angezeigt werden. Eine teilweise berechnete Methode wird als `Unvollständig`
markiert und nicht als konvergiertes Ergebnis bewertet.

## Modale Tabellen-Dialoge

### Value-Table

- bei `ValueIteration`: gespeichertes `V(s)`
- bei `QValueIteration`: `max_a Q(s,a)`
- Start, Ziel und Hindernisse markieren
- CSV-Spalten:

```text
row,column,value,is_start,is_goal,is_obstacle,method,iteration
```

### Q-Table

- zeigt `Q(s,Up)`, `Q(s,Down)`, `Q(s,Left)`, `Q(s,Right)`
- zeigt alle besten Actions als Pfeile
- bei `ValueIteration`: aus `V(s)` abgeleitete Q-Werte
- bei `QValueIteration`: gespeicherte Q-Werte
- CSV-Spalten:

```text
row,column,q_up,q_down,q_left,q_right,best_actions,
is_start,is_goal,is_obstacle,method,iteration
```

Beide Dialoge sind modal und besitzen `CSV exportieren` und `Schließen`.

Modal bedeutet in Tkinter:

```text
dialog.transient(root)
dialog.grab_set()
root.wait_window(dialog)
```

Exporte werden mit Methode, Tabellentyp, Iteration und Zeitstempel im Ordner
`exports/` gespeichert und überschreiben keine Dateien.

- Encoding: UTF-8
- Trennzeichen: Komma
- Fließkommawerte: sechs Nachkommastellen
- Hindernisse werden als eigene Zeilen mit `is_obstacle=True` exportiert
- ein Export verändert keine Planner- oder Rollout-Daten

## Responsivität und Fehlerbehandlung

- lange Berechnungen blockieren die GUI nicht
- Tkinter-Widgets werden nur im Hauptthread aktualisiert
- doppelte Starts werden verhindert
- widersprüchliche Einstellungen werden während eines Laufs deaktiviert
- aktive Berechnungen enden beim Schließen sauber
- ungültige Einstellungen werden nie teilweise angewendet
- `gamma` liegt in `[0, 1]`
- `tolerance > 0`
- Epsilon-Werte liegen in `[0, 1]`
- `epsilon_min <= epsilon_greedy <= epsilon_max`
- `epsilon_min <= epsilon_decay_current <= epsilon_max`
- `0 < epsilon_decay <= 1`
- `agent_loops`, `planner_max_iterations`, `comparison_rollouts` und
  `max_steps` sind positive Ganzzahlen
- Exportfehler bringen die App nicht zum Absturz

Tie-Breaking und epsilon-greedy Action-Auswahl verwenden getrennte
Zufallsgeneratoren. Beide Seeds werden reproduzierbar aus dem eingegebenen
Basis-Seed abgeleitet, damit ein Rollout nicht durch zusätzliche
Tie-Breaking-Aufrufe verändert wird.

## Entry Point

`gridworld_app.py` enthält nur:

1. Tk-Root erzeugen
2. Standard-`GridWorld` erzeugen
3. beide Planner erzeugen
4. `ValueIteration` als Standard auswählen
5. `RolloutAgent` erzeugen
6. GUI mit injizierten Abhängigkeiten erzeugen
7. `root.mainloop()` starten

```bash
conda activate rl-26-08
python gridworld_app.py
```

## Tests

### Verbindliche Logiktests

Teste mindestens:

- Row-/Column-Koordinaten und Actions `0` bis `3`
- deterministische Transitions, Rand und Hindernisse
- `transition()` verändert `current_state` nicht, `step()` dagegen schon
- Fehler bei `step()` nach Episodenende
- Schritt- und Ziel-Reward
- terminalen Zielzustand ohne zukünftigen Wert
- Grid-Validierung, Erreichbarkeit und kürzesten Weg
- synchrone Sweeps beider Methoden
- korrektes `delta` und Konvergenz
- Ableitung von Q aus V und V aus Q
- reproduzierbares Tie-Breaking
- epsilon-greedy und greedy Rollout ohne Wertänderung
- Epsilon Decay nach vollständigem Rollout, aber nicht nach Einzelschritt oder
  greedy Rollout
- getrennte Werte und Zufallsfolgen für `epsilon_greedy` und
  `epsilon_decay_current`
- Fortsetzung eines nach `agent_loops` noch unvollständigen Rollouts
- unvollständiger Rollout erzeugt keine Reward-Statistik und keinen Decay
- Planner-Werte bleiben bei manueller Demo und allen Rollout-Arten unverändert
- Reset und Cancel
- Cancel wird erst nach einem vollständig abgeschlossenen Bellman-Sweep wirksam
- getrennte Planner-Tabellen beim Methodenwechsel
- Reset beider Planner bei Änderung von Grid, Gamma oder Tolerance
- Isolation des Methodenvergleichs
- identische Seed- und Epsilon-Folgen im Reward-Vergleich
- zusätzliche Tie-Breaking-Aufrufe verändern nicht die Zufallsfolge der
  epsilon-greedy Auswahl
- gleichen finalen Startwert beider Methoden innerhalb der Toleranz
- gleiche optimale greedy Policy
- gleiche Mengen optimaler Actions bei mehreren gleichwertigen Lösungen
- optimale Action-Mengen verwenden `action_tolerance` statt exakter
  Fließkomma-Gleichheit

### Verbindliche Exporttests

Teste:

- exakte CSV-Spalten der Value-Table
- exakte CSV-Spalten der Q-Table
- UTF-8 und Komma als Trennzeichen
- Fließkommawerte mit exakt sechs Nachkommastellen
- Hindernisse als Zeilen mit `is_obstacle=True`
- eindeutige Dateinamen ohne Überschreiben bestehender Dateien
- Export verändert weder Planner- noch Rollout-Daten

### Verbindliche GUI-Smoke-Tests

GUI-Tests dürfen klein bleiben, müssen aber prüfen:

- Hauptfenster kann vollständig aufgebaut und wieder geschlossen werden
- alle drei Auswertungs-Tabs sind vorhanden
- Value- und Q-Dialog verwenden `transient()`, `grab_set()` und
  `wait_window()`
- Tabellenansichten erzeugen für jede Grid-Zelle die erwartete Zeile
- nicht relevante oder widersprüchliche Controls werden während eines Laufs
  deaktiviert
- nach Methodenwechsel werden Werte des zuvor gewählten Planners wieder
  unverändert angezeigt
- Grid-, Policy- und Zellanzeige lassen sich ohne Überlappung aktualisieren

```bash
python -m unittest discover -s tests -p "test_gridworld_logic.py" -v
```

## Verbindliche Checkliste der 17 Verbesserungen

Die folgenden Punkte sind Teil der Spezifikation und dürfen bei der Umsetzung
nicht ausgelassen werden:

1. **Planner und Agent trennen:** `planner_max_iterations` steuert
   Bellman-Sweeps; `agent_loops` steuert sichtbare Agentenschritte.
2. **Verhalten nach Konvergenz:** Weitere Sweeps bleiben bis Reset oder
   relevanter Parameteränderung deaktiviert.
3. **Methodenwechsel:** Beide Planner behalten getrennte Tabellen, Historien
   und Iterationszähler.
4. **Tolerance einmal konfigurieren:** `tolerance` liegt im Planner; `run()`
   erhält sie nicht erneut.
5. **Terminalzustand:** `goal` liefert immer `(goal, 0, True)`; `step()` nach
   `done=True` erfordert zuerst `reset()`.
6. **Ergebnisobjekte:** `SweepResult`, `Transition`, `RolloutResult` und
   `ComparisonResult` sind eindeutig definiert und typisiert.
7. **Epsilon Decay:** `epsilon_greedy` und `epsilon_decay_current` sind getrennt;
   nur `epsilon_decay_current` sinkt nach einem vollständig abgeschlossenen
   automatischen Rollout.
8. **Vorgegebene Feldnamen:** `Epsilon (greedy)`, `Epsilon Max (decay)`,
   `Epsilon Min (decay)` und `Decay (decay)` bleiben exakt erhalten und werden
   verständlich erklärt.
9. **Fairer Reward-Vergleich:** Beide Methoden führen nach der Planung gleich
   viele Rollouts mit identischen Epsilon-Folgen und reproduzierbaren Seeds aus.
10. **Reward-Bedeutung:** Kumulativer Reward ist die Summe der Step-Rewards
    innerhalb eines Rollouts, nicht über mehrere Rollouts hinweg.
11. **Isolation der manuellen Demo:** Sie verändert weder Planner-Werte,
    Rollout-Statistik, Epsilon noch Iterationszähler.
12. **Seed-Regeln:** `reset()` startet RNGs nicht neu; `reset(seed=...)` schon.
    Tie-Breaking und epsilon-greedy Auswahl verwenden getrennte RNGs.
13. **Optimale Policy vergleichen:** Verglichen werden Mengen optimaler Actions
    pro Zustand unter Verwendung von `action_tolerance`; einzelne Pfade dürfen
    sich durch Tie-Breaking unterscheiden.
14. **CSV-Details:** UTF-8, Komma, sechs Nachkommastellen, Hinderniszeilen,
    eindeutige Dateinamen und keine Seiteneffekte.
15. **Echte Modalität:** Tabellenfenster verwenden `transient()`, `grab_set()`
    und `wait_window()`.
16. **Steuerbare Zellanzeige:** Checkbox `Policy anzeigen` und Dropdown für
    Keine, `V(s)`, beste Actions oder beide Darstellungen.
17. **Abbruch des Vergleichs:** Vollständige Ergebnisse bleiben sichtbar;
    teilweise berechnete Methoden werden als `Unvollständig` markiert und
    nicht als konvergiert bewertet.

## Akzeptanzkriterien

- Start mit `python gridworld_app.py`
- bestehende modellfreie App bleibt erhalten
- beide Methoden verwenden synchrone Bellman-Sweeps
- Epsilon beeinflusst nur Rollouts
- `Planner single sweep` führt exakt einen Sweep aus
- `Planner bis Konvergenz` stoppt bei Konvergenz oder Iterationslimit
- `Agent single step` führt exakt einen Agentenschritt aus
- `Agent run n loops` führt höchstens `agent_loops` Schritte aus und beginnt
  nach Episodenende keine zweite Episode
- ein nach `agent_loops` unvollständiger Rollout bleibt aktiv und wird beim
  nächsten Agentenbefehl fortgesetzt
- unvollständige Rollouts verändern weder Reward-Statistik noch Epsilon Decay
- Cancel behält fertige Werte
- Grid-Einstellungen werden atomar validiert
- manuelle und greedy Rollouts verändern keine Planner-Werte
- Reward- und Konvergenzdiagramm zeigen getrennte fachliche Aspekte
- beide Methoden sind gemeinsam vergleichbar
- beide konvergieren zum selben optimalen Startwert und zu denselben Mengen
  optimaler Actions unter Verwendung von `action_tolerance`
- modale Tabellen können korrekte CSV-Dateien exportieren
- lange Berechnungen bleiben responsiv
- alle Logiktests laufen ohne GUI
