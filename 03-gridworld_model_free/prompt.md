# Gridworld – Reinforcement-Learning-Demo

Projektordner: `Oliver/03-gridworld_model_free`

## Rolle und Zielgruppe

Erstelle die Anwendung für Reinforcement-Learning-Anfänger. Der Code soll
eigenständig lauffähig, klar strukturiert und verständlich benannt sein.
Kommentiere wichtige Schritte der Algorithmen kurz und hilfreich.

## Ziel

Implementiere eine lokale Gridworld-Anwendung, die tabellarische modellfreie
Reinforcement-Learning-Verfahren in einer Tkinter-GUI demonstriert.

Die modellbasierte Variante mit Value Iteration liegt getrennt unter
`../02-gridworld_model_based`. Dateien und Imports beider Projekte dürfen nicht
vermischt werden.

Die Anwendung unterstützt:

- manuelle Navigation
- schrittweise Interaktion mit einem Agenten
- vollständige Trainingsepisoden
- Vergleich mehrerer Lernalgorithmen
- Visualisierung von Pfaden, Returns, Zustandswerten, Aktionswerten und Policy

Verwende kein neuronales Netz. Alle Algorithmen arbeiten mit tabellarischen
Werten.

## Voraussetzungen

- Python 3.9 oder neuer
- Tkinter-GUI ohne Webserver
- Matplotlib für Diagramme
- Conda als empfohlene lokale Entwicklungsumgebung
- deterministische Aktionen
- reproduzierbare Experimente durch einen optionalen Random Seed
- responsive GUI während des Trainings

## Zu erstellende Dateien

- `gridworld_logic.py`: Environment, Policies, Agent, Training und Vergleichslogik;
  keine GUI-Abhängigkeiten
- `gridworld_gui.py`: Tkinter-Oberfläche und Matplotlib-Visualisierungen
- `gridworld_app.py`: minimaler Einstiegspunkt der Anwendung
- `README.md`: Installation, Bedienung, Algorithmen und Projektstruktur
- `requirements.txt`: externe Abhängigkeiten, insbesondere Matplotlib
- zentrale `../environment.yml` im Ordner `Oliver`: gemeinsames
  Conda-Environment `rl-26-08` für alle Projekte von Oliver
- `tests/test_gridworld_logic.py`: Unit-Tests für Environment und Algorithmen
- `.gitignore`: ignoriert `.conda`, `.venv`, Python-Caches, erzeugte Diagramme,
  CSV-Exporte und macOS-Metadaten

## Architektur und Verantwortlichkeiten

### `GridWorld`

Das Environment verwaltet:

- Größe des Grids
- Start, Ziel und blockierte Zellen
- aktuellen Zustand
- Schrittzähler
- Zustandsübergänge und Rewards
- Ende einer Episode

Es enthält keine Lernlogik und keinen GUI-Code.

### Policies

Jeder Lernalgorithmus wird als eigene Policy-Klasse implementiert:

- `MonteCarloPolicy`
- `SarsaPolicy`
- `ExpectedSarsaPolicy`
- `QLearningPolicy`

Alle Policies verwenden dieselbe Schnittstelle:

```text
select_action(state) -> action
update(state, action, reward, next_state, next_action, done) -> None
end_episode(trajectory) -> None
reset() -> None
get_q_value(state, action) -> float
get_state_value(state) -> float
get_best_action(state) -> action
```

Gemeinsame Konstruktorparameter der TD-Policies:

```text
Policy(
    actions: Sequence[int],
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_min: float,
    epsilon_decay: float,
    seed: Optional[int] = None,
)
```

`MonteCarloPolicy` erhält kein `alpha`, weil sie die Q-Werte wie angegeben
über ein inkrementelles Sample Average aktualisiert. Sie erhält die übrigen
relevanten Parameter wie `actions`, `gamma`, Epsilon-Werte und `seed`.

Policies, die nach jedem Schritt lernen, dürfen in `end_episode()` nichts tun.
Monte Carlo führt das Lernupdate in `end_episode()` anhand der vollständigen
Trajektorie aus.

### `Agent`

Der `Agent` erhält ein `GridWorld` und eine Policy über Dependency Injection.
Er verwaltet:

- Lebenszyklus einer Episode
- aktuelle und zuletzt abgeschlossene Trajektorie
- Kommunikation zwischen Environment und Policy
- Einzelschritte, vollständige Episoden und Training
- Episode-Returns und Trainingsstatistiken

Der Agent stellt mindestens diese Methoden bereit:

```text
set_policy(policy) -> None
start_episode(training=True) -> state
step(training=True) -> transition
run_episode(training=True) -> trajectory
end_episode() -> None
reset_training() -> None
```

Bei SARSA speichert der Agent das für das Update ausgewählte `next_action` und
verwendet es tatsächlich als `action` des folgenden Schritts. Es darf nicht
für denselben Übergang erneut ausgewählt und verworfen werden.

### GUI

Die GUI ist ausschließlich für Darstellung und Benutzereingaben zuständig.
Reward-Regeln, Zustandsübergänge und Lernformeln dürfen nicht in
`gridworld_gui.py` implementiert werden.

## Definition des Gridworld

### Standardkonfiguration

- `width`: `5`
- `height`: `3`
- `start`: `(0, 2)`
- `goal`: `(4, 2)`
- `blocked`: `(2, 1)` und `(2, 2)`
- `max_steps`: `20`

Zulässige Grid-Größe:

- mindestens `2 x 2`
- höchstens `15 x 15`

### Koordinatensystem

- obere linke Zelle: `(0, 0)`
- untere rechte Zelle: `(width - 1, height - 1)`
- Zustandsdarstellung überall: Tupel `(x, y)`

Environment, Policies, Agent, GUI, Tests und CSV-Exporte verwenden dieselbe
Zustandsdarstellung.

### Aktionen

- Action `0`: Up `(0, -1)`
- Action `1`: Down `(0, +1)`
- Action `2`: Left `(-1, 0)`
- Action `3`: Right `(+1, 0)`

Aktionen sind deterministisch. Eine ausgewählte gültige Aktion wird immer
ausgeführt.

Policies wählen grundsätzlich aus allen vier Aktionen. Bewegungen gegen den
Rand oder gegen blockierte Zellen bleiben mögliche Aktionen. Der Agent lernt
anhand des negativen Rewards, diese Bewegungen zu vermeiden.

### Rewards

- jeder normale Schritt: Reward `-1`
- Betreten des Ziels: Reward `0`

Durch diese Reward-Struktur lernt der Agent, das Ziel mit möglichst wenigen
Schritten zu erreichen.

### Ungültige Bewegung

Falls eine Aktion aus dem Grid heraus oder in eine blockierte Zelle führen
würde:

- bleibt der Agent im aktuellen Zustand
- erhält er Reward `-1`
- zählt die Aktion als Schritt
- endet die Episode nur, falls dadurch `max_steps` erreicht wird

### Ende einer Episode

Eine Episode endet, wenn:

- das Ziel erreicht wurde oder
- `max_steps` erreicht wurde

Das von `step()` zurückgegebene `info`-Dictionary enthält einen Grund:

```text
"goal_reached" oder "max_steps"
```

Nach `done=True` verweigert `GridWorld.step()` weitere Aktionen mit einem
verständlichen `RuntimeError`. Vor dem nächsten Schritt muss `reset()`
aufgerufen werden.

### Environment-API

```text
GridWorld(
    width: int,
    height: int,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: List[Tuple[int, int]],
    max_steps: int,
    seed: Optional[int] = None,
)

reset(seed: Optional[int] = None) -> state
step(action: int) -> (next_state, reward: float, done: bool, info: dict)
```

`reset()` ohne Seed setzt Zustand und Schrittzähler zurück, startet den
Zufallsgenerator aber nicht neu. `reset(seed=...)` initialisiert zusätzlich den
Zufallsgenerator reproduzierbar neu. Ein normaler Episodenreset darf nicht
immer wieder dieselbe Zufallsfolge von vorn beginnen.

### Validierung der Konfiguration

Ungültige Konfigurationen dürfen nicht automatisch verändert werden. Lehne
sie mit einer verständlichen Fehlermeldung ab.

Prüfe:

- `width` und `height` liegen jeweils zwischen `2` und `15`
- `start`, `goal` und alle `blocked`-Zellen liegen im Grid
- `start` und `goal` sind verschieden
- `start` und `goal` sind nicht blockiert
- `blocked` enthält keine Duplikate
- `max_steps` ist positiv
- mindestens ein gültiger Weg von `start` zu `goal` existiert
- der kürzeste gültige Weg ist nicht länger als `max_steps`

Verwende eine einfache Breitensuche, um Erreichbarkeit und Länge des kürzesten
Weges zu prüfen. Ist das Ziel nicht innerhalb von `max_steps` erreichbar, wird
die Konfiguration abgelehnt und nicht teilweise übernommen.

## Algorithmen

Alle Algorithmen verwenden eine epsilon-greedy Behavior Policy. Bei mehreren
gleich guten Aktionen wird zufällig zwischen ihnen gewählt.

Terminalzustände besitzen keinen zukünftigen Wert. Alle Q-Werte starten bei
`0.0`.

### Every-Visit Monte Carlo Control

- on-policy
- lernt Aktionswerte `Q(s,a)`, nicht nur `V(s)`
- speichert die vollständige Trajektorie einer Episode
- berechnet die diskontierten Returns nach Episodenende rückwärts
- aktualisiert jedes besuchte State-Action-Paar
- verwendet ein inkrementelles Sample-Average-Update
- berechnet den angezeigten Zustandswert als:

```text
V(s) = max_a Q(s,a)
```

### SARSA

Verwende das On-Policy-TD-Update:

```text
Q(s,a) = Q(s,a) + alpha * (
    reward + gamma * Q(next_state,next_action) - Q(s,a)
)
```

Bei einem terminalen Übergang ist der zukünftige Q-Wert `0`.

### Expected SARSA

Verwende den erwarteten Wert unter der aktuellen epsilon-greedy Policy:

```text
Q(s,a) = Q(s,a) + alpha * (
    reward + gamma * expected_Q(next_state) - Q(s,a)
)
```

Die Berechnung verteilt die Explorationswahrscheinlichkeit korrekt auf alle
Aktionen und die Exploitationswahrscheinlichkeit auf alle gleich guten besten
Aktionen.

### Q-Learning

Verwende das Off-Policy-Update:

```text
Q(s,a) = Q(s,a) + alpha * (
    reward + gamma * max_a Q(next_state,a) - Q(s,a)
)
```

Bei einem terminalen Übergang ist der zukünftige Q-Wert `0`.

## Lernparameter

Standardwerte:

- `episodes`: `100`
- `max_steps`: `20`
- `alpha`: `0.1`
- `gamma`: `0.9`
- `epsilon_start`: `1.0`
- `epsilon_min`: `0.05`
- `epsilon_decay`: algorithmusspezifischer Standardwert

Standardmäßige Epsilon-Verläufe:

```text
Monte Carlo:
epsilon = max(0.05, 1 / (1 + 0.001 * episode))

SARSA:
epsilon = max(0.05, 1 / (1 + 0.001 * episode))

Expected SARSA:
epsilon = max(0.05, exp(-0.0005 * episode))

Q-Learning:
epsilon = max(0.05, exp(-0.005 * episode))
```

Die GUI enthält eine standardmäßig aktivierte Checkbox
`Algorithmus-Standardwerte verwenden`.

Wenn sie aktiviert ist:

- wird das aktuelle `epsilon` angezeigt
- wird der algorithmusspezifische Verlauf verwendet
- werden `epsilon_start`, `epsilon_min` und `epsilon_decay` ausgeblendet

Wenn sie deaktiviert ist, werden diese Parameter angezeigt und folgende
Formel verwendet:

```text
epsilon = max(epsilon_min, epsilon_start * exp(-epsilon_decay * episode))
```

Validierung:

- `0 <= alpha <= 1`
- `0 <= gamma <= 1`
- `0 <= epsilon_min <= epsilon_start <= 1`
- `epsilon_decay > 0`
- `episodes` und `max_steps` sind positive Ganzzahlen

## Aufbau der GUI

### Grid-Konfiguration

Platziere diesen Bereich oben links:

- Grid-Breite und -Höhe
- Start X und Y
- Ziel X und Y
- blockierte Zellen im Format `(x,y), (x,y), ...`
- optionaler Random Seed
- Button `Grid anwenden und zurücksetzen`

Eine gültige neue Konfiguration setzt Environment, alle Policy-Tabellen,
Trajektorien, Returns und Diagramme zurück. Ungültige Eingaben verändern das
laufende Experiment nicht und erzeugen einen verständlichen Fehlerdialog.

### Grid-Darstellung

Platziere das Grid gut sichtbar rechts neben der Konfiguration.

Zeige:

- Zellkoordinaten in Hellgrau
- Startzelle in Grün
- Zielzelle in Gold
- blockierte Zellen in Dunkelgrau
- aktuellen Agenten als farbigen Kreis mit `A`
- zuletzt abgeschlossene Trajektorie
- Richtungspfeile zwischen aufeinanderfolgenden Zuständen
- wiederholte Besuche ohne Bewegung als kleine Schleife oder Zähler

Das Canvas passt sich an gültige Grid-Größen an und hält die Zellen quadratisch.

### Episoden-Konfiguration

Platziere diesen Bereich unter der Grid-Konfiguration:

- Policy-Dropdown
- Anzahl der Trainingsepisoden
- `max_steps`
- `gamma`
- `alpha` nur für SARSA, Expected SARSA und Q-Learning
- Checkbox `Algorithmus-Standardwerte verwenden`
- benutzerdefinierte Epsilon-Felder nur bei deaktivierten Standardwerten
- Auswahl zwischen `Einzelne Methode` und `Methoden vergleichen`
- Button `Parameter anwenden und Training zurücksetzen`

Ein Policy-Wechsel weist dem Agenten eine neue Policy mit leerer Tabelle zu.
Q-Werte dürfen nicht unbemerkt zwischen Algorithmen übernommen werden.
Nicht relevante Parameter werden vollständig ausgeblendet und hinterlassen
keine leeren Lücken. Insbesondere zeigt Monte Carlo kein `alpha`-Feld.

### Manuelle Steuerung

Stelle Buttons und Tastenkürzel bereit für:

- Hoch
- Runter
- Links
- Rechts

Manuelle Schritte:

- aktualisieren Environment und Pfaddarstellung
- verändern keine Q-Werte
- werden als manuelle Interaktion gekennzeichnet
- werden nicht in Trainings-Returns aufgenommen
- dienen ausschließlich der Demonstration und können nicht als
  Trainingsbeispiele übernommen werden

Manuelle Navigation und Agentenepisoden sind getrennte Modi. Beim Wechsel des
Modus wird das Environment auf `start` zurückgesetzt. Während einer aktiven
Agentenepisode sind manuelle Aktionen deaktiviert. Manuelle Aktionen verändern
weder Q-Werte noch Epsilon oder Episodenzähler des Trainings.

### Agentensteuerung

- Button `Einzelschritt`
- Button `Eine Episode ausführen`
- Button `N Episoden trainieren`
- Button `Training stoppen`
- Button `Gelernte Policy ausführen`
- Fortschrittsbalken mit abgeschlossenen und gesamten Episoden
- Button `Value-Tabelle öffnen`
- Button `Q-Tabelle und Policy öffnen`

Verhalten von `Einzelschritt`:

- startet automatisch eine Episode, falls keine aktiv ist
- führt genau einen Übergang aus
- aktualisiert beim Training die ausgewählte Policy
- beendet die Episode am Ziel oder bei `max_steps`
- startet beim nächsten Klick nach Episodenende eine neue Episode

### Greedy-Auswertung

Der Button `Gelernte Policy ausführen` demonstriert die aktuell gelernte
Policy:

- führt eine vollständige Episode mit `training=False` aus
- verwendet ausschließlich greedy Actions, entsprechend `epsilon = 0`
- verändert keine Q-Werte, Visit Counts oder Epsilon-Werte
- wird nicht in Trainings-Returns oder Episodenzähler aufgenommen
- hebt den ausgeführten Pfad im Grid deutlich hervor
- zeigt Return, Episodenlänge und Abbruchgrund an
- kann auch schrittweise animiert werden, ohne die GUI zu blockieren

Bei gleichwertigen besten Actions verwendet die Auswertung weiterhin das
definierte zufällige Tie-Breaking mit reproduzierbarem Seed.

Während des Trainings werden Einstellungen gesperrt, die das Experiment
ungültig machen könnten. Das Training läuft in kleinen Blöcken über `after()`
oder in einem Worker-Thread, damit die GUI responsiv bleibt. Tkinter-Widgets
werden ausschließlich im Hauptthread aktualisiert.

`Training stoppen` beendet den Lauf sicher nach dem aktuellen Schritt oder der
aktuellen Episode. Bereits abgeschlossene Episoden, Returns und Lernwerte
bleiben erhalten. Ein Stop führt keinen automatischen Reset aus.

## Dialog für die Value-Tabelle

Zeichne eine Kopie des Gridworld und zeige in jeder begehbaren Zelle:

```text
V(s) = max_a Q(s,a)
```

Markiere Start, Ziel und blockierte Zellen. Der Dialog aktualisiert sich beim
erneuten Öffnen oder durch den Button `Aktualisieren`.

Stelle einen Button zum CSV-Export der letzten Trajektorie bereit.

## Dialog für Q-Tabelle und Policy

Zeichne eine Kopie des Gridworld. Zeige für jeden begehbaren, nicht terminalen
Zustand:

- alle vier Werte für Up, Down, Left und Right
- den maximalen Wert hervorgehoben
- einen Pfeil für die Greedy-Action
- mehrere Pfeile bei gleichwertigen besten Aktionen

Falls ein Zustand noch nie besucht wurde und alle Q-Werte `0.0` sind, zeigt die
Policy-Ansicht ein `?` statt vier gleichwertiger Richtungspfeile.

Bezeichne `max_a Q(s,a)` allein nicht als Q-Tabelle. Dieser Wert gehört in den
Dialog der Value-Tabelle.

Stelle einen Button zum CSV-Export der letzten Trajektorie bereit.

## CSV-Export der Trajektorie

Spalten:

```text
episode,step,state_x,state_y,action,action_name,
next_state_x,next_state_y,reward,done,termination_reason,policy
```

Der Dateiname enthält:

- normalisierten Namen der Policy
- Episodennummer
- Zeitstempel

Speichere Exporte im Ordner `exports/`. Bestehende Dateien dürfen nicht
überschrieben werden.

## Diagramm der Episode-Returns

Das Diagramm ist Teil des Hauptfensters.

- x-Achse: Episode
- y-Achse: Return der Episode
- dünne helle Linie: Return einzelner Episoden
- dicke Linie: gleitender Mittelwert der letzten 20 Episoden
- Legende mit Algorithmusnamen und zugehörigen Farben
- Button `Diagramm speichern`
- Dateiname enthält Policy und Zeitstempel

Vor dem ersten Training erscheint der Hinweis:

```text
Trainiere den Agenten, um die Lernkurve zu sehen.
```

Bezeichne dieses Diagramm nicht als kumulativen Reward, solange keine
kumulierten Returns dargestellt werden.

## Vergleich der Algorithmen

Der Vergleichsmodus bewertet ausgewählte Algorithmen unter denselben
Bedingungen.

### Vergleichseinstellungen

- auswählbare Policies; standardmäßig sind alle ausgewählt
- `episodes_per_run`: Standardwert `500`
- `repetitions`: Standardwert `20`
- optionaler `base_seed`
- optionale 95-%-Konfidenzintervalle
- Button `Vergleich starten`
- Button `Vergleich stoppen`
- Button `Vergleich zurücksetzen`

### Regeln für einen fairen Vergleich

- jede Policy verwendet dasselbe Grid, `gamma`, `episodes` und `max_steps`
- alle TD-Policies verwenden dasselbe `alpha`; Monte Carlo verwendet wie
  definiert ein Sample-Average-Update ohne `alpha`
- jede Policy verwendet ihren eigenen oben definierten
  algorithmusspezifischen Epsilon-Verlauf
- jede Policy startet mit einer leeren Q-Tabelle
- jede Policy erhält dieselben `episodes` und `max_steps`
- Environment- und Policy-Seeds werden reproduzierbar aus `base_seed` abgeleitet
- mehrere Wiederholungen reduzieren den Einfluss einzelner Zufallsläufe
- Vergleichsläufe verändern weder interaktiven Agenten noch dessen Statistiken
- Policies laufen nacheinander oder in einem Worker-Thread
- Tkinter-Widgets werden ausschließlich im Hauptthread aktualisiert

`Vergleich stoppen` beendet die Berechnung nach der aktuellen Wiederholung.
Vollständig abgeschlossene Wiederholungen dürfen angezeigt werden; eine nur
teilweise berechnete Wiederholung wird nicht in die Statistik aufgenommen.

### Vergleichsdiagramm

Zeige alle ausgewählten Methoden im selben Diagramm:

- x-Achse: Episode
- y-Achse: mittlerer Episode-Return
- eine farbenblind-freundliche Linie pro Methode
- optionales 95-%-Konfidenzband
- deutliche Linie für den gleitenden Mittelwert über 20 Episoden
- verständliche Legende

Berechne das 95-%-Konfidenzintervall für jede Episode über die unabhängigen
Wiederholungen:

```text
confidence_interval = mean ± 1.96 * standard_deviation / sqrt(repetitions)
```

Bei nur einer Wiederholung wird kein Konfidenzintervall angezeigt.

Verwende konsistente Farben wie Blau, Orange, Grün und Violett. Rot und Grün
dürfen nicht die einzigen Unterscheidungsmerkmale sein.

Zeige unterhalb des Diagramms eine kompakte Tabelle mit:

- finalem gleitenden Durchschnitt des Returns
- mittlerer Episodenlänge
- Erfolgsquote beim Erreichen des Ziels
- 95-%-Konfidenzintervall

Erkläre, dass Ergebnisse von Grid-Aufbau, Parametern, Episodenzahl und Zufall
abhängen. Der Vergleich stellt keine allgemeingültige Rangliste dar.

## Reset-Verhalten

`Grid anwenden und zurücksetzen` setzt zurück:

- Environment
- alle Policy-Tabellen
- aktive und letzte Trajektorie
- Episodenzähler
- Returns
- Diagramm

`Parameter anwenden und Training zurücksetzen` behält das gültige Grid, setzt
aber Lernzustand und Trainingsstatistiken vollständig zurück.

Ein manueller Reset verändert eingegebene Einstellungen nicht unbemerkt.

## Fehlerbehandlung

- Fehlerhafte Koordinaten und Listen blockierter Zellen werden abgelehnt.
- Ungültige numerische Parameter werden abgelehnt.
- Die GUI zeigt verständliche Fehlermeldungen.
- Ungültige Konfigurationen werden niemals teilweise angewendet.
- Ein Training kann nicht versehentlich mehrfach gestartet werden.
- Lange Trainings- und Vergleichsläufe können sicher gestoppt werden.
- Fehler beim Datei-Export bringen die Anwendung nicht zum Absturz.

## `gridworld_app.py`

Halte den Einstiegspunkt minimal:

1. Tk-Root erzeugen
2. standardmäßiges `GridWorld` erzeugen
3. vier Policy-Objekte erzeugen
4. `Agent` mit der standardmäßigen `MonteCarloPolicy` erzeugen
5. GUI erzeugen und Root, Environment, Agent und Policies injizieren
6. `root.mainloop()` starten

Start der Anwendung:

```bash
python gridworld_app.py
```

## README

Dokumentiere:

- Ziel des Projekts
- Installation und virtuelles Environment
- Einrichtung und Aktivierung des lokalen Conda-Environments
- Startbefehl
- standardmäßiges Gridworld
- Reward- und Abbruchregeln
- Erklärung aller vier Algorithmen
- manuelle Interaktion und Training
- Vergleichsmodus
- Interpretation von Value-, Q-Tabelle und Policy
- CSV- und Diagrammexport
- Testbefehl

Einmalige Einrichtung im Repository-Root:

```bash
cd /Users/oliver/git/RL-26-08
conda env create -f Oliver/environment.yml
conda activate rl-26-08
python -c "import tkinter; print(tkinter.TkVersion)"
```

Alle Projekte im Ordner `Oliver` verwenden dasselbe Environment `rl-26-08`.
Die dokumentierte Tk-Version muss mindestens `8.6` sein. Conda und `.venv`
dürfen nicht gleichzeitig aktiviert werden.

## Tests

Verwende bei Zufallsverhalten deterministische Seeds. Implementiere Tests für:

- gültige und ungültige Environment-Konfiguration
- Erkennung eines nicht erreichbaren Ziels
- Ablehnung, wenn der kürzeste Weg länger als `max_steps` ist
- Grid-Grenzen und blockierte Zellen
- Abbruch am Ziel und durch `max_steps`
- `RuntimeError` bei `step()` nach Episodenende
- Reset-Verhalten
- Unterschied zwischen `reset()` und `reset(seed=...)`
- deterministisches Verhalten mit Seed
- Action-Nummerierung und State-Koordinaten
- Exploration, Exploitation und zufälliges Tie-Breaking bei epsilon-greedy
- diskontierte Monte-Carlo-Returns und Every-Visit-Updates
- SARSA-Update-Target
- tatsächliche Wiederverwendung von `next_action` im folgenden SARSA-Schritt
- wahrscheinlichkeitsgewichtetes Expected-SARSA-Target
- Q-Learning-Maximum-Target
- terminale Updates ohne zukünftigen Wert
- Epsilon-Verläufe und Mindestwert
- Policy-Wechsel ohne gemeinsam genutzte Lerntabellen
- Struktur der Trajektorie und CSV-Spalten
- reproduzierbare Vergleichsergebnisse
- Isolation des Vergleichs vom interaktiven Agenten
- Isolation manueller Demonstrationen vom Training
- greedy Auswertung ohne Veränderung von Lernwerten oder Trainingsstatistik
- Stoppen eines Trainings oder Vergleichs ohne Verlust abgeschlossener Daten

Tests starten:

```bash
python -m unittest discover -s tests -v
```

## Akzeptanzkriterien

- Die Anwendung startet mit `python gridworld_app.py`.
- Manuelle Navigation und alle Agentensteuerungen funktionieren.
- Manuelle Navigation ist vollständig vom Training isoliert.
- Alle vier Algorithmen trainieren ohne GUI-Abhängigkeiten.
- Monte Carlo lernt `Q(s,a)` anhand vollständiger Episode-Returns.
- Ungültige Grid-Konfigurationen verändern das aktive Grid nicht.
- Der Lebenszyklus bei Einzelschritten ist konsistent und sichtbar.
- SARSA verwendet das ausgewählte `next_action` im tatsächlich folgenden Schritt.
- Die gelernte Policy kann greedy und ohne weitere Updates ausgeführt werden.
- Pfade und Bewegungsrichtungen werden korrekt dargestellt.
- Value- und Q-Dialog zeigen unterschiedliche, korrekt definierte Werte.
- Das Training bleibt responsiv und kann gestoppt werden.
- Der Vergleichsmodus ist reproduzierbar und verändert den interaktiven Lauf nicht.
- Im Vergleich verwendet jede Methode ihren algorithmusspezifischen
  Epsilon-Verlauf.
- Diagramme und CSV-Trajektorien werden unter eindeutigen Dateinamen gespeichert.
- Unit-Tests laufen ohne Öffnen der GUI erfolgreich durch.
