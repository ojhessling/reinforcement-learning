# Allgemeiner Basis-Prompt für Reinforcement-Learning-Workbench-Projekte

## Zweck

Diese Datei enthält verbindliche Grundregeln für alle Reinforcement-Learning-
Workbench-Projekte im Ordner `Oliver`.

Jedes Projekt besitzt zusätzlich einen projektspezifischen Prompt. Dieser
definiert Environment, Algorithmen, Rewards, Hyperparameter und besondere
Funktionen. Die allgemeinen Regeln aus dieser Datei müssen dort nicht
wiederholt werden.

## Verwendung

Ein projektspezifischer Prompt beginnt mit:

```text
Berücksichtige die verbindlichen Regeln aus ../workbench.md.
Projektname: {{project_name}}
Environment: {{environment_name}}
```

Platzhalter werden vor der Implementierung vollständig ersetzt. Verwende
einheitlich `{{placeholder_name}}` und keine Mischung aus eckigen Klammern,
Freitext und unterschiedlichen Schreibweisen.

## Priorität der Anforderungen

Bei Widersprüchen gilt:

1. ausdrückliche aktuelle Benutzeranweisung
2. projektspezifischer Prompt
3. diese allgemeine Workbench-Spezifikation

Ein projektspezifischer Prompt darf allgemeine Regeln nur ausdrücklich und mit
Begründung überschreiben.

## Rolle und Zielgruppe

Erstelle die Anwendung für Developer und Reinforcement-Learning-Anfänger.

- fachlich korrekte Algorithmen
- verständliche, klar getrennte Architektur
- nachvollziehbare Standardwerte
- kurze Kommentare an fachlich wichtigen Stellen
- keine unnötige Abstraktion
- keine vorausgesetzten RL-Kenntnisse in GUI und README

## Sprache und Benennung

- GUI-Texte, Hilfetexte, Statusmeldungen und Fehlermeldungen: Deutsch
- README und projektspezifischer Prompt: Deutsch
- Datei-, Modul-, Klassen-, Methoden- und Variablennamen: Englisch
- etablierte Fachbegriffe wie `Reward`, `Policy`, `Replay Buffer` oder
  `Q-Value` dürfen verwendet werden, müssen für Anfänger aber erklärt werden
- Schreibweisen innerhalb eines Projekts bleiben konsistent

## Ziel

Erstelle eine eigenständig lauffähige lokale Python-Anwendung, mit der sich
Reinforcement-Learning-Verfahren:

- konfigurieren
- trainieren
- schrittweise beobachten
- ohne Lernupdates evaluieren
- vergleichen
- anhand geeigneter Metriken verstehen

lassen.

Die Anwendung läuft ohne Webserver. Abweichungen, beispielsweise eine bewusst
gewählte Weboberfläche, müssen im projektspezifischen Prompt stehen.

## Projektordner und Dateien

Ausgabeordner ist der Ordner des projektspezifischen Prompts.

Mindeststruktur:

```text
{{project_name}}_app.py
{{project_name}}_logic.py
{{project_name}}_gui.py
README.md
requirements.txt
tests/
    test_{{project_name}}_logic.py
```

Optional bei größerem Umfang:

```text
{{project_name}}_agents.py
{{project_name}}_models.py
{{project_name}}_comparison.py
{{project_name}}_export.py
tests/test_{{project_name}}_integration.py
```

Regeln:

- `{{project_name}}_app.py` enthält ausschließlich den Entry Point
- `{{project_name}}_logic.py` und weitere Logikmodule importieren kein Tkinter
- `{{project_name}}_gui.py` enthält keine Lernformeln
- Tests liegen einheitlich unter `tests/`, nicht unter `test/`
- Laufzeitergebnisse liegen nur in ignorierten Ordnern wie `exports/`,
  `plots/` oder `models/`
- keine projektspezifische virtuelle Umgebung im Projektordner

## Python- und Conda-Umgebung

Alle Projekte von Oliver verwenden das zentrale Conda-Environment aus:

```text
Oliver/environment.yml
```

Bei neuen Abhängigkeiten:

- direkte Projektabhängigkeiten in `requirements.txt` dokumentieren
- zentrale `../environment.yml` ergänzen
- keine unnötigen oder ungenutzten Pakete aufnehmen
- Versionskonflikte mit bestehenden Projekten prüfen

## Architektur und Verantwortlichkeiten

### Environment

Das Environment verwaltet ausschließlich:

- Observation- und Action Space
- Zustandsübergänge
- Rewards
- `terminated` und `truncated`
- Reset und Seed
- gegebenenfalls Rendering und Environment-spezifische Statistiken

Es enthält keine GUI- oder Lernlogik.

### Agent oder Policy

Jeder Algorithmus besitzt eine eigene Klasse mit einer möglichst gemeinsamen
Schnittstelle:

```text
reset(seed=None) -> None
select_action(observation, training=True) -> action
observe(transition) -> None
end_episode() -> None
get_metrics() -> Dict[str, object]
save(path) -> None
load(path) -> None
```

Nicht benötigte Methoden dürfen als klar dokumentierte No-Op implementiert
werden. Unterschiedliche Algorithmen dürfen intern verschieden arbeiten, müssen
aber nach außen vergleichbare Ergebnisobjekte liefern.

### Runner

Training, Evaluation und Vergleich werden nicht in Button-Callbacks
implementiert. Verwende getrennte Runner:

- `TrainingRunner`
- `EvaluationRunner`
- `ComparisonRunner`

Sie kapseln Schleifen, Abbruchprüfungen, Fortschritt und Ergebnisaggregation.

### Ergebnisobjekte

Verwende Dataclasses oder eindeutig typisierte Strukturen, beispielsweise:

```text
Transition
EpisodeResult
TrainingPoint
EvaluationResult
ComparisonResult
```

Vermeide uneindeutige Tupel mit positionsabhängiger Bedeutung.

### GUI

Die GUI ist ausschließlich verantwortlich für:

- Eingaben und Validierungsfeedback
- Starten und Stoppen von Runnern
- Visualisierung
- Status und Fortschritt
- Dialoge
- Datei-Auswahl für Exporte und Modelle

Reward-Regeln, Updates und Action-Auswahl gehören nicht in die GUI.

## Fachliche RL-Regeln

### Training und Evaluation trennen

Evaluation:

- verwendet keine Exploration, sofern nicht ausdrücklich anders definiert
- führt keine Lernupdates aus
- verändert weder Tabellen, Netze, Replay Buffer noch Trainingsstatistiken
- verwendet eigene Ergebnislisten
- startet in einem definierten Environment-Zustand

### Episode-Ende

Unterscheide überall:

- `terminated`: fachlich terminaler Zustand
- `truncated`: externes Limit, beispielsweise `max_steps`
- `done = terminated or truncated`

Dokumentiere für jeden Algorithmus ausdrücklich, ob bei Truncation gebootstrapt
wird. Die Entscheidung muss in Implementierung, Tests und README konsistent
sein.

### Zufälliges Tie-Breaking

Bei mehreren gleich guten Actions:

- keine systematische Bevorzugung des kleinsten Action-Index
- reproduzierbare zufällige Auswahl
- numerische Toleranz bei Fließkommawerten
- Policy-Ansicht zeigt alle gleichwertigen Actions

### Unbesuchte Zustände

Unbesuchte Werte werden in Tabellen und Visualisierungen als `—` oder `?`
dargestellt. Ein Initialwert `0` darf nicht fälschlich wie ein gelernter Wert
wirken.

### Reproduzierbarkeit

Seed, sofern verwendet:

- Python `random`
- NumPy
- Environment und Action Space
- Frameworks wie PyTorch
- getrennte Zufallsgeneratoren für unabhängige Aufgaben, beispielsweise
  Exploration, Tie-Breaking und Vergleich

Ein Reset startet Zufallsgeneratoren nur dann neu, wenn dies ausdrücklich über
einen Seed angefordert wird.

## Parameter

### Sichtbarkeit

- allgemeine Parameter immer anzeigen
- algorithmusspezifische Parameter nur für relevante Methoden anzeigen
- irrelevante Felder ausblenden oder deaktivieren
- direkt bei komplexen Parametern kurze Erklärungen anzeigen
- sinnvolle, im projektspezifischen Prompt begründete Standardwerte verwenden

### Übernahme

- geänderte Werte werden beim Start einer Aktion automatisch validiert und
  übernommen
- ein zusätzlicher Anwenden-Button ist optional
- Übernahme erfolgt atomar: entweder alle Eingaben sind gültig oder das aktive
  Experiment bleibt unverändert
- strukturelle Änderungen setzen betroffene Lernzustände bewusst zurück
- Änderungen ohne Einfluss auf gelernte Werte sollen den Lernzustand erhalten
- ein notwendiger Reset wird vor der Ausführung verständlich angezeigt

### Validierung

Prüfe mindestens:

- Wertebereiche
- positive Ganzzahlen
- Beziehungen zwischen Parametern
- erreichbare Ziele, sofern relevant
- Buffer- und Batch-Größen
- Netzwerkarchitekturen
- optionale Seeds
- vorhandene Pfade und kompatible Modelldateien

Fehlermeldungen nennen Feld, ungültigen Wert und gültigen Bereich.

## GUI-Design

Die Oberfläche wirkt wie ein übersichtliches Lernlabor.

### Layout

- Kopfbereich mit Titel, Untertitel und Status
- linkes oder oberes Bedienpanel
- großer Visualisierungsbereich
- Summary mit wichtigsten Metriken
- Tabs für Diagramme und Vergleiche
- scrollbareres Bedienpanel bei kleinen Bildschirmen
- lesbar auf typischen Laptop-Auflösungen
- sinnvolle Mindestgröße

### Layout-Stabilität

- wechselnde Parameter dürfen das Fenster nicht unkontrolliert vergrößern
- Diagramme und Environment teilen den verfügbaren Platz über Gewichte
- lange Texte umbrechen
- große Tabellen besitzen horizontale und vertikale Scrollbars
- Zellen und Beschriftungen überlappen nicht
- Widgets springen während Updates nicht zwischen Positionen

### Zustände der Controls

- während inkompatibler Aktionen Buttons gezielt deaktivieren
- `Stoppen` nur während laufender Arbeit aktivieren
- nach Erfolg, Abbruch oder Exception alle Controls zuverlässig wiederherstellen
- Status unterscheidet `Bereit`, `Läuft`, `Gestoppt`, `Abgeschlossen` und
  `Fehler`
- deaktivierte Oberfläche darf nicht wie ein Programmabsturz wirken

### Bedienungsanleitung

Jede App besitzt einen Button `Bedienungsanleitung`. Der modale Dialog erklärt:

- empfohlenen Ablauf
- Environment und Rewards
- Methoden
- Training gegenüber Evaluation
- Parameter
- Diagramme und Tabellen
- typische Gründe für ausbleibenden Lernerfolg

## Asynchronität und Responsivität

Die GUI darf während Training, Evaluation, Vergleich, Export oder Animation
nicht einfrieren.

- keine langen Schleifen im Tkinter-Thread
- keine `sleep()`-Aufrufe im Tkinter-Thread
- kleine Arbeitspakete mit `after()` oder Worker-Thread plus Queue
- Worker greifen niemals direkt auf Tkinter-Widgets zu
- GUI liest Queues über `after()` in einem angemessenen Intervall, üblicherweise
  `50–200 ms`
- `10 ms` nur, wenn Messungen zeigen, dass es notwendig und effizient ist
- Plot-Updates begrenzen, beispielsweise auf `5–10` Aktualisierungen pro
  Sekunde
- Massentraining ohne Einzelbildanimation
- sichtbare Episoden und Evaluationen besitzen eine Option
  `Animation anzeigen`
- das Animationsintervall ist in Millisekunden einstellbar; Standardwert
  `10 ms`, sofern der projektspezifische Prompt nichts anderes festlegt
- bei deaktivierter Animation ohne Einzelbilder ausführen und nur das
  Endergebnis darstellen
- das Intervall steuert die Zeit zwischen zwei sichtbaren Agentenschritten und
  nicht die Trainingsgeschwindigkeit des Massentrainings
- Abbruch regelmäßig an sicheren Grenzen prüfen
- beim Schließen Worker beenden und Ressourcen freigeben

Fortschrittsmeldungen enthalten nach Möglichkeit:

```text
Methode
Episode oder Trainingsschritt
Wiederholung
Gesamtfortschritt
geschätzter Arbeitsumfang
```

## Visualisierung und Metriken

Zeige nur Metriken, die für das konkrete Environment und den Algorithmus
sinnvoll sind.

Typische Metriken:

- Episode-Return
- gleitender Durchschnitt
- Erfolgsrate
- Episodenlänge
- Exploration beziehungsweise Epsilon
- Environment-Schritte
- Loss bei neuronalen Netzen
- Environment-spezifische Fehler oder Risiken

Regeln:

- Rohwerte dünn und transparent
- geglättete Werte deutlich hervorgehoben
- Achsen, Einheiten und Methode beschriften
- Legende bei mehreren Methoden
- keine künstlichen Nullwerte für noch nicht vorhandene Messungen
- Training und Evaluation optisch unterscheiden

## Methodenvergleich

Ein Vergleich verändert das sichtbare Experiment nicht.

### Fairness

- identische Environment-Konfiguration
- identische Trainingsbudgets
- gleiche, reproduzierbar abgeleitete Seeds
- neuer Lernzustand pro Methode und Wiederholung
- Evaluation ohne Exploration und ohne Lernupdates
- gleiche Evaluationsmetriken
- Laufzeit zusätzlich, aber nicht als alleinige Qualitätsmetrik

Bei sehr unterschiedlichen Algorithmen wird ein gemeinsames Budget verwendet,
bevorzugt Environment-Schritte statt nur Episoden.

### Wiederholungen

Zeige den Gesamtumfang vor dem Start:

```text
Anzahl Methoden × Trainingsbudget × Wiederholungen
```

Aggregiere:

- Mittelwert
- Standardabweichung oder 95-%-Konfidenzintervall
- Erfolgsrate
- relevante Environment-Metriken

Bei einer Wiederholung hat das Konfidenzintervall die Breite `0` und wird nicht
irreführend dargestellt.

### Abbruch

- bereits abgeschlossene Ergebnisse bleiben erhalten
- unvollständige Ergebnisse werden markiert
- sichtbares Experiment bleibt unverändert

## Tabellen, Export und Modelle

### Tabellen

- beim Öffnen aktueller Lernstand
- vollständige Zustands- oder Parameterliste
- besuchte und unbesuchte Zustände unterscheiden
- modaler, scrollbar bedienbarer Dialog
- sinnvolle Hervorhebung von Start, Ziel, terminalen und ungültigen Zuständen

### CSV-Export

- UTF-8
- Komma als Trennzeichen
- stabile Spaltenreihenfolge
- Fließkommazahlen standardmäßig mit sechs Nachkommastellen
- eindeutiger Dateiname
- keine Änderung des Experiments

### Modelle

Wenn Speichern und Laden unterstützt werden:

- Lernzustand plus notwendige Metadaten speichern
- Format-Version aufnehmen
- Methode und Environment-Kompatibilität beim Laden prüfen
- inkompatible Datei verändert den aktiven Zustand nicht
- Dateifehler verständlich melden

## Fehlerbehandlung und Lebenszyklus

- erwartbare Eingabefehler als deutsche Dialogmeldung
- technische Exceptions im Runner auffangen und an GUI melden
- Busy-Zustand auch nach Fehlern zuverlässig beenden
- keine leeren `except`-Blöcke
- Ressourcen über `close()` freigeben
- temporäre Dateien nur in geeigneten temporären Ordnern
- bestehende Dateien nicht ohne Nachfrage überschreiben

## Performance

Optimiere erst nach einer korrekten, getesteten Referenzimplementierung.

Vorgehen:

1. Korrektheit mit kleinen deterministischen Tests sichern.
2. Repräsentativen Trainingslauf messen.
3. tatsächlichen Engpass bestimmen.
4. gezielte Optimierung implementieren.
5. Tests erneut ausführen.
6. Laufzeit und Lernergebnis mit derselben Konfiguration vergleichen.

Keine pauschale Forderung nach mehreren Optimierungsrunden ohne Messung.
Lernerfolg darf nicht durch eine fachlich falsche Abkürzung erkauft werden.

## Tests

Tests werden nicht beim normalen App-Start ausgeführt.

### Unit-Tests

Prüfe mindestens:

- Environment-Übergänge und Rewards
- Termination und Truncation
- Updateformel jedes Algorithmus
- Action-Auswahl und Tie-Breaking
- Parametergrenzen
- Reset-Verhalten
- Ergebnisobjekte
- Exportdaten ohne Dateidialog

### Simulationstests

Für jeden Algorithmus:

- kleines, schnelles und möglichst deterministisches Szenario
- Nachweis, dass ein definierter Lernfortschritt möglich ist
- feste Seeds
- robuste Schwelle statt exakter zufälliger Trajektorie
- sinnvolles Zeitlimit

Ein Simulationstest darf nicht nur prüfen, dass kein Fehler auftritt. Er prüft
eine fachliche Eigenschaft, beispielsweise verbesserter Return, gelernte Action
oder steigende Erfolgsrate.

### Neuronale Netze

Nur wenn das Projekt neuronale Netze verwendet:

- Ein- und Ausgabeformen
- terminale Targets
- Loss und Optimizer-Schritt
- Target-Network-Update
- Replay-Buffer
- Save-/Load-Roundtrip
- kurzer Trainingstest

Tabellarische Projekte benötigen keine künstlichen Netzwerkklassen oder
Netzwerktests.

### Integrations- und Smoke-Tests

- Module importierbar
- App-Komponenten konstruierbar
- kurzer Trainings- und Evaluationslauf
- Vergleich mit minimaler Konfiguration
- GUI-Smoke-Test nur, wenn ein Display verfügbar ist

### Prüfablauf pro Algorithmus

1. Updateformel fachlich prüfen.
2. Termination und Truncation prüfen.
3. relevanten Parameterumfang der GUI prüfen.
4. kurzen Simulationstest ausführen.
5. Lernergebnis und Laufzeit dokumentieren.
6. nur bei nachgewiesenem Bedarf optimieren.
7. Regressionstests vollständig erneut ausführen.

## README

Jedes Projekt besitzt eine README mit:

- Ziel der Anwendung
- Installation und Environment-Aktualisierung
- Startbefehl
- Environment, Actions und Rewards
- Methoden und wesentliche Formeln in verständlicher Sprache
- Parameter und Standardwerte
- empfohlener Bedienablauf
- Interpretation von Metriken und Diagrammen
- Vergleichslogik und Wiederholungen
- Exporte und Modelle
- Testbefehl
- bekannte Grenzen

## Arbeitsweise bei der Implementierung

1. Projektspezifischen Prompt vollständig lesen.
2. Widersprüche und fehlende Entscheidungen benennen.
3. Environment und Ergebnisobjekte implementieren.
4. Algorithmen einzeln implementieren und testen.
5. Training, Evaluation und Vergleich ergänzen.
6. GUI mit dynamischen Parametern anbinden.
7. Exporte und Dokumentation ergänzen.
8. vollständige Testsuite ausführen.
9. repräsentativen Smoke-Test durchführen.
10. Änderungen und bekannte Grenzen knapp dokumentieren.

## Abnahmekriterien

Ein Workbench-Projekt ist abgeschlossen, wenn:

- alle projektspezifischen Methoden korrekt implementiert sind
- Training und Evaluation getrennt sind
- GUI während langer Aktionen bedienbar bleibt
- Abbruch und Fehler Controls zuverlässig wieder aktivieren
- nur relevante Parameter sichtbar sind
- Eingaben automatisch und atomar übernommen werden
- Vergleiche isoliert, reproduzierbar und fair sind
- unbesuchte Werte klar erkennbar sind
- Tests fachliche Eigenschaften und Lernfortschritt prüfen
- alle Tests erfolgreich sind
- README und Bedienungsanleitung vollständig sind
- keine generierten Dateien oder lokalen Umgebungen committed werden
