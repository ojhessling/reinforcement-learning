# Basis-Prompt für Reinforcement-Learning-Workbench-Projekte

## Geltung

Diese Datei enthält die verbindlichen Grundregeln für alle
Reinforcement-Learning-Projekte im Ordner `Oliver`. Der projektspezifische
Prompt definiert Environment, Algorithmen, Rewards, Hyperparameter und
Besonderheiten und beginnt mit:

```text
Berücksichtige die verbindlichen Regeln aus ../workbench.md.
Projektname: {{project_name}}
Environment: {{environment_name}}
```

Bei Widersprüchen gilt folgende Priorität:

1. aktuelle Benutzeranweisung
2. projektspezifischer Prompt
3. diese Workbench-Spezifikation

Abweichungen von dieser Spezifikation müssen im projektspezifischen Prompt
ausdrücklich begründet werden.

## Ziel und Sprache

Erstelle eine eigenständig lauffähige lokale Python-Anwendung für Developer
und RL-Anfänger. Sie ermöglicht das Konfigurieren, Trainieren, Beobachten,
Evaluieren und Vergleichen der vorgegebenen Verfahren. Die Anwendung läuft
ohne Webserver, sofern der projektspezifische Prompt nichts anderes festlegt.

- GUI, Hilfen, Meldungen, README und Prompt: Deutsch
- Code-Bezeichner: Englisch
- Fachbegriffe dürfen verwendet werden, werden für Anfänger aber erklärt
- fachlich wichtige Stellen erhalten kurze, hilfreiche Kommentare

## Projektstruktur und Umgebung

Mindeststruktur im Ordner des projektspezifischen Prompts:

```text
{{project_name}}_app.py
{{project_name}}_logic.py
{{project_name}}_gui.py
README.md
requirements.txt
tests/
    test_{{project_name}}_logic.py
```

- Die App-Datei enthält nur den Entry Point.
- Logikmodule importieren kein Tkinter; die GUI enthält keine Lernformeln.
- Zusätzliche Module werden nur bei erkennbarem Nutzen angelegt.
- Laufzeitergebnisse und lokale virtuelle Umgebungen werden nicht committed.
- Alle Projekte verwenden `Oliver/environment.yml`.
- Direkte Abhängigkeiten stehen zusätzlich in `requirements.txt`; neue Pakete
  werden nur bei Bedarf aufgenommen und auf Konflikte geprüft.
- Neuronale Netze werden in diesem Kurs ausschließlich mit PyTorch umgesetzt.
  TensorFlow und Keras werden nicht verwendet.

## Architektur

### Environment

Das Environment verwaltet Observation- und Action-Space, Zustandsübergänge,
Rewards, Reset, Seed, `terminated`, `truncated` und gegebenenfalls Rendering.
Es enthält keine GUI- oder Lernlogik.

### Agenten und Runner

Jeder Algorithmus besitzt eine eigene Klasse mit möglichst einheitlicher
Schnittstelle für Reset, Action-Auswahl, Lernen, Episodenende, Metriken sowie
gegebenenfalls Speichern und Laden.

Training, Evaluation und Vergleich liegen nicht in Button-Callbacks, sondern
in getrennten Runnern oder gleichwertig klar getrennten Komponenten. Ergebnisse
werden durch Dataclasses oder eindeutig typisierte Strukturen statt
positionsabhängiger Tupel dargestellt.

### GUI

Die GUI ist für Eingaben, Validierungsfeedback, Runner-Steuerung,
Visualisierung, Status und Dialoge verantwortlich. Reward-Regeln, Lernupdates
und Action-Auswahl gehören nicht in die GUI.

## Fachliche RL-Regeln

### Training und Evaluation

Evaluation verwendet keine Exploration und keine Lernupdates. Sie verändert
weder Lernzustand noch Trainingsstatistiken und verwendet eigene Ergebnisse
sowie einen definierten Startzustand.

### Episodenende

Unterscheide durchgängig:

- `terminated`: fachlich terminaler Zustand
- `truncated`: externes Limit
- `done = terminated or truncated`

Ob bei Truncation gebootstrapt wird, muss je Algorithmus in Implementierung,
Tests und README konsistent festgelegt sein.

### Tie-Breaking und unbesuchte Zustände

Gleich gute Actions werden mit numerischer Toleranz reproduzierbar zufällig
ausgewählt. Die Policy-Ansicht zeigt alle gleichwertigen Actions. Unbesuchte
Werte erscheinen als `—` oder `?`, nicht wie gelernte Nullwerte.

### Reproduzierbarkeit

Ein angegebener Seed wird für Python, NumPy, Environment, Action-Space und
verwendete Frameworks gesetzt. Unabhängige Aufgaben erhalten getrennte
Zufallsgeneratoren. Ein Reset setzt Generatoren nur bei ausdrücklich
angegebenem Seed zurück.

## Parameter

- allgemeine Parameter sind immer sichtbar
- algorithmusspezifische Parameter erscheinen nur bei passenden Methoden
- Parameter werden fachlich gruppiert und so kompakt angeordnet, dass sie auf
  typischen Laptop-Auflösungen möglichst ohne Scrollen auf einen Blick sichtbar
  sind; lange, ungegliederte Ein-Spalten-Listen sind zu vermeiden
- Parameterbezeichnungen verwenden die üblichen englischen Fachnamen und,
  sofern vorhanden, zusätzlich das etablierte mathematische Symbol,
  beispielsweise `Learning Rate α`, `Discount Factor γ` oder `Exploration ε`;
  Symbole werden nicht künstlich erfunden, wenn es keine gebräuchliche
  Notation gibt. Die übrige Oberfläche und die Erklärungen bleiben deutsch
- komplexe Parameter erhalten kurze Erklärungen
- Eingaben werden vor einer Aktion vollständig validiert und atomar übernommen
- notwendige Resets des Lernzustands werden verständlich angezeigt
- Validierung berücksichtigt Wertebereiche, Abhängigkeiten sowie kompatible
  Netzwerk-, Buffer-, Batch- und Modelldatei-Konfigurationen
- Fehlermeldungen nennen Feld, ungültigen Wert und gültigen Bereich

### Neuronale Netze

Bei neuronalen Netzen müssen alle verwendeten Hyperparameter in der UI
änderbar sein. Dazu gehören je nach Verfahren insbesondere:

- Anzahl und Größe der Hidden Layers
- Aktivierungsfunktion
- Lernrate und Optimizer-Parameter
- Batch-Größe
- Initialisierung, Normalisierung, Regularisierung und Gradient Clipping
- Target-Network-Update und Anzahl der Gradientenschritte

Standardwerte werden aus einschlägiger Fachliteratur, den
algorithmusspezifischen Voreinstellungen von Stable-Baselines3 oder einem
passenden environmentspezifischen Profil des RL Baselines3 Zoo abgeleitet.
Environment-spezifische Profile haben Vorrang vor allgemeinen Defaults. Prompt
und README nennen Quelle, Algorithmus und Version beziehungsweise Profilstand.
Abweichungen werden fachlich begründet, nicht unterstützte Optionen in
der README dokumentiert.

## GUI-Design

- das Projekt verwendet ein konsistentes helles oder dunkles Farbschema; bei
  Dark Mode besitzen Texte, Eingabewerte, deaktivierte Controls, Achsen,
  Legenden und Statusmeldungen einen gut lesbaren Kontrast
- Kopfbereich mit Titel, Untertitel und Status
- Hauptbereich mit verschiebbarem horizontalem Splitter
- obere und untere Hälfte sind anfangs gleich hoch und frei skalierbar
- im oberen Bereich stehen das kompakt gruppierte Bedienpanel links und die
  Environment-Visualisierung rechts nebeneinander
- verwendet das Bedienpanel ein Drei-Spalten-Raster, enthalten die ersten
  beiden Spalten ausschließlich fachlich gruppierte Parameter; die dritte
  Spalte ist den Steuerungsbuttons vorbehalten
- Steuerungsbuttons stehen in dieser dritten Spalte untereinander, nutzen die
  volle Spaltenbreite und besitzen ausreichend große, einheitliche
  Klickflächen
- Eingabefelder und Auswahlfelder stehen innerhalb ihrer Parametergruppe
  rechtsbündig. Ihre Breite orientiert sich am längsten erwartbaren regulären
  Wert: so schmal wie sinnvoll, aber groß genug, dass typische Werte ohne
  Abschneiden oder horizontales Scrollen lesbar sind
- der untere Bereich nutzt die gesamte Fensterbreite für Diagramme, Vergleiche
  und Summary
- Diagramm beziehungsweise Vergleichsgraph und Summary sind im unteren Bereich
  gleichzeitig nebeneinander sichtbar; die Summary liegt nicht in einem
  separaten Tab und der Graph erhält den deutlich größeren Platzanteil
- Diagramm und Summary können über klar bezeichnete Aktionen exportiert
  werden. Der Graph wird mindestens als PNG in der aktuell dargestellten Form
  gespeichert; die Summary wird als gut lesbare UTF-8-Textdatei exportiert.
  Ein CSV-Export ist nicht erforderlich. Dateidialoge schlagen aussagekräftige
  Dateinamen vor und überschreiben bestehende Dateien nicht unbemerkt
- Bedienpanel und Visualisierung erhalten feste beziehungsweise gewichtete
  Platzanteile, sodass keines der beiden durch die Wunschgröße des anderen
  verdrängt oder auf 1 × 1 Pixel reduziert wird
- die Environment-Animation nutzt den gesamten verbleibenden Platz ihres
  Bereichs. Frames werden unter Beibehaltung ihres Seitenverhältnisses auf die
  größtmögliche vollständig sichtbare Größe skaliert; kein Teil des Frames darf
  abgeschnitten werden
- Tabs für sinnvolle Diagramme und Vergleiche
- alle wesentlichen Parameter und Steuerungsbuttons sind bei der
  Mindestfenstergröße gleichzeitig sichtbar; Scrollen ist nur ein Fallback für
  kleinere Fenster, nicht das Standardlayout
- stabile, lesbare Darstellung auf typischen Laptop-Auflösungen
- große Tabellen mit horizontaler und vertikaler Scrollbar

Fenstergröße und initiale Splitterposition werden nicht blind festgelegt,
sondern aus Bildschirmgröße, Mindestgröße der sichtbaren Controls und einem
Mindestplatz für Visualisierung beziehungsweise Diagramm abgeleitet. Beim
Programmstart darf kein wesentliches Widget abgeschnitten sein. Das Fenster
darf den nutzbaren Bildschirmbereich nicht unnötig überschreiten.

Das Layout wird mit einem realen GUI-Smoke-Test geprüft. Dabei müssen
Visualisierung, alle wesentlichen Buttons und der Diagrammbereich tatsächlich
gemappt sein und innerhalb des sichtbaren Fensters liegen. Dies gilt auch für
alle Eingabefelder, Auswahlfelder, Checkboxen und Fortschrittsanzeigen. Der Test
läuft mit der vorgesehenen Startfenstergröße und initialen Splitterposition.
Eine reine Konstruktion der Widgets reicht nicht als Layout-Test.

Controls spiegeln den Zustand `Bereit`, `Läuft`, `Gestoppt`, `Abgeschlossen`
oder `Fehler` wider. Inkompatible Aktionen werden gezielt deaktiviert und nach
Erfolg, Abbruch oder Fehler wieder freigegeben.

Jede App besitzt eine `Bedienungsanleitung`. Sie erklärt kurz den empfohlenen
Ablauf, Environment und Rewards, Methoden, Training gegenüber Evaluation,
Parameter, Ansichten und typische Ursachen ausbleibenden Lernerfolgs.

## Responsivität

Die GUI bleibt bei Training, Evaluation, Vergleich und Animation bedienbar.
Lange Arbeit läuft in kleinen `after()`-Schritten oder in Worker-Threads mit
Queue; Worker greifen nie direkt auf Tkinter-Widgets zu. Plot- und
Statusaktualisierungen werden auf eine sinnvolle Frequenz begrenzt.

Massentraining läuft ohne Einzelbildanimation. Sichtbare Episoden besitzen eine
abschaltbare Animation mit einer im Projekt sinnvoll festgelegten
Abspielgeschwindigkeit. Ein Feld für `Animation Δt` oder ein vergleichbarer
Geschwindigkeitsparameter wird nicht angeboten. Abbruch wird regelmäßig
geprüft; beim Schließen werden Worker und Ressourcen sauber beendet.

Auf macOS dürfen Tkinter und ein SDL-/Pygame-Renderer nicht im selben Prozess
initialisiert werden, wenn dies zu nativen Abstürzen führen kann. In diesem Fall
läuft ausschließlich das Rendering in einem isolierten, unsichtbaren Prozess
mit headless SDL-Treiber; die GUI erhält nur RGB-Frames. Der Hilfsprozess erzeugt
keinen zusätzlichen Dock-Eintrag und wird beim Schließen beendet.

## Visualisierung und Vergleich

Zeige nur für Environment und Algorithmus sinnvolle Metriken, beispielsweise
Episode-Return, gleitenden Durchschnitt, Erfolgsrate, Episodenlänge,
Exploration, Environment-Schritte und bei neuronalen Netzen den Loss.

- Achsen, Einheiten und Methoden sind beschriftet.
- Wenn neben dem Plot ausreichend Breite vorhanden ist, liegt die Legende in
  einem reservierten Bereich außerhalb der Achsen. Sie darf weder Datenlinien
  verdecken noch am Rand der Figure abgeschnitten werden.
- Rohwerte und geglättete Werte sind unterscheidbar.
- Training und Evaluation werden optisch getrennt.
- Deterministische Evaluationsergebnisse werden in der Summary ausgewiesen und
  müssen nicht zusätzlich im Trainingsgraphen dargestellt werden.
- Längere Trainings- und Vergleichsläufe werden in einem sichtbaren,
  konfigurierbaren Schrittintervall automatisch in einer separaten headless
  Environment deterministisch evaluiert. Dies erzeugt keine Animation und
  verändert weder Modell noch Replay Buffer. Der Graph und die Summary-Tabelle
  werden mit dem Ergebnis live aktualisiert.
- Der beste deterministische Evaluationswert eines Einzeltrainings wird gemerkt.
  Bei jeder Verbesserung werden Modell und zugehöriger Replay Buffer konsistent als
  gemeinsamer Checkpoint gesichert, in der Summary ausgewiesen und über einen
  klar beschrifteten Button wiederherstellbar gemacht. Wiederhergestellt wird der
  vollständige Lernzustand, nicht nur das neuronale Netz.
- Trainings- und Vergleichskurven verwenden einheitlich Episoden auf der
  X-Achse. Das Trainingsbudget und die tatsächlich ausgeführten
  Environment-Schritte bleiben separat in Status und Summary sichtbar.
- Fehlende Daten werden nicht durch künstliche Nullwerte ersetzt.
- Bei episodenbasierten Kurven entstehen Punkte nur für vollständig
  abgeschlossene Episoden. Endet ein Trainingsbudget innerhalb einer Episode,
  darf der letzte Kurvenpunkt deshalb vor dem tatsächlich ausgeführten
  Schrittbudget liegen. GUI und Summary zeigen ausgeführte Schritte,
  angefordertes Budget und diese Bedeutung getrennt und verständlich an.

Enthält ein Projekt mehrere Algorithmen, ist ein gemeinsamer Vergleichsgraph
verpflichtend. Er stellt alle Algorithmen mit derselben aussagekräftigen
X-Achse und derselben Metrik gegenüber. Farben,
Linienstile und Legende unterscheiden die Methoden eindeutig. Bei mehreren
Wiederholungen zeigt der Graph den Mittelwert und zusätzlich
Standardabweichung oder 95-%-Konfidenzintervall als Unsicherheitsband. Bei genau
einem Algorithmus entfällt der Algorithmenvergleich; unterschiedliche
Konfigurationen dürfen optional verglichen werden.

Die GUI erlaubt auszuwählen, welche der verfügbaren Algorithmen in den
Vergleich eingehen. Der Vergleichsgraph erscheint mit dem ersten verfügbaren
Ergebnis und wird während aller Läufe in einem sinnvollen Intervall
fortgeschrieben. Er darf nicht erst nach Abschluss des gesamten Vergleichs
angezeigt oder aktualisiert werden. Ausgewählte Algorithmen starten parallel;
die Fortschrittsanzeige aggregiert ihre tatsächlich ausgeführten Schritte.
Ein erneut gestarteter, kompatibel konfigurierter Vergleich setzt die
Vergleichsmodelle nicht zurück, sondern setzt ihr Training fort und hängt neue
Messpunkte an die vorhandenen Kurven an. Rohwerte werden dezent dargestellt;
pro Algorithmus hebt eine kräftige Linie den gleitenden Durchschnitt der
letzten 20 Episodenergebnisse hervor.
Dasselbe gilt für den normalen
Trainingsgraphen: aktuelle Episodenergebnisse werden während des Trainings
sichtbar. Parallel dazu wird auch die Summary live aktualisiert; sie zeigt für
Training und Vergleich konsistent Episoden, ausgeführte Environment-Schritte,
aktuelle beziehungsweise gemittelte Rewards und Erfolgsrate.

Lange Rohkurven werden nur für die Darstellung auf eine feste, angemessene
Punktzahl verdichtet; die Messdaten selbst bleiben vollständig erhalten. Eine
Min-/Max-Verdichtung ist einfachem Auslassen vorzuziehen, damit lokale Spitzen
und Einbrüche sichtbar bleiben. Plot-Updates werden zeitlich gedrosselt.

Vergleiche verändern das sichtbare Experiment nicht. Methoden erhalten
identische Environment-Konfigurationen, reproduzierbar abgeleitete Seeds und
ein vergleichbares Trainingsbudget, bevorzugt in Environment-Schritten. Jede
Methode und Wiederholung startet mit neuem Lernzustand; die Evaluation erfolgt
ohne Exploration und Lernupdates.

Vor dem Start wird der Gesamtumfang angezeigt. Mehrere Wiederholungen werden
mit Mittelwert und Standardabweichung oder 95-%-Konfidenzintervall aggregiert.
Bei Abbruch bleiben vollständige Ergebnisse erhalten und unvollständige werden
gekennzeichnet.

## Tabellen und Modelle

Tabellen zeigen den aktuellen, vollständigen Lernstand, unterscheiden besuchte
und unbesuchte Zustände und sind scrollbar.

Wenn Modelle gespeichert und geladen werden, enthalten sie Lernzustand,
Format-Version und notwendige Metadaten. Methode und Environment werden beim
Laden auf Kompatibilität geprüft. Fehlerhafte oder inkompatible Dateien dürfen
den aktiven Zustand nicht verändern.

## Fehlerbehandlung und Performance

- erwartbare Eingabefehler erscheinen als deutsche Dialogmeldung
- technische Fehler werden aufgefangen und verständlich gemeldet
- Busy-Zustände werden auch nach Fehlern beendet
- Ressourcen werden sauber freigegeben
- bestehende Dateien werden nicht ohne Nachfrage überschrieben

Zuerst entsteht eine korrekte, getestete Referenzimplementierung. Optimiert wird
nur nach Messung eines repräsentativen Laufs; Tests und Lernergebnis werden
danach erneut geprüft.

## Tests und Abnahme

Tests laufen nicht beim normalen App-Start. Sie prüfen mindestens:

- Environment-Übergänge, Rewards, Termination und Truncation
- Updateformeln, Action-Auswahl und Tie-Breaking jedes Algorithmus
- Parametergrenzen, Reset-Verhalten und Ergebnisobjekte
- Trennung von Training und Evaluation
- reproduzierbaren Lernfortschritt in einem kurzen Simulationsszenario
- kurzen Trainings-, Evaluations- und Vergleichslauf
- Import und Konstruktion der App-Komponenten

Bei neuronalen Netzen werden zusätzlich Ein- und Ausgabeformen, Targets, Loss,
Optimizer-Schritt, Target-Network-Update, Replay-Buffer sowie gegebenenfalls der
Save-/Load-Roundtrip getestet. Ein GUI-Smoke-Test wird nur mit verfügbarem
Display ausgeführt.

Ein Projekt ist abgeschlossen, wenn alle projektspezifischen Verfahren korrekt
implementiert sind, die GUI responsiv bleibt, Vergleiche fair und isoliert
ablaufen, fachlicher Lernfortschritt getestet ist und alle Tests erfolgreich
sind.

## README

Die README enthält:

- Ziel, Installation und Startbefehl
- Environment, Actions und Rewards
- Methoden und wesentliche Formeln in verständlicher Sprache
- Parameter, Standardwerte und Quellen
- Bedienablauf und Interpretation der Ansichten
- Vergleichslogik sowie Speichern und Laden, sofern vorhanden
- Testbefehl und bekannte Grenzen
