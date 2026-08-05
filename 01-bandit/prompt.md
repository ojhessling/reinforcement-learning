# Multi-Armed Bandit

## Rolle und Zielgruppe

Erstelle die Anwendung für einen Reinforcement-Learning-Anfänger. Der Code soll
übersichtlich, gut benannt und an wichtigen Stellen kurz kommentiert sein.

## Ziel

Erstelle eine lokale Python-App mit drei Multi-Armed Bandits. Die Anwendung soll
den Unterschied zwischen Exploration und Exploitation anschaulich und
interaktiv vermitteln.

## Voraussetzungen

- Lokale Python-Anwendung ohne Webserver
- GUI mit Tkinter
- Plot mit Matplotlib
- Drei Bernoulli-Banditen mit festen Erfolgswahrscheinlichkeiten:
  - Bandit 1: 0,2
  - Bandit 2: 0,5
  - Bandit 3: 0,8
- Jeder Pull liefert entweder Reward `1` (Erfolg) oder `0` (Misserfolg).
- Externe Abhängigkeiten werden in `requirements.txt` aufgeführt.

## Projektstruktur

- `bandit_app.py`: ausschließlich Einstiegspunkt; erzeugt Environment, Agent
  und GUI und startet die Anwendung
- `bandit_gui.py`: Tkinter-Oberfläche und Matplotlib-Visualisierung
- `bandit_logic.py`: komplette Spiellogik ohne Abhängigkeit von der GUI
- `requirements.txt`: externe Abhängigkeiten, insbesondere Matplotlib
- `README.md`: kurze Installations- und Startanleitung

## Logik und Klassenstruktur

Implementiere in `bandit_logic.py`:

- eine Environment-Klasse für die drei Banditen
- eine gemeinsame Agent- beziehungsweise Strategie-Schnittstelle mit den
  Methoden:
  - `select_action()`
  - `update(action, reward)`
  - `reset()`
- eine eigene Klasse für jede Strategie:
  - `EpsilonGreedyAgent`
  - `EpsilonDecayAgent`
  - `ThompsonSamplingAgent`
  - `UCBAgent`
  - `SoftmaxAgent`

Alle Strategien sollen dieselben Methoden verwenden, damit die GUI unabhängig
von der gewählten Strategie bleibt.

## Strategien und Parameter

### Epsilon Greedy

- `epsilon`: Standardwert `0.1`

Mit Wahrscheinlichkeit `epsilon` wird zufällig exploriert. Andernfalls wird der
Bandit mit dem aktuell höchsten geschätzten Erwartungswert gewählt.

### Epsilon Decay

- `epsilon_start`: Standardwert `0.999`
- `epsilon_min`: Standardwert `0.001`
- `epsilon_decay`: Standardwert `0.05`

Nach jedem Pull wird Epsilon nach folgender Formel reduziert:

```text
epsilon = max(epsilon_min, epsilon * (1 - epsilon_decay))
```

### Thompson Sampling

- Verwende für jeden Banditen eine Beta-Verteilung mit dem Prior
  `Beta(1, 1)`.
- Aktualisiere die Parameter anhand von Erfolgen und Misserfolgen.

### UCB

- Jeder Bandit wird zu Beginn mindestens einmal gezogen.
- Verwende anschließend UCB1 zur Auswahl des nächsten Banditen.

### Boltzmann/Softmax

- `temperature`: Standardwert `1.0`
- Die Temperatur muss größer als `0` sein.
- Berechne die Auswahlwahrscheinlichkeiten numerisch stabil.

## Agent Memory

- Eingabefeld `Last N Pulls`, Standardwert `0`
- `0` bedeutet: Alle bisherigen Pulls werden berücksichtigt.
- Ein Wert größer als `0` bedeutet: Für Schätzwerte werden nur die letzten
  `N` Ergebnisse je Bandit berücksichtigt.
- Falls diese Begrenzung für eine Strategie fachlich nicht sinnvoll ist, zum
  Beispiel für Thompson Sampling oder UCB, wird das Feld vollständig
  ausgeblendet und die vollständige Historie verwendet.

## GUI

### Gestaltungsziel

Die Oberfläche soll wie ein übersichtliches, modernes Lernexperiment wirken
und nicht wie ein technisches Konfigurationsformular. Verwende eine klare
visuelle Hierarchie, ausreichend Abstände, gut lesbare Beschriftungen und eine
ruhige, zusammenhängende Farbpalette.

Die wichtigsten Aktionen und Ergebnisse müssen ohne Scrollen sichtbar sein.
Technische Parameter werden nur dann angezeigt, wenn sie für die aktuell
gewählte Strategie relevant sind.

### Aufbau der Oberfläche

Gliedere das Hauptfenster von oben nach unten in vier Bereiche:

1. **Kopfbereich**
   - Titel `Multi-Armed Bandit Lab`
   - kurze Erklärung: `Entdecke, wie ein Agent zwischen Ausprobieren und dem
     Nutzen seiner bisherigen Erfahrung entscheidet.`
2. **Experiment-Einstellungen**
   - Strategieauswahl
   - kurze, leicht verständliche Erklärung der ausgewählten Strategie
   - ausschließlich die dazu passenden Parameter
   - Agent Loops, optionaler Random Seed und Start-/Reset-Steuerung
3. **Drei Banditenkarten**
   - drei gleich große Karten nebeneinander
   - jeweils eine grafische Darstellung eines einarmigen Banditen
   - manuelle Interaktion und kompakte Statistik direkt auf der Karte
4. **Auswertung**
   - Umschaltung zwischen Einzellauf und Strategievergleich
   - Reward- beziehungsweise Vergleichsplot
   - zusammengefasste Agentenwerte und aktuelle Statusmeldung

Bei kleineren Fenstern darf sich der Auswertungsbereich unter die
Banditenkarten verschieben. Das Fenster soll eine sinnvolle Mindestgröße haben,
aber nicht größer als der verfügbare Bildschirm geöffnet werden.

### Darstellung der Banditen

- Zeichne für jeden Banditen eine einfache Spielautomaten-Illustration mit
  Tkinter Canvas. Es werden keine externen Bilddateien benötigt.
- Jeder Automat erhält eine eigene Akzentfarbe und die deutlich sichtbare
  Beschriftung `Bandit 1`, `Bandit 2` oder `Bandit 3`.
- Zeige im Display des Automaten nach einem Pull beispielsweise `⭐ Reward 1`
  oder `– Reward 0` an.
- Der Hebel oder ein direkt darunter angeordneter Button `Ziehen` löst einen
  manuellen Pull dieses Banditen aus.
- Beim Pull soll eine kurze visuelle Rückmeldung erfolgen, etwa ein farbiger
  Displaywechsel. Die Animation darf die GUI nicht blockieren.
- Hebe den vom Agenten zuletzt gewählten Banditen kurz visuell hervor.
- Die wahren Erfolgswahrscheinlichkeiten sind standardmäßig verborgen, damit
  der Nutzer zunächst selbst explorieren kann.
- Eine Checkbox `Wahre Wahrscheinlichkeiten anzeigen` blendet die Werte 20 %,
  50 % und 80 % ein oder aus.

### Strategieauswahl und dynamische Parameter

- Dropdown zur Strategieauswahl:
  - Epsilon Greedy
  - Epsilon Decay
  - Thompson Sampling
  - UCB
  - Boltzmann/Softmax
- Zeige unter dem Dropdown einen kurzen Erklärungstext in Alltagssprache, der
  sich beim Strategiewechsel aktualisiert.
- Eingabefeld `Agent Loops`, Standardwert `100`
- Optionales Eingabefeld `Random Seed`; leer bedeutet zufälliger Lauf
- Zeige ausschließlich die Parameter der ausgewählten Strategie:
  - Epsilon Greedy: `Epsilon` und `Last N Pulls`
  - Epsilon Decay: `Epsilon Start`, `Epsilon Minimum`, `Epsilon Decay` und
    `Last N Pulls`
  - Thompson Sampling: keine zusätzlichen Parameter
  - UCB: keine zusätzlichen Parameter
  - Boltzmann/Softmax: `Temperatur` und `Last N Pulls`
- Nicht relevante Eingabefelder dürfen nicht nur deaktiviert, sondern sollen
  vollständig aus dem Layout entfernt werden. Beim Strategiewechsel wird das
  Layout ohne leere Lücken neu aufgebaut.
- Bereits eingegebene Werte bleiben beim Wechsel der Strategie erhalten.
- Ergänze zu jedem sichtbaren Parameter einen kurzen Tooltip oder Hilfetext.

### Steuerung

- Pro Banditenkarte ein Button `Ziehen` für manuelle Pulls
- Primärer Button `Agent starten` für die eingestellte Anzahl Loops
- Sekundärer Button `Einzelschritt`
- Button `Experiment zurücksetzen`
- Während eines Agentenlaufs:
  - manuellen Pull und Strategieänderung vorübergehend sperren
  - Fortschritt als `Schritt X von N` anzeigen
  - GUI weiterhin responsiv halten

Manuelle Pulls aktualisieren die Statistik des Environments, aber nicht das
Wissen oder den Reward-Plot des Agenten. Dadurch bleiben manuelles Ausprobieren
und Agentenlauf klar getrennt.

`Reset` setzt Environment, Agent, Statistiken, aktuellen Reward und Plot zurück.
Die gewählte Strategie und die eingetragenen Einstellungen bleiben erhalten.

## Strategievergleich

Ergänze neben dem interaktiven Einzellauf einen separaten Vergleichsmodus. Der
Nutzer soll damit erkennen können, wie sich die Strategien über viele Schritte
hinweg unterscheiden.

### Umschaltung

- Verwende im Auswertungsbereich zwei Tabs oder eine gut sichtbare Umschaltung:
  - `Aktueller Lauf`
  - `Strategien vergleichen`
- Der aktuelle interaktive Lauf bleibt erhalten, wenn zwischen den Ansichten
  gewechselt wird.
- Der Vergleich läuft getrennt vom interaktiven Environment und verändert
  dessen Statistiken nicht.

### Vergleichseinstellungen

- Checkbox für jede Strategie, standardmäßig sind alle ausgewählt
- Eingabefeld `Schritte pro Lauf`, Standardwert `500`
- Eingabefeld `Wiederholungen`, Standardwert `20`
- Optionaler `Random Seed`; leer bedeutet zufällige Versuche
- Button `Vergleich starten`
- Button `Vergleich zurücksetzen`
- Für den ersten übersichtlichen Vergleich werden die Standardparameter der
  Strategien verwendet.
- Optional können die Parameter der Vergleichsstrategien in einem
  aufklappbaren Bereich `Erweiterte Einstellungen` verändert werden.

### Fairer und reproduzierbarer Vergleich

- Jede ausgewählte Strategie erhält dieselben Banditenwahrscheinlichkeiten und
  dieselbe Anzahl Schritte.
- Führe jede Strategie über mehrere Wiederholungen aus, damit einzelne
  Zufallsergebnisse den Vergleich nicht dominieren.
- Leite für jede Wiederholung reproduzierbare Seeds aus dem eingegebenen
  Basis-Seed ab.
- Verwende für alle Strategien einer Wiederholung dieselben vorbereiteten
  Zufallswerte pro Zeitschritt. Dadurch basieren ihre Rewards auf denselben
  Zufallsbedingungen, obwohl sie unterschiedliche Banditen auswählen.
- Vergleichsläufe dürfen die Agenten oder Statistiken des interaktiven Modus
  nicht verändern.
- Berechne die Vergleichsdaten außerhalb der GUI-Darstellung in einer
  testbaren Hilfsklasse oder Funktion in `bandit_logic.py`.
- Die Berechnung erfolgt in kleinen Blöcken oder in einem Worker-Thread, damit
  die Tkinter-GUI während des Vergleichs bedienbar bleibt.

### Vergleichsdiagramm

- x-Achse: Anzahl der Agent-Pulls
- y-Achse: durchschnittlicher kumulativer Reward
- Zeichne für jede ausgewählte Strategie eine deutlich unterscheidbare,
  farbenblind-freundliche Linie.
- Zeige eine Legende mit den ausgeschriebenen Strategienamen.
- Zeichne zusätzlich eine gestrichelte Referenzlinie `Optimaler Erwartungswert`:

```text
optimaler kumulativer Reward = Anzahl Pulls * 0.8
```

- Zeige um jede mittlere Strategiekurve optional ein dezentes Band für das
  95-%-Konfidenzintervall. Eine Checkbox `Konfidenzintervalle anzeigen`
  schaltet diese Flächen ein oder aus.
- Vermeide zu viele zusätzliche Linien, Marker und Beschriftungen im Plot.
  Detailwerte werden stattdessen in der Tabelle unter dem Diagramm angezeigt.
- Beim Überfahren oder Auswählen einer Strategie soll deren Kurve optisch
  hervorgehoben werden, sofern dies mit vertretbarem Aufwand umsetzbar ist.

### Vergleichskennzahlen

Zeige unterhalb des Diagramms eine kompakte Tabelle mit einer Zeile pro
Strategie und folgenden Werten:

- mittlerer kumulativer Reward am Ende
- 95-%-Konfidenzintervall des End-Rewards
- mittlerer Reward pro Pull
- Anteil der Pulls von Bandit 3 in Prozent
- kumulatives Regret gegenüber der optimalen Strategie

Das Regret wird verständlich erklärt und wie folgt berechnet:

```text
Regret = (Anzahl Pulls * 0.8) - kumulativer Reward
```

Sortiere die Tabelle standardmäßig nach dem höchsten mittleren End-Reward.
Hebe die beste Strategie dezent hervor, ohne andere Ergebnisse als „schlecht“
zu markieren.

### Pädagogische Einordnung

- Zeige unter dem Vergleich einen kurzen dynamischen Erklärungstext, zum
  Beispiel: `UCB erzielte in diesem Versuch den höchsten mittleren Reward.`
- Weise darauf hin, dass Ergebnisse trotz gleicher Einstellungen durch Zufall
  schwanken können.
- Erkläre, dass eine Strategie bei kurzen Läufen besser aussehen kann, während
  eine andere bei langen Läufen stärker wird.
- Der Vergleich soll keine allgemeingültige Rangliste behaupten, sondern das
  Verhalten unter den aktuell gewählten Bedingungen zeigen.

## Summary-Ausgabe

Zeige kompakt auf jeder Banditenkarte:

- Pulls
- letzter Reward
- gesamte Rewards
- beobachtete Erfolgsquote
- Agentenschätzung

Zeige im Auswertungsbereich zusätzlich:

- aktuell gewählte Strategie
- Gesamtzahl der Agent-Pulls
- kumulativen Agent-Reward
- aktuellen Wert von Epsilon beziehungsweise Temperatur, sofern relevant
- Verhältnis von Exploration zu Exploitation, sofern die Strategie diese
  Unterscheidung direkt ermöglicht

## Reward-Plot

Dieser Abschnitt beschreibt den Plot im Tab `Aktueller Lauf`. Der
Strategievergleich verwendet das oben definierte Vergleichsdiagramm.

- Überschrift: `Kumulativer Reward des Agenten`
- x-Achse: Anzahl der Agent-Pulls
- y-Achse: kumulativer Reward
- Der Plot zeigt den aktuellen Lauf der gewählten Strategie.
- Nach jedem einzelnen Agent-Schritt und nach einem Lauf mit N Schleifen wird
  der Plot aktualisiert.
- Bei `Reset` wird der Plot geleert.
- Manuelle Pulls werden nicht in den Agent-Plot aufgenommen.
- Vor dem ersten Agent-Pull zeigt der Plot mittig den Hinweis
  `Starte den Agenten, um den Lernverlauf zu sehen.`
- Verwende passende Farben und eine dezente Rasterung; Beschriftungen dürfen
  sich nicht überlappen.

## Validierung und Fehlerbehandlung

- `Agent Loops` muss eine positive Ganzzahl sein.
- `Last N Pulls` muss eine nicht negative Ganzzahl sein.
- Epsilon-Werte müssen zwischen `0` und `1` liegen.
- `epsilon_min` darf nicht größer als `epsilon_start` sein.
- `epsilon_decay` muss größer als `0` und kleiner oder gleich `1` sein.
- Die Softmax-Temperatur muss größer als `0` sein.
- Ungültige Eingaben werden mit einer verständlichen Meldung in der GUI
  angezeigt und dürfen die Anwendung nicht zum Absturz bringen.
- Während `Agent Run N Loops` darf die GUI nicht dauerhaft einfrieren.
- `Schritte pro Lauf` und `Wiederholungen` im Vergleichsmodus müssen positive
  Ganzzahlen sein.
- Es muss mindestens eine Strategie für den Vergleich ausgewählt sein.
- Ein laufender Vergleich kann nicht versehentlich mehrfach gestartet werden.
- Die Anwendung muss auch auf macOS im Dunkelmodus mit älteren Tk-Versionen
  lesbar bleiben. Verwende deshalb ein plattformunabhängiges ttk-Theme und
  explizite Vorder- und Hintergrundfarben.

## Akzeptanzkriterien

- Die Anwendung startet mit `python bandit_app.py`.
- Alle fünf Strategien lassen sich auswählen und ausführen.
- Single Step und N-Loops-Lauf aktualisieren Statistiken und Plot korrekt.
- Die drei manuellen Pull-Buttons funktionieren unabhängig vom Agenten.
- Reset stellt einen definierten Ausgangszustand her.
- Die drei Spielautomaten werden vollständig und gut sichtbar dargestellt.
- Beim Strategiewechsel sind nur die relevanten Parameter sichtbar und es
  entstehen keine leeren Lücken im Layout.
- Jede Strategie besitzt einen verständlichen Erklärungstext.
- Manuelle und vom Agenten ausgeführte Pulls liefern eine erkennbare visuelle
  Rückmeldung.
- Der Vergleichsmodus kann mindestens zwei Strategien gleichzeitig darstellen.
- Vergleichsläufe verändern den aktuellen interaktiven Lauf nicht.
- Mit demselben Seed und denselben Einstellungen entstehen identische
  Vergleichsergebnisse.
- Die Vergleichsgrafik zeigt Mittelwertkurven, eine optimale Referenzlinie und
  optional Konfidenzintervalle.
- Die Vergleichstabelle zeigt End-Reward, Reward pro Pull, Auswahlanteil von
  Bandit 3 und Regret für jede Strategie.
- Bei genügend vielen Läufen bevorzugen die Strategien in der Regel Bandit 3.
- Die Logik kann ohne gestartete GUI importiert und getestet werden.
- Es gibt keine Spiellogik in `bandit_gui.py` und keine GUI-Abhängigkeit in
  `bandit_logic.py`.
