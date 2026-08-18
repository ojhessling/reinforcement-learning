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

Die Regeln gelten für neu erstellte Projekte. Bereits abgeschlossene Projekte
werden nicht rückwirkend angepasst, sofern die Benutzeranweisung dies nicht
ausdrücklich verlangt.

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
- Modul- und Testdateinamen sind repository-weit eindeutig. Greift ein neues
  Projekt dasselbe Environment wie ein früheres auf, erhält `{{project_name}}`
  ein unterscheidendes Suffix, damit `pytest` vom Repository-Root nicht an
  gleichnamigen Testdateien ohne Paketkontext scheitert.
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

Der projektspezifische Prompt legt die Kennung und alle abweichenden
Konstruktorargumente fest. Eine gemeinsame Factory erzeugt daraus getrennte
Instanzen für Training, deterministische Evaluation und sichtbare Animation.
Rein headless Evaluationen verzichten auf Rendering, solange dieselbe
unveränderte Environment-Spezifikation erhalten bleibt. Environment-Instanzen
werden weder gleichzeitig noch threadübergreifend geteilt. Dynamik,
Startzustandsverteilung, Reward und Abbruchregeln werden nicht verändert;
eigenes Reward Shaping ist unzulässig.

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

- allgemeine Parameter sind in jedem Verfahrenstab sichtbar
- algorithmusspezifische Parameter erscheinen ausschließlich im Tab des
  Verfahrens, das sie besitzt. Parameter, die das dort gewählte Verfahren nicht
  kennt, werden weggelassen und nicht als deaktivierte Felder mitgeschleppt
- ein Verfahrenswechsel im Dropdown lädt die Standardwerte des neu gewählten
  Verfahrens in diesen Tab und setzt ausschließlich den Lernzustand dieses
  Slots zurück; der andere Slot bleibt unberührt
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

## Verfahrenskatalog

Dieser Abschnitt beschreibt Verfahren, die in mehreren Projekten vorkommen,
einmalig: fachlicher Kern, vollständige UI-Parameter, Prüfregeln, Quellen. Der
projektspezifische Prompt nennt **welche** Verfahren ein Projekt verwendet,
ihre environmentspezifischen Standardprofile und begründete Abweichungen – er
wiederholt die Beschreibungen hier nicht. Verwendet ein Projekt ein Verfahren,
das hier fehlt, beschreibt der Projekt-Prompt es vollständig selbst.

Verwendet werden die Implementierungen von Stable-Baselines3 unverändert,
sofern der projektspezifische Prompt nichts anderes verlangt. Eigene Arbeit
liegt dann in Konfiguration, Runnern, Metriken, Vergleich, GUI und Tests, nicht
in einer Neuimplementierung des Algorithmus.

### Gemeinsame Parameter

In **jedem** Verfahrenstab einstellbar:

- `total_timesteps`
- `learning_rate` mit wählbarem Verlauf `konstant` oder `linear fallend`
  (Stable-Baselines3 akzeptiert dafür eine Callable-Schedule)
- `batch_size`
- `gamma`
- `seed`
- Hidden Layers für Actor und Critic getrennt, Aktivierungsfunktion, Optimizer
  sowie dessen `eps` und `weight_decay`

Global außerhalb der Tabs, weil beide Läufe dieselben Stützstellen brauchen:
Intervall und Episodenzahl der deterministischen Zwischenevaluation.

Technische Optionen wie `verbose`, `tensorboard_log`, `device`, die
`policy`-Kennung und `_init_setup_model` gehören nicht in die UI.

### PPO

On-Policy-Verfahren. Es sammelt Rollouts fester Länge, schätzt Vorteile mit
Generalized Advantage Estimation und optimiert über mehrere Epochen auf
denselben Daten ein geclipptes Surrogatziel:

```text
L = E[ min( r(θ)·Â , clip(r(θ), 1-ε, 1+ε)·Â ) ]   mit r(θ) = π_θ(a|s) / π_alt(a|s)
```

Die Daten werden nach dem Update verworfen; einen Replay Buffer gibt es nicht.

Zusätzliche UI-Parameter: `n_steps`, `n_epochs`, `gae_lambda`, `clip_range`,
`clip_range_vf` (leer bedeutet aus), `normalize_advantage`, `ent_coef`,
`vf_coef`, `max_grad_norm`, `target_kl` (leer bedeutet aus), `use_sde`,
`sde_sample_freq`, `log_std_init`.

Prüfregeln: `batch_size` muss `n_steps` teilen, sonst verwirft
Stable-Baselines3 Daten und warnt erst zur Laufzeit; `total_timesteps` muss
mindestens einen vollständigen Rollout zulassen.

Quellen: [Schulman et al., PPO](https://arxiv.org/abs/1707.06347),
[Schulman et al., GAE](https://arxiv.org/abs/1506.02438)

### TD3

Off-Policy-Verfahren mit deterministischem Actor. Es lernt zwei Critics und
bildet das Ziel aus deren Minimum, um Überschätzung zu dämpfen. Auf die
Target-Action kommt geclipptes Rauschen (Target Policy Smoothing); Actor und
Target-Netze werden nur alle `policy_delay` Updates aktualisiert:

```text
ã = clip(π_target(s') + clip(N(0, σ_t), -c, +c), a_min, a_max)
y = r + γ·(1-done)·min( Q₁_target(s', ã), Q₂_target(s', ã) )
```

Zusätzliche UI-Parameter: `buffer_size`, `learning_starts`, `tau`,
`train_freq`, `gradient_steps`, `policy_delay`, `target_policy_noise`,
`target_noise_clip`, Action-Noise-Typ `keins`, `normal` oder
`Ornstein-Uhlenbeck` sowie dessen `σ`.

Prüfregel: Weil der Actor deterministisch ist, exploriert TD3 ohne explizites
Action Noise überhaupt nicht. `keins` wird für TD3 mit einer verständlichen
Meldung abgelehnt.

Quelle: [Fujimoto et al., TD3](https://arxiv.org/abs/1802.09477)

### SAC

Off-Policy-Verfahren mit stochastischem Actor. Es maximiert zusätzlich zur
erwarteten Rendite die Entropie der Policy, gewichtet mit einer Temperatur `α`:

```text
y = r + γ·(1-done)·[ min(Q₁_target, Q₂_target) - α·log π(a'|s') ]
```

Bei `α = auto` wird die Temperatur selbst gelernt, sodass die mittlere Entropie
einer Zielentropie folgt; Stable-Baselines3 verwendet als Standard
`target_entropy = -dim(A)`.

Zusätzliche UI-Parameter: `buffer_size`, `learning_starts`, `tau`,
`train_freq`, `gradient_steps`, `ent_coef` als `auto` oder fester Wert samt
Startwert von `α`, `target_entropy` als `auto` oder Zahl,
`target_update_interval`, `use_sde`, `sde_sample_freq`, `log_std_init` sowie
optionales Action Noise mit `σ`.

Quellen: [Haarnoja et al., SAC](https://arxiv.org/abs/1801.01290),
[Haarnoja et al., SAC mit gelernter Temperatur](https://arxiv.org/abs/1812.05905),
[Raffin et al., gSDE](https://arxiv.org/abs/2005.05719)

### Exploration

PPO, TD3 und SAC explorieren aus der Policy selbst beziehungsweise aus
explizitem Action Noise. Projekte, die ausschließlich diese Verfahren
verwenden, besitzen deshalb **keine** ε-greedy-Parameter wie
`exploration_fraction` oder `exploration_final_eps`.

### Normalisierung

Verlangt ein environmentspezifisches Profil `normalize: true` oder sind die
Observationswerte sehr unterschiedlich skaliert, erhält jeder Verfahrenstab
eine Gruppe `Normalisierung` mit `Beobachtungen normalisieren`,
`Rewards normalisieren` sowie den zugehörigen Clip-Werten. Umgesetzt wird sie
mit `VecNormalize` von Stable-Baselines3. Dabei gilt verbindlich:

- Die laufenden Statistiken werden ausschließlich im Training fortgeschrieben.
  Deterministische Evaluation und Animation verwenden dieselben Statistiken
  eingefroren.
- Graph, Summary und Evaluation zeigen immer den **unnormalisierten**
  Episoden-Return, sonst wären Referenzlinien und Schwellen bedeutungslos.
  Verwende dafür den `Monitor`-Wrapper innerhalb der Vektor-Environment
  (`info["episode"]["r"]`) oder `VecNormalize.get_original_reward()`.
- Die Animation erhält rohe Observationen aus dem Renderprozess und
  normalisiert sie vor `predict()` mit denselben eingefrorenen Statistiken.
- Die Statistiken gehören zum Speicherstand und zum automatischen Checkpoint.
  Ein ohne sie geladenes Modell verhält sich anders als das gespeicherte.

Off-Policy-Verfahren erhalten Normalisierung nicht als Voreinstellung: Ein
Replay Buffer speichert Beobachtungen, deren Normalisierungsstatistik sich
weiter verschiebt, sodass alte Einträge nicht mehr zur aktuellen Normierung
passen. Wählbar bleibt die Option trotzdem.

### Gespeicherter Zustand

- `PPO`: Policy, Value-Netz und Optimizer. Ein Replay Buffer existiert nicht.
- `TD3`: Actor, beide Critics, Target-Netze, Optimizer und Replay Buffer.
- `SAC`: wie TD3, zusätzlich der gelernte Temperaturparameter.
- Bei aktiver Normalisierung zusätzlich die `VecNormalize`-Statistiken.

### Fairness von On-Policy gegen Off-Policy

Ein Vergleich von PPO gegen TD3 oder SAC fällt bei gleichem Schrittbudget
systematisch zugunsten der Off-Policy-Verfahren aus: Diese lernen aus jedem
gespeicherten Übergang mehrfach, PPO verwirft seine Daten nach jedem Update.
Das ist kein Messfehler, sondern eine Eigenschaft der Verfahrensklassen.
Bedienungsanleitung und README sagen das ausdrücklich.

### Verfahrensbezogene Tests

Für jedes eingesetzte Verfahren aus diesem Katalog wird zusätzlich geprüft:

- Die Konstruktorargumente enthalten ausschließlich Schlüssel, die der
  jeweilige Stable-Baselines3-Algorithmus kennt.
- `PPO`: Wirkung von `clip_range`, `gae_lambda` und `n_epochs`; kein Replay
  Buffer vorhanden; Fortsetzen des Trainings ohne Rücksetzen des
  Schrittzählers.
- `TD3`: verzögerte Actor-Updates gemäß `policy_delay`, Clipping des
  Target-Rauschens, Action Noise wirkt im Training und nicht in der
  Evaluation.
- `SAC`: automatische Entropieanpassung verändert `α`, die Zielentropie
  entspricht bei `auto` genau `-dim(A)`.
- Save-/Load-Roundtrip je Verfahren mit genau den Bestandteilen aus
  `Gespeicherter Zustand`.
- Ablehnung einer Modelldatei, deren Algorithmus nicht zum aktiven Slot passt.

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
  beiden Spalten ausschließlich die Verfahrenswahl mit ihren fachlich
  gruppierten Parametern gemäß `Verfahrenswahl und Vergleichstabs`; die dritte
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
- im unteren Bereich Tabs für sinnvolle Diagramme und Vergleiche
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
Eine reine Konstruktion der Widgets reicht nicht als Layout-Test. Geprüft
werden beide Verfahrenstabs, also auch die Eingabefelder des zunächst nicht
sichtbaren Tabs nach dem Umschalten.

Controls spiegeln den Zustand `Bereit`, `Läuft`, `Gestoppt`, `Abgeschlossen`
oder `Fehler` wider. Inkompatible Aktionen werden gezielt deaktiviert und nach
Erfolg, Abbruch oder Fehler wieder freigegeben.

Jede App besitzt eine `Bedienungsanleitung`. Sie erklärt kurz den empfohlenen
Ablauf, Environment und Rewards, Methoden, Training gegenüber Evaluation, die
Bedeutung der beiden Verfahrensslots, Parameter, Ansichten und typische
Ursachen ausbleibenden Lernerfolgs.

### Verfahrenswahl und Vergleichstabs

Projekte mit mehreren Algorithmen bieten immer genau zwei gleichrangige
Verfahrensslots an. Über den beiden Parameterspalten stehen dafür die Dropdowns
`Verfahren 1` und `Verfahren 2`, in denen jeweils einer der verfügbaren
Algorithmen gewählt wird. Beide Slots dürfen denselben Algorithmus enthalten.
Damit lassen sich sowohl zwei verschiedene Verfahren als auch zwei
Parametrisierungen desselben Verfahrens vergleichen.

Darunter liegen zwei gleich aufgebaute Tabs `Verfahren 1` und `Verfahren 2`.
Jeder Tab enthält vollständig und unabhängig die Parameter des dort gewählten
Algorithmus, einschließlich Trainingsbudget, Seed und Netzwerkparametern.
Parameter werden nicht zwischen den Slots geteilt. Global außerhalb der Tabs
bleiben nur Einstellungen, die für einen fairen Vergleich in beiden Läufen
identisch sein müssen, etwa Intervall und Umfang der deterministischen
Zwischenevaluation.

Genau ein Slot ist das aktive Verfahren. Er ergibt sich aus dem gewählten Tab
und wird über den Steuerungsbuttons unübersehbar angezeigt, zum Beispiel
`Aktiv: Verfahren 1 – PPO`. Alle Einzellauf-Aktionen wirken auf dieses
Verfahren. Läuft bereits ein Einzellauf, bleibt dessen Ziel fixiert: Ein
Tabwechsel ändert dann nur die angezeigten Parameter, nicht den laufenden Lauf.

Enthält ein Projekt nur einen Algorithmus, entfallen die Dropdowns; die beiden
Tabs bleiben und vergleichen zwei Parametrisierungen desselben Verfahrens.

### Steuerungsbuttons

Die dritte Spalte enthält, soweit im Projekt fachlich sinnvoll, diese
Steuerungsbuttons in dieser Reihenfolge:

1. `Training starten / fortsetzen`
2. `Stoppen`
3. `Deterministisch evaluieren`
4. `Sichtbare Episode abspielen`
5. `Vergleich starten / fortsetzen`
6. `Bestes Modell wiederherstellen`
7. `Neues Modell`

Die Buttons 1, 3, 4, 6 und 7 wirken auf das aktive Verfahren, Button 5 immer
auf beide Slots. Buttons zum manuellen Speichern und Laden gibt es nicht: Der
Lernzustand wird über den automatischen Checkpoint des besten
Evaluationsergebnisses gesichert und mit Button 6 zurückgeholt.

Darunter folgen die Animationssteuerung gemäß `Animation`, die
Fortschrittsanzeige und die Statuszeile.

Buttonbeschriftungen benennen nur Bestandteile, die jedes Verfahren des
Projekts tatsächlich besitzt. Besitzt beispielsweise nur ein Teil der Verfahren
einen Replay Buffer, taucht er in keiner Beschriftung auf; welche Bestandteile
ein Button tatsächlich betrifft, erklären Statusmeldung, Bedienungsanleitung
und README.

## Responsivität

Die GUI bleibt bei Training, Evaluation, Vergleich und Animation bedienbar.
Lange Arbeit läuft in kleinen `after()`-Schritten oder in Worker-Threads mit
Queue; Worker greifen nie direkt auf Tkinter-Widgets zu. Plot- und
Statusaktualisierungen werden auf eine sinnvolle Frequenz begrenzt.

Abbruch wird regelmäßig geprüft; beim Schließen werden Worker und Ressourcen
sauber beendet.

### Animation

Die Animation besitzt genau **einen** Schalter und ein Eingabefeld, beide
global neben den Steuerungsbuttons und keinem Verfahrensslot zugeordnet:

- `Animation zeigen` schaltet die Einzelbildanimation ein und aus – jederzeit,
  auch mitten in einem laufenden Trainings- oder Vergleichslauf. Eingeschaltet
  zeigt sie sowohl einzeln abgespielte Episoden als auch den laufenden Lauf.
- `Bildrate (FPS)` legt die Abspielgeschwindigkeit fest. Standardwert ist die
  environment-eigene Bildrate `env.metadata["render_fps"]`; gültig sind ganze
  Zahlen von 1 bis 120. Die Validierung nennt wie überall Feld, Wert und
  Bereich. Eine Änderung wirkt spätestens mit der nächsten sichtbaren Episode,
  auch während eines Laufs.

Während eines Trainings- oder Vergleichslaufs zeigt die Animation fortlaufend
Episoden, die der **aktuelle Lernstand** im isolierten Renderprozess spielt.
Die Trainingsschleife selbst rendert nicht: Sie liefe sonst im Takt der
Darstellung und würde massiv ausgebremst.

Damit lernendes Netz und Darstellung sich nicht in die Quere kommen, arbeitet
die Animation auf einer **Kopie der Policy**, die zu Beginn jeder sichtbaren
Episode gezogen wird. Sie greift nie aus dem GUI-Thread in das Netz, das ein
Worker gerade trainiert. Jede sichtbare Episode zeigt damit den Lernstand zu
ihrem Beginn, nicht den fortlaufend aktualisierten.

Läuft ein Einzeltraining, zeigt die Animation dessen fixierten Slot. Läuft ein
Vergleich, zeigt sie **beide Verfahren gleichzeitig**, jedes mit eigenem Bild
und eigener Messwertanzeige. Außerhalb eines Laufs ist genau das Anzeigefeld des
aktiven Verfahrens sichtbar. Die Größe der Einzelbilder leitet sich aus dem
verfügbaren Platz und der Zahl der Anzeigen ab, nicht aus der Größe des zuletzt
gezeigten Bildes – sonst behielte ein einmal großes Bild seinen Platz und
verdrängte die zweite Anzeige.

Die Animation kostet Rechenzeit und verlangsamt das Training spürbar. Die
eingestellte Bildrate ist dabei eine Obergrenze: Während eines Laufs
konkurrieren Training und Rendern um Rechenzeit, sodass die tatsächliche Rate
darunter liegen kann. Weise in der Bedienungsanleitung auf beides hin und halte
den Lauf ohne Animation voll funktionsfähig.

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
- Der beste deterministische Evaluationswert eines Einzeltrainings wird je
  Verfahrensslot getrennt gemerkt. Bei jeder Verbesserung wird der vollständige
  Lernzustand des Slots konsistent als gemeinsamer Checkpoint gesichert, in der
  Summary ausgewiesen und über einen klar beschrifteten Button
  wiederherstellbar gemacht. Welche Bestandteile dazugehören, hängt vom
  Verfahren ab: Bei Off-Policy-Verfahren gehört der Replay Buffer dazu, bei
  On-Policy-Verfahren treten Policy-, Value- und Optimizerzustand an seine
  Stelle. Wiederhergestellt wird der vollständige Lernzustand, nicht nur das
  neuronale Netz.
- Trainings- und Vergleichskurven verwenden einheitlich Episoden auf der
  X-Achse. Das Trainingsbudget und die tatsächlich ausgeführten
  Environment-Schritte bleiben separat in Status und Summary sichtbar.
- Fehlende Daten werden nicht durch künstliche Nullwerte ersetzt.
- Bei episodenbasierten Kurven entstehen Punkte nur für vollständig
  abgeschlossene Episoden. Endet ein Trainingsbudget innerhalb einer Episode,
  darf der letzte Kurvenpunkt deshalb vor dem tatsächlich ausgeführten
  Schrittbudget liegen. GUI und Summary zeigen ausgeführte Schritte,
  angefordertes Budget und diese Bedeutung getrennt und verständlich an.

Der Vergleich stellt immer die beiden Verfahrensslots gegenüber. Ein
gemeinsamer Vergleichsgraph ist verpflichtend; er zeigt beide Läufe mit
derselben aussagekräftigen X-Achse und derselben Metrik. Legende und
Beschriftung benennen Slot und Algorithmus, etwa `V1 – PPO` und `V2 – SAC`.
Enthalten beide Slots denselben Algorithmus, nennt das Label zusätzlich den
wichtigsten abweichenden Parameter; Farbe und Linienstil unterscheiden die
beiden Kurven in jedem Fall eindeutig. Bei mehreren Wiederholungen zeigt der
Graph den Mittelwert und zusätzlich Standardabweichung oder
95-%-Konfidenzintervall als Unsicherheitsband.

Der Vergleichsgraph erscheint mit dem ersten verfügbaren Ergebnis und wird
während beider Läufe in einem sinnvollen Intervall fortgeschrieben. Er darf
nicht erst nach Abschluss des gesamten Vergleichs angezeigt oder aktualisiert
werden. Beide Slots starten parallel; die Fortschrittsanzeige aggregiert ihre
tatsächlich ausgeführten Schritte. Ein erneut gestarteter, kompatibel
konfigurierter Vergleich setzt die Vergleichsmodelle nicht zurück, sondern
setzt ihr Training fort und hängt neue Messpunkte an die vorhandenen Kurven an.
Rohwerte werden dezent dargestellt; je Slot hebt eine kräftige Linie den
gleitenden Durchschnitt der letzten 20 Episodenergebnisse hervor.
Dasselbe gilt für den normalen
Trainingsgraphen: aktuelle Episodenergebnisse werden während des Trainings
sichtbar. Parallel dazu wird auch die Summary live aktualisiert; sie zeigt für
Training und Vergleich konsistent Episoden, ausgeführte Environment-Schritte,
aktuelle beziehungsweise gemittelte Rewards und Erfolgsrate.

Die Vergleichs-Summary besitzt genau zwei Ergebnisspalten, `Verfahren 1` und
`Verfahren 2`, mit dem jeweiligen Algorithmusnamen in der Kopfzeile. Sie
enthält zusätzlich einen Abschnitt, der genau die Parameter auflistet, in denen
sich die beiden Konfigurationen unterscheiden. Ohne diesen Abschnitt ist ein
Vergleich zweier Parametrisierungen desselben Algorithmus nicht
interpretierbar. Unterscheiden sich die Trainingsbudgets der beiden Slots,
weist die GUI vor dem Start sichtbar darauf hin; unzulässig ist es nicht.

Lange Rohkurven werden nur für die Darstellung auf eine feste, angemessene
Punktzahl verdichtet; die Messdaten selbst bleiben vollständig erhalten. Eine
Min-/Max-Verdichtung ist einfachem Auslassen vorzuziehen, damit lokale Spitzen
und Einbrüche sichtbar bleiben. Plot-Updates werden zeitlich gedrosselt.

Vergleiche verändern das sichtbare Experiment nicht. Beide Slots erhalten
identische Environment-Konfigurationen und reproduzierbar abgeleitete Seeds.
Trainingsbudget und Seed stammen aus dem jeweiligen Tab und sind bewusst frei
wählbar, damit auch Budget- und Seed-Vergleiche möglich sind; das Budget wird
bevorzugt in Environment-Schritten angegeben. Jeder Slot und jede Wiederholung
startet mit neuem Lernzustand; die Evaluation erfolgt ohne Exploration und
Lernupdates.

Vor dem Start wird der Gesamtumfang angezeigt. Mehrere Wiederholungen werden
mit Mittelwert und Standardabweichung oder 95-%-Konfidenzintervall aggregiert.
Bei Abbruch bleiben vollständige Ergebnisse erhalten und unvollständige werden
gekennzeichnet.

## Tabellen und Modelle

Tabellen zeigen den aktuellen, vollständigen Lernstand, unterscheiden besuchte
und unbesuchte Zustände und sind scrollbar.

Wenn Modelle gespeichert und geladen werden, enthalten sie Lernzustand,
Format-Version und notwendige Metadaten. Zu den Metadaten gehören Algorithmus,
vollständige Slot-Konfiguration und die Environment-Kennung samt aller
abweichenden Konstruktorargumente. Methode und Environment werden beim Laden
auf Kompatibilität geprüft. Fehlerhafte oder inkompatible Dateien dürfen
den aktiven Zustand nicht verändern.

Gespeichert wird genau der Zustand, den das jeweilige Verfahren besitzt. Ein
Replay Buffer gehört dazu, wenn das Verfahren einen führt, und wird andernfalls
nicht erwähnt oder durch leere Platzhalter ersetzt. Geladen wird immer in den
aktiven Verfahrensslot; passt die Datei nicht zu dem dort gewählten
Algorithmus, wird sie verständlich abgelehnt.

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
- einen Vergleichslauf mit zweimal demselben Algorithmus und unterschiedlichen
  Parametern: Beide Slots besitzen getrennte Lernzustände, Ergebnisse und
  Kurven und beeinflussen sich nicht
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
- Verfahrensslots und Vergleichslogik sowie Speichern und Laden, sofern
  vorhanden
- Testbefehl und bekannte Grenzen
