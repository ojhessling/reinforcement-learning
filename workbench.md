# Basis-Prompt für Reinforcement-Learning-Workbench-Projekte

## 1 Geltung

Verbindliche Grundregeln für alle RL-Projekte im Ordner `Oliver`. Der
projektspezifische Prompt beginnt mit:

```text
Berücksichtige die verbindlichen Regeln aus ../workbench.md.
Projektname: {{project_name}}
Environment: {{environment_name}}
```

- Priorität bei Widersprüchen: aktuelle Benutzeranweisung → Projekt-Prompt →
  diese Datei.
- Abweichungen werden im Projekt-Prompt ausdrücklich begründet.
- Die Regeln gelten für neue Projekte. Abgeschlossene Projekte werden nicht
  rückwirkend angepasst, sofern der Benutzer das nicht ausdrücklich verlangt.

### 1.1 Was in den Projekt-Prompt gehört

Nur diese sieben Punkte:

1. Environment-Kennung und alle Konstruktorargumente
2. Action- und Observation-Space, Reward, Episodenende
3. Erfolgsdefinitionen und die daraus folgenden Metriken
4. Standardprofile der Verfahren samt Quelle und begründeten Abweichungen
5. Startbelegung der Slots und Standardwerte der globalen Einstellungen
6. Bildgröße des Renderframes und Inhalt der Einblendung neben der Animation
7. environmentbezogene Tests und Abnahmekriterien

Alles andere – GUI, Bedienelemente, Farben, Linienstile, Beschriftungen,
Export, Animationsraster, Rendering, Checkpoints, allgemeine Tests – steht
**ausschließlich** hier und wird im Prompt nicht wiederholt, auch nicht
zusammenfassend: Eine Kopie veraltet, sobald diese Datei sich ändert. Verweise
genügen.

## 2 Projekt

### 2.1 Ziel und Sprache

Eigenständig lauffähige lokale Python-Anwendung für Developer und RL-Anfänger
zum Konfigurieren, Trainieren, Beobachten, Evaluieren und Vergleichen. Kein
Webserver, sofern der Prompt nichts anderes sagt.

- GUI, Hilfen, Meldungen, README, Prompt: Deutsch
- Code-Bezeichner: Englisch
- Fachbegriffe erlaubt, für Anfänger erklärt
- fachlich wichtige Stellen erhalten kurze Kommentare

### 2.2 Dateien und Umgebung

```text
{{project_name}}_app.py
{{project_name}}_logic.py
{{project_name}}_gui.py
README.md
requirements.txt
tests/
    test_{{project_name}}_logic.py
```

- App-Datei: nur der Entry Point.
- Logikmodule importieren kein Tkinter; die GUI enthält keine Lernformeln.
- Zusätzliche Module nur bei erkennbarem Nutzen.
- Modul- und Testdateinamen sind repository-weit eindeutig – sonst scheitert
  `pytest` vom Repository-Root an gleichnamigen Testdateien ohne Paketkontext.
  Greift ein Projekt ein früheres Environment erneut auf, bekommt
  `{{project_name}}` ein unterscheidendes Suffix.
- Laufzeitergebnisse und lokale venvs werden nicht committed.
- Alle Projekte nutzen `Oliver/environment.yml`; direkte Abhängigkeiten stehen
  zusätzlich in `requirements.txt`. Neue Pakete nur bei Bedarf und auf
  Konflikte geprüft.
- Neuronale Netze ausschließlich mit PyTorch. Kein TensorFlow, kein Keras.

### 2.3 Architektur

- **Environment**: Spaces, Übergänge, Rewards, Reset, Seed, `terminated`,
  `truncated`, ggf. Rendering. Keine GUI- und keine Lernlogik.
- **Agenten**: je Algorithmus eine Klasse mit einheitlicher Schnittstelle für
  Reset, Action-Auswahl, Lernen, Episodenende, Metriken, Speichern/Laden.
- **Runner**: Training, Evaluation und Vergleich liegen nicht in
  Button-Callbacks, sondern in getrennten Komponenten. Ergebnisse sind
  Dataclasses oder typisierte Strukturen, keine positionsabhängigen Tupel.
- **GUI**: Eingaben, Validierung, Runner-Steuerung, Visualisierung, Status,
  Dialoge. Keine Reward-Regeln, Lernupdates oder Action-Auswahl.

## 3 Fachliche RL-Regeln

### 3.1 Environment-Factory

- Der Prompt legt Kennung und alle abweichenden Konstruktorargumente fest.
- Eine gemeinsame Factory erzeugt getrennte Instanzen für Training,
  deterministische Evaluation und Animation. Headless Evaluationen verzichten
  auf Rendering, behalten aber dieselbe Environment-Spezifikation.
- Instanzen werden weder gleichzeitig noch threadübergreifend geteilt.
- Dynamik, Startzustandsverteilung, Reward und Abbruchregeln bleiben
  unverändert; eigenes Reward Shaping ist unzulässig.

### 3.2 Training und Evaluation

Evaluation nutzt keine Exploration und keine Lernupdates, verändert weder
Lernzustand noch Trainingsstatistiken und arbeitet mit eigenen Ergebnissen und
definiertem Startzustand.

### 3.3 Episodenende

- `terminated`: fachlich terminaler Zustand · `truncated`: externes Limit ·
  `done = terminated or truncated`
- Ob bei Truncation gebootstrapt wird, ist je Algorithmus in Implementierung,
  Tests und README konsistent festzulegen.
- Liefert ein Environment **immer** `terminated=False`, ist Bootstrapping die
  einzig richtige Behandlung – Abschneiden behauptete ein Episodenende, das die
  Umgebung nicht kennt. Der Prompt hält das ausdrücklich fest.

### 3.4 Reproduzierbarkeit und Tie-Breaking

- Ein angegebener Seed wird für Python, NumPy, Environment, Action-Space und
  verwendete Frameworks gesetzt; unabhängige Aufgaben erhalten getrennte
  Generatoren. Ein Reset setzt Generatoren nur bei ausdrücklichem Seed zurück.
- Gleich gute Actions werden mit numerischer Toleranz reproduzierbar zufällig
  gewählt; die Policy-Ansicht zeigt alle gleichwertigen Actions.
- Unbesuchte Werte erscheinen als `—` oder `?`, nicht als gelernte Nullwerte.

## 4 Parameter

- Allgemeine Parameter sind in jedem Verfahrenstab sichtbar.
- Algorithmusspezifische Parameter erscheinen nur im Tab des Verfahrens, das
  sie besitzt. Unbekannte Parameter werden **weggelassen**, nicht deaktiviert
  mitgeschleppt.
- Ein Verfahrenswechsel im Dropdown lädt die Standardwerte des neuen Verfahrens
  und setzt nur den Lernzustand dieses Slots zurück; andere Slots bleiben
  unberührt.
- Parameter werden fachlich gruppiert und kompakt angeordnet – auf typischen
  Laptop-Auflösungen möglichst ohne Scrollen sichtbar. Keine langen,
  ungegliederten Ein-Spalten-Listen.
- Bezeichnungen nutzen die üblichen englischen Fachnamen plus etabliertes
  Symbol, etwa `Learning Rate α`, `Discount Factor γ`, `Exploration ε`. Symbole
  werden nicht erfunden; die übrige Oberfläche bleibt deutsch.
- Komplexe Parameter erhalten kurze Erklärungen.
- Eingaben werden vor einer Aktion vollständig validiert und atomar übernommen;
  Validierung prüft Wertebereiche, Abhängigkeiten sowie Netzwerk-, Buffer-,
  Batch- und Modelldatei-Kompatibilität.
- Fehlermeldungen nennen Feld, ungültigen Wert und gültigen Bereich.
- Notwendige Resets des Lernzustands werden verständlich angezeigt.

### 4.1 Neuronale Netze

Alle verwendeten Hyperparameter sind in der UI änderbar, je nach Verfahren
insbesondere: Zahl und Größe der Hidden Layers, Aktivierung, Lernrate und
Optimizer-Parameter, Batch-Größe, Initialisierung, Normalisierung,
Regularisierung, Gradient Clipping, Target-Network-Update und Zahl der
Gradientenschritte.

Standardwerte stammen aus Fachliteratur, den SB3-Voreinstellungen oder einem
environmentspezifischen Profil des RL Baselines3 Zoo; **Profile haben Vorrang**.
Prompt und README nennen Quelle, Algorithmus und Version bzw. Profilstand.
Abweichungen werden begründet, nicht unterstützte Optionen in der README
dokumentiert.

## 5 Verfahrenskatalog

Beschreibt mehrfach vorkommende Verfahren einmalig: fachlicher Kern,
UI-Parameter, Prüfregeln, Quellen. Der Prompt nennt nur **welche** Verfahren
ein Projekt nutzt, ihre Profile und Abweichungen. Fehlt ein Verfahren hier,
beschreibt der Prompt es vollständig selbst.

Verwendet werden die SB3-Implementierungen unverändert. Eigene Arbeit liegt in
Konfiguration, Runnern, Metriken, Vergleich, GUI und Tests – nicht in einer
Neuimplementierung.

### 5.1 Gemeinsame Parameter

In jedem Verfahrenstab: `total_timesteps`, `learning_rate` mit Verlauf
`konstant` oder `linear fallend` (SB3 akzeptiert eine Callable-Schedule),
`batch_size`, `gamma`, `seed`, Hidden Layers für Actor und Critic getrennt,
Aktivierung, Optimizer samt `eps` und `weight_decay`.

Global außerhalb der Tabs, weil alle Läufe dieselben Stützstellen brauchen:
`Anzahl Verfahren` sowie Intervall und Episodenzahl der Zwischenevaluation.

Nicht in die UI: `verbose`, `tensorboard_log`, `device`, `policy`-Kennung,
`_init_setup_model`.

Standardwerte:

- `total_timesteps` ist für **alle** Verfahren eines Projekts gleich – sonst
  wäre der Vergleich schon ohne Zutun unfair. Vorzugswürdig ist das Budget des
  Profils: Nur damit erreichen die Verfahren die Ergebnisse, für die sie getunt
  wurden. Ist es interaktiv nicht abwartbar, nennt der Prompt einen begründet
  kleineren Wert und sagt, was verloren geht.
- Der Prompt nennt in jedem Fall die zu erwartende **Laufzeit**, damit niemand
  versehentlich einen Mehrstundenlauf startet.
- Das Evaluationsintervall beträgt rund ein Zehntel des Schrittbudgets – etwa
  zehn Stützstellen je Standardlauf.
- Der Prompt nennt beide Zahlen ausdrücklich.

### 5.2 PPO

On-Policy: Rollouts fester Länge, Vorteile per GAE, mehrere Epochen auf
denselben Daten mit geclipptem Surrogatziel. Daten werden nach dem Update
verworfen; kein Replay Buffer.

```text
L = E[ min( r(θ)·Â , clip(r(θ), 1-ε, 1+ε)·Â ) ]   mit r(θ) = π_θ(a|s) / π_alt(a|s)
```

- Zusätzliche UI-Parameter: `n_steps`, `n_epochs`, `gae_lambda`, `clip_range`,
  `clip_range_vf` (leer = aus), `normalize_advantage`, `ent_coef`, `vf_coef`,
  `max_grad_norm`, `target_kl` (leer = aus), `use_sde`, `sde_sample_freq`,
  `log_std_init`, `ortho_init`.
- Prüfregeln: `batch_size` muss `n_steps` teilen – sonst verwirft SB3 Daten und
  warnt erst zur Laufzeit; `total_timesteps` muss einen vollständigen Rollout
  zulassen.
- Quellen: [PPO](https://arxiv.org/abs/1707.06347),
  [GAE](https://arxiv.org/abs/1506.02438)

### 5.3 TD3

Off-Policy mit deterministischem Actor, zwei Critics und Minimum als Ziel gegen
Überschätzung. Geclipptes Rauschen auf die Target-Action; Actor und Target-Netze
nur alle `policy_delay` Updates.

```text
ã = clip(π_target(s') + clip(N(0, σ_t), -c, +c), a_min, a_max)
y = r + γ·(1-done)·min( Q₁_target(s', ã), Q₂_target(s', ã) )
```

- Zusätzliche UI-Parameter: `buffer_size`, `learning_starts`, `tau`,
  `train_freq`, `gradient_steps`, `policy_delay`, `target_policy_noise`,
  `target_noise_clip`, Action-Noise-Typ `keins` / `normal` /
  `Ornstein-Uhlenbeck` samt `σ`.
- Prüfregel: Mit deterministischem Actor exploriert TD3 ohne Action Noise gar
  nicht. `keins` wird für TD3 mit verständlicher Meldung abgelehnt.
- Quelle: [TD3](https://arxiv.org/abs/1802.09477)

### 5.4 SAC

Off-Policy mit stochastischem Actor; maximiert zusätzlich die Entropie,
gewichtet mit Temperatur `α`. Bei `α = auto` wird sie gelernt, sodass die
mittlere Entropie einer Zielentropie folgt; SB3-Standard
`target_entropy = -dim(A)`.

```text
y = r + γ·(1-done)·[ min(Q₁_target, Q₂_target) - α·log π(a'|s') ]
```

- Zusätzliche UI-Parameter: `buffer_size`, `learning_starts`, `tau`,
  `train_freq`, `gradient_steps`, `ent_coef` als `auto` oder fester Wert samt
  Startwert von `α`, `target_entropy` als `auto` oder Zahl,
  `target_update_interval`, `use_sde`, `sde_sample_freq`, `log_std_init`,
  optionales Action Noise mit `σ`.
- Quellen: [SAC](https://arxiv.org/abs/1801.01290),
  [SAC mit gelernter Temperatur](https://arxiv.org/abs/1812.05905),
  [gSDE](https://arxiv.org/abs/2005.05719)

### 5.5 Exploration

PPO, TD3 und SAC explorieren aus der Policy selbst oder aus Action Noise.
Projekte mit ausschließlich diesen Verfahren haben **keine** ε-greedy-Parameter
wie `exploration_fraction` oder `exploration_final_eps`.

### 5.6 Normalisierung

Verlangt ein Profil `normalize: true` oder sind die Observationswerte sehr
unterschiedlich skaliert, erhält jeder Tab eine Gruppe `Normalisierung` mit
`Beobachtungen normalisieren`, `Rewards normalisieren` und den Clip-Werten;
umgesetzt mit `VecNormalize`.

- Laufende Statistiken wachsen **nur im Training**. Evaluation und Animation
  nutzen sie eingefroren.
- Graph, Summary und Evaluation zeigen immer den **unnormalisierten**
  Episoden-Return – sonst wären Referenzlinien bedeutungslos. Quelle dafür:
  `Monitor` innerhalb der Vektor-Environment (`info["episode"]["r"]`) oder
  `VecNormalize.get_original_reward()`.
- Die Animation erhält rohe Observationen aus dem Renderprozess und
  normalisiert sie vor `predict()` mit denselben eingefrorenen Statistiken.
- Die Statistiken gehören zum Speicherstand und zum Checkpoint.
- Off-Policy-Verfahren normalisieren **nicht** per Voreinstellung: Der Replay
  Buffer speichert Beobachtungen, deren Statistik sich weiter verschiebt.
  Wählbar bleibt die Option.

### 5.7 Gespeicherter Zustand

- `PPO`: Policy, Value-Netz, Optimizer. Kein Replay Buffer.
- `TD3`: Actor, beide Critics, Target-Netze, Optimizer, Replay Buffer.
- `SAC`: wie TD3 plus gelernter Temperaturparameter.
- Bei aktiver Normalisierung zusätzlich die `VecNormalize`-Statistiken.

### 5.8 Fairness On-Policy gegen Off-Policy

Bei gleichem Schrittbudget fällt ein Vergleich systematisch zugunsten der
Off-Policy-Verfahren aus – sie lernen aus jedem Übergang mehrfach, PPO verwirft
seine Daten nach jedem Update. Das gilt erst recht, wenn drei oder vier Slots
ein On-Policy-Verfahren gegen mehrere Off-Policy-Verfahren stellen. Kein
Messfehler, sondern eine Eigenschaft der Verfahrensklassen;
Bedienungsanleitung und README sagen das ausdrücklich.

### 5.9 Verfahrensbezogene Tests

- Konstruktorargumente enthalten nur Schlüssel, die der jeweilige
  SB3-Algorithmus kennt.
- `PPO`: Wirkung von `clip_range`, `gae_lambda`, `n_epochs`; kein Replay
  Buffer; Fortsetzen ohne Rücksetzen des Schrittzählers.
- `TD3`: verzögerte Actor-Updates gemäß `policy_delay`, Clipping des
  Target-Rauschens, Action Noise wirkt im Training und nicht in der Evaluation.
- `SAC`: automatische Entropieanpassung verändert `α`; Zielentropie bei `auto`
  genau `-dim(A)`.
- Save-/Load-Roundtrip je Verfahren mit genau den Bestandteilen aus 5.7.
- Ablehnung einer Modelldatei, deren Algorithmus nicht zum aktiven Slot passt.

## 6 Oberfläche

### 6.1 Aufbau

- konsistentes helles oder dunkles Farbschema; bei Dark Mode gut lesbarer
  Kontrast für Texte, Eingabewerte, deaktivierte Controls, Achsen, Legenden,
  Statusmeldungen
- Kopfbereich mit Titel, Untertitel, Status
- Hauptbereich mit verschiebbarem horizontalem Splitter; beide Hälften anfangs
  gleich hoch und frei skalierbar
- oben links das kompakt gruppierte Bedienpanel, rechts die
  Environment-Visualisierung
- Bedienpanel als Drei-Spalten-Raster: Spalte 1 und 2 tragen ausschließlich die
  Verfahrenswahl mit ihren Parametergruppen (6.2), Spalte 3 die
  Steuerungsbuttons untereinander über die volle Spaltenbreite mit einheitlich
  großen Klickflächen
- Eingabe- und Auswahlfelder stehen in ihrer Gruppe rechtsbündig; Breite am
  längsten erwartbaren regulären Wert orientiert – so schmal wie sinnvoll, aber
  ohne Abschneiden
- unten über die volle Fensterbreite: Tabs für Diagramme und Vergleiche,
  daneben gleichzeitig sichtbar die Summary. Die Summary liegt **nicht** in
  einem eigenen Tab; der Graph bekommt den deutlich größeren Anteil
- Bedienpanel und Visualisierung erhalten feste bzw. gewichtete Platzanteile,
  sodass keines das andere auf 1 × 1 Pixel drückt
- die Animation nutzt den gesamten verbleibenden Platz; Frames werden unter
  Beibehaltung des Seitenverhältnisses größtmöglich skaliert, nie beschnitten
- alle wesentlichen Parameter und Buttons sind bei Mindestfenstergröße
  gleichzeitig sichtbar; Scrollen ist Fallback, nicht Standardlayout
- große Tabellen mit horizontaler und vertikaler Scrollbar

Fenstergröße und initiale Splitterposition werden aus Bildschirmgröße,
Mindestgröße der Controls und Mindestplatz für Visualisierung und Diagramm
abgeleitet, nicht blind gesetzt. Beim Start darf kein wesentliches Widget
abgeschnitten sein, und das Fenster überschreitet den nutzbaren
Bildschirmbereich nicht.

Controls spiegeln `Bereit`, `Läuft`, `Gestoppt`, `Abgeschlossen` oder `Fehler`.
Inkompatible Aktionen werden gezielt deaktiviert und nach Erfolg, Abbruch oder
Fehler wieder freigegeben.

Jede App besitzt eine `Bedienungsanleitung`: empfohlener Ablauf, Environment
und Rewards, Methoden, Training gegenüber Evaluation, Bedeutung der Slots und
ihrer Anzahl, Parameter, Ansichten, typische Ursachen ausbleibenden
Lernerfolgs. Sie sagt außerdem, dass die Werte neben der Animation erst beim
Überfahren erscheinen – sonst sucht man sie vergeblich unter dem Bild.

### 6.2 Verfahrenswahl und Vergleichstabs

Projekte mit mehreren Algorithmen bieten bis zu **vier** gleichrangige Slots:
mehrere Algorithmen, mehrere Parametrisierungen desselben Algorithmus oder eine
Mischung.

- `Anzahl Verfahren` (Werte `2`, `3`, `4`) liegt global außerhalb der Tabs. Der
  Prompt nennt Standardwert und Startbelegung.
- Belegt die Startbelegung zwei Slots mit demselben Algorithmus, bekommt einer
  einen abweichenden Startwert – sonst wären sie identisch und der Vergleich
  zeigte nichts. Das gilt nur für die Startbelegung; später hinzugefügte Slots
  starten mit den unveränderten Standardwerten ihres Algorithmus.
- Über den Parameterspalten stehen die Dropdowns `Verfahren 1` bis
  `Verfahren 4`; mehrere Slots dürfen denselben Algorithmus enthalten.
- Darunter gleich aufgebaute Tabs `Verfahren 1` bis `Verfahren 4`, jeder mit
  den vollständigen und unabhängigen Parametern seines Algorithmus samt Budget,
  Seed und Netzwerkparametern. Parameter werden nicht geteilt.
- Global bleiben nur Einstellungen, die für einen fairen Vergleich in allen
  Läufen identisch sein müssen: `Anzahl Verfahren`, Intervall und Umfang der
  Zwischenevaluation.
- Nicht aktive Slots verschwinden vollständig – weder Dropdown noch Tab wird
  erzeugt. Deaktivierte Karteileichen widersprechen der Regel aus Abschnitt 4.

Beim Ändern von `Anzahl Verfahren`:

- kleiner und ein wegfallender Slot hat Lernzustand oder Messdaten → GUI fragt
  nach und verwirft erst nach Bestätigung
- größer → neue Slots starten mit den Standardwerten ihres Algorithmus und ohne
  Lernzustand
- vorhandene Slots bleiben in beiden Fällen unberührt
- war der aktive Slot ein wegfallender, wird `Verfahren 1` aktiv
- während eines laufenden Laufs ist das Feld gesperrt

Genau ein Slot ist das **aktive Verfahren**, bestimmt durch den gewählten Tab
und über den Steuerungsbuttons unübersehbar angezeigt (`Aktiv: Verfahren 1 –
PPO`). Alle Einzellauf-Aktionen wirken darauf. Läuft bereits ein Einzellauf,
bleibt dessen Ziel fixiert; ein Tabwechsel ändert nur die angezeigten
Parameter.

Jeder Slot besitzt eine feste Farbe, durchgängig in Vergleichsgraph, Legende,
Summary-Kopfzeile und Animationsbeschriftung:

| Slot | Farbe |
| --- | --- |
| `Verfahren 1` | Blau |
| `Verfahren 2` | Rot |
| `Verfahren 3` | Gelb |
| `Verfahren 4` | Grün |

Enthält ein Projekt nur einen Algorithmus, entfallen die Dropdowns; die Tabs
bleiben und vergleichen Parametrisierungen.

### 6.3 Steuerungsbuttons

In Spalte 3, soweit fachlich sinnvoll, in dieser Reihenfolge:

1. `Training starten / fortsetzen`
2. `Stoppen`
3. `Deterministisch evaluieren`
4. `Sichtbare Episode abspielen`
5. `Vergleich starten / fortsetzen`
6. `Bestes Modell wiederherstellen`
7. `Neues Modell`

- 1, 3, 4, 6, 7 wirken auf das aktive Verfahren, 5 auf alle aktiven Slots.
- Buttons zum manuellen Speichern und Laden gibt es nicht: Der Lernzustand wird
  über den automatischen Checkpoint gesichert und mit Button 6 zurückgeholt.
- Darunter folgen Animationssteuerung (7.1), Fortschrittsanzeige und
  Statuszeile.
- Beschriftungen benennen nur Bestandteile, die **jedes** Verfahren des
  Projekts besitzt. Besitzt nur ein Teil einen Replay Buffer, taucht er in
  keiner Beschriftung auf; was ein Button betrifft, erklären Statusmeldung,
  Bedienungsanleitung und README.

### 6.4 Responsivität

- Die GUI bleibt bei Training, Evaluation, Vergleich und Animation bedienbar.
- Lange Arbeit läuft in kleinen `after()`-Schritten oder in Worker-Threads mit
  Queue; Worker greifen nie direkt auf Tkinter-Widgets zu.
- Plot- und Statusaktualisierungen werden auf eine sinnvolle Frequenz begrenzt.
- Abbruch wird regelmäßig geprüft; beim Schließen werden Worker und Ressourcen
  sauber beendet.

## 7 Animation

### 7.1 Steuerung und Inhalt

Global neben den Steuerungsbuttons, keinem Slot zugeordnet: genau **ein**
Schalter und **ein** Eingabefeld.

- `Animation zeigen` schaltet die Einzelbildanimation jederzeit ein und aus,
  auch mitten in einem Lauf. Eingeschaltet zeigt sie einzeln abgespielte
  Episoden ebenso wie den laufenden Lauf.
- `Bildrate (FPS)`: Standard ist `env.metadata["render_fps"]`, gültig `1` bis
  `250`. Die Obergrenze liegt bewusst über jeder üblichen Environment-Rate –
  läge sie darunter, wäre der Standardwert selbst ungültig. Eine Änderung wirkt
  spätestens mit der nächsten sichtbaren Episode, auch während eines Laufs.
  Dauert eine Episode bei der nativen Rate ungewöhnlich lange, nennt die
  Bedienungsanleitung einen brauchbaren höheren Startwert.

Gezeigt wird ausschließlich der offizielle, von `env.render()` gelieferte
RGB-Frame. Keine eigene Grafik, kein separates Fenster; der Prompt nennt nur
die Bildgröße.

Während eines Laufs zeigt die Animation fortlaufend Episoden des **aktuellen
Lernstands** im isolierten Renderprozess. Die Trainingsschleife rendert nicht –
sie liefe sonst im Takt der Darstellung. Die Animation arbeitet auf einer
**Kopie der Policy**, gezogen zu Beginn jeder sichtbaren Episode, und greift
nie aus dem GUI-Thread in das lernende Netz.

Die Animation kostet Rechenzeit und verlangsamt das Training spürbar; die
eingestellte Bildrate ist eine Obergrenze. Bedienungsanleitung erwähnt beides,
und der Lauf bleibt ohne Animation voll funktionsfähig.

### 7.2 Raster

Einzeltraining zeigt den fixierten Slot; ein Vergleich zeigt **alle aktiven
Verfahren gleichzeitig**, jedes mit eigenem Bild und eigener Beschriftung.
Außerhalb eines Laufs ist genau das Feld des aktiven Verfahrens sichtbar.

| Sichtbare Anzeigen | Raster |
| --- | --- |
| 1 | eine Anzeige über den gesamten Bereich |
| 2 | zwei nebeneinander in einer Zeile |
| 3 | zwei in der ersten Zeile, eine in der zweiten |
| 4 | zwei je Zeile und zwei je Spalte |

- Höchstens zwei Spalten und zwei Zeilen; alle Zellen gleich groß, Zeilen und
  Spalten gleich gewichtet.
- Die Beschriftung unter dem Bild bekommt ihren Platz **vor** dem Bild
  zugeteilt – ein Bild, das den Rest füllt, drückt sie sonst unbemerkt aus
  einer knappen Zelle. Der Layout-Test prüft sie ausdrücklich mit.
- Die Bildgröße leitet sich aus dem verfügbaren Platz und der Zahl der Anzeigen
  ab, nicht aus dem zuletzt gezeigten Bild – sonst behielte ein einmal großes
  Bild seinen Platz.
- Zellen behalten ihre Position, während Episoden beginnen und enden.

### 7.3 Welchen Lernstand eine Anzeige zeigt

Je Anzeige höchstens ein weiteres, **slotgebundenes** Bedienelement: die Wahl
des gezeigten Lernstands. Sie wirkt nur auf ihre eigene Anzeige und steht
**über** oder neben dem Bild, nie darunter. Zur Wahl stehen genau zwei Stände:

- `aktuell` (Standard): der laufende Lernstand. Die Beschriftung nennt die
  Nummer der zuletzt trainierten bzw. verglichenen Episode – dieselbe Nummer
  wie auf der X-Achse der Graphen. Ein eigener, bei jedem Einschalten wieder
  bei 1 beginnender Animationszähler ist **unzulässig**.
- `beste`: der Lernstand der bisher besten Episode, gemessen am explorativen
  Return. Abgespielt mit **festem Seed**, damit die Wiederholung jedes Mal
  gleich aussieht; die Beschriftung macht erkennbar, dass nicht der aktuelle
  Stand läuft. Ist `beste` gewählt, aber noch keine Episode abgeschlossen,
  läuft der aktuelle Stand weiter und die Statuszeile sagt das.

Gesichert wird dafür je Slot **genau ein** zusätzlicher Lernstand. Ein Verlauf
über alle Episoden scheidet aus: Bei großen Budgets entstehen Tausende
Episoden, deren Policy-Kopien Gigabytes belegten. Der Stand entsteht im
Worker-Thread unmittelbar nach dem Episodenende – nur dort gehört die Policy zu
dieser Episode – als losgelöste Kopie, die der Optimizer nicht mehr verändert.
Ein neues Modell verwirft ihn mit.

### 7.4 Beschriftung und Messwerte

Jede Anzeige trägt einen Titel mit Slot und Algorithmus, etwa
`Verfahren 3 – SAC`. Teilen sich mehrere Slots einen Algorithmus, ergänzt der
Titel – mit demselben Wortlaut wie die Legende des Vergleichsgraphen – den
wichtigsten abweichenden Parameter: `Verfahren 3 – SAC (Lernrate α 0.0003)`.
Titel und Legende stammen aus derselben Quelle.

Unter jedem Bild steht dauerhaft eine kurze Zeile mit genau diesen drei Angaben
und nichts sonst: **Episode**, **Schritt** innerhalb der Episode, bisher
kumulierter **Return**.

- Das Verfahren gehört in den Titel, nicht in diese Zeile – in einer schmalen
  Zelle ist der Platz knapp, und der Titel steht ohnehin darüber.
- Abkürzen ist erlaubt, etwa `E: 57 · S: 354/1000 · R: 2.700,0`.
- Fester Aufbau und feste Höhe, damit die Bilder beim Weiterzählen nicht
  springen.
- Weitere Messwerte gehören nicht darunter: Bei vier Anzeigen bliebe sonst kein
  Platz für die Bilder.

Alle übrigen Messwerte – insbesondere die Action, je nach Projekt die
Observationswerte – erscheinen nur, solange der Mauszeiger über dem Bild steht,
und werden **neben** dem Bild eingeblendet:

- Die Einblendung verdeckt ihr eigenes Bild nicht und schneidet es nicht ab.
- Erscheinen und Verschwinden verändern Größe und Position der Bilder nicht:
  Entweder ist der Platz dauerhaft reserviert, oder die Einblendung liegt als
  Overlay über dem Nachbarbereich.
- Sie gehört zu genau dem überfahrenen Bild und nennt dessen Slot; steht der
  Zeiger über keinem Bild, ist keine Einblendung sichtbar.
- Sie aktualisiert sich mit der Bildfrequenz und zeigt den gerade
  dargestellten Frame, nicht einen älteren.
- Gut lesbar: fester Zeichensatz, ausreichender Kontrast, feste Spaltenbreiten.
- Ihren Inhalt legt der Projekt-Prompt fest.

### 7.5 Renderprozesse

- Auf macOS dürfen Tkinter und ein nativer Grafikkontext – SDL-/Pygame-Fenster
  ebenso wie OpenGL-Kontext – nicht im selben Prozess initialisiert werden,
  wenn das zu nativen Abstürzen führen kann. Dann läuft ausschließlich das
  Rendering in einem isolierten, unsichtbaren Prozess mit headless
  Grafiktreiber; die GUI erhält nur RGB-Frames.
- Jede sichtbare Anzeige erhält ihren **eigenen** Prozess mit eigener
  Environment-Instanz; bei vier Anzeigen also vier Hilfsprozesse. Die
  Bedienungsanleitung sagt, dass mehr Animationen den Lauf stärker ausbremsen.
- Die Hilfsprozesse erzeugen weder ein eigenes Fenster noch einen zusätzlichen
  Dock- oder Programmeintrag und werden beim Schließen beendet. Genügt der
  naheliegende Treiber dem nicht, wird ein passender gewählt und die Wahl
  begründet – ein Dock-Eintrag je Anzeige ist ein Fehler, kein hinnehmbarer
  Nebeneffekt.
- Für MuJoCo regelt 7.6 Treiber und Umgebungsvariable abschließend; bei anderen
  Environment-Familien nennt sie der Projekt-Prompt.

### 7.6 MuJoCo-Environments

Gilt für alle MuJoCo-Projekte, damit es in keinem Prompt erneut steht:

- `MUJOCO_GL` wird gesetzt, **bevor** `mujoco` bzw. `gymnasium.envs.mujoco`
  importiert wird: macOS `cgl`, headless Linux `egl`, ersatzweise `osmesa`.
  Eine bereits gesetzte Variable bleibt unverändert.
- `glfw` ist auf macOS **unzulässig**, obwohl es funktioniert und Gymnasiums
  Standardliste es führt: `glfw.init()` meldet den Prozess beim Window Server
  als Vordergrund-App an, sodass je Anzeige ein Programm- und Dock-Eintrag
  entsteht. Nachprüfbar mit `lsappinfo list`.
- Gymnasiums `MujocoRenderer` führt `cgl` nicht in seiner Backend-Tabelle,
  obwohl MuJoCo den Kontext unter `mujoco.cgl` mitbringt. Der Renderprozess
  ergänzt den Eintrag vor dem Erzeugen des Environments; die mitgelieferten
  Backends bleiben unangetastet.
- Schlägt der Import von `mujoco` oder der Grafikkontext fehl, meldet die
  Anwendung das verständlich auf Deutsch mit Hinweis auf
  `pip install "gymnasium[mujoco]"` und `MUJOCO_GL`, statt abzustürzen.
  Training und Evaluation ohne Animation bleiben nutzbar.

## 8 Diagramme und Summary

### 8.1 Metrikwahl

Zeige nur für Environment und Algorithmus sinnvolle Metriken, etwa
Episode-Return, gleitenden Durchschnitt, Erfolgsrate, Episodenlänge,
Exploration, Environment-Schritte und bei neuronalen Netzen den Loss.

Metriken, Anzeigen und Hilfetexte werden aus dem **aktuellen** Environment
abgeleitet, nie aus einem Vorgängerprojekt übernommen. Diese Projekte ähneln
einander stark, und genau daraus entstehen die hartnäckigsten Fehler: eine
Sturzquote ohne terminalen Zustand, ein Clipping-Hinweis, wo nichts geclippt
wird, ein Überlebensbonus, den es nicht gibt.

- Jede Kennzahl muss sich aus den Regeln des aktuellen Environments begründen
  lassen. Passt sie nicht, entfällt sie ersatzlos.
- Eine im Environment **konstante** Kennzahl gehört nicht als Summary-Zeile,
  sondern einmal in den Text und in einen Test.
- Environmentkonstanten – Übersetzungen, Skalierungen, Grenzen, Bildraten –
  werden aus dem Environment gelesen oder als Tabelle hinterlegt und gegen das
  Environment getestet. Nichts wird geraten oder kopiert.

### 8.2 Achsen, Legende, Linien

- Achsen, Einheiten und Methoden sind beschriftet; die Achsenbeschriftung
  bleibt knapp. Wertungen wie „höher ist besser" und Schwellen gehören in
  Legende, Summary und Hilfetexte, nicht an die Achse.
- Über den Achsen steht keine Überschrift, wenn der Tab- oder Gruppentitel
  bereits sagt, was zu sehen ist – der Platz gehört den Kurven.
- Die Legende liegt bei ausreichender Breite in einem reservierten Bereich
  außerhalb der Achsen, verdeckt keine Datenlinien und wird am Figure-Rand
  nicht abgeschnitten. Die Breite dieses Bereichs wird aus der **tatsächlichen**
  Legendenbreite abgeleitet, nicht fest gewählt: Labels mit abweichendem
  Parameter sprengen jede feste Reserve.
- Slots werden allein über die **Farbe** unterschieden (6.2), nicht über den
  Linienstil. Jede hervorgehobene Slotkurve ist **durchgezogen**; gestrichelte
  oder gepunktete Slotkurven sind unzulässig, weil sie bei vier Verfahren
  unlesbar werden. Die Farbwerte heben sich deutlich voneinander und vom
  Hintergrund ab.
- Referenz- und Schwellenlinien sind **weiß und gestrichelt** – weder in der
  Farbe noch im Strich mit einer Datenlinie zu verwechseln.
- Rohkurven nutzen dieselbe Slotfarbe mit deutlich verringerter Deckkraft und
  Strichstärke; Rohwerte und geglättete Werte bleiben unterscheidbar.
- Training und Evaluation werden optisch getrennt. Deterministische
  Evaluationsergebnisse stehen in der Summary und müssen nicht zusätzlich im
  Graphen erscheinen.
- Lange Rohkurven werden nur für die Darstellung auf eine feste Punktzahl
  verdichtet; die Messdaten bleiben vollständig. Min-/Max-Verdichtung ist
  einfachem Auslassen vorzuziehen. Plot-Updates werden gedrosselt.
- Fehlende Daten werden nicht durch künstliche Nullwerte ersetzt.

### 8.3 Gleitender Durchschnitt

Je Slot hebt eine kräftige Linie den gleitenden Durchschnitt der
Episodenergebnisse hervor. Seine Fensterbreite ist **einstellbar**:

- Die Einstellung liegt am Diagramm selbst, nicht in den Verfahrenstabs – etwa
  in derselben Leiste wie die Exportschaltfläche –, ist mit `Glättung`
  beschriftet und benennt damit die Wirkung, nicht die Mechanik. Sie gilt
  **global für alle Slots und beide Graphen**: Unterschiedlich stark geglättete
  Kurven wären nicht vergleichbar.
- Standard `20` Episoden, gültig `1` bis `500`. Bei `1` fällt die geglättete
  Kurve mit den Rohwerten zusammen.
- Eine Änderung wirkt **sofort**, auch mitten in einem Lauf, und ändert
  ausschließlich die Darstellung: Messdaten bleiben vollständig, der Lauf wird
  nicht berührt.
- Eine ungültige Eingabe lässt die zuletzt gültige Breite stehen und wird in
  der Statuszeile erklärt. Ein modaler Dialog ist hier unzulässig – er erschiene
  bei jedem Tastendruck.

### 8.4 X-Achse, Zwischenevaluation, Checkpoint

- Trainings- und Vergleichskurven führen einheitlich **Episoden** auf der
  X-Achse. Budget und tatsächlich ausgeführte Schritte bleiben separat in
  Status und Summary sichtbar.
- Punkte entstehen nur für vollständig abgeschlossene Episoden. Endet ein
  Budget innerhalb einer Episode, liegt der letzte Punkt vor dem ausgeführten
  Schrittbudget; GUI und Summary zeigen ausgeführte Schritte, angefordertes
  Budget und diese Bedeutung getrennt und verständlich.
- Längere Läufe werden in einem sichtbaren, konfigurierbaren Schrittintervall
  automatisch in einer separaten headless Environment deterministisch
  evaluiert. Das erzeugt keine Animation und verändert weder Modell noch Replay
  Buffer; Graph und Summary werden live aktualisiert.
- Der beste deterministische Evaluationswert wird je Slot getrennt gemerkt. Bei
  jeder Verbesserung wird der **vollständige** Lernzustand des Slots konsistent
  als gemeinsamer Checkpoint gesichert, in der Summary ausgewiesen und über
  einen klar beschrifteten Button wiederherstellbar. Welche Bestandteile
  dazugehören, hängt vom Verfahren ab (5.7); wiederhergestellt wird der
  vollständige Lernzustand, nicht nur das Netz.

### 8.5 Vergleich

Der Vergleich stellt immer alle aktiven Slots gegenüber. Ein gemeinsamer
Vergleichsgraph ist verpflichtend, mit derselben X-Achse und derselben Metrik
für alle Läufe.

- Legende und Beschriftung benennen Slot und Algorithmus (`V1 – PPO`). Teilen
  sich mehrere Slots einen Algorithmus, nennt das Label zusätzlich den
  wichtigsten abweichenden Parameter.
- Der Graph erscheint mit dem ersten Ergebnis und wird während aller Läufe in
  einem sinnvollen Intervall fortgeschrieben – nicht erst nach Abschluss.
- Alle aktiven Slots starten parallel; die Fortschrittsanzeige aggregiert ihre
  ausgeführten Schritte.
- Ein erneut gestarteter, kompatibel konfigurierter Vergleich setzt die
  Vergleichsmodelle nicht zurück, sondern setzt ihr Training fort und hängt
  neue Messpunkte an.
- Vergleiche verändern das sichtbare Experiment nicht. Alle Slots erhalten
  identische Environment-Konfigurationen und reproduzierbar abgeleitete Seeds.
  Budget und Seed stammen aus dem jeweiligen Tab und sind frei wählbar, damit
  auch Budget- und Seed-Vergleiche möglich sind; das Budget wird bevorzugt in
  Environment-Schritten angegeben.
- Jeder Slot und jede Wiederholung startet mit neuem Lernzustand; Evaluation
  ohne Exploration und Lernupdates.
- Vor dem Start wird der Gesamtumfang angezeigt. Mehrere Wiederholungen werden
  mit Mittelwert und Standardabweichung oder 95-%-Konfidenzintervall
  aggregiert; der Graph zeigt sie als Unsicherheitsband. Bei Abbruch bleiben
  vollständige Ergebnisse erhalten, unvollständige werden gekennzeichnet.

### 8.6 Summary

- Sie wird live aktualisiert und zeigt für Training und Vergleich konsistent
  Episoden, ausgeführte Environment-Schritte, aktuelle bzw. gemittelte Rewards
  und Erfolgsrate.
- Die Vergleichs-Summary besitzt genau eine Ergebnisspalte je aktivem Slot,
  `Verfahren 1` bis `Verfahren 4`, mit dem Algorithmusnamen in der Kopfzeile.
- Sie enthält einen Abschnitt mit genau den Parametern, in denen sich die
  Konfigurationen unterscheiden – je Parameter eine Zeile mit dem Wert aller
  Slots. Ohne ihn ist ein Vergleich mehrerer Parametrisierungen desselben
  Algorithmus nicht interpretierbar.
- Unterscheiden sich die Trainingsbudgets, weist die GUI vor dem Start sichtbar
  darauf hin; unzulässig ist es nicht.
- Bei drei oder vier Spalten bleibt sie vollständig lesbar: notfalls mit
  horizontaler Scrollbar, statt Werte abzuschneiden.

### 8.7 Export

- Diagramm und Summary werden über klar bezeichnete Aktionen exportiert: der
  Graph mindestens als PNG in der dargestellten Form, die Summary als gut
  lesbare UTF-8-Textdatei. CSV ist nicht erforderlich.
- Dateidialoge schlagen aussagekräftige Namen vor und überschreiben bestehende
  Dateien nicht unbemerkt.
- Die Schaltflächen bekommen **keine** eigene Kopfzeile: Sie liegen kompakt in
  einer ohnehin vorhandenen Leiste – etwa der Tableiste des Diagramms –, damit
  die übrige Höhe der Darstellung gehört, und verdecken weder Kurven noch
  Legende noch Text.

### 8.8 Tabellen und Modelldateien

- Tabellen zeigen den vollständigen aktuellen Lernstand, unterscheiden besuchte
  und unbesuchte Zustände und sind scrollbar.
- Gespeicherte Modelle enthalten Lernzustand, Format-Version und Metadaten:
  Algorithmus, vollständige Slot-Konfiguration, Environment-Kennung samt aller
  abweichenden Konstruktorargumente.
- Methode und Environment werden beim Laden auf Kompatibilität geprüft.
  Fehlerhafte oder inkompatible Dateien verändern den aktiven Zustand nicht.
- Gespeichert wird genau der Zustand, den das Verfahren besitzt (5.7). Ein
  Replay Buffer wird andernfalls nicht erwähnt und nicht durch leere
  Platzhalter ersetzt.
- Geladen wird immer in den aktiven Slot; passt die Datei nicht zum dort
  gewählten Algorithmus, wird sie verständlich abgelehnt.

## 9 Qualität

### 9.1 Fehlerbehandlung und Performance

- erwartbare Eingabefehler als deutsche Dialogmeldung
- technische Fehler abfangen und verständlich melden
- Busy-Zustände auch nach Fehlern beenden
- Ressourcen sauber freigeben
- bestehende Dateien nicht ohne Nachfrage überschreiben
- zuerst eine korrekte, getestete Referenzimplementierung; optimiert wird erst
  nach Messung eines repräsentativen Laufs, danach werden Tests und
  Lernergebnis erneut geprüft

### 9.2 Tests

Tests laufen nicht beim App-Start. Sie prüfen mindestens:

- Environment-Übergänge, Rewards, Termination, Truncation
- Updateformeln, Action-Auswahl, Tie-Breaking jedes Algorithmus
- Parametergrenzen, Reset-Verhalten, Ergebnisobjekte
- Trennung von Training und Evaluation
- reproduzierbaren Lernfortschritt in einem kurzen Szenario
- kurzen Trainings-, Evaluations- und Vergleichslauf
- Vergleich mit zweimal demselben Algorithmus und unterschiedlichen Parametern:
  getrennte Lernzustände, Ergebnisse und Kurven, keine gegenseitige
  Beeinflussung
- Vergleich über die vom Projekt unterstützte Höchstzahl an Slots: je Slot
  getrennte Zustände und Kurven, je Slot eine Summary-Spalte, aggregierte
  Fortschrittsanzeige
- Verkleinern und Vergrößern von `Anzahl Verfahren`: Verwerfen erst nach
  Bestätigung, neue Slots mit Standardwerten, bestehende unverändert
- Farbzuordnung und Linienstil: je Slot die vorgesehene Farbe, hervorgehobene
  Slotkurven durchgezogen, Referenzlinie weiß und gestrichelt
- die Beschriftung unter dem Animationsbild enthält genau Episode, Schritt und
  Return; Action- und Observationswerte erscheinen nur in der Einblendung
- Import und Konstruktion der App-Komponenten

Bei neuronalen Netzen zusätzlich: Ein- und Ausgabeformen, Targets, Loss,
Optimizer-Schritt, Target-Network-Update, Replay Buffer und ggf.
Save-/Load-Roundtrip.

**GUI-Smoke-Test** (nur mit verfügbarem Display): Visualisierung, alle
wesentlichen Buttons, der Diagrammbereich sowie alle Eingabe-, Auswahlfelder,
Checkboxen und Fortschrittsanzeigen sind tatsächlich gemappt und liegen im
sichtbaren Fenster. Der Test läuft mit vorgesehener Startfenstergröße und
initialer Splitterposition; reine Widget-Konstruktion genügt nicht. Geprüft
werden alle aktiven Verfahrenstabs – auch die zunächst nicht sichtbaren nach
dem Umschalten – bei der größten unterstützten Zahl von Verfahren, sowie das
Animationsraster mit einer, zwei, drei und vier Anzeigen: alle Bilder
vollständig im Fenster, keine Zelle auf unbrauchbare Größe gedrückt.

### 9.3 Abnahme

Ein Projekt ist abgeschlossen, wenn alle projektspezifischen Verfahren korrekt
implementiert sind, die GUI responsiv bleibt, Vergleiche fair und isoliert
ablaufen, fachlicher Lernfortschritt getestet ist und alle Tests erfolgreich
sind.

### 9.4 README

- Ziel, Installation, Startbefehl
- Environment, Actions, Rewards
- Methoden und wesentliche Formeln in verständlicher Sprache
- Parameter, Standardwerte, Quellen. Nennt das Projekt eine Gelöst-Schwelle,
  nennt die README zusätzlich, welche Ergebnisse die verwendeten Profile laut
  Referenzquelle tatsächlich erreichen – auch und gerade dann, wenn sie die
  Schwelle verfehlen
- Bedienablauf und Interpretation der Ansichten
- Verfahrensslots, ihre Anzahl, Vergleichslogik sowie Speichern und Laden
- Testbefehl und bekannte Grenzen
