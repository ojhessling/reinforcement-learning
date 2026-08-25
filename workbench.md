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
genügen. Ein Verweis nennt **Nummer und Überschrift**, etwa
`Workbench 5.9 (Fairness zwischen Verfahrensklassen)` – die Nummer ist bequem,
die Überschrift bleibt auch nach einer Umstrukturierung eindeutig.

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

Die Anwendung zerfällt sichtbar in einen **Konfigurator** – das Bedienpanel mit
Verfahrenswahl, Parametern und globalen Einstellungen (6.1, 6.2) – und einen
**ausführenden Teil**: Runner für Training, Evaluation und Vergleich samt
Animation, Diagramm und Summary. Beide teilen sich ein Fenster, aber keine
Zuständigkeiten.

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

Die Netzgröße darf sich zwischen den Verfahren unterscheiden, wenn die
Mathematik des Verfahrens das erzwingt – etwa weil ein Optimierer über der Zahl
der Parameter quadratisch oder kubisch skaliert und ein Netz mit
Zehntausenden Gewichten schlicht nicht mehr rechenbar ist. Der Prompt nennt
Grund und Größenordnung. Fairness bemisst sich nach dem Schrittbudget, nicht
nach gleicher Architektur.

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

Verwendet werden die SB3-Implementierungen unverändert. Kennt Stable-Baselines3
ein Verfahren nicht, tritt an seine Stelle eine **etablierte
Referenzbibliothek**; der Prompt nennt sie samt Version. Eine Eigen-
implementierung des Optimierers ist auch dann unzulässig. Eigene Arbeit liegt
in Konfiguration, Runnern, Metriken, Vergleich, GUI und Tests.

### 5.1 Gemeinsame Parameter

In jedem Verfahrenstab, **soweit das Verfahren den Parameter kennt**:
`total_timesteps`, `learning_rate` mit konstantem oder linear fallendem Verlauf
(SB3 akzeptiert eine Callable-Schedule), `batch_size`, `gamma`, `seed`, Hidden
Layers, Aktivierung, Optimizer samt `eps` und `weight_decay`.

Ausnahmslos in **jedem** Tab stehen nur `total_timesteps`, `Episoden` und
`seed`: Sie definieren Budget und Reproduzierbarkeit und existieren in jeder
Verfahrensklasse. Alles andere folgt der Regel aus Abschnitt 4 – ein Parameter,
den das gewählte Verfahren nicht besitzt, wird weggelassen und nicht als
deaktiviertes Feld mitgeschleppt. Ein gradientenfreies Verfahren hat etwa weder
`learning_rate` noch `batch_size` noch `gamma`.

Global außerhalb der Tabs, weil sie für alle Läufe gleich gelten müssen:
`Anzahl Verfahren` und die Fensterbreite des gleitenden Durchschnitts (8.3).

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
- `Episoden` steht standardmäßig auf `1000`.
- Beide Standardwerte müssen **zueinander passen**. Liegt der eine um
  Größenordnungen über dem anderen, greift immer dieselbe Grenze und die andere
  ist reine Dekoration. Der Prompt rechnet vor, welcher Schrittzahl die
  Episodenvorgabe im konkreten Environment ungefähr entspricht, und wählt den
  Schritt-Standard in derselben Größenordnung. Das Budget der eigentlichen
  Messläufe darf davon abweichen und wird gesondert genannt.

**Zwei Budgetgrenzen.** Jeder Slot führt `total_timesteps` **und** `Episoden`.

- Der Lauf endet, **was zuerst eintritt**. Status und Summary nennen beide
  Grenzen und weisen aus, **welche** den Lauf tatsächlich beendet hat.
- Beide Grenzen gelten **relativ zum bereits Gelaufenen**. Ein erneut
  gestarteter Lauf setzt nichts zurück, sondern hängt erneut das **volle**
  Budget an; Kurve und Summary wachsen weiter. Bei Schritten erledigt das
  Stable-Baselines3 selbst, sofern `reset_num_timesteps=False` gesetzt ist; die
  Episodengrenze muss ausdrücklich um die bereits gelaufenen Episoden
  verschoben werden – sonst wäre sie beim zweiten Start sofort überschritten
  und der Lauf endete nach einer einzigen Episode.
- `Episoden = 0` bedeutet unbegrenzt; dann greift allein das Schrittbudget.
- Für den **Vergleich** mehrerer Verfahren ist das Schrittbudget die faire
  Grenze. Eine Episodengrenze bevorzugt systematisch das Verfahren, das besser
  lernt: Überall dort, wo Scheitern eine Episode vorzeitig beendet, werden
  Episoden mit dem Lernfortschritt länger, und der bessere Agent sammelt bei
  gleicher Episodenzahl mehr Environment-Schritte – also mehr Trainingsdaten.
  Sein Vorsprung im Diagramm wäre dann teilweise nur ein Vorsprung an
  Erfahrung. Der Prompt sagt ausdrücklich, welche Grenze die Messläufe steuert.
- Wie stark die beiden Grenzen auseinanderfallen, hängt am Environment: Der
  Prompt nennt die mittlere Episodenlänge einer untrainierten Policy und
  rechnet vor, welcher Schrittzahl die Episodenvorgabe im besten und im
  schlechtesten Fall entspricht.

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

### 5.5 CMA-ES

Gradientenfreies, populationsbasiertes Verfahren aus der Familie der
Evolutionsstrategien. Es optimiert **direkt die Policy-Gewichte**: Aus einer
mehrdimensionalen Normalverteilung `N(m, σ²·C)` werden λ Kandidaten gezogen,
jeder wird über ganze Episoden bewertet, und aus den nach Rendite sortierten
Kandidaten werden Mittelwert `m`, Schrittweite `σ` und Kovarianz `C` neu
geschätzt.

```text
xₖ ~ m + σ·N(0, C),  k = 1 … λ        Kandidaten einer Generation
m ← Σ wᵢ · x_{i:λ}                    gewichteter Mittelwert der besten μ
σ, C ← aus den erfolgreichen Schritten fortgeschrieben
```

Damit unterscheidet es sich in jeder Hinsicht von den SB3-Verfahren:

- **Kein Gradient, kein Replay Buffer, kein Critic.** Es gibt keine
  `learning_rate`, keine `batch_size` und kein `gamma` – optimiert wird die
  **undiskontierte** Episodenrendite, also genau die Größe, die der Graph zeigt.
- **Rangbasiert.** Nur die Reihenfolge der Kandidaten zählt, nicht der Abstand
  ihrer Renditen. Jede monotone Umskalierung der Rendite ist wirkungslos – eine
  Reward-Normalisierung ist deshalb nicht nur unnötig, sondern sinnlos.
- **Episodenweise.** Ein Kandidat wird über vollständige Episoden bewertet;
  Information einzelner Schritte wird nicht genutzt.
- **Deterministische Policy.** Exploriert wird im Parameterraum, nicht über die
  Action.

Verbindlich gilt:

- Verwendet wird eine etablierte Referenzimplementierung, üblicherweise `pycma`.
  Der Optimierer wird nicht selbst geschrieben.
- Die Policy ist deterministisch und bildet auf den Action-Space ab; bei
  `Box(-1, 1)` bietet sich ein abschließendes `tanh` an.
- Die Zahl der Policy-Parameter `n` ist der **entscheidende** Wert: CMA-ES führt
  eine `n × n`-Kovarianzmatrix. Bei `n = 70.000` wären das über 40 GB und eine
  Eigenzerlegung in `O(n³)`. Praktikabel ist die Vollmatrix bis in den unteren
  vierstelligen Bereich; darüber nur die Diagonalvariante. Der Prompt nennt die
  Netzgröße und die daraus folgende Parameterzahl (Abschnitt 4.1).
- Ein Schrittbudget wird wie bei allen Verfahren in **Environment-Schritten**
  gezählt, nicht in Generationen. Der Lauf endet, sobald das Budget erschöpft
  ist, notfalls mitten in einer Generation.
- Jede Kandidatenbewertung ist eine Episode und erzeugt einen Kurvenpunkt.
- Die deterministische Evaluation verwendet den **Verteilungsmittelwert** `m`,
  nicht einen gezogenen Kandidaten: Das ist hier das Gegenstück zu „ohne
  Exploration".

UI-Parameter: `total_timesteps`, `seed`, Hidden Layers und Aktivierung der
Policy, `sigma0` (anfängliche Schrittweite), `popsize` λ (`auto` entspricht
`4 + ⌊3·ln n⌋`), Episoden je Kandidat sowie ein Schalter für die
Diagonalvariante. Beobachtungsnormalisierung nach 5.7, Reward-Normalisierung
entfällt.

Prüfregeln: `popsize` ≥ 2 und `sigma0` > 0; das Budget muss mindestens eine
vollständige Generation zulassen, sonst entsteht kein einziger Update-Schritt.

Quellen: [Hansen, CMA-ES Tutorial](https://arxiv.org/abs/1604.00772),
[pycma](https://github.com/CMA-ES/pycma),
[Salimans et al., ES als Alternative zu RL](https://arxiv.org/abs/1703.03864),
[Rajeswaran et al., lineare Policies für MuJoCo](https://arxiv.org/abs/1703.02660)

### 5.6 Exploration

- PPO, TD3 und SAC explorieren aus der Policy selbst oder aus Action Noise.
- Populationsbasierte Verfahren wie CMA-ES explorieren im **Parameterraum**:
  Die Streuung der gezogenen Gewichtsvektoren ist ihre Exploration. Eine
  Action-Noise- oder Entropieeinstellung gibt es dort nicht.
- Projekte mit ausschließlich diesen Verfahren haben **keine**
  ε-greedy-Parameter wie `exploration_fraction` oder `exploration_final_eps`.

### 5.7 Normalisierung

Verlangt ein Profil `normalize: true` oder sind die Observationswerte sehr
unterschiedlich skaliert, erhält jeder Tab eine Gruppe `Normalisierung` mit
`Beobachtungen normalisieren`, `Rewards normalisieren` und den Clip-Werten; bei
SB3-Verfahren umgesetzt mit `VecNormalize`, sonst mit einer gleichwertigen
eigenen laufenden Statistik, für die dieselben Regeln gelten.

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
- Bei rangbasierten Verfahren wie CMA-ES entfällt `Rewards normalisieren`
  ersatzlos: Sie werten nur die Reihenfolge der Renditen aus, jede monotone
  Umskalierung ist wirkungslos. Ein solches Feld wäre eine Attrappe.

### 5.8 Gespeicherter Zustand

- `PPO`: Policy, Value-Netz, Optimizer. Kein Replay Buffer.
- `TD3`: Actor, beide Critics, Target-Netze, Optimizer, Replay Buffer.
- `SAC`: wie TD3 plus gelernter Temperaturparameter.
- `CMA-ES`: der vollständige Zustand der Suchverteilung – Mittelwert,
  Schrittweite und Kovarianz –, dazu der beste bisher gefundene
  Parametervektor. Ein Netz allein genügt nicht: Ohne die Verteilung ließe sich
  das Training nicht fortsetzen.
- Bei aktiver Normalisierung zusätzlich die Statistiken.

### 5.9 Fairness zwischen Verfahrensklassen

Ein Vergleich bei gleichem Schrittbudget ist fair im Sinne des Budgets, nicht
im Sinne gleicher Voraussetzungen. Das ist kein Messfehler, sondern eine
Eigenschaft der Verfahrensklassen; Bedienungsanleitung und README sagen das
ausdrücklich:

- **Off-Policy gegen On-Policy:** SAC und TD3 lernen aus jedem gespeicherten
  Übergang mehrfach, PPO verwirft seine Daten nach jedem Update. Der Vergleich
  fällt systematisch zugunsten der Off-Policy-Verfahren aus – erst recht, wenn
  mehrere Slots sie gegen ein einzelnes On-Policy-Verfahren stellen.
- **Gradientenfrei gegen gradientenbasiert:** Ein Verfahren wie CMA-ES nutzt je
  Episode genau eine Zahl – die Rendite – und verwirft alles, was zwischen den
  Schritten passiert ist. Die gradientenbasierten Verfahren werten jeden
  einzelnen Übergang aus. Bei gleichem Schrittbudget liegt CMA-ES deshalb
  typischerweise deutlich zurück; das sagt nichts über die Qualität des
  Verfahrens, sondern über die Informationsmenge je Schritt. Dafür braucht es
  keine Differenzierbarkeit, keinen Critic und keine Diskontierung.

### 5.10 Verfahrensbezogene Tests

- Konstruktorargumente enthalten nur Schlüssel, die der jeweilige
  SB3-Algorithmus kennt.
- `PPO`: Wirkung von `clip_range`, `gae_lambda`, `n_epochs`; kein Replay
  Buffer; Fortsetzen ohne Rücksetzen des Schrittzählers.
- `TD3`: verzögerte Actor-Updates gemäß `policy_delay`, Clipping des
  Target-Rauschens, Action Noise wirkt im Training und nicht in der Evaluation.
- `SAC`: automatische Entropieanpassung verändert `α`; Zielentropie bei `auto`
  genau `-dim(A)`.
- `CMA-ES`: die Zahl der Policy-Parameter entspricht der Netzarchitektur; eine
  Generation zieht `popsize` Kandidaten und verändert danach Mittelwert und
  Schrittweite; die deterministische Evaluation nutzt den Verteilungsmittelwert
  und nicht einen gezogenen Kandidaten; das Schrittbudget wird in
  Environment-Schritten eingehalten, auch wenn es mitten in einer Generation
  endet; ein Save-/Load-Zyklus setzt das Training mit derselben Verteilung
  fort.
- Save-/Load-Roundtrip je Verfahren mit genau den Bestandteilen aus 5.8.
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
- Untereinanderstehende Wertefelder sind **exakt gleich breit**, unabhängig vom
  Widgettyp. Eine Zeichenbreite allein genügt dafür nicht: Ein Auswahlfeld
  rechnet seinen Aufklapp-Pfeil zusätzlich und bliebe immer einen Tick breiter
  als ein Eingabefeld daneben. Die Breite gibt deshalb das Raster vor – feste
  Mindestbreite und gemeinsame Uniform-Gruppe für die Wertespalten, Felder
  dehnen sich darin
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

Die Anleitung steht in einem **eigenen, scrollbaren Fenster**, nicht in einem
Meldungsdialog. Ein Dialog wächst mit seinem Text, bis die Schaltfläche unter
den Bildschirmrand rutscht – dann lässt er sich nicht mehr schließen. Das
Fenster leitet seine Größe aus dem Inhalt ab, begrenzt sie am Bildschirm,
scrollt und schließt auf **Escape** ebenso wie über seine Schaltfläche. Die
Breite wird an der **längsten Zeile selbst** gemessen; Zeichenzahl mal
Zeichenbreite geht daneben, sobald die Darstellung nicht exakt gleich breit ist.

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
- Über den Parameterspalten steht ein Block mit den globalen Einstellungen und
  darunter die Belegung der Slots: `Verfahren 1` bis `Verfahren 4`, jeweils mit
  ihrem Algorithmus und der Wahl der Animation (7.3) daneben. Mehrere Slots
  dürfen denselben Algorithmus enthalten. Die Reihenfolge folgt der Wirkung:
  Erst die Regeln, die für alle gelten, dann die Belegung, die sie ausfüllt.
- Auswahlfelder werden nicht breiter gemacht als nötig. Ein Dropdown mit den
  Werten `PPO`, `TD3`, `SAC` braucht keine zehn Zeichen.
- Eine Auswahl aus wenigen benachbarten Zahlen ist ein **Zahlenfeld mit
  Pfeilen**, keine Aufklappliste: Die Pfeile führen direkt zum Nachbarwert,
  eine Liste mit drei Einträgen lohnt den Klick nicht.
- Auswahlwerte werden **kurz** gehalten. Ein Wert, der nur in ein breiteres Feld
  passt, zwingt dieses Feld aus der Flucht aller übrigen – die Bezeichnung wird
  dann gekürzt und in Bedienungsanleitung und README erklärt, statt das Layout
  danach zu richten.
- Jede Spalte aus gleichartigen Bedienelementen trägt eine **Überschrift**.
  Ohne sie ist bei einer Spalte aus Auswahlfeldern nicht erkennbar, worauf sich
  ihre Werte beziehen.
- Darunter gleich aufgebaute Tabs `Verfahren 1` bis `Verfahren 4`, jeder mit
  den vollständigen und unabhängigen Parametern seines Algorithmus samt Budget,
  Seed und Netzwerkparametern. Parameter werden nicht geteilt.
- Global bleiben nur Einstellungen, die in allen Läufen identisch sein müssen:
  `Anzahl Verfahren` und die Fensterbreite des gleitenden Durchschnitts.
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

In Spalte 3, in dieser Reihenfolge:

1. `Training starten / fortsetzen`
2. `Vergleich starten / fortsetzen`
3. `Stoppen`
4. `Zurücksetzen`

- 1 und 4 wirken auf das aktive Verfahren, 2 auf alle aktiven Slots.
- Mehr braucht der Ablauf nicht. Jede weitere Schaltfläche kostet Platz in der
  schmalsten Spalte und muss sich rechtfertigen: Die Animation läuft während
  der Läufe ohnehin mit und wird über 7.1 und 7.3 gesteuert, nicht über einen
  eigenen Knopf.
- Buttons zum manuellen Speichern und Laden gibt es nicht.
- Darunter folgen Animationssteuerung (7.1), Fortschrittsanzeige und
  Statuszeile.
- Während eines Laufs gehört die Statuszeile dem **Lauf**: Sie nennt die
  beteiligten Verfahren und ihr Budget. Meldungen der Animation überschreiben
  sie nicht – sie erscheinen nur, wenn gerade nichts läuft. Sonst wäre die
  Information, auf die es ankommt, nach dem ersten Einzelbild verschwunden. Die Fortschrittsanzeige zählt in **Episoden**: Das ist die
  Größe, die der Benutzer vorgibt und im Graphen wiederfindet. Nur wenn die
  Episodengrenze auf unbegrenzt steht, fehlt der Nenner; dann zählt sie
  ersatzweise Schritte.
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

Global, keinem Slot zugeordnet, genügt **ein** Eingabefeld für die Bildrate.

Einen zusätzlichen globalen Schalter „Animation zeigen" gibt es **nicht**,
sobald die Wahl je Anzeige (7.3) einen Zustand `inaktiv` kennt: Alle Anzeigen
darauf zu stellen ist dasselbe, und zwei Bedienelemente für dieselbe Sache
können einander nur widersprechen. Beim Start steht jede Anzeige auf ihrem
Standardwert und ist damit sichtbar – die Animation ist der sichtbare Zweck der
Anwendung und wird abgeschaltet, wenn Rechenzeit wichtiger wird, nicht
umgekehrt.
- `Bildrate (FPS)`: gültig `1` bis `250`. Die Obergrenze liegt bewusst über
  jeder üblichen Environment-Rate – läge sie darunter, wäre ein
  environment-eigener Standardwert selbst ungültig. Eine Änderung wirkt
  spätestens mit der nächsten sichtbaren Episode, auch während eines Laufs.
- Der **Standardwert** ist eine Anzeigeentscheidung, keine Umgebungskonstante.
  Die environment-eigene Rate gibt Echtzeit wieder; das ist nicht immer das
  Ziel. Bei schnellen oder kurzlebigen Environments ist eine deutlich
  niedrigere Rate vorzuziehen: Sie zeigt das Geschehen in Zeitlupe und kostet
  zugleich weniger Rechenzeit, die dem Training zugutekommt. Der Prompt nennt
  den gewählten Wert und begründet ihn, wenn er von der Environment-Rate
  abweicht.

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

- Die **Spaltenzahl folgt der Fläche**, sie ist nicht fest. Gewählt wird die
  Aufteilung, die das größte Bild ergibt. Eine feste Aufteilung verschenkt
  Platz, sobald der Bereich nicht zufällig dasselbe Seitenverhältnis hat wie
  das Raster: Quadratische Frames sind in einem breiten, flachen Bereich durch
  die **Höhe** begrenzt – mehr Breite bringt dann nichts, eine zusätzliche
  Zeile kostet dagegen sofort. Drei Anzeigen nebeneinander sind dort deutlich
  größer als zwei über zwei.
- In die Rechnung geht ein, was je Zelle **neben** dem Bild Platz braucht. Diese
  Höhe fällt je Zeile an und macht Zeilen teurer als Spalten.
- Alle belegten Zellen sind **gleich groß**, Zeilen und Spalten gleich
  gewichtet. Die letzte Anzeige über eine freie Nachbarzelle zu spannen ist
  verlockend, bringt bei quadratischen Frames aber nichts – der Gewinn wäre
  null, der Verlust an Gleichmäßigkeit sichtbar.
- Die Beschriftung unter dem Bild bekommt ihren Platz **vor** dem Bild
  zugeteilt – ein Bild, das den Rest füllt, drückt sie sonst unbemerkt aus
  einer knappen Zelle. Der Layout-Test prüft sie ausdrücklich mit.
- Die Bildgröße leitet sich aus dem verfügbaren Platz und der Zahl der Anzeigen
  ab, nicht aus dem zuletzt gezeigten Bild – sonst behielte ein einmal großes
  Bild seinen Platz.
- Zellen behalten ihre Position, während Episoden beginnen und enden.

### 7.3 Welchen Lernstand eine Anzeige zeigt

Je Slot ein Bedienelement für den gezeigten Lernstand. Es gehört **zur
Verfahrenszeile im Konfigurator**, nicht an das Bild: Dort steht es neben dem
Verfahren, zu dem es gehört, es bleibt auch dann erreichbar, wenn die Anzeige
gerade ausgeblendet ist, und unter dem Bild bleibt ausschließlich die eine
Messwertzeile (7.4). Jede Zelle gewinnt dadurch Höhe fürs Bild. Zur Wahl
stehen vier Möglichkeiten:

- der **laufende Lernstand** (Standard). Die Beschriftung nennt die
  Nummer der zuletzt trainierten bzw. verglichenen Episode – dieselbe Nummer
  wie auf der X-Achse der Graphen. Ein eigener, bei jedem Einschalten wieder
  bei 1 beginnender Animationszähler ist **unzulässig**.
- die **beste Episode**, gemessen am explorativen Return, **exakt
  nachgespielt** statt mit der Policy nachgerechnet – siehe unten.
- der **Lernstand** dieser Episode, deterministisch und mit festem Seed. Er
  beantwortet die andere Frage: nicht „was ist damals passiert", sondern „wie
  gut ist dieser Stand ohne das Glück explorativer Züge". Beide Werte
  nebeneinander sind aussagekräftig – ihr Abstand misst, wie viel des
  Spitzenwerts Zufall war.

  Beide Beschriftungen machen erkennbar, dass nicht der aktuelle Stand läuft.
  Ist eine von beiden gewählt, aber noch keine Episode abgeschlossen, läuft der
  aktuelle Stand weiter und die Statuszeile sagt das.
- `inaktiv`: Diese eine Anzeige entfällt; die übrigen rücken nach und bekommen
  ihren Platz. Bei drei oder vier gleichzeitig sichtbaren Verfahren kostet jede
  laufende Anzeige Rechenzeit und einen eigenen Renderprozess – wer nur eines
  beobachten will, schaltet die übrigen einzeln ab und sieht das verbleibende
  entsprechend größer. Die übrigen Anzeigen laufen unberührt weiter; ein
  Anhalten, das alle Slots trifft, gibt es nicht mehr. Stehen alle auf
  `inaktiv`, bleibt der Anzeigebereich leer.

Gesichert wird dafür je Slot **genau ein** zusätzlicher Lernstand. Ein Verlauf
über alle Episoden scheidet aus: Bei großen Budgets entstehen Tausende
Episoden, deren Policy-Kopien Gigabytes belegten.

**Gesichert wird der Stand vom Episoden*beginn*, nicht vom Episodenende.** Das
ist der springende Punkt und leicht falsch zu machen: Off-Policy-Verfahren
aktualisieren die Policy bei `train_freq = 1` nach **jedem** Schritt. Am Ende
einer Episode ist sie eine andere als die, die diese Episode erzeugt hat –
gemessen rund 2 % Gewichtsänderung je Episode, was den deterministischen Return
spürbar verschiebt. Wer am Episodenende sichert, zeigt unter der Nummer der
besten Episode einen Lernstand, den es damals noch gar nicht gab.

- Praktisch heißt das: zu Beginn jeder Episode in einen **vorab angelegten**
  Puffer kopieren und ihn erst übernehmen, wenn die Episode die beste ist. Je
  Episode neu zu allokieren kostete bei großen Netzen Megabytes.
- Normalisiert das Projekt die Beobachtungen, gehören deren **Statistiken zum
  Snapshot**. Eine alte Policy mit heutigen Statistiken sähe die Beobachtungen
  anders als im Training.
- Auch damit reproduziert die Wiederholung den Trainings-Return **nicht** exakt:
  Die Episode lief explorativ, die Wiederholung läuft deterministisch und von
  einem anderen Startzustand. Bedienungsanleitung und README sagen das.

Der Stand entsteht im Worker-Thread als losgelöste Kopie, die der Optimizer
nicht mehr verändert. Ein neues Modell verwirft ihn mit.

**Die beste Episode wird aufgezeichnet, nicht nachgerechnet.** Die Policy
wiederherzustellen genügt nicht: Die beste Episode ist das Maximum über
Hunderte Episoden und verdankt ihren Wert zum Teil glücklichen
Explorationszügen und ihrem Startzustand. Eine deterministische Wiederholung
derselben Policy erreicht gemessen nur 80 bis 90 % ihres Returns – sie
beantwortet die Frage „wie gut ist diese Policy", nicht „was ist damals
passiert". Wer unter der Nummer der besten Episode einen deutlich kleineren
Return zeigt, hat aus Sicht des Benutzers einen Fehler.

Aufgezeichnet werden deshalb je Episode:

- der **Simulatorzustand am Anfang** – ohne ihn läuft dieselbe Folge von
  Actions in eine andere Episode;
- **jede ausgeführte Action**, und zwar die, die tatsächlich an das Environment
  ging, nicht die aus der Policy nachgerechnete.

Beides wandert beim neuen Bestwert in vorab angelegte Puffer, wie der
Lernstand. Der Platzbedarf ist gering: Ein Budget von 1000 Schritten und einer
zweistelligen Zahl Actionwerte bleibt deutlich unter 100 kB je Episode.

Die Wiedergabe setzt den Zustand und spielt die Actions ab. Sie reproduziert
Bewegung und Return **exakt** – jedes Mal. Dafür greift der Renderprozess auf
`unwrapped` des Environments zu; für **Messwerte** bleibt das unzulässig (3.1),
fürs Nachspielen gibt es keinen anderen Weg, und die Abweichung wird im Prompt
benannt.

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
- Führt das Environment **keinen** `reward_threshold`, gibt es auch keine
  offizielle Gelöst-Schwelle. Der Prompt darf dann eine begründete
  Referenzmarke setzen, muss sie aber als projektintern kennzeichnen und darf
  sie nicht „gelöst" nennen. Erfundene offizielle Schwellen sind unzulässig.

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
- Die Y-Achse folgt den **Daten**, nicht der Referenzlinie. Liegt die Marke weit
  über dem erreichten Bereich – der Normalfall, wenn das Budget unter dem
  Profilwert liegt –, wird sie **nicht** in den sichtbaren Bereich erzwungen:
  Sonst drängt eine einzelne gestrichelte Linie alle Kurven in einen Bruchteil
  der Bildhöhe und macht genau die Unterschiede unlesbar, um die es geht. Die
  Marke erscheint dann nicht im Bild, sondern als Wert in Legende und Summary,
  zusammen mit dem Abstand zum besten erreichten Return.
- Rohkurven nutzen dieselbe Slotfarbe mit deutlich verringerter Deckkraft und
  Strichstärke; Rohwerte und geglättete Werte bleiben unterscheidbar.
- **Jede** gezeichnete Kurve ist auf eine feste Punktzahl gedeckelt – die
  geglättete ebenso wie die rohe. Ohne diese Grenze wächst die Zeichenzeit
  linear mit der Episodenzahl; bei mehreren Slots und zehntausenden Episoden
  wird die Oberfläche unbenutzbar. Die Messdaten bleiben vollständig, nur die
  Darstellung wird verdichtet.
  - Für **Rohkurven** ist Min-/Max-Verdichtung einfachem Auslassen vorzuziehen:
    Sie erhält Ausreißer, auf die es gerade ankommt.
  - Für die **geglättete** Kurve gilt das Gegenteil: Min/Max machte eine
    bewusst glatte Linie wieder zackig. Hier wird gleichmäßig ausgedünnt, der
    letzte Punkt bleibt erhalten.
- Plot-Updates werden gedrosselt.
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

### 8.4 X-Achse und abgeschlossene Episoden

- Trainings- und Vergleichskurven führen einheitlich **Episoden** auf der
  X-Achse. Budget und tatsächlich ausgeführte Schritte bleiben separat in
  Status und Summary sichtbar.
- Punkte entstehen nur für vollständig abgeschlossene Episoden. Endet ein
  Budget innerhalb einer Episode, liegt der letzte Punkt vor dem ausgeführten
  Schrittbudget; GUI und Summary zeigen ausgeführte Schritte, angefordertes
  Budget und diese Bedeutung getrennt und verständlich.
- Eine automatische **Zwischenevaluation** ist nicht erforderlich. Sie kostet
  Rechenzeit, erzeugt eine zweite Stützstellenreihe neben der eigentlichen
  Lernkurve und beantwortet dieselbe Frage wie ein Mittelwert über die letzten
  Episoden (8.6) – nur teurer. Wo eine explorationsfreie Messung gebraucht
  wird, bleibt sie als Funktion des Logikmoduls verfügbar, ohne Bedienelement.
- Damit entfallen auch die daran gekoppelten Platten-Checkpoints und ein
  Button zum Wiederherstellen: Ohne Evaluation gibt es keinen Auslöser, der
  sagen könnte, welcher Stand der beste ist.
- Was **bleibt**, ist der Lernstand der besten Episode je Slot im Speicher
  (7.3). Er hängt am explorativen Episoden-Return, nicht an einer Evaluation,
  und wird für die Animation gebraucht.

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

### 8.6 Einzelgraph je Verfahren

Neben dem gemeinsamen Vergleichsgraphen (8.5) lässt sich **jeder aktive Slot
einzeln** als eigener Reward-Plot darstellen und exportieren.

- Der Einzelgraph zeigt genau einen Slot – Rohkurve, gleitenden Durchschnitt
  und Referenzlinie – in der Slotfarbe aus 6.2, mit derselben X-Achse, Metrik
  und Glättung wie der Vergleichsgraph. Nur so sind Einzel- und Vergleichsbild
  nebeneinander lesbar.
- Er ist auch **nach** einem Vergleichslauf verfügbar, ohne dass ein Slot neu
  trainiert werden muss: Die Messdaten liegen bereits vor, es ist eine Frage
  der Darstellung.
- Die Legende benennt Slot, Algorithmus und – bei mehrfach belegtem Algorithmus
  – den abweichenden Parameter, wie in 8.5.
- Der Export schreibt wahlweise den sichtbaren Einzelgraphen oder in **einer**
  Aktion je aktivem Slot eine eigene PNG-Datei mit sprechendem, den Slot und
  den Algorithmus benennenden Namen.

Begründung: Ein Vergleichsgraph beantwortet die Frage „welches Verfahren ist
besser", ein Einzelgraph die Frage „wie verlief dieses eine Training". Berichte
brauchen beides, und eine Parameterstudie über drei Ausprägungen verlangt drei
**getrennte** Plots, nicht drei Kurven in einem Bild.

### 8.7 Summary

- Sie wird live aktualisiert und zeigt für Training und Vergleich konsistent
  Episoden, ausgeführte Environment-Schritte, beide Budgetgrenzen samt der
  Angabe, welche den Lauf beendet hat, sowie Rewards und Erfolgsraten.
- **Mittelwerte umfassen die letzten Episoden, nicht den ganzen Lauf.** Das
  Fenster ist dasselbe wie die Glättung des Graphen (8.3): Was die Kurve zeigt
  und was die Summary mittelt, soll dieselbe Aussage sein. Über den gesamten
  Lauf gemittelt hinge jede Kennzahl noch am untrainierten Anfang und bewegte
  sich kaum – gerade der Fortschritt, den man sehen will, verschwände im
  Mittel. Eine **hervorgehobene** Zwischenüberschrift nennt die tatsächlich
  einbezogene Zahl; liegen weniger Episoden vor, wird über alle vorhandenen
  gemittelt.
- Die Zeilen unterhalb dieser Überschrift tragen **kein** Mittelwertzeichen
  mehr: Die Überschrift sagt es bereits für alle. Es an jeder Zeile zu
  wiederholen kostet nur Breite in einer ohnehin schmalen Spalte.
- Die Fußzeile bleibt so kurz wie möglich – im Zweifel eine Zeile. Erklärungen
  gehören in die Bedienungsanleitung; unter der Tabelle zwingen sie zum
  Scrollen und verdecken genau das, worum es geht.
- Kennzahlen, die sich auf den **gesamten** Lauf beziehen – etwa die beste
  Episode – stehen oberhalb dieser Überschrift. „Beste der letzten zwanzig"
  wäre keine sinnvolle Größe.
- Die Vergleichs-Summary besitzt genau eine Ergebnisspalte je aktivem Slot,
  `Verfahren 1` bis `Verfahren 4`, mit dem Algorithmusnamen in der Kopfzeile.
- Belegt ein Algorithmus **mehr als einen** aktiven Slot, folgt ein Abschnitt
  mit genau den Parametern, in denen sich die Konfigurationen unterscheiden –
  je Parameter eine Zeile mit dem Wert aller Slots. Ohne ihn wäre eine
  Parameterstudie nicht interpretierbar: Die Kopfzeile nennt dann dreimal
  denselben Algorithmus und sagt nicht, welche Spalte welchen Wert hatte. Die
  Überschrift des Abschnitts nennt den Grund seines Erscheinens.
- Tragen alle Slots **verschiedene** Algorithmen, entfällt der Abschnitt. Die
  Kopfzeile erklärt dort bereits alles, und der Block schöbe die Tabelle nur
  aus dem sichtbaren Bereich. Die Summary soll ohne Scrollen lesbar sein.
- Unterscheiden sich die Trainingsbudgets, weist die GUI vor dem Start sichtbar
  darauf hin; unzulässig ist es nicht.
- Bei drei oder vier Spalten bleibt sie vollständig lesbar: notfalls mit
  horizontaler Scrollbar, statt Werte abzuschneiden.

### 8.8 Export

- Diagramm und Summary werden über klar bezeichnete Aktionen exportiert: der
  Graph mindestens als PNG in der dargestellten Form, die Summary als gut
  lesbare UTF-8-Textdatei. CSV ist nicht erforderlich. Zum Export einzelner
  Verfahren siehe 8.6 (Einzelgraph je Verfahren).
- Dateidialoge schlagen aussagekräftige Namen vor und überschreiben bestehende
  Dateien nicht unbemerkt.
- Die Schaltflächen bekommen **keine** eigene Kopfzeile: Sie liegen kompakt in
  einer ohnehin vorhandenen Leiste – etwa der Tableiste des Diagramms –, damit
  die übrige Höhe der Darstellung gehört, und verdecken weder Kurven noch
  Legende noch Text.

### 8.9 Tabellen und Modelldateien

- Tabellen zeigen den vollständigen aktuellen Lernstand, unterscheiden besuchte
  und unbesuchte Zustände und sind scrollbar.
- Gespeicherte Modelle enthalten Lernzustand, Format-Version und Metadaten:
  Algorithmus, vollständige Slot-Konfiguration, Environment-Kennung samt aller
  abweichenden Konstruktorargumente.
- Methode und Environment werden beim Laden auf Kompatibilität geprüft.
  Fehlerhafte oder inkompatible Dateien verändern den aktiven Zustand nicht.
- Gespeichert wird genau der Zustand, den das Verfahren besitzt (5.8). Ein
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
- Budgetlogik: Ein Lauf endet an der zuerst erreichten der beiden Grenzen aus
  5.1; bei `Episoden = 0` greift allein das Schrittbudget; Summary und Status
  weisen aus, welche Grenze gegriffen hat
- Fortsetzen hängt an: Ein zweiter Start liefert erneut das **volle** Budget an
  Episoden und Schritten, nicht eine einzelne weitere Episode und keinen Reset
- Summary-Mittelwerte umfassen genau die letzten Episoden des Glättungsfensters
  und ändern sich mit ihm; die beste Episode bezieht sich weiter auf den ganzen
  Lauf
- Der Unterschiedsblock erscheint bei mehrfach belegtem Algorithmus und
  entfällt, wenn alle Slots verschiedene Algorithmen tragen
- Jede gezeichnete Kurve – auch die geglättete – bleibt unter der Punktgrenze
  aus 8.2, auch bei zehntausenden Episoden
- Jede Anzeige startet sichtbar; die Wahl `inaktiv` blendet genau eine aus
  und lässt die übrigen laufen. Einen globalen Schalter gibt es nicht
- Einzelgraph je Verfahren (8.6): nach einem Vergleichslauf ist jeder aktive
  Slot einzeln darstellbar, ohne erneutes Training; der Sammelexport erzeugt je
  aktivem Slot genau eine PNG-Datei; Einzel- und Vergleichsgraph nutzen
  dieselbe Glättung
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

Unterliegt ein Projekt zusätzlich externen Vorgaben – etwa einer
Aufgabenstellung –, führt sein Prompt eine Tabelle, die **jede einzelne**
Vorgabe auf die Stelle abbildet, an der sie erfüllt wird. Abgenommen wird gegen
diese Tabelle, nicht gegen die Erinnerung an die Vorgaben.

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
