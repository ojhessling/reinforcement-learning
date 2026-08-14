# LunarLander Rainbow-DDQN Workbench

Lokale deutschsprachige PyTorch-/Tkinter-Anwendung für Gymnasiums
`LunarLander-v3`. Sie trainiert und vergleicht sieben Varianten: Double DQN,
fünf einzelne Rainbow-Bausteine (Noisy, Prioritized Experience Replay,
Dueling, Multi-Step-Returns, C51/distributional RL) und Rainbow DDQN, das alle
fünf gleichzeitig in einem Agenten kombiniert.

## Installation und Start

`LunarLander-v3` benötigt **Box2D** – die einzige neue Abhängigkeit gegenüber
den Vorgängerprojekten. `swig` muss vor `box2d` installiert sein, weil das
Box2D-Paket es zum Bauen benötigt:

```bash
conda env update -f ../environment.yml --prune
conda activate rl-26-08
pip install swig && pip install "gymnasium[box2d]"   # falls noch nicht vorhanden
python lunarlander_app.py
```

Ohne Box2D startet das Projekt nicht; Gymnasium meldet dann
`DependencyNotInstalled`. Direkte Abhängigkeiten stehen zusätzlich in
`requirements.txt`. Es wird nur PyTorch verwendet, nicht TensorFlow oder Keras.

## Environment und Reward

Alle Trainings- und Darstellungsinstanzen entstehen mit Gymnasium:

```python
gymnasium.make("LunarLander-v3", render_mode="rgb_array")
```

Alle optionalen Konstruktorargumente bleiben auf ihren Standardwerten
(`continuous=False`, `gravity=-10.0`, `enable_wind=False`, `wind_power=15.0`,
`turbulence_power=1.5`). Die Observation ist achtdimensional: Position `x`/`y`
relativ zur Landeplattform, Geschwindigkeit `vₓ`/`v_y`, Winkel, Winkel-
geschwindigkeit und zwei Bodenkontakt-Flags für die Beine. Die vier Actions
sind „nichts tun“, „linkes Steuertriebwerk“, „Haupttriebwerk“ und „rechtes
Steuertriebwerk“.

Der Reward belohnt Annäherung an die Plattform, geringe Geschwindigkeit,
geringe Schräglage und Beinkontakt und bestraft Treibstoffverbrauch (`-0,3` je
Frame Haupttriebwerk, `-0,03` je Frame Steuertriebwerk). Die Episode endet mit
`+100` bei sicherer Landung und `-100` bei Absturz oder Verlassen des
sichtbaren Bereichs; nach 1000 Schritten wird trunkiert.

Anders als in den rein negativ belohnten Vorgängerprojekten ist der Return hier
**zweiseitig**: höhere Werte sind besser, realistisch etwa zwischen `-400` und
`+320`. Die Referenzlinie bei `+200` ist Gymnasiums offizieller
`reward_threshold`. Die Anwendung weist zwei Quoten getrennt aus, weil eine
sichere Landung mit hohem Treibstoffverbrauch die Lösungsschwelle verfehlen
kann:

- **Landequote**: Anteil Episoden mit sicherer Landung (`terminated`, kein
  Absturz, kein Zeitlimit)
- **Gelöst-Quote**: Anteil Episoden mit Return `≥ 200`

Zusätzlich wird die Absturzquote gezeigt. Die Winkelgeschwindigkeit der
Observation ist in Einheiten von `0,4 rad/s` angegeben; die Anzeige rechnet sie
mit Faktor `2,5` in `rad/s` um, den Winkel in Grad.

Quelle: [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

## Algorithmen

- **DDQN:** Das Online-Netz wählt die nächste Action, das Target-Netz bewertet
  genau diese Action. Das reduziert die für DQN typische Überschätzung.
- **Noisy DDQN:** Faktorisierte gaußsche NoisyLinear-Layer erzeugen lernbare
  Exploration. Zusätzliches Epsilon-Greedy ist deaktiviert; bei der
  deterministischen Evaluation wird das Rauschen ausgeschaltet.
- **PER DDQN:** Übergänge werden proportional zum absoluten TD-Fehler (bei
  C51-basierten Varianten zum Kreuzentropie-Verlust) priorisiert.
  Importance-Sampling-Gewichte korrigieren den Bias; `β` steigt bis `1`.
- **Dueling DDQN:** Getrennte Value- und Advantage-Streams werden als
  `Q(s,a) = V(s) + A(s,a) - mean(A(s,·))` zusammengeführt.
- **Multi-Step DDQN:** n-Schritt-Returns über `n_step` Übergänge; endet eine
  Episode im Fenster, verkürzt sich der Return, ohne über das Episodenende
  hinaus zu akkumulieren.
- **C51 DDQN:** Statt eines skalaren Q-Werts eine kategoriale Verteilung über
  `n_atoms` Atome in `[V_min, V_max]`, trainiert per Kreuzentropie gegen die
  projizierte Bellman-Zielverteilung. Die Double-DQN-Actionauswahl nutzt
  weiterhin den Erwartungswert der Online-Verteilung.
- **Rainbow DDQN:** Alle fünf Bausteine gleichzeitig.

Alle sieben Varianten teilen dieselbe DDQN-Zielberechnung und dasselbe
`LunarLanderDQNPolicy`/`RainbowCapableQNetwork`-Paar; Noisy, Dueling und C51
sind dort unabhängige Flags. Multi-Step und PER sind kombinierbare
Replay-Buffer-Wrapper (`NStepReplayBuffer` umschließt bei Bedarf einen
`PrioritizedReplayBuffer`, der den rohen SB3-Buffer umschließt). Rainbow DDQN
dupliziert damit keinen Baustein.

Quellen: [Double DQN](https://arxiv.org/abs/1509.06461),
[NoisyNet](https://arxiv.org/abs/1706.10295),
[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952),
[Dueling Networks](https://arxiv.org/abs/1511.06581),
[Multi-Step-Returns (Sutton)](https://link.springer.com/article/10.1007/BF00115009),
[C51 (Bellemare et al.)](https://arxiv.org/abs/1707.06887).

## Standardprofil und Parameter

Die gemeinsamen Ausgangswerte stammen aus dem getunten DQN-Profil des RL
Baselines3 Zoo für `LunarLander-v3`: `100.000` Schritte, Lernrate `6,3e-4`,
Buffer `50.000`, Lernstart `0`, Batch `128`, `γ=0,99`, Training alle 4
Schritte, Target-Update alle 250 Schritte, Epsilon-Abklinganteil `0,12`,
finales Epsilon `0,1`, Hidden Layers `256,256`. Der Zoo-Wert
`gradient_steps=-1` bedeutet „ein Gradientenschritt je gesammeltem
Environment-Schritt“; bei `train_freq=4` entspricht das konkret `4`
Gradientenschritten je Update, was als Standardwert übernommen wurde.

Nicht im Zoo-Profil festgelegte Werte folgen Stable-Baselines3 2.9.0 oder den
Originalarbeiten: Noisy `σ₀=0,5`, PER `α=0,6`, `β₀=0,4`, `ε=1e-6`,
Dueling-Streams mit je 64 Neuronen, Multi-Step `n=3` und C51 `n_atoms=51`.
`V_min=-400` / `V_max=400` decken den realistisch erreichbaren, zweiseitigen
Returnbereich symmetrisch ab; sie sind bewusst keine harten
Environment-Grenzen, sondern eine begründete Kurswahl. Sämtliche Netzwerk- und
Algorithmusparameter sind in der UI änderbar; unpassende Variantenfelder sind
deaktiviert.

Quelle: [RL Baselines3 Zoo DQN-Hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml).

## Bedienung und Ansichten

Oben stehen Parameter und Steuerung neben dem offiziellen RGB-Frame, unten
Diagramme und Live-Summary nebeneinander. Der Diagrammbereich hat zwei Tabs:
`Training` zeigt Episodenwerte, den gleitenden Mittelwert der letzten 20
Episoden und – optisch klar getrennt als gestrichelte Linie mit Markern – die
deterministischen Evaluationen; `Vergleich` stellt die ausgewählten Algorithmen
gegenüber. Beide enthalten die Referenzlinie bei `+200`.

Während Training und Vergleich wird im einstellbaren Schrittintervall
automatisch in einem separaten headless Environment deterministisch evaluiert.
Das erzeugt keine Animation und verändert weder Modell noch Replay Buffer.
`Intervall 0` schaltet die automatische Evaluation ab. Verbessert sich der
mittlere Evaluations-Return, werden Modell, Target-Netz, Optimizer und Replay
Buffer gemeinsam als Checkpoint gesichert; `Bestes Modell wiederherstellen`
lädt diesen vollständigen Lernzustand.

Die Animation läuft mit der environment-eigenen Bildrate von 50 FPS (20 ms je
Frame). Gemäß Workbench gibt es dafür **kein** Eingabefeld – die Animation ist
lediglich über `Animation zeigen` abschaltbar; ist sie aus, läuft die sichtbare
Episode ohne Einzelbilder durch und zeigt nur das Endbild.

Zur flüssigen Anzeige wird höchstens alle zwei Sekunden neu gezeichnet und die
Rohkurve ab 2.000 Punkten per Min-/Max-Verdichtung dargestellt; intern bleiben
alle Messwerte erhalten. Die scrollbare Summary zeigt Episoden, ausgeführte
Environment-Schritte, mittleren Return sowie Lande-, Gelöst- und Absturzquote.

Über `Diagramm exportieren (PNG)` und `Summary exportieren (TXT)` lässt sich
der aktuelle Stand sichern. Beide Dialoge öffnen `10-lunarlander/exports/`
(wird automatisch angelegt und ist per `.gitignore` von Commits
ausgeschlossen). Exportiert wird das gerade sichtbare Tab; der Textexport
enthält zusätzlich die vollständige Konfiguration. Beide Vorschlagsnamen teilen
denselben Stamm aus Algorithmus/Vergleich und Zeitstempel, solange sich
zwischen den Exporten keine Episode geändert hat – Diagramm und
Konfigurationsdatei bleiben so eindeutig einander zuordenbar.

Auf macOS läuft der offizielle Gymnasium-/Pygame-Renderer in einem separaten
unsichtbaren Prozess, damit Tkinter und SDL sich keinen Prozess teilen.

## Speichern und Laden

Ein Speicherstand besteht aus drei zusammengehörenden Dateien:

- `<name>.zip`: SB3-Modell mit Online-/Target-Netz und Optimizer
- `<name>_replay.pkl`: Replay Buffer inklusive PER-Prioritäten; der
  unvollständige Multi-Step-Fensterzustand gehört nicht dazu und beginnt nach
  dem Laden leer
- `<name>_metadata.json`: Variante und vollständige Konfiguration

PER- und Multi-Step-Baustein hüllen den SB3-Replay-Buffer in eigene
Wrapper-Klassen, die keine `ReplayBuffer`-Unterklassen sind. Speichern und
Laden verwenden deshalb bewusst die generischen SB3-Pickle-Helfer
(`save_to_pkl`/`load_from_pkl`) statt `model.save_replay_buffer()`, dessen
interne `isinstance`-Prüfung diese Wrapper ablehnen würde. Unvollständige oder
zu einem anderen Environment gehörende Dateisätze werden mit einer
verständlichen Meldung abgelehnt.

## Tests und Grenzen

```bash
python -m unittest discover -s tests -v
```

Die Tests prüfen Environment und unveränderte Standardargumente,
Double-Zielwert, NoisyNet, PER, Dueling-Netz, n-Schritt-Returns (volle und an
Episodengrenzen verkürzte Fenster, Kombination mit PER), die kategoriale
C51-Projektion (Massenerhaltung, exakter Atomtreffer, terminale Übergänge), die
Unterscheidung von Landung, Absturz und Zeitlimit, alle sieben kurzen
Trainingsläufe, unveränderte Evaluation, die automatische Evaluation im
konfigurierten Intervall, Save/Load/Fortsetzen je Variante, Ablehnung
inkompatibler Checkpoints, Renderer-Prozesstrennung, Export von Diagramm und
Summary sowie das Startlayout auf beiden Diagramm-Tabs. Zusätzlich prüft ein
Test explizit, dass es **kein** Eingabefeld für die Animationsgeschwindigkeit
gibt.

Kurze Läufe lösen LunarLander nicht zuverlässig; für belastbare Ergebnisse ist
das volle Budget nötig. Paralleltraining mehrerer Varianten beansprucht
entsprechend mehr CPU. Der Kursstand unterstützt nur die diskrete Variante und
die hier aufgeführten sieben Algorithmen.
