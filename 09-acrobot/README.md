# Acrobot Rainbow-DDQN Workbench

Lokale deutschsprachige PyTorch-/Tkinter-Anwendung für Gymnasiums
`Acrobot-v1`. Sie trainiert und vergleicht sieben Varianten: Double DQN sowie
fünf Einzelbausteine (Noisy, Prioritized Experience Replay, Dueling,
Multi-Step-Returns, C51/distributional RL) und Rainbow DDQN, das alle fünf
Bausteine gleichzeitig in einem Agenten kombiniert.

## Installation und Start

Das Projekt verwendet die gemeinsame Kursumgebung:

```bash
conda env update -f ../environment.yml --prune
conda activate rl-26-08
python acrobot_app.py
```

Direkte Abhängigkeiten stehen zusätzlich in `requirements.txt`. Es wird nur
PyTorch verwendet, nicht TensorFlow oder Keras.

## Environment und Reward

Alle Trainings- und Darstellungsinstanzen entstehen mit Gymnasium:

```python
gymnasium.make("Acrobot-v1", render_mode="rgb_array")
```

Die Observation enthält `cos`/`sin` beider Gelenkwinkel sowie beide
Winkelgeschwindigkeiten. Die drei Actions wenden ein Drehmoment von `-1`, `0`
oder `+1` Nm am Gelenk zwischen den beiden Armsegmenten an. Jeder Schritt
kostet `-1` Reward, Zielerreichung `0`. Die Episode endet, sobald das freie
Ende der Kette die Zielhöhe überschreitet (`-cos θ₁ - cos(θ₂+θ₁) > 1,0`), oder
wird nach 500 Schritten trunkiert. Deshalb sind weniger negative Rewards
besser; der Episoden-Return liegt stets im Bereich `[-500; 0]`. Die gelbe
Referenzlinie bei `-100` ist Gymnasiums offizieller `reward_threshold` für
dieses Environment, kein theoretisches Optimum und keine Änderung des
Environments. Reward Shaping wird nicht verwendet.

Quelle: [Gymnasium Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/)

## Algorithmen

- **DDQN:** Das Online-Netz wählt die nächste Action, das Target-Netz bewertet
  genau diese Action. Dadurch wird die für DQN typische Überschätzung reduziert.
- **Noisy DDQN:** Factorisierte gaußsche NoisyLinear-Layer erzeugen lernbare
  Exploration. Zusätzliches Epsilon-Greedy ist deaktiviert; bei der
  deterministischen Evaluation wird das Rauschen ausgeschaltet.
- **PER DDQN:** Übergänge werden proportional zu ihrem absoluten TD-Fehler
  (bei C51-basierten Varianten: proportional zum Kreuzentropie-Verlust)
  priorisiert. Importance-Sampling-Gewichte korrigieren den Sampling-Bias;
  `β` steigt während des Trainings bis `1`.
- **Dueling DDQN:** Getrennte Value- und Advantage-Streams werden als
  `Q(s,a) = V(s) + A(s,a) - mean(A(s,·))` zusammengeführt.
- **Multi-Step DDQN:** Statt des 1-Schritt-TD-Fehlers wird ein n-Schritt-Return
  über `n_step` aufeinanderfolgende Übergänge akkumuliert; endet eine Episode
  innerhalb des Fensters, verkürzt sich der Return entsprechend, ohne über das
  Episodenende hinweg zu akkumulieren.
- **C51 DDQN:** Der skalare Q-Wert wird durch eine kategoriale Verteilung über
  `n_atoms` feste Rückgabewerte in `[V_min, V_max]` ersetzt. Die Zielverteilung
  entsteht durch Bellman-Update der Atome und anschließende kategoriale
  Projektion (Bellemare et al.) auf das feste Atom-Gitter; trainiert wird mit
  Kreuzentropie. Die Double-DQN-Actionauswahl verwendet weiterhin den aus der
  Online-Verteilung gebildeten Erwartungswert.
- **Rainbow DDQN:** Kombiniert alle fünf Bausteine oben gleichzeitig in einem
  Agenten – ein einziges `RainbowCapableQNetwork` mit `noisy=True,
  dueling=True, distributional=True`, ein `NStepReplayBuffer`, der einen
  `PrioritizedReplayBuffer` umschließt.

Alle sieben Varianten teilen sich dieselbe DDQN-Zielwertberechnung und
dasselbe `AcrobotDQNPolicy`/`RainbowCapableQNetwork`-Paar; Noisy-, Dueling- und
C51-Baustein sind dort unabhängig zu- oder abschaltbare Flags. Multi-Step und
PER sind unabhängig kombinierbare Replay-Buffer-Wrapper
(`NStepReplayBuffer` umschließt bei Bedarf einen `PrioritizedReplayBuffer`,
der wiederum den rohen SB3-Buffer umschließt) – Rainbow DDQN dupliziert damit
keinen der fünf Bausteine.

Quellen: [Double DQN](https://arxiv.org/abs/1509.06461),
[NoisyNet](https://arxiv.org/abs/1706.10295),
[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952),
[Dueling Networks](https://arxiv.org/abs/1511.06581),
[Multi-Step-Returns (Sutton)](https://link.springer.com/article/10.1007/BF00115009),
[C51 / Distributional RL (Bellemare et al.)](https://arxiv.org/abs/1707.06887).

## Standardprofil und Parameter

Die gemeinsamen Ausgangswerte stammen aus dem DQN-Profil des RL Baselines3
Zoo für `Acrobot-v1`: `100.000` Schritte, Lernrate `6,3e-4`, Buffer `50.000`,
Lernstart `0`, Batch `128`, `γ=0,99`, Training alle 4 Schritte, Target-Update
alle 250 Schritte, Epsilon-Abklinganteil `0,12`, finales Epsilon `0,1` und
Hidden Layers `256,256`. Der Zoo-Wert `gradient_steps=-1` bedeutet "ein
Gradientenschritt je im Rollout gesammeltem Environment-Schritt"; bei
`train_freq=4` entspricht das konkret `4` Gradientenschritten je Update, was
als Standardwert übernommen wurde.

Nicht im Zoo-Profil festgelegte gemeinsame Werte folgen Stable-Baselines3
2.9.0. Variantenwerte folgen den Originalarbeiten beziehungsweise verbreiteten
Ausgangswerten: Noisy `σ₀=0,5`, PER `α=0,6`, `β₀=0,4`, `ε=1e-6`, Dueling-
Streams mit je 64 Neuronen, Multi-Step `n=3` (Rainbow-Paper-Standardwert) und
C51 `n_atoms=51` mit `V_min=-500`, `V_max=0` (dem tatsächlichen Wertebereich
des Acrobot-Episoden-Returns). Rainbow DDQN verwendet exakt dieselben
Variantenwerte wie die fünf Einzelvarianten. Sämtliche verwendeten Netzwerk-
und Algorithmusparameter sind in der UI änderbar; unpassende Variantenfelder
sind deaktiviert.

Quelle: [RL Baselines3 Zoo DQN-Hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml).

## Bedienung und Ansichten

Oben stehen Parameter und Steuerung neben dem offiziellen RGB-Frame. Unten
bleiben Diagramm und Live-Summary gleichzeitig sichtbar. Einzeltraining kann
fortgesetzt werden. Im Vergleich starten die ausgewählten Varianten parallel;
ein weiterer kompatibler Vergleich hängt an die vorhandenen Läufe an.

Der Graph zeigt transparente Episodenwerte und den gleitenden Mittelwert der
letzten 20 Episoden. Zur flüssigen Anzeige wird höchstens alle zwei Sekunden
neu gezeichnet und die Rohkurve ab 2.000 Punkten per Min-/Max-Verdichtung
dargestellt; intern bleiben alle Messwerte erhalten. Die kompakte Summary zeigt
Episoden, Environment-Schritte, mittleren Reward und Erfolgsrate. Alle sieben
Varianten bleiben gleichzeitig lesbar.

Über die Buttons `Diagramm exportieren (PNG)` und `Summary exportieren (TXT)`
oberhalb von Graph und Summary lässt sich der jeweils aktuell dargestellte
Stand sichern. Beide Dateidialoge öffnen standardmäßig `09-acrobot/exports/`
(wird bei Bedarf automatisch angelegt und ist wie andere Laufzeitergebnisse
per `.gitignore` von Commits ausgeschlossen); ein anderer Speicherort lässt
sich im Dialog frei wählen. Der PNG-Export speichert das Diagramm exakt in der
aktuell gezeigten Form; der Textexport enthält zusätzlich zur Summary-Tabelle
die vollständige aktuelle Konfiguration (Algorithmus beziehungsweise
verglichene Algorithmen und alle Hyperparameter). Beide Vorschlagsdateinamen
teilen sich denselben Stamm aus Algorithmus/Vergleich und Zeitstempel
(`acrobot_<algorithmus>_<zeitstempel>.png` und `..._config.txt`), solange sich
zwischen den beiden Exporten keine weitere Episode geändert hat – so bleiben
Diagramm und zugehörige Konfigurationsdatei eindeutig einander zuordenbar.
Bestehende Dateien werden nicht unbemerkt überschrieben.

Training läuft ohne Rendering und ohne automatische Zwischen-Evaluationen.
Eine manuell gestartete Evaluation nutzt ein separates headless Environment,
lernt nicht und verändert weder Parameter noch Replay Buffer. Sie meldet
zusätzlich mittlere und beste Schrittzahl bis zur Zielerreichung; ohne
erfolgreiche Episode bleibt dieser Wert `—` statt einer künstlichen Null. Bei
einem besseren mittleren Evaluations-Reward werden Modell, Target-Netz,
Optimizer und Replay Buffer gemeinsam temporär gesichert. Der Button
`Bestes Modell wiederherstellen` lädt diesen vollständigen Lernzustand.

Auf macOS läuft der offizielle Gymnasium-/Pygame-Renderer in einem separaten
unsichtbaren Prozess. So teilen sich Tkinter und SDL keinen Prozess und es
entsteht weder ein nativer GIL-Absturz noch ein zweiter Dock-Eintrag.

## Speichern und Laden

Ein Speicherstand besteht aus drei zusammengehörenden Dateien:

- `<name>.zip`: SB3-Modell einschließlich Online-/Target-Netz und Optimizer
- `<name>_replay.pkl`: Replay Buffer einschließlich PER-Prioritäten (PER- und
  Rainbow-basierte Varianten); der unvollständige Multi-Step-Fensterzustand
  gehört nicht dazu, das Fenster beginnt nach dem Laden leer
- `<name>_metadata.json`: Variante und vollständige Konfiguration

PER- und Multi-Step-Baustein hüllen den SB3-Replay-Buffer in eigene
Wrapper-Klassen (`PrioritizedReplayBuffer`, `NStepReplayBuffer`), die keine
`ReplayBuffer`-Unterklassen sind. Speichern/Laden verwendet deshalb bewusst
die generischen SB3-Pickle-Helfer (`save_to_pkl`/`load_from_pkl`) anstelle von
`model.save_replay_buffer()`/`load_replay_buffer()`, deren interne
`isinstance`-Prüfung diese Wrapper sonst ablehnen würde. Nur ein vollständiger
kompatibler Dateisatz kann geladen und weitertrainiert werden.

## Tests und Grenzen

```bash
python -m unittest discover -s tests -v
```

Die Tests prüfen Environment, Double-Zielwert, NoisyNet, PER, Dueling-Netz,
die n-Schritt-Return-Bildung des Multi-Step-Bausteins (vollständige und durch
Episodenende verkürzte Fenster, Kombination mit PER), die kategoriale
C51-Projektion (Massenerhaltung, exakter Atomtreffer ohne Wahrscheinlichkeits-
verlust, Double-DQN-Actionauswahl über den Erwartungswert), die gleichzeitige
Aktivierung aller fünf Bausteine bei Rainbow DDQN, alle sieben kurzen
Trainingsläufe, unveränderte Evaluation, Save/Load/Fortsetzen für jede
Variante, Renderer-Prozesstrennung, den PNG-/TXT-Export von Diagramm und
Summary sowie das sichtbare Startlayout.

Kurze Läufe lösen Acrobot nicht zuverlässig. Paralleltraining mehrerer
Varianten beansprucht entsprechend mehr CPU. Der Kursstand unterstützt nur
diskrete Actions und die hier aufgeführten sieben Varianten.
