# MountainCar DDQN Workbench

Lokale deutschsprachige PyTorch-/Tkinter-Anwendung für Gymnasiums
`MountainCar-v0`. Sie trainiert und vergleicht vier getrennte Varianten:
Double DQN, Noisy Double DQN, Double DQN mit Prioritized Experience Replay
und Dueling Double DQN. Die Erweiterungen sind bewusst nicht zu einem
Rainbow-Agenten kombiniert.

## Installation und Start

Das Projekt verwendet die gemeinsame Kursumgebung:

```bash
conda env update -f ../environment.yml --prune
conda activate rl-26-08
python mountaincar_app.py
```

Direkte Abhängigkeiten stehen zusätzlich in `requirements.txt`. Es wird nur
PyTorch verwendet, nicht TensorFlow oder Keras.

## Environment und Reward

Alle Trainings- und Darstellungsinstanzen entstehen mit Gymnasium:

```python
gymnasium.make("MountainCar-v0", render_mode="rgb_array")
```

Die Observation enthält Position und Geschwindigkeit. Die drei Actions
beschleunigen nach links, gar nicht oder nach rechts. Jeder Schritt kostet
`-1` Reward. Die Episode endet beim Erreichen der Position `0,5` oder wird nach
200 Schritten trunkiert. Deshalb sind weniger negative Rewards besser. Die
gelbe Linie bei `-110` ist eine verbreitete Gymnasium-Referenzschwelle, kein
theoretisches Optimum und keine Änderung des Environments. Reward Shaping wird
nicht verwendet.

Quelle: [Gymnasium MountainCar](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

## Algorithmen

- **DDQN:** Das Online-Netz wählt die nächste Action, das Target-Netz bewertet
  genau diese Action. Dadurch wird die für DQN typische Überschätzung reduziert.
- **Noisy DDQN:** Factorisierte gaußsche NoisyLinear-Layer erzeugen lernbare
  Exploration. Zusätzliches Epsilon-Greedy ist deaktiviert; bei der
  deterministischen Evaluation wird das Rauschen ausgeschaltet.
- **PER DDQN:** Übergänge werden proportional zu ihrem absoluten TD-Fehler
  priorisiert. Importance-Sampling-Gewichte korrigieren den Sampling-Bias;
  `β` steigt während des Trainings bis `1`.
- **Dueling DDQN:** Getrennte Value- und Advantage-Streams werden als
  `Q(s,a) = V(s) + A(s,a) - mean(A(s,·))` zusammengeführt.

Quellen: [Double DQN](https://arxiv.org/abs/1509.06461),
[NoisyNet](https://arxiv.org/abs/1706.10295),
[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952),
[Dueling Networks](https://arxiv.org/abs/1511.06581).

## Standardprofil und Parameter

Die gemeinsamen Ausgangswerte stammen aus dem DQN-Profil des RL Baselines3
Zoo für `MountainCar-v0`: `120.000` Schritte, Lernrate `0,004`, Buffer
`10.000`, Lernstart `1.000`, Batch `128`, `γ=0,98`, Training alle 16 Schritte
mit 8 Gradientenupdates, Target-Update alle 600 Schritte, Epsilon-Abklinganteil
`0,2` und finales Epsilon `0,07`. Die Hidden Layers wurden für den kleinen,
zweidimensionalen Zustandsraum bewusst von `256,256` auf `64,64` reduziert,
damit Training und paralleler Vergleich schneller laufen. Das ist eine
begründete Kursabweichung vom Zoo-Profil.

Nicht im Zoo-Profil festgelegte gemeinsame Werte folgen Stable-Baselines3
2.9.0. Variantenwerte folgen den Originalarbeiten beziehungsweise verbreiteten
Ausgangswerten: Noisy `σ₀=0,5`, PER `α=0,6`, `β₀=0,4`, `ε=1e-6` und Dueling-
Streams mit je 64 Neuronen. Sämtliche verwendeten Netzwerk- und
Algorithmusparameter sind in der UI änderbar; unpassende Variantenfelder sind
deaktiviert.

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
Episoden, Environment-Schritte, mittleren Reward und Erfolgsrate. Alle vier
Varianten bleiben gleichzeitig lesbar.

Training läuft ohne Rendering und ohne automatische Zwischen-Evaluationen.
Eine manuell gestartete Evaluation nutzt ein separates headless Environment,
lernt nicht und verändert weder Parameter noch Replay Buffer. Bei einem
besseren mittleren Evaluations-Reward werden Modell, Target-Netz, Optimizer und
Replay Buffer gemeinsam temporär gesichert. Der Button
`Bestes Modell wiederherstellen` lädt diesen vollständigen Lernzustand.

Auf macOS läuft der offizielle Gymnasium-/Pygame-Renderer in einem separaten
unsichtbaren Prozess. So teilen sich Tkinter und SDL keinen Prozess und es
entsteht weder ein nativer GIL-Absturz noch ein zweiter Dock-Eintrag.

## Speichern und Laden

Ein Speicherstand besteht aus drei zusammengehörenden Dateien:

- `<name>.zip`: SB3-Modell einschließlich Online-/Target-Netz und Optimizer
- `<name>_replay.pkl`: Replay Buffer einschließlich PER-Prioritäten
- `<name>_metadata.json`: Variante und vollständige Konfiguration

Nur ein vollständiger kompatibler Satz kann geladen und weitertrainiert werden.

## Tests und Grenzen

```bash
python -m unittest discover -s tests -v
```

Die Tests prüfen Environment, Double-Zielwert, NoisyNet, PER, Dueling-Netz,
alle vier kurzen Trainingsläufe, unveränderte Evaluation, Save/Load/Fortsetzen,
Renderer-Prozesstrennung und das sichtbare Startlayout.

Kurze Läufe lösen MountainCar nicht zuverlässig. Paralleltraining mehrerer
Varianten beansprucht entsprechend mehr CPU. Der Kursstand unterstützt nur
diskrete Actions und die hier aufgeführten vier Varianten.
