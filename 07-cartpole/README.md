# CartPole DQN Workbench

Lokale deutschsprachige Tkinter-Anwendung zum Trainieren und Verstehen eines
Deep-Q-Networks auf Gymnasiums `CartPole-v1`. Angeboten werden
`stable_baselines3.DQN` und darauf aufbauendes Double DQN (`DDQN`) mit
`MlpPolicy`.

## Installation und Start

Das Projekt verwendet die zentrale Conda-Umgebung:

```bash
conda env update -f ../environment.yml --prune
conda activate rl-26-08
python cartpole_app.py
```

Alternativ stehen die direkten Abhängigkeiten in `requirements.txt`.

## Environment

Die Anwendung erzeugt getrennte Instanzen mit:

```python
gymnasium.make("CartPole-v1", render_mode="rgb_array")
```

Observation:

1. Wagenposition in Metern
2. Wagengeschwindigkeit in Metern pro Sekunde
3. Stangenwinkel in Radiant (in der GUI zusätzlich als Grad)
4. Winkelgeschwindigkeit in Radiant pro Sekunde (GUI: Grad pro Sekunde)

Action `0` beschleunigt nach links, Action `1` nach rechts. Jeder Schritt gibt
Reward `+1`. Eine Episode terminiert bei etwa `±12°` Stangenwinkel oder `±2,4 m`
Wagenposition. Das Zeitlimit truncatiert nach 500 Schritten; dies gilt in der
Workbench als erfolgreiche Episode.

Quelle: [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

## DQN und Standardwerte

Die Implementierung stammt aus Stable-Baselines3 2.9.0. Die Ausgangswerte
verwenden ein gegen starke Policy-Einbrüche konservativeres CartPole-Profil:

- Trainingsbudget `100.000` Environment-Schritte
- Lernrate `0.0005`
- Replay Buffer `200.000`
- Learning Starts `1.000`
- Batch-Größe `128`
- Tau `1.0`, Gamma `0.99`
- Training alle `4` Environment-Schritte mit `1` Gradientenschritt
- Target-Network-Update alle `1.000` Schritte
- Epsilon von `1.0` auf `0.05` während der ersten `20 %` des Trainings
- maximale Gradientennorm `10`
- MLP mit zwei Hidden Layers zu je `64` Neuronen und ReLU als schneller
  Kompromiss für die vierdimensionalen CartPole-Observations
- Adam-Optimizer mit numerischer Stabilität `1e-5`

Alle fachlichen DQN- und Netzwerkparameter sind in der GUI änderbar. Die
Workbench verwendet für DQN die unveränderte SB3-Implementierung. Bei DDQN
wählt das Online-Netz die nächste Action, während das Target-Netz deren Wert
bestimmt. Dueling DQN und Prioritized Experience Replay sind nicht enthalten.

Die GUI gruppiert die Parameter fachlich. Verwendete Kurzzeichen:

- `N`: Trainingsschritte, `t₀`: Lernstart, `|D|`: Replay-Buffer-Größe
- `B`: Batch-Größe, `fₜ`: Trainingsfrequenz, `G`: Gradientenschritte
- `α`: Lernrate, `γ`: Diskontfaktor, `τ`: Soft-Update-Koeffizient
- `C`: Target-Network-Update-Intervall, `‖g‖ₘₐₓ`: maximale Gradientennorm
- `ε₀`: initiale und ε<sub>min</sub>: finale Explorationswahrscheinlichkeit
- f<sub>ε</sub>: Anteil des Trainings für Epsilon-Decay
- ε<sub>opt</sub>: numerische Stabilität des Optimizers, `λ`: Weight Decay
- `h`: Hidden-Layer-Architektur, `φ`: Aktivierungsfunktion
- `λ`: Weight Decay, `s`: Seed, `M`: Evaluations-Episoden

Quelle: [Stable-Baselines3 DQN 2.9.0](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/dqn.html)

## Bedienablauf

1. Parameter prüfen oder ändern.
2. `Training starten / fortsetzen` wählen.
3. Bei Bedarf kontrolliert stoppen.
4. Mit mehreren Episoden deterministisch evaluieren.
5. Modell, Replay Buffer und Metadaten gemeinsam speichern.

Training, Evaluation und Animation verwenden getrennte Environments.
Evaluation ruft `predict(..., deterministic=True)` auf, führt keine Lernupdates
aus und verändert weder Modell noch Replay Buffer.

Auf macOS läuft Gymnasiums Pygame/SDL-Rendering in einem eigenen Prozess. Die
Tkinter-Anwendung erhält ausschließlich die fertigen RGB-Frames. Dadurch werden
Tk und SDL nicht im selben Prozess initialisiert. Der Renderprozess verwendet
einen unsichtbaren SDL-Treiber und erzeugt keinen zusätzlichen Dock-Eintrag.

## Ansichten und Metriken

Der verschiebbare horizontale Splitter trennt die obere Arbeitsfläche von den
unteren Auswertungen. Der offizielle Gymnasium-RGB-Frame zeigt CartPole. Daneben
stehen alle vier Observation-Werte. Das Diagramm zeigt Reward beziehungsweise
Episodenlänge über den Episoden sowie einen gleitenden Mittelwert.

Die Evaluation meldet mittleren Reward, Standardabweichung, mittlere
Episodenlänge und den Anteil erfolgreicher 500-Schritte-Episoden. Sie wird
nicht im Trainingsgraphen dargestellt, sondern in den Summary-Tabellen
ausgewiesen. Zusätzlich wird
während Training und Vergleich standardmäßig alle `5.000` Trainingsschritte
headless und ohne Animation deterministisch evaluiert. Intervall `E` und Anzahl
der Episoden `M` (standardmäßig `10`) sind in der GUI einstellbar; Ergebnisse erscheinen live im
Graphen und als letzter deterministischer Wert in der Summary-Tabelle. Steigt
der mittlere Evaluations-Reward, werden Modell und Replay Buffer dieses exakten
Zustands als temporärer Best-Checkpoint gesichert. Mit `Bestes Modell wiederherstellen` wird dieser
Lernzustand einschließlich Replay Buffer geladen und kann weitertrainiert werden.

Der Algorithmenvergleich erlaubt die Auswahl von DQN und DDQN. Jeder
Algorithmus besitzt genau einen Lauf. Der Graph erscheint sofort mit dem ersten Episodenergebnis
und wird während des Vergleichs zur Begrenzung der Darstellungsarbeit höchstens
alle zwei Sekunden aktualisiert. Pro Rohkurve zeichnet die GUI maximal `2.000`
Punkte; eine Min-/Max-Verdichtung erhält lokale Spitzen und Einbrüche, während
alle Messwerte intern vollständig erhalten bleiben. Die ausgewählten
Algorithmen starten parallel. Ein erneuter Vergleich setzt die Modelle nicht zurück, sondern
trainiert sie weiter und hängt die neuen Ergebnisse an. Rohwerte und
Streuungsband sind dezent; eine kräftige Linie zeigt je Algorithmus den
gleitenden Durchschnitt der letzten 20 Episodenergebnisse. Die Summary wird
bei Training und Vergleich bereits während des Laufs aktualisiert. Im Vergleich
stellt eine kompakte Tabelle die Algorithmen anhand von Episoden, Schritten,
durchschnittlichem Reward und Trainingserfolgsrate gegenüber.

## Speichern und Laden

Zu einem Speicherstand gehören drei Dateien:

- `<name>.zip`: SB3-Modell
- `<name>_replay.pkl`: Replay Buffer
- `<name>_metadata.json`: Version, Environment und Konfiguration

Nur ein vollständiger, kompatibler Satz wird geladen. Danach kann das Training
fortgesetzt werden. Bestehende Dateien sollten nur nach bewusster Auswahl im
Dateidialog überschrieben werden.

## Tests

```bash
python -m unittest discover -s tests -v
```

Die Tests prüfen Environment-Konfiguration, Parametergrenzen, Callback-Abbruch,
Training, unveränderte Evaluation, getrennte Animation sowie den
Save-/Load-/Weitertrainieren-Roundtrip.

## Bekannte Grenzen

- Nur DQN, DDQN und `MlpPolicy` werden unterstützt.
- Replay Buffer können bei großen Konfigurationen viel Speicherplatz belegen.
- Ein kurzer Trainingslauf löst CartPole nicht zwingend; Lernergebnisse hängen
  von Trainingsbudget, Hyperparametern und Seed ab.
