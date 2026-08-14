# LunarLander – RL-Workbench

Projektordner: `Oliver/10-lunarlander`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `lunarlander`
Environment: `LunarLander-v3`

Dieser Prompt enthält nur die LunarLander-spezifischen Anforderungen. Alles
Allgemeine – Projektstruktur, GUI-Aufbau, Responsivität, Vergleichslogik,
Export, Tests und Abnahme – steht ausschließlich in `../workbench.md` und wird
hier bewusst nicht wiederholt.

Dieses Projekt enthält **keine Abweichungen** von der Workbench. Insbesondere
gelten ohne Einschränkung:

- Es gibt **kein** Eingabefeld für Animationsgeschwindigkeit, `Animation Δt`
  oder einen vergleichbaren Parameter. Die Abspielgeschwindigkeit ist im
  Projekt fest vorgegeben (siehe `Darstellung`) und nur ein- und ausschaltbar.
- Längere Trainings- und Vergleichsläufe werden im sichtbaren, konfigurierbaren
  Schrittintervall automatisch headless deterministisch evaluiert; Graph und
  Summary werden live mitgeschrieben.
- Diagramm und Summary sind exportierbar.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung für Gymnasiums diskretes
`LunarLander-v3`. Vergleiche dieselben sieben PyTorch-basierten
Double-DQN-Varianten wie in `Oliver/09-acrobot`, die zusammen alle sechs
Rainbow-Bausteine einzeln und kombiniert abdecken:

1. `DDQN`
2. `Noisy DDQN` (NoisyNet-Exploration)
3. `PER DDQN` (Prioritized Experience Replay)
4. `Dueling DDQN` (Dueling-Netzarchitektur)
5. `Multi-Step DDQN` (n-Schritt-Returns)
6. `C51 DDQN` (distributional RL, kategoriale Rückgabeverteilung)
7. `Rainbow DDQN` (alle fünf Erweiterungen gleichzeitig)

`Rainbow DDQN` verwendet exakt dieselben Bausteine wie die Einzelvarianten
(2–6), kombiniert in einem Agenten, ohne deren Implementierung zu duplizieren.
Alle sieben Varianten verwenden dieselbe DDQN-Target-Berechnung: Das
Online-Netz wählt die nächste Action (bei `C51 DDQN`/`Rainbow DDQN` anhand des
aus der Verteilung gebildeten Erwartungswerts), das Target-Netz bewertet genau
diese Action.

## Environment

Erzeuge das Environment zwingend mit:

```python
import gymnasium

env = gymnasium.make("LunarLander-v3", render_mode="rgb_array")
```

Eine gemeinsame Factory erzeugt getrennte Instanzen für Training,
deterministische Evaluation und sichtbare Animation. Verwende für rein
headless Evaluationen kein Rendering, wenn dadurch dieselbe unveränderte
Environment-Spezifikation erhalten bleibt. Environment-Instanzen werden weder
gleichzeitig noch threadübergreifend geteilt.

Verändere Dynamik, Startzustandsverteilung, Reward oder Abbruchregeln nicht.
Belasse insbesondere alle optionalen Konstruktorargumente auf ihren
Standardwerten: `continuous=False`, `gravity=-10.0`, `enable_wind=False`,
`wind_power=15.0`, `turbulence_power=1.5`. Die kontinuierliche Variante
(`LunarLanderContinuous-v3`) gehört nicht zum Projekt.

Actions (`Discrete(4)`):

- `0`: nichts tun
- `1`: linkes Steuertriebwerk zünden
- `2`: Haupttriebwerk zünden
- `3`: rechtes Steuertriebwerk zünden

Observation (8-dimensional, `float32`):

1. Position `x` relativ zur Landeplattform, Bereich `[-2,5; 2,5]`
2. Position `y` relativ zur Landeplattform, Bereich `[-2,5; 2,5]`
3. Geschwindigkeit `vₓ`, Bereich `[-10; 10]`
4. Geschwindigkeit `v_y`, Bereich `[-10; 10]`
5. Winkel `θ` des Landers, Bereich `[-2π; 2π]`
6. Winkelgeschwindigkeit `θ̇`, Bereich `[-10; 10]`
7. Bodenkontakt linkes Bein, `0` oder `1`
8. Bodenkontakt rechtes Bein, `0` oder `1`

Die Landeplattform liegt immer bei `(0, 0)`. Die Einheiten der Observation sind
normalisiert und nicht durchgängig SI-konform; die Winkelgeschwindigkeit ist in
Einheiten von `0,4 rad/s` angegeben und muss für eine Anzeige in `rad/s` mit
`2,5` multipliziert werden. Rechne Winkel für die Anzeige in Grad um.

Der Reward setzt sich pro Schritt zusammen aus:

- einer Differenz zweier Shaping-Terme, die Annäherung an die Plattform,
  geringere Geschwindigkeit, geringere Schräglage sowie Beinkontakt
  (`+10` je Bein) belohnen
- `-0,3` je Frame mit gezündetem Haupttriebwerk
- `-0,03` je Frame mit gezündetem Steuertriebwerk

Zusätzlich endet die Episode mit `-100` bei Absturz oder Verlassen des
sichtbaren Bereichs (`|x| ≥ 1`) und mit `+100`, sobald der Lander zur Ruhe
kommt. Nach 1000 Schritten wird die Episode trunkiert. Dieses Shaping ist Teil
des Original-Environments; eigenes Reward Shaping ist nicht zulässig.

Anders als in den vorherigen Projekten ist der Reward hier **nicht** rein
negativ: Höhere Werte sind besser, der Episoden-Return liegt realistisch etwa
zwischen `-400` und `+320`. Gymnasium nennt `200` als `reward_threshold`; eine
Episode ab `200` Punkten gilt als gelöst.

Erfolgsdefinition für dieses Projekt:

- **Landung**: Episode endet mit `terminated` und ruhendem Lander (`+100`),
  also weder Absturz noch Zeitlimit.
- **Gelöst**: Episoden-Return `≥ 200`.

Beide Quoten werden getrennt ausgewiesen, weil eine sichere Landung mit hohem
Treibstoffverbrauch die Lösungsschwelle verfehlen kann.

Quelle: [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

## Gemeinsame DDQN-Basis

Verwende Stable-Baselines3 als Infrastruktur für Environment-Anbindung,
Training, Policies, Target-Netz, Logging und Speichern, implementiere die
benötigten Erweiterungen jedoch fachlich nachvollziehbar in eigenen kleinen
Klassen. Kopiere keine unabhängige vollständige DQN-Implementierung, wenn eine
gezielte Unterklasse, Policy, Q-Network- oder Replay-Buffer-Erweiterung reicht.

Die DDQN-Zielwertberechnung lautet:

```text
a* = argmax_a Q_online(s', a)
y  = r + gamma * (1 - done) * Q_target(s', a*)
```

Behandle `terminated` und `truncated` entsprechend den Regeln der Workbench und
der Replay-Buffer-Semantik von Stable-Baselines3 konsistent. Beachte, dass
`LunarLander` selbst immer `truncated=False` liefert und das Zeitlimit
ausschließlich vom `TimeLimit`-Wrapper aus `gymnasium.make` gesetzt wird. Teste
explizit, dass Online-Netz und Target-Netz bei der DDQN-Berechnung
unterschiedliche Aufgaben erfüllen.

Gemeinsame, in der UI änderbare Parameter:

- `total_timesteps`
- `learning_rate`
- `buffer_size`
- `learning_starts`
- `batch_size`
- `tau`
- `gamma`
- `train_freq`
- `gradient_steps`
- `target_update_interval`
- `exploration_fraction`
- `exploration_initial_eps`
- `exploration_final_eps`
- `max_grad_norm`
- `seed`
- Hidden Layers, Aktivierung, Optimizer und relevante Optimizer-Parameter

Technische Optionen wie `verbose`, `tensorboard_log`, `device` und
`_init_setup_model` gehören nicht in die UI.

## Noisy-Baustein

Ersetzt die linearen Schichten des Q-Netzes durch trainierbare faktorisierte
NoisyNet-Schichten. Das Rauschen wird während des Trainings passend neu
gezogen; die deterministische Evaluation arbeitet ohne Rauschen. NoisyNet
ersetzt die ε-greedy-Exploration in jeder Variante, die diesen Baustein nutzt
(`Noisy DDQN`, `Rainbow DDQN`). Setze dort die effektive ε-Exploration auf null
und deaktiviere die zugehörigen Eingaben verständlich.

Zusätzlich in der UI: initiale Noisy-Standardabweichung `σ₀`, Standardwert `0,5`.

Quelle: [Fortunato et al., Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

## PER-Baustein

Implementiert proportional Prioritized Experience Replay. Neue Transitionen
erhalten zunächst die höchste vorhandene Priorität. Sampling erfolgt
proportional zu `priority ** α`. Korrigiere den Sampling-Bias mit
normalisierten Importance-Sampling-Gewichten; `β` steigt während des Trainings
von `β₀` bis 1. Nach jedem Lernupdate werden die Prioritäten aus den absoluten
TD-Fehlern (bei `C51 DDQN`/`Rainbow DDQN` aus dem Kreuzentropie-Verlust je
Sample) aktualisiert. Dieser Baustein wird in `PER DDQN` und `Rainbow DDQN`
verwendet.

Zusätzlich in der UI:

- Priorisierungsstärke `α_PER`, Standardwert `0,6`
- initiale Bias-Korrektur `β₀`, Standardwert `0,4`
- Annealing-Schritte für `β`, standardmäßig das Trainingsbudget `N`
- Prioritätskonstante `ε_PER`, Standardwert `1e-6`

Teste Sampling-Verteilung, Importance-Sampling-Gewichte und
Prioritätsaktualisierung unabhängig von der GUI.

Quelle: [Schaul et al., Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

## Dueling-Baustein

Verwendet nach einem gemeinsamen Feature-Netz getrennte Value- und
Advantage-Streams. Führe sie identifizierbar zusammen mit:

```text
Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
```

Value- und Advantage-Stream-Größen sind in der UI einstellbar. Der gemeinsame
Feature-Teil verwendet die allgemeinen Netzwerkparameter. Dieser Baustein wird
in `Dueling DDQN` und `Rainbow DDQN` verwendet; bei gleichzeitig aktivem
C51-Baustein bildet dieselbe Formel je Atom der Rückgabeverteilung. Teste Form,
Zentrierung und Gradientenfluss der Zusammenführung.

Quelle: [Wang et al., Dueling Network Architectures](https://arxiv.org/abs/1511.06581)

## Multi-Step-Baustein

Ersetzt den 1-Schritt-TD-Fehler durch einen n-Schritt-Return:

```text
R_t^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k}
y = R_t^(n) + gamma^n * (1 - done) * Q_target(s_{t+n}, a*)
```

Endet eine Episode innerhalb der n Schritte, verkürzt sich der Return
entsprechend; es wird nicht über das Episodenende hinweg akkumuliert.
Truncation-Fälle bootstrappen konsistent zu den übrigen Bausteinen.
Implementiere dies als eigenständige, unabhängig testbare
Replay-Buffer-Erweiterung, die sich mit dem PER-Baustein kombinieren lässt,
ohne dessen Prioritäten-Buchhaltung zu verändern. Dieser Baustein wird in
`Multi-Step DDQN` und `Rainbow DDQN` verwendet.

Zusätzlich in der UI: Schrittanzahl `n_step`, Standardwert `3`.

Teste die Return- und Discount-Berechnung für vollständige n-Schritt-Fenster,
für durch Episodenende verkürzte Fenster und das Zusammenspiel mit PER.

Quelle: [Sutton, Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009)

## C51-Baustein (Distributional RL)

Ersetzt den skalaren Q-Wert durch eine kategoriale Verteilung über feste
Rückgabewerte (`Atome`) `z_i`, gleichmäßig verteilt in `[V_min, V_max]`. Das
Netz gibt je Action eine Wahrscheinlichkeit pro Atom aus (Softmax über die
Atome); der Q-Wert ergibt sich als `Q(s,a) = Σ_i p_i(s,a) · z_i` und wird für
Action-Auswahl und `predict()` verwendet. Die Zielverteilung entsteht durch
Anwenden der Bellman-Gleichung auf die Atome und anschließende Projektion auf
das feste Atom-Gitter. Die Double-DQN-Regel gilt weiterhin: Das Online-Netz
wählt `a*` über den aus seiner eigenen Verteilung gebildeten Erwartungswert,
das Target-Netz liefert die zu projizierende Verteilung für genau dieses `a*`.
Trainiert wird mit Kreuzentropie. Dieser Baustein wird in `C51 DDQN` und
`Rainbow DDQN` verwendet.

Zusätzlich in der UI:

- Anzahl Atome `n_atoms`, Standardwert `51`
- `V_min`, Standardwert `-400`
- `V_max`, Standardwert `400`

Anders als bei den rein negativ belohnten Vorgängerprojekten ist der
LunarLander-Return zweiseitig. `V_min`/`V_max` decken deshalb symmetrisch den
realistisch erreichbaren Returnbereich ab und sind keine harten
Environment-Grenzen; begründe den gewählten Bereich in der README.

Teste, dass projizierte Zielverteilungen normiert bleiben, dass exakt auf ein
Atom fallende Zielwerte ihre Masse korrekt an genau diesem Atom sammeln und
dass die Double-DQN-Actionauswahl weiterhin über das Online-Netz erfolgt.

Quelle: [Bellemare et al., A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)

## Rainbow DDQN

`Rainbow DDQN` kombiniert Noisy-, PER-, Dueling-, Multi-Step- und C51-Baustein
gleichzeitig in einem einzigen Agenten, ohne deren Implementierung zu ändern
oder zu duplizieren. Alle zugehörigen UI-Parameter erscheinen auch im
Rainbow-Modus. Teste zusätzlich zu den Einzelbaustein-Tests, dass alle fünf
Bausteine gleichzeitig aktiv sind und sich das Zusammenspiel korrekt in
Netzarchitektur, Sampling, Zielverteilungsberechnung und Prioritäten-Update
niederschlägt.

## Standardprofil

Nutze als gemeinsame Ausgangswerte das getunte LunarLander-v3-DQN-Profil des
RL Baselines3 Zoo:

- Trainingsschritte `N = 100.000`
- Lernrate `α = 6,3e-4`
- Replay Buffer `50.000`
- Lernstart `0`
- Batch-Größe `128`
- Diskontfaktor `γ = 0,99`
- Target-Update-Intervall `250`
- Trainingsfrequenz `4`
- Gradientenschritte: ein Schritt je gesammeltem Environment-Schritt (Zoo-Wert
  `-1`); dokumentiere die daraus abgeleitete konkrete Zahl in der README
- Explorationsanteil `0,12`
- finales Epsilon `0,1`
- Hidden Layers `256,256`

Nicht im Profil definierte Werte stammen aus der verwendeten
Stable-Baselines3-Version oder den oben genannten Originalarbeiten.
Dokumentiere Quelle und jede begründete Abweichung in der README.

Quelle: [RL Baselines3 Zoo, DQN-Hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)

## Evaluation

Zeige im Evaluationsergebnis mindestens:

- mittleren Return und Standardabweichung
- mittlere Episodenlänge
- Landequote (Anteil sicher gelandeter Episoden)
- Gelöst-Quote (Anteil Episoden mit Return `≥ 200`)
- Absturzquote
- letzte und beste deterministische Evaluation

## Vergleich

Alle sieben Algorithmen sind einzeln für den gemeinsamen Vergleich auswählbar;
mindestens zwei müssen ausgewählt sein. Der Vergleichsgraph zeigt den
explorativen Episoden-Return gegen die Episodennummer und enthält eine
Referenzlinie bei `+200` (`reward_threshold`, gelöst).

Die Vergleichs-Summary ist eine kompakte, live aktualisierte Tabelle, in der
alle sieben Algorithmen gleichzeitig sichtbar sind. Sie enthält Algorithmen als
Spalten und diese Zeilen:

- Episoden
- Environment-Schritte
- durchschnittlicher Return
- Landequote
- Gelöst-Quote

## Darstellung

Zeige ausschließlich den offiziellen, von `env.render()` gelieferten
Gymnasium-RGB-Frame (600 × 400 Pixel). Erstelle keine eigene
LunarLander-Grafik und öffne kein separates Pygame-Fenster. Beachte die
macOS-Prozesstrennung aus der Workbench.

Die Animation läuft mit der environment-eigenen Bildrate von 50 FPS
(`env.metadata["render_fps"]`), also rund 20 ms je Frame. Dieser Wert ist fest
im Code hinterlegt und erscheint gemäß Workbench **nicht** als Eingabefeld; die
Animation ist lediglich abschaltbar.

Zeige neben der Animation:

- Position `x` und `y`
- Geschwindigkeit `vₓ` und `v_y`
- Winkel `θ` in Grad und Winkelgeschwindigkeit `θ̇` in `rad/s`
  (Observationswert × 2,5)
- Bodenkontakt beider Beine als verständliche Ja/Nein-Anzeige
- gewählte Action als `nichts tun`, `links`, `Haupttriebwerk` oder `rechts`
- aktuellen Episodenschritt
- bisher kumulierten Return

LunarLander-spezifische Metriken sind Episoden-Return, Episodenlänge,
Landequote, Gelöst-Quote und Absturzquote. Achsen, Summary und Hilfetexte
erklären eindeutig, dass höhere Werte besser sind und `200` als gelöst gilt.

## Speichern und Laden

Gespeicherte Modelle müssen später mit derselben Variante weitertrainiert
werden können. Speichere Modell, Target-Netz, Optimizer, Replay Buffer,
Algorithmusvariante, vollständige Konfiguration und Metadaten. Für PER-basierte
Varianten gehören Prioritäten, aktueller `β`-Fortschritt und Buffer-Zeiger zum
gespeicherten Zustand; für Multi-Step-basierte Varianten gehört der
unvollständige n-Schritt-Fensterzustand nicht dazu. Für Noisy-, Dueling- und
C51-basierte Varianten muss die exakte Netzarchitektur inklusive Atom-Gitter
rekonstruierbar sein. Lehne inkompatible oder unvollständige Dateien
verständlich ab.

Hinweis: Umhüllen eigene Wrapper den SB3-Replay-Buffer, greifen
`model.save_replay_buffer()` und `model.load_replay_buffer()` wegen einer
internen `isinstance`-Prüfung nicht. Verwende in diesem Fall die generischen
SB3-Helfer `save_to_pkl`/`load_from_pkl` und stelle das Device nach dem Laden
wieder her.

## Abhängigkeiten

`LunarLander-v3` benötigt zusätzlich zur bisherigen Kursumgebung **Box2D**. Das
ist die einzige neue Abhängigkeit gegenüber den Vorgängerprojekten:

```bash
pip install swig
pip install "gymnasium[box2d]"
```

Ergänze `swig` und `box2d` in `requirements.txt` sowie im `pip`-Abschnitt von
`../environment.yml`; `swig` muss dabei vor `box2d` stehen, weil das
Box2D-Paket es zum Bauen benötigt. Verwende ansonsten PyTorch, Gymnasium und
Stable-Baselines3 aus der gemeinsamen Kursumgebung und führe keine zusätzliche
Deep-Learning-Bibliothek ein. Die README nennt den Installationsschritt
ausdrücklich, da das Projekt ohne Box2D nicht startet.

## Abnahme

Zusätzlich zu allen Abnahmekriterien aus `../workbench.md` gilt: Fertig, wenn
alle sieben Varianten fachlich korrekt trainieren, deterministisch evaluiert
und fair verglichen werden; algorithmusspezifische Parameter nur in den
passenden Modi erscheinen; NoisyNet ohne zusätzliches epsilon-greedy arbeitet;
PER priorisiert sampelt und gewichtet; Dueling Value und Advantage korrekt
zusammenführt; Multi-Step an Episodengrenzen korrekt verkürzt; C51 normierte,
korrekt projizierte Zielverteilungen erzeugt; `Rainbow DDQN` alle fünf
Erweiterungsbausteine gleichzeitig kombiniert; Lande- und Gelöst-Quote getrennt
ausgewiesen werden; beste Lernzustände vollständig wiederherstellbar sind; die
offizielle Gymnasium-Animation mit fester Bildrate und ohne
Geschwindigkeits-Eingabefeld eingebettet ist; und Box2D als neue Abhängigkeit
dokumentiert ist.
