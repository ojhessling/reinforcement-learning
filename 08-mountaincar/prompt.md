# MountainCar – RL-Workbench

Projektordner: `Oliver/08-mountaincar`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `mountaincar`
Environment: `MountainCar-v0`

Dieser Prompt enthält nur die MountainCar-spezifischen Anforderungen.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung für Gymnasiums diskretes
`MountainCar-v0`. Vergleiche vier PyTorch-basierte Double-DQN-Varianten:

1. `DDQN`
2. `Noisy DDQN` (`DDQN` mit NoisyNet-Exploration)
3. `PER DDQN` (`DDQN` mit Prioritized Experience Replay)
4. `Dueling DDQN` (`DDQN` mit Dueling-Netzarchitektur)

Dies sind vier getrennte Verfahren. Kombiniere NoisyNet, PER und Dueling nicht
ungefragt zu einem Rainbow-ähnlichen Gesamtalgorithmus. Alle Varianten verwenden
dieselbe DDQN-Target-Berechnung: Das Online-Netz wählt die nächste Action, das
Target-Netz bewertet genau diese Action.

## Environment

Erzeuge das Environment zwingend mit:

```python
import gymnasium

env = gymnasium.make("MountainCar-v0", render_mode="rgb_array")
```

Eine gemeinsame Factory erzeugt getrennte Instanzen für Training,
deterministische Evaluation und sichtbare Animation. Verwende für rein
headless Evaluationen kein Rendering, wenn dadurch dieselbe unveränderte
Environment-Spezifikation erhalten bleibt. Environment-Instanzen werden weder
gleichzeitig noch threadübergreifend geteilt.

Verändere Dynamik, Startzustände, `goal_velocity`, Reward oder Abbruchregeln
nicht. Insbesondere ist kein Reward Shaping zulässig.

Actions:

- `0`: nach links beschleunigen
- `1`: nicht beschleunigen
- `2`: nach rechts beschleunigen

Observation:

1. Position des Wagens im Bereich `[-1,2; 0,6]`
2. Geschwindigkeit im Bereich `[-0,07; 0,07]`

Der originale Reward beträgt in jedem Schritt `-1`. Eine Episode terminiert,
sobald die Position mindestens `0,5` erreicht, und wird nach 200 Schritten
trunkiert. Als Erfolg gilt ausschließlich das Erreichen der Zielposition vor
dem Zeitlimit. Ein höherer, also weniger negativer Reward ist besser; der
optimale Episoden-Reward liegt wegen des zufälligen Startzustands nicht als
einheitliche feste Zahl vor.

Quelle: [Gymnasium MountainCar](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

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
der Replay-Buffer-Semantik von Stable-Baselines3 konsistent. Teste explizit,
dass Online-Netz und Target-Netz bei der DDQN-Berechnung unterschiedliche
Aufgaben erfüllen.

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

## Noisy DDQN

Ersetze in `Noisy DDQN` die linearen Schichten des Q-Netzes durch trainierbare
faktorisierte NoisyNet-Schichten. Das Rauschen wird während des Trainings
passend neu gezogen; die deterministische Evaluation arbeitet ohne Rauschen.
NoisyNet ersetzt die ε-greedy-Exploration dieser Variante. Setze deshalb im
Noisy-Modus die effektive ε-Exploration auf null und deaktiviere die
zugehörigen Eingaben verständlich, statt beide Explorationsarten unbemerkt zu
mischen.

Zusätzlich in der UI:

- initiale Noisy-Standardabweichung `σ₀`, Standardwert `0,5`
- Rauschtyp: faktorisierte gaußsche NoisyNet-Schichten; keine wirkungslose
  Auswahl anbieten, solange kein zweiter Typ implementiert ist

Quelle: [Fortunato et al., Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

## PER DDQN

Implementiere proportional Prioritized Experience Replay. Neue Transitionen
erhalten zunächst die höchste vorhandene Priorität. Sampling erfolgt
proportional zu `priority ** α`. Korrigiere den Sampling-Bias mit
normalisierten Importance-Sampling-Gewichten; `β` steigt während des
Trainings von `β₀` bis 1. Nach jedem Lernupdate werden die Prioritäten aus
den absoluten TD-Fehlern aktualisiert.

Zusätzlich in der UI:

- Priorisierungsstärke `α_PER`, Standardwert `0,6`
- initiale Bias-Korrektur `β₀`, Standardwert `0,4`
- Annealing-Schritte für `β`, standardmäßig das Trainingsbudget `N`
- Prioritätskonstante `ε_PER`, Standardwert `1e-6`

Teste Sampling-Verteilung, Importance-Sampling-Gewichte und
Prioritätsaktualisierung unabhängig von der GUI.

Quelle: [Schaul et al., Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

## Dueling DDQN

Verwende nach einem gemeinsamen Feature-Netz getrennte Value- und
Advantage-Streams. Führe sie identifizierbar zusammen mit:

```text
Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
```

Value- und Advantage-Stream-Größen sind in der UI einstellbar. Der gemeinsame
Feature-Teil verwendet die allgemeinen Netzwerkparameter. Teste Form,
Zentrierung und Gradientenfluss der Zusammenführung.

Quelle: [Wang et al., Dueling Network Architectures](https://arxiv.org/abs/1511.06581)

## Standardprofil

Nutze als gemeinsame Ausgangswerte das getunte MountainCar-v0-DQN-Profil des
RL Baselines3 Zoo:

- Trainingsschritte `N = 120.000`
- Lernrate `α = 0,004`
- Replay Buffer `10.000`
- Lernstart `1.000`
- Batch-Größe `128`
- Diskontfaktor `γ = 0,98`
- Target-Update-Intervall `600`
- Trainingsfrequenz `16`
- Gradientenschritte `8`
- Explorationsanteil `0,2`
- finales Epsilon `0,07`
- Hidden Layers `64,64` als bewusst kleineres, schnelleres Kursprofil für den
  zweidimensionalen Zustandsraum

Nicht im Profil definierte Werte stammen aus der verwendeten
Stable-Baselines3-Version oder den oben genannten Originalarbeiten. Dokumentiere
Quelle und jede begründete Abweichung in der README.

Quelle: [RL Baselines3 Zoo, DQN-Hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)

## Training und Evaluation

Training nutzt einen Callback für Abbruch, Fortschritt und Episodenmetriken.
Während Training und Vergleich finden keine automatischen
Zwischen-Evaluationen statt. Eine ausdrücklich gestartete Evaluation verwendet ausschließlich die
deterministische Policy und verändert weder Modell, Optimizer, Replay Buffer
noch Trainingsstatistiken. Bei Noisy DDQN ist das Rauschen dabei deaktiviert.

Die Anzahl der manuell gestarteten Evaluations-Episoden ist in der UI
einstellbar. Zeige im Evaluationsergebnis mindestens:

- mittleren Reward und Standardabweichung
- mittlere Episodenlänge
- Erfolgsrate als Anteil erreichter Zielpositionen
- mittlere und beste erreichte Endposition
- letzte und beste deterministische Evaluation

Merke beim Einzeltraining den besten deterministischen Evaluationszustand und
sichere bei Verbesserung Modell, Target-Netz, Optimizer-Zustand und passenden
Replay Buffer. Der Button `Bestes Modell wiederherstellen` stellt den gesamten
Lernzustand wieder her.

## Vergleich

Alle vier Algorithmen sind einzeln für den gemeinsamen Vergleich auswählbar;
mindestens zwei müssen ausgewählt sein. Jeder Algorithmus besitzt genau einen
Lauf. Ausgewählte Algorithmen starten gleichzeitig und verwenden identische
gemeinsame Hyperparameter, dasselbe Trainingsbudget und fair abgeleitete Seeds.
Ein erneuter kompatibler Vergleich trainiert dieselben Vergleichsmodelle weiter
und hängt Ergebnisse an.

Der Vergleichsgraph zeigt ausschließlich den explorativen Episoden-Reward gegen
die Episodennummer. Rohwerte sind stark transparent; je Algorithmus zeigt eine
schmale, kräftigere Linie den gleitenden Durchschnitt der letzten 20 Episoden.
Deterministische Evaluationen erscheinen nur in der Summary, nicht im Graphen.
Die Legende steht in einem kompakten reservierten Bereich neben dem Plot und
nennt nur die Algorithmusnamen sowie eine sinnvolle MountainCar-Referenzlinie,
falls eine fachlich begründete feste Referenz verwendet wird.

Die Vergleichs-Summary ist eine kompakte, live aktualisierte Tabelle, in der
alle vier Algorithmen gleichzeitig sichtbar sind. Sie enthält Algorithmen als
Spalten und diese Zeilen:

- Episoden
- Environment-Schritte
- durchschnittlicher Reward
- Trainingserfolgsrate

## Darstellung

Zeige ausschließlich den offiziellen, von `env.render()` gelieferten
Gymnasium-RGB-Frame. Erstelle keine eigene MountainCar-Grafik und öffne kein
separates Pygame-Fenster. Beachte die macOS-Prozesstrennung aus der Workbench.

Zeige neben der Animation:

- aktuelle Position
- aktuelle Geschwindigkeit
- gewählte Action als `links`, `keine Beschleunigung` oder `rechts`
- aktuellen Episodenschritt
- bisher kumulierten Reward
- Zielposition `0,5`

MountainCar-spezifische Metriken sind Reward, Episodenlänge, Erfolgsrate,
maximal erreichte Position und Zielerreichung. Der Reward ist negativ; Achsen,
Summary und Hilfetexte erklären eindeutig, dass Werte näher an null besser sind.

## Speichern und Laden

Gespeicherte Modelle müssen später mit derselben Variante weitertrainiert
werden können. Speichere Modell, Target-Netz, Optimizer, Replay Buffer,
Algorithmusvariante, vollständige Konfiguration und Metadaten. Für PER gehören
Prioritäten, aktueller `β`-Fortschritt und Buffer-Zeiger zum gespeicherten
Zustand. Für Noisy und Dueling muss die exakte Netzarchitektur rekonstruierbar
sein. Lehne inkompatible oder unvollständige Dateien verständlich ab.

## Abhängigkeiten

Verwende PyTorch, Gymnasium und Stable-Baselines3 aus der gemeinsamen
Kursumgebung. Führe keine zusätzliche Deep-Learning-Bibliothek ein. Dokumentiere
direkte Abhängigkeiten in `requirements.txt` und bei Bedarf in
`../environment.yml`.

## Abnahme

Fertig, wenn alle vier Varianten fachlich korrekt trainieren, deterministisch
evaluiert und fair verglichen werden; algorithmusspezifische Parameter nur im
passenden Modus erscheinen; NoisyNet ohne zusätzliches epsilon-greedy arbeitet;
PER tatsächlich priorisiert sampelt und gewichtet; Dueling Value und Advantage
korrekt zusammenführt; beste Lernzustände vollständig wiederherstellbar sind;
die offizielle Gymnasium-Animation eingebettet ist; und alle allgemeinen Tests,
Sichtbarkeitsprüfungen und Abnahmekriterien aus `../workbench.md` erfüllt sind.
