# Acrobot – RL-Workbench

Projektordner: `Oliver/09-acrobot`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `acrobot`
Environment: `Acrobot-v1`

Dieser Prompt enthält nur die Acrobot-spezifischen Anforderungen.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung für Gymnasiums diskretes
`Acrobot-v1`. Vergleiche sieben PyTorch-basierte Double-DQN-Varianten, die
zusammen alle sechs Rainbow-Bausteine (van Hasselt et al. 2016, Wang et al.
2016, Schaul et al. 2016, Sutton 1988, Bellemare et al. 2017, Fortunato et al.
2018) einzeln und kombiniert abdecken:

1. `DDQN`
2. `Noisy DDQN` (`DDQN` mit NoisyNet-Exploration)
3. `PER DDQN` (`DDQN` mit Prioritized Experience Replay)
4. `Dueling DDQN` (`DDQN` mit Dueling-Netzarchitektur)
5. `Multi-Step DDQN` (`DDQN` mit n-Schritt-Returns)
6. `C51 DDQN` (`DDQN` mit distributional RL / kategorialer Rückgabeverteilung)
7. `Rainbow DDQN` (alle sechs Bausteine gleichzeitig: NoisyNet, Prioritized
   Experience Replay, Dueling-Netzarchitektur, Multi-Step-Returns und
   distributional RL, zusätzlich zur Double-DQN-Basis)

`Rainbow DDQN` verwendet exakt dieselben Bausteine wie die fünf
Einzelvarianten (2–6), kombiniert in einem Agenten, ohne deren jeweilige
Implementierung zu duplizieren. Alle sieben Varianten verwenden dieselbe
DDQN-Target-Berechnung: Das Online-Netz wählt die nächste Action (bei `C51
DDQN`/`Rainbow DDQN` anhand des aus der Verteilung gebildeten Erwartungswerts),
das Target-Netz bewertet genau diese Action.

## Environment

Erzeuge das Environment zwingend mit:

```python
import gymnasium

env = gymnasium.make("Acrobot-v1", render_mode="rgb_array")
```

Eine gemeinsame Factory erzeugt getrennte Instanzen für Training,
deterministische Evaluation und sichtbare Animation. Verwende für rein
headless Evaluationen kein Rendering, wenn dadurch dieselbe unveränderte
Environment-Spezifikation erhalten bleibt. Environment-Instanzen werden weder
gleichzeitig noch threadübergreifend geteilt.

Verändere Dynamik, Startzustandsverteilung, Reward, Abbruchregeln oder die
Dynamikvariante nicht. Belasse `book_or_nips` auf dem Standardwert `"book"`.

Actions:

- `0`: Drehmoment `-1 Nm` am Gelenk zwischen den beiden Armsegmenten
- `1`: kein Drehmoment (`0 Nm`)
- `2`: Drehmoment `+1 Nm`

Observation (6-dimensional):

1. `cos(θ₁)`
2. `sin(θ₁)`
3. `cos(θ₂)`
4. `sin(θ₂)`
5. Winkelgeschwindigkeit `θ̇₁`, Bereich ca. `[-12,567; 12,567]` rad/s
6. Winkelgeschwindigkeit `θ̇₂`, Bereich ca. `[-28,274; 28,274]` rad/s

`θ₁` ist der Winkel des ersten Gelenks (`0` = Arm hängt senkrecht nach unten),
`θ₂` der Winkel des zweiten Gelenks relativ zum ersten.

Der originale Reward beträgt in jedem Schritt `-1`, bei Zielerreichung `0`.
Eine Episode terminiert, sobald das freie Ende der Kette die Zielhöhe
erreicht (`-cos(θ₁) - cos(θ₂ + θ₁) > 1,0`), und wird nach 500 Schritten
trunkiert. Als Erfolg gilt ausschließlich das Erreichen der Zielhöhe vor dem
Zeitlimit. Ein höherer, also weniger negativer Reward ist besser; Gymnasium
nennt `-100` als `reward_threshold` für eine als gelöst geltende Performance.
Der Episoden-Return liegt daher stets im Bereich `[-500; 0]`.

Quelle: [Gymnasium Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/)

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

## Noisy-Baustein

Ersetzt die linearen Schichten des Q-Netzes durch trainierbare faktorisierte
NoisyNet-Schichten. Das Rauschen wird während des Trainings passend neu
gezogen; die deterministische Evaluation arbeitet ohne Rauschen. NoisyNet
ersetzt die ε-greedy-Exploration in jeder Variante, die diesen Baustein nutzt
(`Noisy DDQN`, `Rainbow DDQN`). Setze dort die effektive ε-Exploration auf null
und deaktiviere die zugehörigen Eingaben verständlich, statt beide
Explorationsarten unbemerkt zu mischen.

Zusätzlich in der UI:

- initiale Noisy-Standardabweichung `σ₀`, Standardwert `0,5`

Quelle: [Fortunato et al., Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)

## PER-Baustein

Implementiert proportional Prioritized Experience Replay. Neue Transitionen
erhalten zunächst die höchste vorhandene Priorität. Sampling erfolgt
proportional zu `priority ** α`. Korrigiere den Sampling-Bias mit
normalisierten Importance-Sampling-Gewichten; `β` steigt während des
Trainings von `β₀` bis 1. Nach jedem Lernupdate werden die Prioritäten aus
den absoluten TD-Fehlern (bei `C51 DDQN`/`Rainbow DDQN` aus dem
Kreuzentropie-Verlust je Sample) aktualisiert. Dieser Baustein wird in
`PER DDQN` und `Rainbow DDQN` verwendet.

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
C51-Baustein bildet dieselbe Formel je Atom der Rückgabeverteilung. Teste
Form, Zentrierung und Gradientenfluss der Zusammenführung.

Quelle: [Wang et al., Dueling Network Architectures](https://arxiv.org/abs/1511.06581)

## Multi-Step-Baustein

Ersetzt den 1-Schritt-TD-Fehler durch einen n-Schritt-Return. Statt einer
einzelnen Transition wird über `n` aufeinanderfolgende Schritte akkumuliert:

```text
R_t^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k}
y = R_t^(n) + gamma^n * (1 - done) * Q_target(s_{t+n}, a*)
```

Endet eine Episode innerhalb der n Schritte, verkürzt sich der Return
entsprechend auf die tatsächlich verbleibenden Schritte; es wird nicht über
das Episodenende hinweg akkumuliert. Truncation-Fälle bootstrappen dabei
konsistent zu den übrigen Bausteinen dieses Projekts. Implementiere dies als
eigenständige, unabhängig testbare Replay-Buffer-Erweiterung (kein Eingriff in
die 1-Schritt-Logik der übrigen Bausteine), die sich mit dem PER-Baustein
kombinieren lässt, ohne dessen Prioritäten-Buchhaltung zu verändern. Dieser
Baustein wird in `Multi-Step DDQN` und `Rainbow DDQN` verwendet.

Zusätzlich in der UI:

- Schrittanzahl `n_step`, Standardwert `3`

Teste die Return- und Discount-Berechnung für vollständige n-Schritt-Fenster,
für durch Episodenende verkürzte Fenster und das Zusammenspiel mit PER
unabhängig von der GUI.

Quelle: [Sutton, Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009)

## C51-Baustein (Distributional RL)

Ersetzt den skalaren Q-Wert durch eine kategoriale Verteilung über feste
Rückgabewerte (`Atome`) `z_i` gleichmäßig verteilt in `[V_min, V_max]`. Das
Netz gibt je Action eine Wahrscheinlichkeit pro Atom aus (Softmax über die
Atome); der Q-Wert ergibt sich als Erwartungswert `Q(s,a) = Σ_i p_i(s,a) · z_i`
und wird für Action-Auswahl und `predict()` verwendet. Die Zielverteilung
entsteht durch Anwenden der Bellman-Gleichung auf die Atome und anschließende
Projektion auf das feste Atom-Gitter (kategoriale Projektion nach Bellemare et
al.). Die Double-DQN-Regel gilt weiterhin: Das Online-Netz wählt `a*` über den
aus seiner eigenen Verteilung gebildeten Erwartungswert, das Target-Netz
liefert die zu projizierende Verteilung für genau dieses `a*`. Trainiert wird
mit Kreuzentropie zwischen projizierter Zielverteilung und aktueller
Verteilung. Dieser Baustein wird in `C51 DDQN` und `Rainbow DDQN` verwendet.

Zusätzlich in der UI:

- Anzahl Atome `n_atoms`, Standardwert `51`
- `V_min`, Standardwert `-500` (minimal möglicher Episoden-Return)
- `V_max`, Standardwert `0` (maximal möglicher Episoden-Return)

Teste, dass projizierte Zielverteilungen normiert bleiben (Wahrscheinlichkeiten
summieren je Sample zu `1`), dass exakt auf ein Atom fallende Zielwerte ihre
Masse korrekt an genau diesem Atom sammeln (kein Wahrscheinlichkeitsverlust bei
Ganzzahl-Trefferindex) und dass die Double-DQN-Actionauswahl weiterhin über das
Online-Netz erfolgt.

Quelle: [Bellemare et al., A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)

## Rainbow DDQN

`Rainbow DDQN` kombiniert Noisy-, PER-, Dueling-, Multi-Step- und
C51-Baustein gleichzeitig in einem einzigen Agenten, ohne deren jeweilige
Implementierung zu verändern oder zu duplizieren. Alle zugehörigen
UI-Parameter (`σ₀`, `α_PER`, `β₀`, Beta-Annealing, `ε_PER`, Stream-Größen,
`n_step`, `n_atoms`, `V_min`, `V_max`) erscheinen auch im Rainbow-Modus. Teste
zusätzlich zu den Einzelbaustein-Tests, dass alle fünf Bausteine gleichzeitig
aktiv sind und sich das Zusammenspiel korrekt in Netzarchitektur, Sampling,
Zielverteilungsberechnung und Prioritäten-Update niederschlägt.

## Standardprofil

Nutze als gemeinsame Ausgangswerte das getunte Acrobot-v1-DQN-Profil des
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
Stable-Baselines3-Version oder den oben genannten Originalarbeiten. Dokumentiere
Quelle und jede begründete Abweichung in der README.

Quelle: [RL Baselines3 Zoo, DQN-Hyperparameter](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml)

## Training und Evaluation

Training nutzt einen Callback für Abbruch, Fortschritt und Episodenmetriken.
Während Training und Vergleich finden keine automatischen
Zwischen-Evaluationen statt. Eine ausdrücklich gestartete Evaluation verwendet
ausschließlich die deterministische Policy und verändert weder Modell,
Optimizer, Replay Buffer noch Trainingsstatistiken. Bei allen Noisy-basierten
Varianten ist das Rauschen dabei deaktiviert.

Die Anzahl der manuell gestarteten Evaluations-Episoden ist in der UI
einstellbar. Zeige im Evaluationsergebnis mindestens:

- mittleren Reward und Standardabweichung
- mittlere Episodenlänge
- Erfolgsrate als Anteil der Episoden mit Zielerreichung vor dem Zeitlimit
- letzte und beste deterministische Evaluation

Merke beim Einzeltraining den besten deterministischen Evaluationszustand und
sichere bei Verbesserung Modell, Target-Netz, Optimizer-Zustand und passenden
Replay Buffer. Der Button `Bestes Modell wiederherstellen` stellt den gesamten
Lernzustand wieder her.

## Vergleich

Alle sieben Algorithmen sind einzeln für den gemeinsamen Vergleich auswählbar;
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
nennt die sieben Algorithmusnamen sowie eine sinnvolle Acrobot-Referenzlinie,
falls eine fachlich begründete feste Referenz verwendet wird.

Die Vergleichs-Summary ist eine kompakte, live aktualisierte Tabelle, in der
alle sieben Algorithmen gleichzeitig sichtbar sind. Sie enthält Algorithmen als
Spalten und diese Zeilen:

- Episoden
- Environment-Schritte
- durchschnittlicher Reward
- Trainingserfolgsrate

## Darstellung

Zeige ausschließlich den offiziellen, von `env.render()` gelieferten
Gymnasium-RGB-Frame. Erstelle keine eigene Acrobot-Grafik und öffne kein
separates Pygame-Fenster. Beachte die macOS-Prozesstrennung aus der Workbench.

Zeige neben der Animation:

- aktuelle Winkel `θ₁` und `θ₂` (aus `cos`/`sin` zurückgerechnet, in Grad)
- aktuelle Winkelgeschwindigkeiten `θ̇₁` und `θ̇₂`
- gewählte Action als `Drehmoment −1`, `kein Drehmoment` oder `Drehmoment +1`
- aktuellen Episodenschritt
- bisher kumulierten Reward

Acrobot-spezifische Metriken sind Reward, Episodenlänge, Erfolgsrate und
Schritte bis zur Zielerreichung. Der Reward ist negativ; Achsen, Summary und
Hilfetexte erklären eindeutig, dass Werte näher an null besser sind.

## Speichern und Laden

Gespeicherte Modelle müssen später mit derselben Variante weitertrainiert
werden können. Speichere Modell, Target-Netz, Optimizer, Replay Buffer,
Algorithmusvariante, vollständige Konfiguration und Metadaten. Für PER-basierte
Varianten gehören Prioritäten, aktueller `β`-Fortschritt und Buffer-Zeiger zum
gespeicherten Zustand. Für Multi-Step-basierte Varianten gehört der
unvollständige n-Schritt-Fensterzustand nicht zum gespeicherten Zustand (nach
dem Laden beginnt das Fenster leer). Für Noisy-, Dueling- und C51-basierte
Varianten muss die exakte Netzarchitektur inklusive Atom-Gitter
rekonstruierbar sein. Lehne inkompatible oder unvollständige Dateien
verständlich ab.

## Abhängigkeiten

Verwende PyTorch, Gymnasium und Stable-Baselines3 aus der gemeinsamen
Kursumgebung. Führe keine zusätzliche Deep-Learning-Bibliothek ein. Dokumentiere
direkte Abhängigkeiten in `requirements.txt` und bei Bedarf in
`../environment.yml`.

## Abnahme

Fertig, wenn alle sieben Varianten fachlich korrekt trainieren, deterministisch
evaluiert und fair verglichen werden; algorithmusspezifische Parameter nur in
den passenden Modi erscheinen; NoisyNet ohne zusätzliches epsilon-greedy
arbeitet; PER tatsächlich priorisiert sampelt und gewichtet; Dueling Value und
Advantage korrekt zusammenführt; Multi-Step n-Schritt-Returns korrekt bildet
und an Episodengrenzen korrekt verkürzt; C51 normierte, korrekt projizierte
Zielverteilungen erzeugt und über den Erwartungswert konsistent mit Double-DQN
handelt; `Rainbow DDQN` alle fünf Erweiterungsbausteine gleichzeitig und
korrekt kombiniert; beste Lernzustände vollständig wiederherstellbar sind; die
offizielle Gymnasium-Animation eingebettet ist; und alle allgemeinen Tests,
Sichtbarkeitsprüfungen und Abnahmekriterien aus `../workbench.md` erfüllt sind.
