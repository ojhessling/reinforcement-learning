# Walker2d – RL-Workbench

Projektordner: `Oliver/15-walker2d`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `walker2d`
Environment: `Walker2d-v5`

Dieser Prompt enthält nur die sieben Punkte aus Workbench 1.1 (Was in den
Projekt-Prompt gehört). Alles Allgemeine steht dort und wird hier nicht
wiederholt. Keine Abweichungen von der Workbench.

## Ziel

Tkinter-Lernanwendung, die einen zweibeinigen Roboter im Seitenprofil aufrecht
laufen lässt. Drei Verfahren, davon zwei bekannte und ein neues:

1. `TD3`
2. `SAC`
3. `CMA-ES`

**Das Neue an diesem Projekt** ist nicht das Environment, sondern die dritte
Verfahrensklasse. `CMA-ES` ist gradientenfrei und populationsbasiert: Es
optimiert die Policy-Gewichte direkt, kennt weder `learning_rate` noch
`batch_size` noch `gamma`, bewertet Kandidaten über ganze Episoden und wertet
nur deren Rangfolge aus. Fachlicher Kern, UI-Parameter und Quellen stehen in
Workbench 5.5 (CMA-ES); dieser Prompt legt nur das environmentspezifische
Profil fest.

`PPO` ist bewusst **nicht** dabei. Dadurch stehen sich hier genau zwei Klassen
gegenüber – zwei gradientenbasierte Off-Policy-Verfahren gegen ein
gradientenfreies –, und der Vergleich hat nur **eine** systematische
Asymmetrie statt zwei. Die On-Policy-Benachteiligung aus Workbench 5.9
(Fairness zwischen Verfahrensklassen) entfällt; was bleibt, ist der Unterschied
in der Informationsmenge je Schritt, und der ist damit sauber ablesbar.

## Startbelegung und globale Einstellungen

- `Anzahl Verfahren` = `3`: `V1 = TD3`, `V2 = SAC`, `V3 = CMA-ES`. Jeder
  Algorithmus kommt in der Startbelegung genau einmal vor; die Regel aus
  Workbench 6.2 (Verfahrenswahl und Vergleichstabs) zum abweichenden Startwert
  bei doppelter Belegung greift deshalb nicht. Wählt der Benutzer einen vierten
  Slot, muss sie greifen, weil dann zwangsläufig ein Algorithmus doppelt
  vorkommt.
- `Eval-Intervall = 100.000` (ein Zehntel des Budgets, zehn Stützstellen),
  `Eval-Episoden M = 5`, `Glättung = 20`.

## Environment

```python
import gymnasium

env = gymnasium.make(
    "Walker2d-v5",
    render_mode="rgb_array",   # entfällt bei headless Evaluation
    width=480,
    height=480,
)
```

- `forward_reward_weight`, `ctrl_cost_weight`, `healthy_reward`,
  `terminate_when_unhealthy`, `healthy_z_range`, `healthy_angle_range`,
  `reset_noise_scale` und `exclude_current_positions_from_observation` werden
  **nicht** übergeben und nicht verändert – das wäre Reward Shaping bzw. eine
  Änderung der Abbruchregeln.
- `width`/`height` entsprechen den Gymnasium-Voreinstellungen, werden aber
  mitgegeben, damit die Framegröße eindeutig ist. Die Kamera folgt dem Roboter.
- `v5` statt `v4`: gepflegte Fassung mit dokumentierter
  `observation_structure`. Die Zoo-Profile sind für `v4` hinterlegt.

### Actions

`Box(-1.0, 1.0, (6,), float32)`. Alle sechs Motoren haben dieselbe Übersetzung
`gear = 100` – anders als bei HalfCheetah, wo sie sich unterscheiden:

| Index | Gelenk (XML) |
| --- | --- |
| `a₀` | rechter Oberschenkel (`thigh_joint`) |
| `a₁` | rechter Unterschenkel (`leg_joint`) |
| `a₂` | rechter Fuß (`foot_joint`) |
| `a₃` | linker Oberschenkel (`thigh_left_joint`) |
| `a₄` | linker Unterschenkel (`leg_left_joint`) |
| `a₅` | linker Fuß (`foot_left_joint`) |

Vorzeichen = Drehrichtung, Betrag mal `100` = Moment in N·m. `0` heißt „kein
Moment". Werte außerhalb `[-1, 1]` clippt das Environment. Die Übersetzung wird
gegen `model.actuator_gear` getestet; die Aktuatoren tragen im XML **keine**
Namen, die Zuordnung erfolgt daher über die Reihenfolge der Gelenke.

### Observation

17 Werte, `float64`, `Box(-inf, inf, (17,))`:

| Index | Bedeutung | Einheit |
| --- | --- | --- |
| 0 | Höhe des Rumpfes (`rootz`) | m |
| 1 | Neigungswinkel des Rumpfes (`rooty`) | rad |
| 2–7 | Winkel der sechs Gelenke, Reihenfolge wie bei den Actions | rad |
| 8–9 | Geschwindigkeiten `vₓ`, `v_z` | m/s |
| 10 | Winkelgeschwindigkeit des Rumpfes | rad/s |
| 11–16 | Winkelgeschwindigkeiten der sechs Gelenke | rad/s |

- Die **x-Position** fehlt; für Anzeige und Metriken steht sie in
  `info["x_position"]`.
- Die Geschwindigkeiten sind auf `[-10, 10]` **geclippt** – wie beim Hopper und
  anders als bei HalfCheetah. Ein Wert von genau `±10` heißt „mindestens so
  schnell". Entsprechend beschriften.

### Reward und Episodenende

```text
reward = healthy_reward + forward_reward − ctrl_cost
       = 1,0            + 1,0 · vₓ       − 0,001 · Σ aᵢ²
```

- `healthy_reward = 1,0` je gesundem Schritt; über eine volle Episode `+1000`.
- `forward_reward = 1,0 · vₓ` mit `vₓ = Δx / dt`, `dt = 0,008 s`.
- `ctrl_cost = 0,001 · Σ aᵢ²`, höchstens `0,006` je Schritt – vernachlässigbar
  klein, anders als bei HalfCheetah.

Die drei Anteile stehen in `info` als `reward_survive`, `reward_forward` und
`reward_ctrl` (bereits negativ) und werden von dort übernommen.

Die Episode endet:

- mit `terminated`, sobald der Roboter **ungesund** wird: Höhe außerhalb
  `(0,8; 2,0) m` oder Rumpfwinkel außerhalb `(−1,0; 1,0) rad`. Umgangssprachlich
  ein Sturz.
- mit `truncated` nach `1000` Schritten.

Es gibt **weder Terminalbonus noch Terminalstrafe**: Ein Sturz kostet nur die
Rewards der Schritte, die nicht mehr stattfinden. Der gesunde Höhenbereich ist
nach **oben** begrenzt – ein Sprung über 2,0 m beendet die Episode ebenfalls.

### Erfolgsdefinitionen

`Walker2d-v5` führt in der Gymnasium-Registry **keinen** `reward_threshold`.
Eine offizielle Gelöst-Schwelle existiert also nicht. Gemäß Workbench 8.1
(Metrikwahl) setzt dieses Projekt deshalb eine **projektinterne Referenzmarke**
und nennt sie ausdrücklich nicht „gelöst":

- **Zielmarke `4000`**, begründet durch die Benchmarkwerte des RL Baselines3
  Zoo (siehe unten, 3479 bis 4718). Die Metrik heißt **Zielquote**, nicht
  Gelöst-Quote, und README wie Bedienungsanleitung sagen in einem Satz, dass
  die Marke projektintern gesetzt ist.
- **Durchgehalten**: Episode endet mit `truncated` nach 1000 Schritten.
- **Sturz**: Episode endet mit `terminated`.

Durchhalte- und Sturzquote sind komplementär, werden aber getrennt
ausgewiesen. Höhere Returns sind besser; der Return liegt realistisch zwischen
etwa `+5` (sofortiger Sturz) und `+5000`.

Quelle: [Gymnasium Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/)

## Standardprofile

`Walker2d-v4`-Profile des RL Baselines3 Zoo, Stand geprüft am 21.08.2026; nicht
enthaltene Werte sind SB3-Defaults der Version aus `../environment.yml`.

`SAC` – der Zoo setzt nur `learning_starts`, alles Weitere SB3-Default:

- `N = 1.000.000`, `learning_starts = 10.000`, `buffer_size = 1.000.000`
- `learning_rate = 3e-4` konstant, `gamma = 0,99`, `tau = 0,005`
- `batch_size = 256`, `train_freq = 1`, `gradient_steps = 1`
- `ent_coef = auto`, `target_entropy = auto`, `target_update_interval = 1`
- Netz `256,256`, Aktivierung `ReLU`

`TD3`:

- `N = 1.000.000`, `learning_rate = 1e-3` konstant, `gamma = 0,99`,
  `tau = 0,005`
- `learning_starts = 10.000`, `batch_size = 256`, `buffer_size = 1.000.000`
- `train_freq = 1`, `gradient_steps = 1`, Action Noise `normal` mit `σ = 0,1`
- Netz `400,300` gemäß Profil, Aktivierung `ReLU`
- übrige SB3-Defaults, insbesondere `policy_delay = 2`,
  `target_policy_noise = 0,2`, `target_noise_clip = 0,5`

`CMA-ES` – für dieses Verfahren gibt es kein Zoo-Profil; die Werte folgen den
Voreinstellungen von `pycma` und der Literatur:

- `N = 1.000.000` Environment-Schritte, identisch zu den beiden anderen
- Policy **linear**: `a = tanh(W·s + b)` ohne Hidden Layer, also
  `6 · 17 + 6 = 108` Parameter. Hidden Layers bleiben in der UI änderbar,
  Standard ist die leere Liste.
- `sigma0 = 0,5`
- `popsize λ = auto`, was `4 + ⌊3·ln 108⌋ = 18` ergibt
- Episoden je Kandidat: `1`
- Diagonalvariante: aus
- Beobachtungen normalisieren: **an**. Ohne laufende Beobachtungsstatistik
  arbeitet eine lineare Policy auf den sehr unterschiedlich skalierten
  17 Werten kaum. Rewards normalisieren entfällt ersatzlos – siehe
  Workbench 5.7 (Normalisierung).

**Normalisierung insgesamt:** Anders als in `13-hopper` und `14-halfcheetah`
verlangt hier **kein** Zoo-Profil `normalize: true` – das wäre das PPO-Profil
gewesen, und PPO ist nicht dabei. `SAC` und `TD3` normalisieren deshalb nicht,
`CMA-ES` normalisiert die Beobachtungen aus eigenem Bedarf. Die Gruppe
`Normalisierung` erscheint trotzdem in jedem Tab, weil die 17 Werte sehr
unterschiedlich skaliert sind; bei `CMA-ES` ohne das Feld für Rewards.

Zur Netzgröße von `CMA-ES` – die Begründung nach Workbench 4.1 (Neuronale
Netze): Die Kovarianzmatrix hat `n × n` Einträge. Bei `256,256` wären das rund
`70.000` Parameter, also über `4,9 · 10⁹` Matrixeinträge und mehr als 40 GB,
dazu eine Eigenzerlegung in `O(n³)`. Mit `108` Parametern sind es `11.664`
Einträge. Dass eine lineare Policy für MuJoCo-Laufaufgaben ausreicht, zeigen
Rajeswaran et al.; eine unterlegene CMA-ES-Kurve ist deshalb **nicht** auf ein
zu kleines Netz zu schieben, sondern auf die Informationsmenge je Schritt.

### CMA-ES im Ablauf

Mehrere Workbench-Regeln setzen stillschweigend ein gradientenbasiertes
Verfahren voraus. Für `CMA-ES` gilt deshalb ausdrücklich:

- **Was die Animation zeigt.** Workbench 7.1 (Steuerung und Inhalt) verlangt
  den „aktuellen Lernstand". Das ist hier der **Verteilungsmittelwert** `m`,
  derselbe Stand, den auch die deterministische Evaluation nutzt – nicht ein
  gezogener Kandidat. Ein Kandidat wäre bei jeder Episode ein anderer und
  spränge sichtbar hin und her, ohne den Lernfortschritt zu zeigen.
- **Wann die Beobachtungsstatistik wächst.** Sie sammelt während der
  Kandidatenbewertungen, wird aber erst **nach** Abschluss einer Generation
  fortgeschrieben. Innerhalb einer Generation sehen damit alle Kandidaten
  dieselbe Normalisierung – sonst wäre ihre Rangfolge verfälscht. In Evaluation
  und Animation bleibt sie eingefroren. Clip-Wert wie bei den übrigen
  Verfahren `10,0`.
- **Mehrere Episoden je Kandidat.** Die Fitness ist der **Mittelwert** der
  Renditen; jede einzelne Episode ist trotzdem ein eigener Kurvenpunkt, denn
  sie hat tatsächlich stattgefunden.
- **Budgetende mitten in einer Generation.** Die bereits gelaufenen Episoden
  bleiben als Kurvenpunkte erhalten, aber die unvollständige Generation wird
  **nicht** an den Optimierer gemeldet: Ohne vollständige Population gibt es
  kein Update. Die Zeile `Generationen` zählt nur abgeschlossene Generationen.
- **Zeitpunkt der Zwischenevaluation.** Das Schrittintervall lässt sich nicht
  mitten in einer Episode prüfen. Geprüft wird nach jeder abgeschlossenen
  Kandidatenbewertung; die Stützstellen liegen dadurch etwas hinter dem
  nominellen Intervall.
- **Was der Checkpoint der besten Episode enthält.** Den Parametervektor, der
  diese Episode erzeugt hat – bei 108 Werten ein kostenloser Schnappschuss. Der
  reguläre Checkpoint des besten Evaluationsergebnisses enthält dagegen die
  vollständige Suchverteilung, siehe Workbench 5.8 (Gespeicherter Zustand).

Die Zeile `Generationen` erscheint auch in der Trainings-Summary, nicht nur im
Vergleich.

Die Netzgrößen von `SAC` (`256,256`) und `TD3` (`400,300`) werden in diesem
Projekt **nicht** vereinheitlicht, anders als in `13-hopper` und
`14-halfcheetah`. Dort waren alle Verfahren derselben Klasse, eine gemeinsame
Architektur also erreichbar und aussagekräftig. Hier ist sie schon durch
`CMA-ES` ausgeschlossen: Dessen Policy muss um Größenordnungen kleiner sein.
Ein getuntes Profil zu überschreiben brächte deshalb nichts. Fairness bemisst
sich nach dem Schrittbudget – siehe Workbench 4.1 (Neuronale Netze).

**Laufzeit:** `1e6` Schritte dauern auf einem Laptop Stunden, bei drei Slots
parallel länger. Als Einstieg `100.000` Schritte mit `Eval-Intervall = 10.000` nennen.
Bei `CMA-ES` ist zusätzlich zu bedenken, dass eine Generation `popsize`
Episoden kostet; mit `λ = 18` und früh kurzen Episoden entstehen anfangs viele
Generationen, später deutlich weniger.

**Erwartung:** Zoo-Benchmark für `Walker2d` nach `1e6` Schritten:

| Verfahren | Return | Std |
| --- | --- | --- |
| `SAC` | 3863,2 | 254,3 |
| `TD3` | 4717,8 | 46,3 |

Für `CMA-ES` führt der Zoo keine Werte; es ist bei diesem Budget ein deutlicher
Rückstand zu erwarten, weil jede Episode nur eine einzige Zahl beiträgt. Nenne
in der README keine erfundene Vergleichszahl, sondern beschreibe den
Mechanismus.

Quellen: [benchmark.md](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md),
[ppo.yml](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml),
[sac.yml](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml),
[td3.yml](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/td3.yml)

## Metriken

Episoden-Return, Episodenlänge, Durchhaltequote, Sturzquote, Zielquote,
mittleres Tempo `vₓ` und zurückgelegte Strecke. Tempo und Strecke aus
`info["x_velocity"]` und `info["x_position"]`.

Vergleichs-Summary je Slot: Algorithmus · Episoden · Environment-Schritte
ausgeführt und angefordert · Ø Return · beste Episode mit Nummer und Return ·
Ø Länge · Ø Tempo in m/s · Ø Strecke in m · Durchhaltequote · Sturzquote ·
Zielquote.

Bei `CMA-ES` kommt eine Zeile **Generationen** hinzu; bei den drei anderen
Verfahren steht dort `—`. Sie ist die einzige Kennzahl, die nicht alle
Verfahren besitzen, und macht sichtbar, wie wenige Update-Schritte hinter einer
CMA-ES-Kurve stehen.

Evaluationsergebnis: dieselben Größen plus Standardabweichung des Returns sowie
letzte und beste deterministische Evaluation.

Referenzlinie der Graphen: `+4000`, weiß und gestrichelt, in der Legende
ausdrücklich als **Zielmarke** und nicht als Gelöst-Schwelle bezeichnet.

## Einblendung neben der Animation

- die sechs Rohwerte `a₀` bis `a₅` sowie Drehrichtung und Moment in N·m
  (`aᵢ · 100`)
- Höhe des Rumpfes mit dem gesunden Bereich `0,8` bis `2,0` m
- Rumpfwinkel mit dem gesunden Bereich `±1,0` rad
- die sechs Gelenkwinkel, nach rechtem und linkem Bein getrennt
- `vₓ`, `v_z` und Winkelgeschwindigkeit des Rumpfes mit dem Hinweis auf die
  Clippung bei `±10`
- Zerlegung des letzten Rewards in `reward_survive`, `reward_forward` und
  `reward_ctrl`
- Strecke aus `info["x_position"]`

**Bildrate:** environment-eigen `125` FPS, eine volle Episode dauert damit acht
Sekunden.

## Tests

Zusätzlich zu Workbench 5.10 (Verfahrensbezogene Tests) und 9.2 (Tests):

- Environment-Kennwerte: Action `Box(-1, 1, (6,))`, Observation `(17,)`
  `float64`, `max_episode_steps = 1000`, `render_fps = 125`, `dt = 0,008`,
  Frame `480 × 480`; die Factory fordert genau diese Argumente an und übergibt
  **keinen** Physik-, Reward- oder Abbruchparameter
- `Walker2d-v5` führt **keinen** `reward_threshold`; die Zielmarke `4000` ist
  eine Projektkonstante und wird nicht als offizielle Schwelle ausgegeben
- alle sechs Übersetzungen sind `100` und stimmen mit `model.actuator_gear`
  überein
- Reward-Zerlegung: `reward_survive + reward_forward + reward_ctrl` aus `info`
  ergibt exakt den Reward; Steuerkosten `0,001 · Σ aᵢ²`
- Episodenausgänge: Sturz als `terminated` **ohne** Terminalstrafe, Durchhalten
  als `truncated` nach genau 1000 Schritten; der gesunde Bereich ist nach oben
  **und** unten begrenzt
- die Geschwindigkeiten in der Observation sind auf `±10` geclippt
- x-Position stammt aus `info["x_position"]`, nicht aus der Observation
- SAC-Zielentropie bei `auto` genau `-6`
- `CMA-ES`: lineare Policy hat genau `108` Parameter; `popsize auto` ergibt
  `18`; eine Generation verändert Mittelwert und Schrittweite; die
  deterministische Evaluation nutzt den Verteilungsmittelwert und liefert bei
  gleichem Seed zweimal dasselbe Ergebnis; das Schrittbudget wird in
  Environment-Schritten eingehalten; ein Save-/Load-Zyklus setzt die Suche mit
  identischer Verteilung fort; ohne `learning_rate`, `batch_size`, `gamma`,
  `buffer_size` und Action-Noise-Feldern in der Konfiguration
- die Beobachtungsstatistik von `CMA-ES` wächst im Training, bleibt in der
  Evaluation eingefroren und gehört zum Checkpoint
- Ablehnung eines Speicherstands mit Kennung `Walker2d-v4`, `Hopper-v5` oder
  `HalfCheetah-v5`

Bei kurzen Testläufen für `CMA-ES` beachten: Das Budget muss mindestens eine
vollständige Generation zulassen, sonst entsteht kein einziger Update-Schritt.
Mit kleinem `popsize` und einem gestürzten Agenten sind das wenige Hundert
Schritte.

## Abhängigkeiten

Neu ist **`cma`** (pycma) als Referenzimplementierung für CMA-ES; einzutragen
in `../environment.yml` und in die `requirements.txt` dieses Projekts. MuJoCo
und `imageio` stehen bereits dort. Alle Dateien tragen das Präfix `walker2d_`
bzw. `test_walker2d_`.

## Abnahme

Zusätzlich zu Workbench 9.3 (Abnahme): alle drei Verfahren trainieren,
evaluieren und vergleichen mit den obigen Profilen; der `CMA-ES`-Tab enthält **keine**
Felder für `learning_rate`, `batch_size`, `gamma`, Replay Buffer oder Action
Noise, dafür `sigma0`, `popsize` und Episoden je Kandidat; die Zielmarke `4000`
wird überall als projektintern und nicht als offizielle Gelöst-Schwelle
ausgewiesen; die sechs Übersetzungen von `100` werden gegen das Environment
geprüft; die Geschwindigkeitsclippung bei `±10` ist dokumentiert; Durchhalte-,
Sturz- und Zielquote werden getrennt ausgewiesen und die Zeile `Generationen`
nur für `CMA-ES` gefüllt; ein Speicherstand fremder Environment-Kennung wird
abgelehnt.
