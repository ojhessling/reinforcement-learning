# Humanoid – RL-Workbench (Abschlussprojekt)

Projektordner: `Oliver/Projekt`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `humanoid`
Environment: `Humanoid-v5`

Dieser Prompt enthält die sieben Punkte aus Workbench 1.1 (Was in den
Projekt-Prompt gehört). Zusätzlich – und nur, weil dieses Projekt eine
benotete Kursabgabe ist – die Abschnitte **Abweichungen**, **Experimente für
den Bericht** und **Abgabe**. Alles Allgemeine steht in der Workbench und wird
hier nicht wiederholt.

## Ziel

Abschlussprojekt der Projektwoche, Kurs D21195UYS, AlfaTraining, Dozent
Manfred Messing. Zugeteilt laut Aufgabenstellung: **`Humanoid-v5` mit PPO, TD3
und SAC**.

Tkinter-Lernanwendung, die eine dreidimensionale humanoide Figur von 42 kg
aufrecht halten und vorwärts laufen lassen soll. Drei Verfahren:

1. `PPO`
2. `TD3`
3. `SAC`

**Das Neue an diesem Projekt** ist weder ein Verfahren noch eine
Verfahrensklasse – alle drei sind aus 13-hopper und 14-halfcheetah bekannt –,
sondern der Sprung in der Schwierigkeit des Environments:

| | Hopper | HalfCheetah | Walker2d | **Humanoid** |
| --- | --- | --- | --- | --- |
| Observation | 11 | 17 | 17 | **348** |
| Actions | 3 | 6 | 6 | **17** |
| Raum | 2D | 2D | 2D | **3D** |
| Rewardanteile | 3 | 2 | 3 | **4** |

Daraus folgen drei Dinge, die dieses Projekt von den Vorgängern unterscheiden
und an den jeweiligen Stellen unten geregelt sind: Die Einblendung kann die
Observation nicht mehr vollständig zeigen (siehe *Einblendung*), die Figur kann
seitlich abdriften statt nur vorwärts (siehe *Metriken*), und die Rechenzeit
zwingt zu einem Budget unterhalb des Profilwerts (siehe *Abweichungen*).

## Abweichungen von der Workbench

Drei begründete Abweichungen; sonst keine. Das doppelte Budget aus
Episoden- und Schrittgrenze ist **keine** Abweichung mehr – es steht seit
der Anpassung für dieses Projekt in Workbench 5.1 (Gemeinsame Parameter).

### 1 Ordnerlayout und Dateinamen

Alle Dateien liegen direkt in `Oliver/Projekt`, **ohne** Nummernpräfix im
Ordnernamen, abweichend vom Muster `NN-name` der Projekte 00 bis 15. Grund: Die
Aufgabenstellung schreibt diesen Ordner als Abgabeordner vor („Unterordner
*Projekt* unterhalb des eigenen Namens"), samt Bericht und allen Bildern.
Code und Abgabe in zwei Ordner zu trennen würde bedeuten, dass Bildpfade aus
dem Abgabeordner herauszeigen.

Dateipräfix bleibt beim Muster: `humanoid_logic.py`, `humanoid_gui.py`,
`humanoid_render.py`, `humanoid_app.py`, Tests `test_humanoid_*.py`.

### 2 Thread-Zahl von PyTorch

`torch.set_num_threads` wird beim Start gesetzt und ist global einstellbar
(Default `4`). Grund: gemessen auf der Zielmaschine (6 Kerne, davon 2
Performance-Kerne), SAC auf `Humanoid-v5`:

| Threads | Schritte/s | 500.000 Schritte |
| --- | --- | --- |
| Standard (6) | 16 | 8,7 h |
| 1 | 27 | 5,1 h |
| 2 | 46 | 3,0 h |
| **4** | **49,5** | **2,8 h** |

`torch.set_num_threads` bestimmt, über wie viele Kerne PyTorch **eine
einzelne** Matrixmultiplikation verteilt – nicht, wie viele Verfahren
gleichzeitig laufen. Im Standard nimmt PyTorch alle sechs Kerne; an jedem
Synchronisationspunkt warten dann die zwei schnellen Threads auf die vier
langsamen. Das ist eine Optimierung nach Messung im Sinne von Workbench 9.1
(Fehlerbehandlung und Performance).

Die Einstellung ist **prozessweit**: Ein Vergleich lässt seine Slots als
Threads **eines** Prozesses laufen, alle teilen sich denselben Thread-Pool. Der
Wert gilt also für die ganze Anwendung, nicht je Lauf, und wird beim Start
eines Laufs übernommen – eine Änderung während eines laufenden Trainings wirkt
nicht mehr auf dieses.

### 3 Speicherbedarf des Replay Buffers

`buffer_size` wird bei `TD3` und `SAC` auf das Schrittbudget gesetzt
(`500.000` statt `1e6`), und die Observation wird als `float32` gepuffert.

Grund: Die Zielmaschine hat **8 GB RAM**. SB3 legt den Replay Buffer beim
Erzeugen vollständig an und speichert bei `optimize_memory_usage = False`
sowohl `observations` als auch `next_observations`. Bei 348 Werten in
`float64` sind das gemessen:

| `buffer_size` | dtype | nur Observations |
| --- | --- | --- |
| `1e6` (Profil) | `float64` | **5,19 GB** |
| `500.000` | `float64` | 2,60 GB |
| `500.000` | `float32` | **1,30 GB** |

- Die Verkleinerung auf `500.000` ist **verhaltensneutral**: Bei einem Budget
  von 500.000 Schritten enthält der Buffer nie mehr als 500.000 Übergänge, es
  wird also in keiner Variante je ein Übergang verdrängt. Der Inhalt ist
  identisch, nur die Allokation kleiner.
- `float32` ist eine echte, aber folgenlose Abweichung: SB3 wandelt beim
  Sampeln ohnehin in `float32`-Tensoren um, bevor irgendetwas das Netz
  erreicht. Die zusätzliche Genauigkeit von `float64` wird nirgends genutzt,
  sie kostet nur Speicher.
- `optimize_memory_usage = True` wäre die naheliegende Alternative, ist hier
  aber **unzulässig**: Sie verträgt sich nicht mit
  `handle_timeout_termination`, und Humanoid trunkiert nach 1000 Schritten.
  Ohne korrekte Timeout-Behandlung würde das Abschneiden der Episode als
  echtes Episodenende gelernt.

**Folge für parallele Läufe:** Drei Verfahren gleichzeitig mit dem Profilbuffer
wären 15,6 GB und auf dieser Maschine nicht lauffähig. Mit den obigen Werten
belegen die beiden Off-Policy-Slots zusammen 2,6 GB, dazu rund 1,5 GB für die
drei Prozesse – zusammen etwa 4 GB von 8 GB. Ein Durchgang mit drei Slots läuft
damit wie in Workbench 8.5 (Vergleich) vorgesehen **parallel**, und Vorgabe
1.2d ist erfüllt.

Zwei Einschränkungen: Bei parallelem Betrieb bekommt jeder Prozess `2` statt `4`
Threads, weil nur zwei Performance-Kerne vorhanden sind. Und es läuft immer nur
**ein** Durchgang gleichzeitig – Durchgang A und B nacheinander, nie
überlappend, sonst verdoppelt sich der Speicherbedarf.

## Budgetgrenzen

Workbench 5.1 (Gemeinsame Parameter) verlangt beide Grenzen je Slot und
verpflichtet den Prompt, für das konkrete Environment vorzurechnen, wie weit
sie auseinanderfallen. Für Humanoid:

Eine Episode endet beim Sturz oder nach 1000 Schritten. Untrainiert fällt die
Figur nach **im Mittel 24,5 Schritten** – gemessen über 200 Episoden einer
Zufallspolicy: Median 22, min 16, max 56. Theoretisch reicht die Spannweite
damit von ~24.500 Schritten (nie gelernt) bis 1.000.000 (läuft durch), also
Faktor 40.

Gemessen wurde zusätzlich, wie schnell die Episodenlänge tatsächlich wächst –
SAC mit Profilparametern über 60.000 Schritte:

| Episoden | Ø Länge | Ø Return |
| --- | --- | --- |
| 0–50 | 21,3 | 98,7 |
| 50–150 | 28,0 | 135,5 |
| 150–300 | 61,9 | 309,3 |
| 300–802 | 93,2 | 461,4 |

**In 60.000 Schritten entstehen bereits 802 Episoden.** Hochgerechnet werden die
vorgegebenen 1000 Episoden nach rund **80.000 Schritten** erreicht – etwa ein
Sechstel des größten Messbudgets (500.000), nach knapp einer halben Stunde. Ein Lauf mit
`Episoden = 1000` würde also bei einem Return um 500 abbrechen, während das
Verfahren noch deutlich steigt.

**Die Messläufe für den Bericht steuert deshalb das Schrittbudget**, mit
`Episoden = 0` und einem festen `total_timesteps` je Block (300.000 im
Verfahrensvergleich, 500.000 in der Parameterstudie, siehe *Experimente*) –
abweichend von der Startbelegung, die für einen ersten Eindruck gedacht ist. Der zweite Grund
ist die Fairness-Regel aus Workbench 5.1 (Gemeinsame Parameter): Bei
Humanoid beendet ein Sturz die Episode vorzeitig, Episoden werden mit dem
Lernfortschritt also länger. SAC sammelt bis Episode 1000 rund 80.000 Schritte;
ein Verfahren, das langsamer lernt und bei ~25 Schritten je Episode bleibt,
käme auf ~25.000. Dasselbe Episodenbudget hieße dann **dreifach ungleiche
Trainingsdaten** – der Vorsprung im Diagramm wäre teilweise nur ein Vorsprung an
Erfahrung.

Die Episodeneingabe bleibt bedienbar und steht wie gefordert auf `1000`. Die
X-Achse führt Episoden (Workbench 8.4), was sich mit der Vorgabe „Verlauf des
Rewards über die Episoden" deckt.

## Startbelegung und globale Einstellungen

- `Anzahl Verfahren` = `3`: `V1 = PPO`, `V2 = TD3`, `V3 = SAC`. Jeder
  Algorithmus kommt genau einmal vor; die Regel aus Workbench 6.2
  (Verfahrenswahl und Vergleichstabs) zum abweichenden Startwert bei doppelter
  Belegung greift erst, wenn der Benutzer einen vierten Slot wählt.
- `total_timesteps = 100.000` für **alle** Verfahren, `Episoden = 1000`,
  `seed = 0`. Warum die beiden Grenzen in derselben Größenordnung liegen, steht
  unter *Budgetgrenzen*.
- `Glättung = 20`, `Threads = 4`, `Bildrate = 20` FPS.
- Animationswahl je Slot auf `akt. Ep.`; damit sind beim Start alle drei
  Anzeigen sichtbar.

Die Glättung bestimmt zugleich, über wie viele Episoden die Summary mittelt
(Workbench 8.7, Summary). Eine automatische Zwischenevaluation gibt es nicht.

**Laufzeit bei 500.000 Schritten** (Workbench 5.1 verlangt die Angabe):

| Verfahren | Schritte/s | Laufzeit |
| --- | --- | --- |
| PPO | 481 | ~17 min |
| TD3 | ~50 | ~2,8 h |
| SAC | ~50 | ~2,8 h |

Vor dem Start eines Laufs über 100.000 Schritten weist die GUI auf die
erwartete Dauer hin.

**Warum weit unter dem Profilbudget:** Die Zoo-Profile sehen 2.000.000
Schritte (TD3, SAC) bzw. 10.000.000 (PPO) vor. Das wären bei gemessener
Geschwindigkeit rund 11 h je Off-Policy-Lauf und 5,8 h für PPO – bei drei
Verfahren, zwei Seeds und einer dreiwertigen Parameterstudie nicht bis zur
Abgabe zu schaffen. Was dadurch verloren geht, ist ausdrücklich zu benennen:
**Die Figur wird mit 300.000 bis 500.000 Schritten nicht laufen.** Erwartbar ist, dass sie
sich zunehmend länger aufrecht hält und schwankend vorwärts kommt. Die
Referenzwerte des Zoo (unten) werden nicht erreicht; Bericht und README sagen
das an sichtbarer Stelle. Vergleichbar bleiben die Verfahren trotzdem, weil
alle dasselbe Budget bekommen.

## Environment

```python
import gymnasium

env = gymnasium.make(
    "Humanoid-v5",
    render_mode="rgb_array",   # entfällt bei headless Evaluation
    width=480,
    height=480,
)
```

- `forward_reward_weight`, `ctrl_cost_weight`, `healthy_reward`,
  `contact_cost_weight`, `contact_cost_range`, `terminate_when_unhealthy`,
  `healthy_z_range`, `reset_noise_scale`,
  `exclude_current_positions_from_observation` und die vier
  `include_*_in_observation`-Schalter werden **nicht** übergeben und nicht
  verändert – das wäre Reward Shaping bzw. eine Änderung der Abbruchregeln oder
  des Beobachtungsraums.
- `width`/`height` entsprechen den Gymnasium-Voreinstellungen, werden aber
  mitgegeben, damit die Framegröße eindeutig ist. Die Kamera folgt der Figur.
- `v5` statt `v4`: gepflegte Fassung mit dokumentierter
  `observation_structure`. Die Zoo-Profile sind für `v4` hinterlegt.

### Actions

`Box(-0.4, 0.4, (17,), float32)`.

**Achtung, Abweichung von allen Vorgängerprojekten:** Der Wertebereich ist
`±0,4`, **nicht** `±1,0` wie bei Hopper, HalfCheetah und Walker2d. Jede
Anzeige, Skalierung, Normierung und jeder Test liest die Grenzen aus
`env.action_space` und setzt sie **nicht** als `±1` voraus.

Die Übersetzungen unterscheiden sich je Gelenkgruppe – Moment in N·m ist
`aᵢ · gearᵢ`:

| Index | Gelenk (XML) | `gear` |
| --- | --- | --- |
| 0 | `abdomen_y` | 100 |
| 1 | `abdomen_z` | 100 |
| 2 | `abdomen_x` | 100 |
| 3 | `right_hip_x` | 100 |
| 4 | `right_hip_z` | 100 |
| 5 | `right_hip_y` | 300 |
| 6 | `right_knee` | 200 |
| 7 | `left_hip_x` | 100 |
| 8 | `left_hip_z` | 100 |
| 9 | `left_hip_y` | 300 |
| 10 | `left_knee` | 200 |
| 11–13 | `right_shoulder1/2`, `right_elbow` | 25 |
| 14–16 | `left_shoulder1/2`, `left_elbow` | 25 |

Die Beine sind also bis zu zwölfmal kräftiger übersetzt als die Arme. Maximales
Moment: `0,4 · 300 = 120` N·m an der Hüfte, `0,4 · 25 = 10` N·m am Ellbogen.

### Observation

`Box(-inf, inf, (348,), float64)`, geprüft gegen `observation_structure`:

| Bereich | Größe | Inhalt |
| --- | --- | --- |
| `0` | 1 | Rumpfhöhe `z` (`qpos[2]`) |
| `1:5` | 4 | Rumpforientierung als Quaternion |
| `5:22` | 17 | Gelenkwinkel |
| `22:45` | 23 | `qvel`: 3 Linear-, 3 Winkel-, 17 Gelenkgeschwindigkeiten |
| `45:175` | 130 | `cinert`: Trägheitstensoren, 13 Körper × 10 |
| `175:253` | 78 | `cvel`: Körpergeschwindigkeiten, 13 × 6 |
| `253:270` | 17 | `qfrc_actuator`: Aktuatorkräfte |
| `270:348` | 78 | `cfrc_ext`: externe Kontaktkräfte, 13 × 6 |

Die x- und y-Position sind ausgeschlossen (`skipped_qpos = 2`); sie stehen in
`info`. Die zwei Sehnen des Modells (`ntendon = 2`) sind in `v5`
**nicht** Teil der Observation (`ten_length = ten_velocity = 0`), obwohl
`info` die Schlüssel `tendon_length` und `tendon_velocity` führt.

Anders als bei Walker2d gibt es **keine** Clippung der Geschwindigkeiten auf
`±10`; ein entsprechender Hinweis wäre hier falsch.

### Reward und Episodenende

```text
reward = healthy_reward + forward_reward − ctrl_cost − contact_cost
       = 5,0            + 1,25 · vₓ      − 0,1 · Σ aᵢ²  − 5e−7 · Σ cfrc_ext²
```

- `healthy_reward = 5,0` je gesundem Schritt; über eine volle Episode `+5000`.
  **Das ist der dominante Anteil** – bei Weitem größer als bei allen
  Vorgängerprojekten und der Schlüssel zum Verständnis der Skala.
- `forward_reward = 1,25 · vₓ` mit `vₓ = Δx / dt`, `dt = 0,015 s`
  (`frame_skip = 5`).
- `ctrl_cost = 0,1 · Σ aᵢ²`, höchstens `0,1 · 17 · 0,4² = 0,272` je Schritt.
- `contact_cost = 5e−7 · Σ cfrc_ext²`, nach oben auf `10` begrenzt
  (`contact_cost_range = (-inf, 10)`). In der Praxis meist nahe null, bei
  hartem Aufprall spürbar.

Alle vier Anteile stehen in `info` als `reward_survive`, `reward_forward`,
`reward_ctrl` und `reward_contact` (die beiden Kosten bereits negativ) und
werden von dort übernommen, nicht nachgerechnet.

Die Episode endet:

- mit `terminated`, sobald die Rumpfhöhe `z` den gesunden Bereich
  `(1,0; 2,0) m` verlässt. Umgangssprachlich ein Sturz. Anders als bei Walker2d
  gibt es **keine** Winkelbedingung – die Figur darf beliebig verdreht sein,
  solange die Höhe stimmt.
- mit `truncated` nach `1000` Schritten.

**Weder Terminalbonus noch Terminalstrafe**: Ein Sturz kostet nur die Rewards
der Schritte, die nicht mehr stattfinden – bei `healthy_reward = 5,0` ist das
aber ein sehr scharfes Signal.

### Erfolgsdefinitionen

`Humanoid-v5` führt in der Gymnasium-Registry **keinen** `reward_threshold`
(geprüft: `spec.reward_threshold is None`). Eine offizielle Gelöst-Schwelle
existiert nicht. Gemäß Workbench 8.1 (Metrikwahl) setzt dieses Projekt deshalb
eine **projektinterne Referenzmarke** und nennt sie ausdrücklich nicht
„gelöst":

- **Zielmarke `5000`**, begründet als `1000 Schritte · healthy_reward 5,0`:
  Genau diesen Return erreicht eine Figur, die eine volle Episode lang nicht
  stürzt, ohne sich vorwärts zu bewegen. Die Marke trennt also „bleibt
  aufrecht" von „stürzt" und ist damit direkt lesbar – ein Vorzug gegenüber
  einer aus Benchmarkwerten gegriffenen Zahl. Die Metrik heißt **Zielquote**,
  nicht Gelöst-Quote; README und Bedienungsanleitung sagen in einem Satz, dass
  die Marke projektintern gesetzt ist.
- **Durchgehalten**: Episode endet mit `truncated` nach 1000 Schritten.
- **Sturz**: Episode endet mit `terminated`.

Durchhalte- und Sturzquote sind komplementär, werden aber getrennt
ausgewiesen. Höhere Returns sind besser. Realistische Spannweite: etwa `+120`
(sofortiger Sturz nach ~24 Schritten) bis über `+6000` (laufende Figur).

Quelle: [Gymnasium Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/)

## Standardprofile

`Humanoid-v4`-Profile des RL Baselines3 Zoo, Stand geprüft am 24.08.2026; nicht
für `v5` hinterlegt, die Umgebungen sind aber bis auf die oben genannten
Punkte gleich. Einzige Abweichung von den Profilen ist `n_timesteps`, oben
begründet.

### PPO

Vollständig getuntes Profil (nicht geerbt):

| Parameter | Wert |
| --- | --- |
| `normalize` | `true` |
| `n_envs` | `1` |
| `batch_size` | `256` |
| `n_steps` | `512` |
| `gamma` | `0.95` |
| `learning_rate` | `3.56987e-05` |
| `ent_coef` | `0.00238306` |
| `clip_range` | `0.3` |
| `n_epochs` | `5` |
| `gae_lambda` | `0.9` |
| `max_grad_norm` | `2` |
| `vf_coef` | `0.431892` |
| `log_std_init` | `-2` |
| `ortho_init` | `False` |
| `activation_fn` | `ReLU` |
| `net_arch` | `pi=[256, 256]`, `vf=[256, 256]` |
| Profilbudget | `1e7` |

`gamma = 0.95` ist auffällig niedrig und `normalize: true` verpflichtend –
beides aus dem Profil übernehmen, nicht „korrigieren".

### TD3

Erbt aus `mujoco-defaults`:

| Parameter | Wert |
| --- | --- |
| `learning_starts` | `10000` |
| `noise_type` | `normal` |
| `noise_std` | `0.1` |
| `train_freq` | `1` |
| `gradient_steps` | `1` |
| `learning_rate` | `1e-3` |
| `batch_size` | `256` |
| `net_arch` | `[400, 300]` |
| `buffer_size` | `500.000` statt `1e6` (siehe Abweichung 3) |
| Profilbudget | `2e6` |

### SAC

Erbt aus `mujoco-defaults`, das für SAC nur drei Schlüssel setzt – alles
Übrige sind SB3-Voreinstellungen:

| Parameter | Wert |
| --- | --- |
| `learning_starts` | `10000` |
| `learning_rate` | `3e-4` (SB3-Default) |
| `batch_size` | `256` (SB3-Default) |
| `buffer_size` | `500.000` statt `1e6` (siehe Abweichung 3) |
| `tau` | `0.005` (SB3-Default) |
| `gamma` | `0.99` (SB3-Default) |
| `ent_coef` | `auto` (SB3-Default) |
| `net_arch` | `[256, 256]` (SB3-Default) |
| Profilbudget | `2e6` |

SAC-Zielentropie bei `auto`: `−17` (negative Action-Dimension).

### Referenzwerte

Benchmark des Zoo bei **2.000.000** Schritten:

| Verfahren | Return |
| --- | --- |
| SAC | `6232,3 ± 279,9` |
| TD3 | `5566,7 ± 14,5` |
| PPO | kein Eintrag für Humanoid |

Beide Werte liegen über der Zielmarke `5000`, die Figur bleibt dort also
aufrecht **und** bewegt sich vorwärts. Mit den Budgets dieser Arbeit sind sie
nicht erreichbar; sie dienen als Einordnung, nicht als Erwartung.

**Ungleicher Abstand zum Profilbudget – für den Bericht wichtig.** Die drei
Profile sind für unterschiedlich lange Läufe getunt:

| Verfahren | Profilbudget | 300.000 sind davon | 500.000 sind davon |
| --- | --- | --- | --- |
| PPO | `1e7` | **3 %** | **5 %** |
| TD3 | `2e6` | 15 % | 25 % |
| SAC | `2e6` | 15 % | 25 % |

Alle Verfahren bekommen nach Workbench 5.1 dasselbe Budget, und das ist
bezogen auf die **Datenmenge** auch fair. PPO tritt damit aber relativ weiter
von seinem vorgesehenen Arbeitspunkt entfernt an als die beiden anderen. Ein
schwaches PPO-Ergebnis belegt deshalb **nicht** ohne Weiteres, dass PPO für
Humanoid schlechter geeignet ist – es belegt zunächst nur, dass PPO bei 3 bis
5 % seines Budgets weiter zurückliegt. Der Bericht muss das benennen, statt die
Rangfolge unkommentiert zu übernehmen.

## Metriken

Episoden-Return, Episodenlänge, Durchhaltequote, Sturzquote, Zielquote,
mittleres Tempo `vₓ`, zurückgelegte Strecke in x sowie – neu gegenüber allen
Vorgängerprojekten – die **seitliche Abweichung**.

Begründung der seitlichen Abweichung nach Workbench 8.1: Humanoid ist das erste
3D-Environment dieser Reihe. Die Figur kann in y abdriften, der
`forward_reward` bewertet aber **nur** `vₓ`. Seitliches Weglaufen ist also
weder belohnt noch bestraft und bleibt ohne eigene Kennzahl unsichtbar. Quelle:
`info["y_position"]` als Betrag, dazu `info["distance_from_origin"]` als
zurückgelegte Gesamtstrecke in der Ebene.

Vergleichs-Summary je Slot: Algorithmus · Episoden · Environment-Schritte
ausgeführt und angefordert · **welche Grenze den Lauf beendet hat** · Ø Return
· beste Episode mit Nummer und Return · Ø Länge · Ø Tempo in m/s · Ø Strecke x
in m · Ø seitliche Abweichung in m · Durchhaltequote · Sturzquote · Zielquote.

Alle Mittelwerte umfassen die **letzten** Episoden im Glättungsfenster
(Workbench 8.6); die beste Episode bezieht sich weiter auf den ganzen Lauf.

Referenzlinie der Graphen: `+5000`, weiß und gestrichelt, in der Legende als
**Zielmarke** und ausdrücklich nicht als Gelöst-Schwelle bezeichnet. Was die
Marke bedeutet, sagt die Fußzeile der Summary in einem Satz: `Zielmarke 5.000 =
1000 Schritte aufrecht`.

## Einblendung neben der Animation

348 Observationswerte lassen sich nicht anzeigen. Die Einblendung zeigt eine
begründete Auswahl – alles, was das Verhalten erklärt, nichts, was nur Masse
macht. Die drei großen Blöcke `cinert`, `cvel` und `cfrc_ext` (286 der 348
Werte) erscheinen **nicht** einzeln, sondern verdichtet:

- die 17 Rohwerte `a₀` bis `a₁₆`, gruppiert nach Rumpf, rechtem Bein, linkem
  Bein, rechtem Arm, linkem Arm, jeweils mit Moment in N·m (`aᵢ · gearᵢ`) und
  dem Hinweis, dass der Bereich `±0,4` ist
- Rumpfhöhe `z` mit dem gesunden Bereich `1,0` bis `2,0` m
- Rumpforientierung: Quaternion, zusätzlich als Neigung gegen die Senkrechte
  in Grad, weil das Quaternion für sich nicht lesbar ist
- die 17 Gelenkwinkel, nach denselben fünf Gruppen getrennt
- `vₓ`, `v_y`, `v_z` und die drei Winkelgeschwindigkeiten des Rumpfes
- Zerlegung des letzten Rewards in `reward_survive`, `reward_forward`,
  `reward_ctrl` und `reward_contact`
- Position aus `info["x_position"]` und `info["y_position"]`, dazu
  `distance_from_origin`
- verdichtet: Summe der Quadrate von `cfrc_ext` als Treiber der Kontaktkosten,
  mit dem Hinweis auf die Deckelung bei `10`

**Zugriff auf `unwrapped` für die exakte Wiedergabe.** Workbench 7.3 (Welchen
Lernstand eine Anzeige zeigt) verlangt, die beste Episode aufzuzeichnen und
nachzuspielen statt sie mit der Policy nachzurechnen, und lässt dafür den
Zugriff auf Environment-Interna zu. Für `Humanoid-v5` heißt das konkret:

- Aufgezeichnet werden `data.qpos` (24 Werte) und `data.qvel` (23 Werte) zu
  Beginn jeder Episode sowie alle ausgeführten Actions – höchstens
  `1000 × 17` `float32`, also **68 kB** je Episode.
- Die Wiedergabe ruft `env.unwrapped.set_state(qpos, qvel)` im Renderprozess
  auf und spielt die Actions ab.
- Für **Messwerte** bleibt der Zugriff auf `unwrapped` unzulässig; Tempo,
  Positionen und Rewardanteile stammen weiterhin ausschließlich aus `info`.

Belegt ist der Nutzen: Die Policy deterministisch nachzurechnen erreicht bei
Humanoid nur 80 bis 91 % des Returns der besten Episode, weil diese explorativ
und aus einem anderen Startzustand lief.

**Bildrate:** Die environment-eigene Rate ist `67` FPS bei `dt = 0,015 s`, eine
volle Episode dauert damit 15 Sekunden Echtzeit. Der Standardwert der Animation
ist trotzdem **`20` FPS** – die von Workbench 7.1 (Steuerung und Inhalt)
verlangte Begründung: Ein stürzender Humanoid ist bei Echtzeit kaum zu
verfolgen, 20 FPS ergeben rund dreifache Zeitlupe. Zugleich kostet die
langsamere Anzeige weniger Rechenzeit, die bei ~50 Schritten/s dem Training
zugutekommt.

## Experimente für den Bericht

Zwei Blöcke, jeweils **zwei Seeds** (`0` und `1`) je Konfiguration und
`Episoden = 0`. Das Schrittbudget unterscheidet sich zwischen den Blöcken und
ist deshalb bei jedem angegeben:

| Block | Durchgänge | Schritte je Lauf | Ergebnis |
| --- | --- | --- | --- |
| 2.2 Verfahrensvergleich | Seed `0` und `1` | **300.000** | vollständig, beide Durchgänge |
| 2.3 Parameterstudie | Seed `0` und `1` | **300.000** (Durchgang 1 geplant 500.000, von Hand bei ~300.400 gestoppt) | vollständig, beide Durchgänge |

Innerhalb eines Blocks bekommt jeder Lauf dasselbe Budget – nur so ist der
Vergleich fair; in 2.3 unterscheiden sich die drei Slots durch den Handstopp um
0,1 %. Da beide Blöcke bei rund 300.000 Schritten stehen, sind ihre Zahlen
ausnahmsweise auch untereinander vergleichbar – der Bericht nutzt das für eine
Probe auf die Reproduzierbarkeit.

Vorgabe 2.3d verlangt nur **ein** Training je Ausprägung; gefahren wurden
zwei, und das hat sich gelohnt – zwischen den beiden kleineren Lernraten dreht
sich die Reihenfolge mit dem Seed. Der Bericht sagt an der Auswertung, welche
Aussage belastbar ist und welche nicht.

Warum zwei Seeds: RL-Training ist zufällig in Initialisierung und Exploration.
Aus einer einzelnen Kurve je Verfahren lässt sich nicht sagen, ob ein
Unterschied echt oder Zufall ist – die Aufgabenstellung bewertet aber
ausdrücklich „Stabilität des Lernfortschritts (Variabilität)". Mit zwei Seeds
lässt sich die Streuung *innerhalb* eines Verfahrens gegen den Abstand
*zwischen* den Verfahren halten.

### Teil 2.2 – Verfahrensvergleich

`PPO`, `TD3`, `SAC` mit den obigen Profilen, je `300.000` Schritte. Sechs
Läufe, gemessen rund 7 h, in **zwei Durchgängen zu drei Slots**:

| Durchgang | Slots | Seed |
| --- | --- | --- |
| A | `V1 = PPO`, `V2 = TD3`, `V3 = SAC` | `0` |
| B | dieselben drei | `1` |

Jeder Durchgang erfüllt Vorgabe 1.2g unmittelbar: drei Läufe, drei Farben, ein
Diagramm. Erzeugte Bilder:

| Datei | Inhalt | Vorgabe |
| --- | --- | --- |
| `2-2-vergleich-seed0.png` | alle drei Verfahren, Seed 0 | 1.2g, 2.2e–g |
| `2-2-vergleich-seed1.png` | alle drei Verfahren, Seed 1 | 1.2g, 2.2e–g |
| `2-2-ppo-seed0.png` … `-seed1.png` | nur PPO, je Seed | 2.2d |
| `2-2-td3-seed0.png` … `-seed1.png` | nur TD3, je Seed | 2.2d |
| `2-2-sac-seed0.png` … `-seed1.png` | nur SAC, je Seed | 2.2d |

Dazu je Durchgang die Summary als `2-2-summary-seed0.txt` beziehungsweise
`-seed1.txt`; sie trägt die Konfiguration aller drei Slots und belegt, dass
sich die Durchgänge nur im Seed unterscheiden.

Die Einzelbilder entstehen über Workbench 8.6 (Einzelgraph je Verfahren) aus
denselben Läufen – **ohne** erneutes Training. Im Bericht bekommt jedes
Verfahren einen Abschnitt mit seinen beiden Einzelplots; die Vergleichsplots
tragen die Gegenüberstellung.

Bewertet wird nach den drei Kriterien der Aufgabenstellung – Lernkurve und
Konvergenzgeschwindigkeit, Stabilität und Variabilität, maximal erreichter
Reward –, jeweils mit einer konkreten Beobachtung aus den Plots belegt.

**Gegenprobe zum Budget (nicht geplant, nachträglich ergänzt).** Der Bericht
hält in 2.2.4 fest, dass 300.000 Schritte bei PPO nur 3 % seines Profilbudgets
sind, bei TD3 und SAC dagegen 15 %. Weil PPO rund zehnmal schneller rechnet,
lässt sich dieser Vorbehalt billig prüfen: zwei Läufe über je 1,5 Mio Schritte –
ebenfalls 15 % seines Profilbudgets –, Seeds `0` und `1`, sonst unverändertes
Profil, rund 50 Minuten je Lauf. Dateien: `2-2-ppo-1500k-vergleich.png`,
`2-2-ppo-1500k-seed0/1.png`, `2-2-ppo-1500k-summary.txt`. Das Ergebnis steht in
2.2.5 des Berichts.

### Teil 2.3 – Parameterstudie

Für das in 2.2 beste Verfahren **ein** besonders sensitiver Parameter in drei
Ausprägungen: empfohlener Wert, kleiner, größer. Je Durchgang drei Slots
gleichzeitig, gemessen rund 5,5 h für 300.000 Schritte.

**Kriterium für „bestes Verfahren", vor den Läufen festgelegt:** mittlerer
Return über die **letzten 50 Episoden**, gemittelt über beide Seeds. Der
Mittelwert statt des Maximums, weil ein einzelner Ausreißer sonst die Wahl
bestimmt; über beide Seeds, weil ein Glückslauf sonst dasselbe täte. Bei einem
Abstand unter der Streuung zwischen den Seeds entscheidet der niedrigere
Rechenaufwand – und der Bericht sagt, dass die Verfahren nicht trennbar waren.
Festgelegt wird das **vorher**, damit die Wahl nicht nachträglich zum Ergebnis
passend begründet wird.

Empfehlung ist die **Lernrate**: Sie existiert in allen drei Verfahren, wirkt
unmittelbar auf Konvergenzgeschwindigkeit *und* Stabilität – also auf zwei der
drei Bewertungskriterien – und liefert einen sauberen Dreischritt um den
Profilwert. Die konkrete Wahl steht erst nach 2.2 fest:

| Bestes Verfahren | kleiner | empfohlen | größer |
| --- | --- | --- | --- |
| SAC | `1e-4` | `3e-4` | `1e-3` |
| TD3 | `3e-4` | `1e-3` | `3e-3` |
| PPO | `1.2e-5` | `3.56987e-05` | `1e-4` |

Gewählt ist jeweils etwa Faktor 3 nach unten und oben – groß genug für einen
sichtbaren Unterschied, klein genug, dass der mittlere Lauf nicht als einziger
etwas lernt. Weicht die Beobachtung davon ab, wird das im Bericht benannt statt
die Werte nachträglich zu schönen.

**Entschieden nach 2.2:** `SAC` gewinnt nach dem oben festgelegten Kriterium
mit Abstand (2.711,6 gegen 491,9 bei PPO und 407,3 bei TD3, gemittelt über
beide Durchgänge). Die Studie läuft also mit `SAC` und den Lernraten `1e-4`,
`3e-4` und `1e-3`. Ausgewertet wurden die letzten **20** statt 50 Episoden –
das ist das Glättungsfenster, mit dem die Workbench die Summary schreibt; bei
einem Abstand von Faktor 5,5 ändert das die Wahl nicht. Der Bericht benennt
diese Abweichung.

Ablauf wie in 2.2, zwei Durchgänge zu drei Slots – die drei Slots tragen hier
denselben Algorithmus mit den drei Lernraten, Workbench 6.2 sorgt für die
Kennzeichnung des abweichenden Parameters in der Legende:

| Datei | Inhalt | Vorgabe |
| --- | --- | --- |
| `2-3-vergleich-seed0.png` … `-seed1.png` | alle drei Ausprägungen, je Durchgang | 1.2g |
| `2-3-lr-klein-seed0/1.png` | nur die kleine Lernrate `1e-4` | **2.3c**, 2.3d |
| `2-3-lr-empfohlen-seed0/1.png` | nur der Profilwert `3e-4` | **2.3c**, 2.3d |
| `2-3-lr-gross-seed0/1.png` | nur die große Lernrate `1e-3` | **2.3c**, 2.3d |

Dazu je Durchgang `2-3-summary-seed0.txt` beziehungsweise `-seed1.txt` mit den
Kennzahlen und der vollständigen Konfiguration aller drei Slots.

Vorgabe 2.3c verlangt ausdrücklich **unterschiedliche** Reward-Plots je
Ausprägung, nicht drei Kurven in einem Bild – die Einzelgraphen sind daher
Pflicht, der Vergleichsplot kommt hinzu.

Dateinamen mit `lr` gelten, falls die Lernrate gewählt wird; fällt die Wahl
nach 2.2 auf einen anderen Parameter, wird der Namensbestandteil entsprechend
ersetzt.

## Tests

Zusätzlich zu Workbench 5.10 (Verfahrensbezogene Tests) und 9.2 (Tests):

- Environment-Kennwerte: Action `Box(-0.4, 0.4, (17,))` – ausdrücklich
  **nicht** `±1` –, Observation `(348,)` `float64`,
  `max_episode_steps = 1000`, `render_fps = 67`, `dt = 0,015`,
  `frame_skip = 5`, Frame `480 × 480`; die Factory fordert genau diese
  Argumente an und übergibt **keinen** Physik-, Reward-, Abbruch- oder
  Observationsparameter
- `Humanoid-v5` führt **keinen** `reward_threshold`; die Zielmarke `5000` ist
  eine Projektkonstante, wird nicht als offizielle Schwelle ausgegeben und
  entspricht `1000 · healthy_reward`
- die Observation zerfällt genau in die acht oben tabellierten Bereiche; jeder
  wird gegen `observation_structure` und gegen `data.qpos`/`qvel`/`cinert`/
  `cvel`/`qfrc_actuator`/`cfrc_ext` geprüft
- die Übersetzungen stimmen mit `model.actuator_gear` überein: `100` für Rumpf
  und Hüfte x/z, `300` für Hüfte y, `200` für Knie, `25` für alle sechs
  Armgelenke
- Reward-Zerlegung: `reward_survive + reward_forward + reward_ctrl +
  reward_contact` aus `info` ergibt exakt den Reward; `healthy_reward` ist
  `5,0`; Steuerkosten `0,1 · Σ aᵢ²`; Kontaktkosten `5e−7 · Σ cfrc_ext²`, nach
  oben auf `10` gedeckelt
- Episodenausgänge: Sturz als `terminated` **ohne** Terminalstrafe, ausgelöst
  **allein** durch die Höhe außerhalb `(1,0; 2,0)`; es gibt **keine**
  Winkelbedingung; Durchhalten als `truncated` nach genau 1000 Schritten
- die Geschwindigkeiten sind **nicht** geclippt (Gegentest zu Walker2d)
- x- und y-Position stammen aus `info`, nicht aus der Observation
- SAC-Zielentropie bei `auto` genau `−17`
- TD3-Action-Noise `noise_std = 0.1` wirkt im auf `±1` normierten Raum von SB3
  und entspricht damit `0,04` in Action-Einheiten; die resultierenden Actions
  bleiben innerhalb `±0,4`
- `torch.set_num_threads` wird mit dem eingestellten Wert aufgerufen
- `buffer_size` entspricht dem Schrittbudget; der Buffer verdrängt bei einem
  vollständigen Lauf keinen einzigen Übergang; die gepufferte Observation ist
  `float32` und stimmt im Rahmen der `float32`-Genauigkeit mit der
  `float64`-Observation des Environments überein
- `optimize_memory_usage` bleibt `False`, `handle_timeout_termination` bleibt
  `True`
- Ablehnung eines Speicherstands mit Kennung `Humanoid-v4`, `Walker2d-v5`,
  `Hopper-v5` oder `HalfCheetah-v5`

## Abhängigkeiten

Keine neuen. MuJoCo, `imageio`, Stable-Baselines3 und PyTorch stehen bereits in
`../environment.yml`; `cma` wird hier nicht gebraucht. Die `requirements.txt`
dieses Projekts führt die tatsächlich importierten Pakete.

## Abgabe

Zusätzlich zu Workbench 9.3 (Abnahme) und 9.4 (README):

- Bericht `KLR-339-2026-08-Hessling_Oliver.md` in diesem Ordner, in einfacher
  Sprache, mit Screenshot der Anwendung und **allen** Reward-Plots
- Bilder liegen direkt in diesem Ordner, damit die relativen Links auch dann
  gültig bleiben, wenn die Datei einzeln weitergereicht wird
- jede Bewertung wird mit einer konkreten Beobachtung aus den Läufen belegt,
  nicht mit allgemeinen Aussagen über die Verfahren
- ausdrücklich benannt wird, dass das Budget unter dem Profilwert liegt und die
  Figur deshalb nicht läuft
- Abgabe als PDF bis **28.08.2026, 15:00**

## Abnahme

Zusätzlich zu Workbench 9.3 (Abnahme): alle drei Verfahren trainieren und
vergleichen mit den obigen Profilen; der Wertebereich `±0,4` wird nirgends als
`±1` behandelt; die Zielmarke `5000` wird überall als projektintern und nicht
als offizielle Gelöst-Schwelle ausgewiesen; die vier Rewardanteile werden
getrennt ausgewiesen; die seitliche Abweichung erscheint in Summary und
Einblendung; Episoden- und Schrittgrenze wirken beide, ein zweiter Start hängt
das volle Budget an, und die beendende Grenze ist ablesbar; die Summary mittelt
über das Glättungsfenster und passt ohne Scrollen ins Feld; die
Unterschiedszeilen erscheinen genau dann, wenn ein Algorithmus mehrere Slots
belegt; jedes Verfahren lässt sich einzeln darstellen und exportieren; die
Einblendung zeigt die verdichtete Auswahl statt aller 348 Werte; die Animation
lässt sich je Slot auf `inaktiv` stellen und blendet dann genau diese Anzeige
aus; `beste Ep.` spielt die Episode exakt nach und erreicht denselben Return wie im
Training; ein Speicherstand fremder
Environment-Kennung wird abgelehnt.

## Vorgabenabgleich

Workbench 9.3 (Abnahme) verlangt für Projekte unter externer Vorgabe eine
Tabelle, die **jede einzelne** Vorgabe auf ihre Fundstelle abbildet. Abgenommen
wird gegen diese Tabelle. Quelle ist die Aufgabenstellung
`2026_08_KLR-339_Projektaufgabe.pdf`; sie ist bewusst nicht im Repository
(`.gitignore`), die Vorgaben sind hier wörtlich genug wiedergegeben.

### Einleitung und Teil 1

| # | Vorgabe | erfüllt durch |
| --- | --- | --- |
| E1 | Workbench aus **Konfigurator und ausführendem Teil** | Workbench 2.3 (Architektur) |
| 1.1a | Python-Programm mit **Tkinter**-UI | Workbench 2.2, 6 (Oberfläche) |
| 1.1b | zugewiesene Animation und Methoden nutzen | `Humanoid-v5`, PPO/TD3/SAC – *Ziel*, *Environment* |
| 1.1c | beste Methode wählen, sensitiver Parameter in 3 Ausprägungen | *Experimente*, Teil 2.3 |
| 1.2a | Animation im Training **anzeigen** | Workbench 7.1, 7.2 |
| 1.2b | während des Trainings **an- und abschaltbar** | Workbench 7.3 – Wahl je Verfahren, `inaktiv` blendet aus; beim Start alle sichtbar |
| 1.2c | Eingabe **Anzahl Episoden, Default 1000** | Workbench 5.1 (Gemeinsame Parameter), *Budgetgrenzen* |
| 1.2d | Methoden **idealerweise parallel** | Workbench 8.5 – alle Slots starten parallel |
| 1.2e | episodenweiser Reward **dünn** | Workbench 8.2 – Rohkurve, verringerte Deckkraft und Strichstärke |
| 1.2f | Episodendurchschnitt **fett** | Workbench 8.3 – kräftige Linie, Fenster einstellbar |
| 1.2g | **eigene Farbe je Durchlauf, selbes Diagramm** | Workbench 8.2, 8.5, 6.2 – Slotfarben, gemeinsamer Vergleichsgraph |
| 1.2h | Reward-Anzeige **als Image speicherbar** | Workbench 8.7 – PNG-Export; je Slot einzeln über 8.9 |
| 1.2i | **qualifizierte Legende** | Workbench 8.2, 8.5 – Slot, Algorithmus, abweichender Parameter |

### Teil 2

| # | Vorgabe | erfüllt durch |
| --- | --- | --- |
| 2.2a | zugeteilte Animation implementieren | dieses Projekt |
| 2.2b | Workbench um fehlende Methoden erweitern | nicht erforderlich – PPO, TD3 und SAC sind aus 13/14 vorhanden |
| 2.2c | **empfohlene optimale** Hyperparameter | *Standardprofile* – RL Baselines3 Zoo, Abweichungen begründet |
| 2.2d | **für jede Methode ein** Reward-Plot über die Episoden | Workbench 8.6 (Einzelgraph je Verfahren), X-Achse Episoden nach 8.4 |
| 2.2e | Bewertung **Lernkurve / Konvergenzgeschwindigkeit** | *Experimente*, Teil 2.2 |
| 2.2f | Bewertung **Stabilität / Variabilität** | *Experimente* – zwei Seeds je Konfiguration |
| 2.2g | Bewertung **maximal erreichter Reward** | Summary-Zeile „beste Episode", *Metriken* |
| 2.2h | Analyse und Bewertung dokumentieren | Bericht, *Abgabe* |
| 2.3a | sensitiven Parameter wählen **und begründen** | *Experimente*, Teil 2.3 – Lernrate, mit Begründung |
| 2.3b | **3 Ausprägungen**: empfohlen, kleiner, größer | *Experimente* – Tabelle je Verfahren, Faktor ~3 |
| 2.3c | in **unterschiedlichen** Reward-Plots darstellen | Workbench 8.6 – drei getrennte Einzelgraphen, nicht drei Kurven in einem Bild |
| 2.3d | je Ausprägung **ein Training** mit Reward-Plot | *Experimente* – 3 Werte × 2 Seeds |
| 2.3e | Bewertung: Einfluss, beste Einstellung, unerwartete Effekte | Bericht |
| 2.3f | Analyse und Bewertung dokumentieren | Bericht |

### Teil 3 und Abgabe

| # | Vorgabe | erfüllt durch |
| --- | --- | --- |
| 3.1 | kurzer, aussagekräftiger Bericht in **Markdown** | `KLR-339-2026-08-Hessling_Oliver.md` |
| 3.2 | kurze **Präsentation** | `praesentation.md` – 17 Folien für 10–12 min, Sprechtext in `praesentation-notizen.md`; Export als PDF und HTML |
| H1 | Beschreibung der Applikation **mit Screenshot** | *Abgabe* |
| H2 | **alle** Reward-Plots übersichtlich einfügen | *Abgabe* – Plots liegen im selben Ordner |
| H3 | Beobachtungen klar und nachvollziehbar | *Abgabe* |
| H4 | **einfache, verständliche Sprache** | *Abgabe* |
| H5 | Bewertungen mit **konkreten Beobachtungen aus den Plots** belegen | *Abgabe*, *Experimente* |
| A1 | Abgabe als **PDF bis 28.08.2026, 15:00** | *Abgabe* |
| A2 | Unterordner **`Projekt`** unterhalb des eigenen Namens | `Oliver/Projekt` – siehe Abweichung 1 |
| A3 | Datei **`KLR-339-2026-08-Nachname_Vorname.md`** | vorhanden |
| A4 | **alle** Bilder, Videos, Grafiken in dieses Verzeichnis | *Abgabe* – Bilder direkt im Ordner, keine Unterordner |

Bewertungskriterien der Aufgabenstellung, die keine einzelne Vorgabe sind,
sondern die Ausführung betreffen: Vollständigkeit und Korrektheit des
Konfigurators, Qualität und Klarheit der Reward-Plots, Nachvollziehbarkeit und
Tiefe der Vergleiche, Struktur der Dokumentation, eigenständige Analyse und
kritische Reflexion, Verständlichkeit der Präsentation.
