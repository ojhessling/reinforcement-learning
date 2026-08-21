# Walker2d Workbench

Lokale Tkinter-Anwendung zum Konfigurieren, Trainieren, Beobachten, Evaluieren
und Vergleichen von **TD3**, **SAC** und **CMA-ES** auf dem MuJoCo-Environment
`Walker2d-v5`.

Grundlage sind [`../workbench.md`](../workbench.md) und [`prompt.md`](prompt.md).

## Ziel

Ein zweibeiniger Roboter im Seitenprofil soll aufrecht laufen.

**Das Neue an diesem Projekt** ist nicht das Environment, sondern die dritte
Verfahrensklasse. `CMA-ES` ist gradientenfrei und populationsbasiert: Es
optimiert die Policy-Gewichte direkt, kennt weder Lernrate noch Batch-Größe
noch Diskontfaktor, bewertet Kandidaten über ganze Episoden und wertet nur
deren Rangfolge aus.

`PPO` fehlt bewusst. Dadurch stehen sich genau zwei Klassen gegenüber – zwei
gradientenbasierte Off-Policy-Verfahren gegen ein gradientenfreies –, und der
Vergleich hat nur **eine** systematische Asymmetrie statt zwei. Was bleibt, ist
der Unterschied in der Informationsmenge je Schritt, und der ist damit sauber
ablesbar.

## Installation und Start

```bash
conda env create -f ../environment.yml     # oder: conda env update -f ../environment.yml
conda activate rl-26-08
python walker2d_app.py
```

Neu gegenüber den MuJoCo-Vorgängerprojekten ist **`cma`** (pycma), die
Referenzimplementierung von CMA-ES. Fehlt sie in einer bestehenden Umgebung:

```bash
pip install cma
```

MuJoCo und `imageio` stehen bereits in der Umgebung; ansonsten
`pip install "gymnasium[mujoco]"`. Zum Rendering (Backend `cgl` auf macOS, kein
Dock-Eintrag) siehe Workbench 7.6.

## Environment

```python
env = gymnasium.make("Walker2d-v5", render_mode="rgb_array", width=480, height=480)
```

Alle physikalischen Parameter bleiben unangetastet – sie zu verstellen wäre
Reward Shaping beziehungsweise eine Änderung der Abbruchregeln.

### Actions

`Box(-1.0, 1.0, (6,), float32)`. Alle sechs Motoren haben **dieselbe**
Übersetzung `gear = 100` – anders als bei HalfCheetah, wo sie sich
unterscheiden:

| Index | Gelenk (XML) |
| --- | --- |
| `a₀` | rechter Oberschenkel (`thigh_joint`) |
| `a₁` | rechter Unterschenkel (`leg_joint`) |
| `a₂` | rechter Fuß (`foot_joint`) |
| `a₃` | linker Oberschenkel (`thigh_left_joint`) |
| `a₄` | linker Unterschenkel (`leg_left_joint`) |
| `a₅` | linker Fuß (`foot_left_joint`) |

Vorzeichen = Drehrichtung, Betrag mal `100` = Moment in N·m. Werte außerhalb
`[-1, 1]` clippt das Environment. Ein Test vergleicht die Übersetzung mit
`model.actuator_gear` und die Gelenkreihenfolge mit der Aktuatorreihenfolge.

### Observation

17 Werte, `float64`: Höhe und Neigungswinkel des Rumpfes, sechs Gelenkwinkel,
`vₓ`, `v_z`, Winkelgeschwindigkeit des Rumpfes und sechs Gelenktempi.

- Die **x-Position** fehlt; für Anzeige und Metriken liest die App sie aus
  `info["x_position"]`.
- Die neun Geschwindigkeiten sind auf `[-10, 10]` **geclippt** – wie beim
  Hopper und anders als bei HalfCheetah.

### Reward und Episodenende

```text
reward = healthy_reward + forward_reward − ctrl_cost
       = 1,0            + 1,0 · vₓ       − 0,001 · Σ aᵢ²
```

Der Überlebensbonus von `+1` je gesundem Schritt dominiert früh: Ein Agent, der
nur stehen bleibt, sammelt bereits `+1000`. Die Steuerkosten sind mit
höchstens `0,006` je Schritt vernachlässigbar. Alle drei Anteile stehen in
`info` und werden von dort übernommen.

Die Episode endet mit `terminated`, sobald der Roboter **ungesund** wird: Höhe
außerhalb `(0,8; 2,0) m` oder Rumpfwinkel außerhalb `(−1,0; 1,0) rad`. Der
gesunde Höhenbereich ist nach **oben und unten** begrenzt – ein Sprung über
2,0 m beendet die Episode ebenso wie ein Sturz. Nach `1000` Schritten greift
`truncated`. Es gibt weder Terminalbonus noch Terminalstrafe.

### Erfolgsdefinitionen

`Walker2d-v5` führt in der Gymnasium-Registry **keinen** `reward_threshold`.
Eine offizielle Gelöst-Schwelle existiert nicht.

| Ausgang | Bedeutung |
| --- | --- |
| **Durchgehalten** | `truncated` nach 1000 Schritten |
| **Sturz** | `terminated`, ungesund geworden |
| **Ziel erreicht** | Return `≥ 4000` – eine **projektinterne** Marke |

Die `4000` sind durch die Zoo-Benchmarkwerte begründet (SAC 3863, TD3 4718) und
liegen zwischen ihnen. Die Metrik heißt deshalb **Zielquote**, nicht
Gelöst-Quote, und die Referenzlinie im Graphen heißt **Zielmarke**.

Quelle: [Gymnasium Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/)

## Methoden

**TD3** und **SAC** sind gradientenbasiert, off-policy, mit zwei Critics und
Replay Buffer – TD3 mit deterministischem Actor und Action Noise, SAC mit
stochastischem Actor und gelernter Entropie (`target_entropy = −dim(A) = −6`).

**CMA-ES** ist eine Evolutionsstrategie. Aus einer Normalverteilung
`N(m, σ²·C)` werden λ Gewichtsvektoren gezogen, jeder über ganze Episoden
bewertet; aus den nach Rendite sortierten Kandidaten werden Mittelwert,
Schrittweite und Kovarianz neu geschätzt.

```text
xₖ ~ m + σ·N(0, C),  k = 1 … λ
m ← Σ wᵢ · x_{i:λ}
```

Es unterscheidet sich in jeder Hinsicht von den beiden anderen:

- **Kein Gradient, kein Replay Buffer, kein Critic.** Keine `learning_rate`,
  keine `batch_size`, kein `gamma` – optimiert wird die **undiskontierte**
  Rendite, also genau die Größe, die der Graph zeigt.
- **Rangbasiert.** Nur die Reihenfolge der Kandidaten zählt; jede monotone
  Umskalierung ist wirkungslos. Eine Reward-Normalisierung wäre sinnlos und
  fehlt deshalb im Tab.
- **Episodenweise.** Ein Kandidat wird über vollständige Episoden bewertet;
  Information einzelner Schritte wird nicht genutzt.
- **Exploration im Parameterraum** statt über die Action.

Quellen: [TD3](https://arxiv.org/abs/1802.09477),
[SAC](https://arxiv.org/abs/1801.01290),
[SAC mit gelernter Temperatur](https://arxiv.org/abs/1812.05905),
[Hansen, CMA-ES Tutorial](https://arxiv.org/abs/1604.00772),
[pycma](https://github.com/CMA-ES/pycma),
[Salimans et al., ES als Alternative zu RL](https://arxiv.org/abs/1703.03864),
[Rajeswaran et al., lineare Policies für MuJoCo](https://arxiv.org/abs/1703.02660)

### Fairness zwischen den Klassen

Ein Vergleich bei gleichem Schrittbudget ist fair im Sinne des Budgets, nicht
im Sinne gleicher Voraussetzungen. CMA-ES nutzt je Episode genau **eine** Zahl –
die Rendite – und verwirft alles, was zwischen den Schritten passiert ist. TD3
und SAC werten jeden einzelnen Übergang aus. Bei gleichem Budget liegt CMA-ES
deshalb typischerweise zurück; das sagt nichts über die Qualität des
Verfahrens, sondern über die Informationsmenge je Schritt. Dafür braucht es
keine Differenzierbarkeit, keinen Critic und keine Diskontierung.

Die Summary-Zeile **`Generationen`** macht das sichtbar: Sie ist nur bei
CMA-ES gefüllt und zeigt, wie wenige Update-Schritte hinter der Kurve stehen.

## Parameter, Standardwerte und Quellen

`Walker2d-v4`-Profile des RL Baselines3 Zoo, Stand geprüft am 21.08.2026;
nicht enthaltene Werte sind SB3-Defaults 2.9.0. Für CMA-ES gibt es kein
Zoo-Profil; die Werte folgen `pycma` und der Literatur.

| Parameter | TD3 | SAC | CMA-ES |
| --- | --- | --- | --- |
| `total_timesteps` | 1.000.000 | 1.000.000 | 1.000.000 |
| `learning_rate` | 1e-3 | 3e-4 | – |
| `gamma` | 0,99 | 0,99 | – |
| `batch_size` | 256 | 256 | – |
| `buffer_size` / `learning_starts` | 1.000.000 / 10.000 | 1.000.000 / 10.000 | – |
| `tau` | 0,005 | 0,005 | – |
| `train_freq` / `gradient_steps` | 1 / 1 | 1 / 1 | – |
| Action Noise | normal, σ = 0,1 | keins | – |
| `ent_coef` | – | `auto` | – |
| `sigma0` | – | – | 0,5 |
| `popsize` λ | – | – | `auto` → 18 |
| Episoden je Kandidat | – | – | 1 |
| Policy | 400,300 ReLU | 256,256 ReLU | **linear**, Tanh |
| Normalisierung (Obs / Reward) | aus / aus | aus / aus | **an** / entfällt |

Global: `Anzahl Verfahren` **3** (V1 TD3, V2 SAC, V3 CMA-ES – jeder genau
einmal), `Eval-Intervall` **100.000**, `Eval-Episoden M` **5**,
`Glättung` **20**.

### Warum die Netzgrößen nicht vereinheitlicht sind

CMA-ES führt eine `n × n`-Kovarianzmatrix. Bei `256,256` wären das rund
**72.000** Parameter, also über `5 · 10⁹` Matrixeinträge und **41 GB**
Speicher, dazu eine Eigenzerlegung in `O(n³)`. Mit der linearen Policy sind es
`6 · 17 + 6 = 108` Parameter und `11.664` Matrixeinträge. Dass eine lineare
Policy für MuJoCo-Laufaufgaben ausreicht, zeigen Rajeswaran et al.; eine
unterlegene CMA-ES-Kurve ist deshalb **nicht** dem kleinen Netz anzulasten.

Damit ist eine projektweit gemeinsame Architektur ohnehin ausgeschlossen. Ein
getuntes Zoo-Profil zu überschreiben brächte also nichts – anders als in
`13-hopper` und `14-halfcheetah`, wo alle Verfahren derselben Klasse angehörten
und eine gemeinsame Architektur erreichbar war. Fairness bemisst sich nach dem
Schrittbudget, nicht nach gleicher Architektur.

### Kein Profil verlangt Reward-Normalisierung

Das wäre das PPO-Profil gewesen, und PPO ist nicht dabei. `TD3` und `SAC`
normalisieren gar nicht; `CMA-ES` normalisiert die Beobachtungen aus eigenem
Bedarf – ohne laufende Statistik arbeitet eine lineare Policy auf den 17 sehr
unterschiedlich skalierten Werten kaum. Die Statistik wird erst **nach** jeder
Generation fortgeschrieben, damit alle Kandidaten einer Generation dieselbe
Normalisierung sehen; sonst wäre ihre Rangfolge verfälscht.

> **Laufzeit:** `1e6` Schritte dauern auf einem Laptop **Stunden**, bei drei
> Slots parallel länger. Für einen ersten Eindruck `Trainingsschritte N` auf
> **100.000** und `Eval-Intervall` auf **10.000** setzen. Bei CMA-ES kostet
> eine Generation `popsize` Episoden; mit λ = 18 und früh kurzen Episoden
> entstehen anfangs viele Generationen, später deutlich weniger. Das Budget
> muss mindestens **eine vollständige Generation** zulassen, sonst gibt es
> keinen einzigen Update-Schritt.

## CMA-ES im Ablauf

- **Die Animation zeigt den Verteilungsmittelwert**, nicht einen gezogenen
  Kandidaten – derselbe Stand, den auch die deterministische Evaluation nutzt.
  Ein Kandidat wäre bei jeder Episode ein anderer und spränge sichtbar herum.
- **Jede Kandidatenbewertung ist eine Episode** und erzeugt einen Kurvenpunkt.
  Bei mehreren Episoden je Kandidat ist die Fitness ihr Mittelwert, aber jede
  Episode bleibt ein eigener Punkt.
- **Budgetende mitten in einer Generation:** Die gelaufenen Episoden bleiben
  als Kurvenpunkte, aber die unvollständige Generation wird nicht an den
  Optimierer gemeldet. `Generationen` zählt nur abgeschlossene.
- **Zwischenevaluation** wird nach jeder abgeschlossenen Kandidatenbewertung
  geprüft; die Stützstellen liegen dadurch leicht hinter dem Intervall.
- **Der Checkpoint** enthält die vollständige Suchverteilung – Mittelwert,
  Schrittweite, Kovarianz – plus die Beobachtungsstatistik. Ein Netz allein
  genügt nicht: Ohne die Verteilung ließe sich das Training nicht fortsetzen.
  Der Schnappschuss der besten Episode ist dagegen nur ihr Parametervektor.

## Bedienablauf

1. `Anzahl Verfahren` wählen (2 bis 4, Standard 3 – jedes Verfahren einmal).
   Ein vierter Slot doppelt zwangsläufig eines und bekommt deshalb einen
   abweichenden Startwert.
2. Je Slot einen Algorithmus wählen. Ein Dropdown-Wechsel lädt das Profil und
   setzt **nur** diesen Slot zurück.
3. Im jeweiligen Tab die Parameter setzen. Der `CMA-ES`-Tab enthält keine
   Felder für Lernrate, Batch-Größe, Diskontfaktor, Replay Buffer oder Action
   Noise – dafür `Sigma σ₀`, `Popul. λ`, `Ep./Kand.` und `Diagonal`.
4. `Training starten / fortsetzen` trainiert das aktive Verfahren,
   `Vergleich starten / fortsetzen` alle aktiven Slots parallel.

Gesichert wird automatisch der beste deterministische Evaluationsstand je Slot;
`Bestes Modell wiederherstellen` holt ihn zurück. Ein Stand mit fremder
Environment-Kennung wird abgelehnt.

## Interpretation der Ansichten

Slotfarben: V1 blau, V2 rot, V3 gelb, V4 grün, alle Kurven durchgezogen; die
Zielmarke bei `4000` ist weiß gestrichelt. `Glättung` rechts in der Tableiste
stellt die Breite des gleitenden Durchschnitts ein – global für alle Slots.

Unter jedem Animationsbild steht nur `E: … · S: … · R: …`; das Verfahren steht
im Titel. Alle weiteren Messwerte – die sechs Actions mit Moment in N·m, Höhe
und Winkel mit ihren gesunden Bereichen, Gelenkwinkel und -tempi nach rechtem
und linkem Bein getrennt, die Reward-Zerlegung – erscheinen beim Überfahren mit
der Maus neben dem Bild.

Die Bildrate ist einstellbar (1 bis 250, Standard **125 FPS**); eine volle
Episode dauert damit acht Sekunden.

## Grenzen und Erwartung

Zoo-Benchmark für `Walker2d` nach `1e6` Schritten:

| Verfahren | mittlerer Return | Standardabweichung |
| --- | --- | --- |
| `SAC` | 3863,2 | 254,3 |
| `TD3` | 4717,8 | 46,3 |

Für **CMA-ES führt der Zoo keine Werte**. Zu erwarten ist bei diesem Budget ein
deutlicher Rückstand – nicht wegen der kleinen Policy, sondern weil jede
Episode nur eine einzige Zahl beiträgt.

Weitere Grenzen:

- Früh dominiert der Überlebensbonus: Eine Policy, die einfach stehen bleibt,
  sammelt schon `+1000` und schlägt jede noch wacklige Laufbewegung. Der Return
  steigt erst darüber hinaus, wenn der Roboter tatsächlich vorankommt. Bei
  CMA-ES ist dieses lokale Optimum besonders früh sichtbar.
- Drei gleichzeitige Animationen kosten spürbar Rechenzeit: drei
  Renderprozesse mit je eigener MuJoCo-Instanz.
- Mehrere Wiederholungen je Slot (Seed-Mittelung mit Unsicherheitsband) sind
  nicht implementiert.
- Trainiert wird mit **einer** Environment-Instanz.

## Tests

```bash
python -m pytest tests -q
```

161 Tests. Geprüft werden unter anderem: Environment-Kennwerte, **dass
`Walker2d-v5` keinen `reward_threshold` führt**, die einheitliche Übersetzung
gegen `model.actuator_gear`, die Übereinstimmung von Gelenk- und
Actionreihenfolge, die Reward-Zerlegung aus `info`, der nach oben **und** unten
begrenzte gesunde Höhenbereich, die Geschwindigkeitsclippung, Sturz ohne
Terminalstrafe, die SAC-Zielentropie `−6`, Save/Load-Roundtrips für alle drei
Verfahren – und für CMA-ES speziell: Parameterzahl `108`, `popsize auto` = 18,
dass eine Generation Mittelwert und Schrittweite verändert, dass die
Evaluation den Verteilungsmittelwert nutzt und sich wiederholt, dass das
Budget in Environment-Schritten eingehalten wird, dass die
Beobachtungsstatistik nur zwischen Generationen wächst und in der Evaluation
eingefroren bleibt, und dass ein Roundtrip dieselbe Suchverteilung fortsetzt.

Tests, die MuJoCo brauchen, werden ohne das Paket übersprungen; der
GUI-Smoke-Test ohne Display.

## Dateien

| Datei | Inhalt |
| --- | --- |
| `walker2d_app.py` | Entry Point |
| `walker2d_logic.py` | Environment-Factory, Konfiguration, SB3- und CMA-ES-Slot, Metriken, Checkpoints |
| `walker2d_gui.py` | Tkinter-Oberfläche, Diagramme, Animation, Summary, Export |
| `walker2d_render.py` | isolierter MuJoCo-Renderprozess |
| `tests/` | Logik- und GUI-Tests |
| `prompt.md` | projektspezifische Anforderungen |
