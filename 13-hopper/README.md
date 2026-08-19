# Hopper Workbench

Lokale Tkinter-Anwendung zum Konfigurieren, Trainieren, Beobachten, Evaluieren
und Vergleichen von **PPO**, **SAC** und **TD3** auf dem MuJoCo-Environment
`Hopper-v5`. Bis zu **vier** Verfahren laufen gleichzeitig nebeneinander.

Grundlage sind die verbindlichen Regeln aus [`../workbench.md`](../workbench.md)
und die projektspezifischen Vorgaben aus [`prompt.md`](prompt.md).

## Ziel

Ein einbeiniger Hüpfroboter soll vorwärts springen, ohne umzufallen. Hopper ist
ein Balance-Environment: Jeder Schritt, den der Roboter „gesund" übersteht,
bringt Reward – zusätzlich zur Vorwärtsgeschwindigkeit. Die Aufgabe ist damit
eine andere als bei den Box2D-Vorgängerprojekten, in denen ein Sturz hart
bestraft wurde.

## Installation und Start

Das Projekt nutzt die gemeinsame Umgebung [`../environment.yml`](../environment.yml):

```bash
conda env create -f ../environment.yml     # oder: conda env update -f ../environment.yml
conda activate rl-26-08
```

`Hopper-v5` braucht **MuJoCo**. Das ist die erste neue Abhängigkeit seit Box2D;
sie steht in `environment.yml` und in [`requirements.txt`](requirements.txt).
Falls die Umgebung schon existiert, genügt:

```bash
pip install "gymnasium[mujoco]"    # installiert mujoco und imageio
```

`imageio` ist kein Zubehör: `gymnasium.envs.mujoco.mujoco_rendering` importiert
es, sodass ohne dieses Paket bereits das Erzeugen des Environments scheitert.
Für MuJoCo gibt es fertige Binärräder für macOS, Linux und Windows; ein
Compiler ist nicht nötig. Das ältere `mujoco-py` wird **nicht** verwendet und
unterstützt `Hopper-v5` auch nicht.

Start:

```bash
python hopper_app.py
```

### Rendering und `MUJOCO_GL`

MuJoCo rendert über einen eigenen OpenGL-Kontext, nicht über SDL. Die
Anwendung setzt `MUJOCO_GL` plattformabhängig, **bevor** MuJoCo importiert
wird:

| Plattform | Backend | Bemerkung |
| --- | --- | --- |
| macOS | `cgl` | reines Offscreen-Rendering, kein Fenster, kein Dock-Eintrag |
| Linux | `egl` | headless; Ersatz bei fehlendem EGL: `osmesa` |

Auf macOS ist `glfw` bewusst **nicht** in Gebrauch, obwohl es funktioniert und
Gymnasiums Standardliste es führt: `glfw.init()` meldet seinen Prozess beim
Window Server als Vordergrund-App an. Da jede Anzeige einen eigenen Prozess
bekommt, entstünden bei vier Animationen vier zusätzliche Programm- und
Dock-Einträge neben der eigentlichen Anwendung. Nachprüfbar mit
`lsappinfo list`: unter `glfw` steht dort jeder Renderprozess als
`type="Foreground"`, unter `cgl` keiner.

Gymnasiums `MujocoRenderer` führt in seiner Tabelle nur `glfw`, `egl` und
`osmesa` und lehnt `cgl` sonst ab, obwohl MuJoCo den Kontext unter
`mujoco.cgl` mitbringt. `hopper_render.register_backend()` ergänzt den Eintrag,
bevor das Environment entsteht; die mitgelieferten Backends bleiben
unangetastet.

Eine bereits gesetzte Variable bleibt unverändert – wer ein anderes Backend
braucht, setzt `MUJOCO_GL` vor dem Start selbst.

Jede sichtbare Anzeige läuft in einem **eigenen Prozess** mit eigener
Environment-Instanz; Tkinter und ein nativer Grafikkontext dürfen unter macOS
nicht im selben Prozess leben. Bei vier Anzeigen laufen also vier
Hilfsprozesse. Sie erzeugen weder ein Fenster noch einen Dock-Eintrag und
werden beim Schließen beendet. Scheitert MuJoCo, meldet die Anwendung das verständlich, schaltet die
Animation ab und bleibt für Training und Evaluation voll nutzbar.

## Environment

Erzeugt wird ausschließlich so:

```python
env = gymnasium.make("Hopper-v5", render_mode="rgb_array", width=480, height=480)
```

`width` und `height` entsprechen den Gymnasium-Voreinstellungen und betreffen
nur die Bildgröße. Alle physikalischen Parameter – `forward_reward_weight`,
`ctrl_cost_weight`, `healthy_reward`, `terminate_when_unhealthy`, die drei
`healthy_*_range`, `reset_noise_scale` und
`exclude_current_positions_from_observation` – bleiben unangetastet. Sie zu
verstellen wäre Reward Shaping beziehungsweise eine Änderung der Abbruchregeln.

Verwendet wird `Hopper-v5`, nicht `Hopper-v4`: `v5` ist die gepflegte Fassung
mit vollständigem `info`-Dictionary. Die Hyperparameterprofile des RL
Baselines3 Zoo sind für `Hopper-v4` hinterlegt; die Unterschiede betreffen
Detailkorrekturen der Umgebung, nicht die Größenordnung der Hyperparameter.

### Actions

`Box(-1.0, 1.0, (3,), float32)` – Drehmomente für drei Gelenke. Alle drei
Motoren haben in `hopper.xml` die Übersetzung `gear = 200`:

| Index | Gelenk | Moment |
| --- | --- | --- |
| `a₀` | Hüfte (`thigh_joint`) | `a₀ · 200 N·m` |
| `a₁` | Knie (`leg_joint`) | `a₁ · 200 N·m` |
| `a₂` | Sprunggelenk (`foot_joint`) | `a₂ · 200 N·m` |

Das Vorzeichen bestimmt die Drehrichtung, der Betrag das Moment. `0` heißt
„kein Moment", nicht „Gelenk hält die Position". Werte außerhalb `[-1, 1]`
clippt das Environment.

### Observation

11 Werte, `float64`:

| Index | Bedeutung | Einheit |
| --- | --- | --- |
| 0 | Höhe des Rumpfes | m |
| 1 | Neigungswinkel des Rumpfes | rad |
| 2–4 | Winkel von Hüfte, Knie, Sprunggelenk | rad |
| 5–6 | Geschwindigkeiten `vₓ`, `v_z` | m/s |
| 7 | Winkelgeschwindigkeit des Rumpfes | rad/s |
| 8–10 | Winkelgeschwindigkeiten der drei Gelenke | rad/s |

Zwei Eigenheiten:

- Die **x-Position** fehlt in der Observation
  (`exclude_current_positions_from_observation=True`). Der Agent weiß nicht,
  wie weit er gekommen ist – nur, wie schnell er ist. Für Anzeige und Metriken
  liest die Anwendung sie aus `info["x_position"]`; auf Interna wie
  `env.unwrapped.data.qpos` greift sie nicht zu.
- Alle sechs Geschwindigkeiten sind im Environment auf `[-10, 10]` **geclippt**.
  Ein Wert von genau `±10` heißt „mindestens so schnell", nicht „genau so
  schnell".

### Reward und Episodenende

```text
reward = healthy_reward + forward_reward − ctrl_cost
       = 1,0            + 1,0 · vₓ       − 0,001 · Σ aᵢ²
```

- `healthy_reward = 1,0` je gesundem Schritt – der Überlebensbonus. Bloßes
  Stehenbleiben bringt über eine volle Episode bereits `+1000`.
- `forward_reward = 1,0 · vₓ` mit `vₓ = Δx / dt`, `dt = 0,008 s`.
- `ctrl_cost = 0,001 · Σ aᵢ²`, höchstens `0,003` je Schritt.

Die drei Anteile stehen einzeln in `info` als `reward_survive`,
`reward_forward` und `reward_ctrl` (bereits negativ) und werden von dort
übernommen, nicht nachgerechnet.

Die Episode endet

- mit `terminated`, sobald der Roboter **ungesund** wird: Höhe `z ≤ 0,7 m`,
  Rumpfwinkel außerhalb `(−0,2; 0,2) rad`, ein Zustandswert außerhalb
  `(−100; 100)` oder ein nicht endlicher Wert – umgangssprachlich: er fällt um;
- mit `truncated` nach `1000` Schritten durch den `TimeLimit`-Wrapper.

Es gibt **weder Terminalbonus noch Terminalstrafe**. Ein Sturz kostet nichts
weiter als alle Rewards der Schritte, die nicht mehr stattfinden. Genau das ist
die Lernaufgabe: lange leben und dabei schnell sein. `Hopper-v5` selbst liefert
immer `truncated=False`; das Zeitlimit setzt ausschließlich der Wrapper.

Erfolgsdefinitionen:

| Ausgang | Bedeutung |
| --- | --- |
| **Durchgehalten** | `truncated` nach 1000 Schritten – hier der **gute** Ausgang |
| **Sturz** | `terminated`, ungesund geworden |
| **Gelöst** | Episoden-Return `≥ 3800` (offizieller `reward_threshold`) |

Durchhalte- und Sturzquote sind komplementär; beide werden getrennt
ausgewiesen, damit die Summary ohne Kopfrechnen lesbar bleibt. Höhere Werte
sind besser; der Return liegt realistisch zwischen etwa `+15` (sofortiger
Sturz) und `+4000`.

Quelle: [Gymnasium Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/)

## Methoden

Verwendet werden die Implementierungen von Stable-Baselines3 unverändert.

**PPO** ist on-policy: Es sammelt Rollouts fester Länge, schätzt Vorteile mit
Generalized Advantage Estimation und optimiert über mehrere Epochen auf
denselben Daten ein geclipptes Surrogatziel

```text
L = E[ min( r(θ)·Â , clip(r(θ), 1−ε, 1+ε)·Â ) ]   mit r(θ) = π_θ(a|s) / π_alt(a|s)
```

Die Daten werden danach verworfen; einen Replay Buffer gibt es nicht.

**TD3** ist off-policy mit deterministischem Actor. Es lernt zwei Critics und
bildet das Ziel aus deren Minimum; auf die Target-Action kommt geclipptes
Rauschen, Actor und Target-Netze werden nur alle `policy_delay` Updates
aktualisiert:

```text
ã = clip(π_target(s') + clip(N(0, σ_t), −c, +c), a_min, a_max)
y = r + γ·(1−done)·min( Q₁_target(s', ã), Q₂_target(s', ã) )
```

Ohne explizites Action Noise exploriert TD3 überhaupt nicht; `keins` wird
deshalb mit einer Meldung abgelehnt.

**SAC** ist off-policy mit stochastischem Actor und maximiert zusätzlich die
Entropie der Policy:

```text
y = r + γ·(1−done)·[ min(Q₁_target, Q₂_target) − α·log π(a'|s') ]
```

Bei `α = auto` wird die Temperatur gelernt, sodass die mittlere Entropie einer
Zielentropie folgt. Der SB3-Standard ist `target_entropy = −dim(A)`, bei drei
Actions also **−3**.

Quellen: [PPO](https://arxiv.org/abs/1707.06347),
[GAE](https://arxiv.org/abs/1506.02438),
[TD3](https://arxiv.org/abs/1802.09477),
[SAC](https://arxiv.org/abs/1801.01290),
[SAC mit gelernter Temperatur](https://arxiv.org/abs/1812.05905),
[gSDE](https://arxiv.org/abs/2005.05719)

### Fairness von On-Policy gegen Off-Policy

Bei gleichem Schrittbudget fällt der Vergleich systematisch zugunsten von SAC
und TD3 aus: Sie lernen aus jedem gespeicherten Übergang mehrfach, PPO
verwirft seine Daten nach jedem Update. Das ist kein Messfehler, sondern eine
Eigenschaft der Verfahrensklassen – und wiegt umso schwerer, wenn ein Vergleich
mit drei oder vier Slots ein On-Policy-Verfahren gegen mehrere
Off-Policy-Verfahren stellt.

Immerhin entsteht hier **kein zusätzlicher** Nachteil durch die eine
Environment-Instanz: Das Zoo-Profil für Hopper verwendet selbst `n_envs = 1`,
der Rollout von 512 Schritten je Update entspricht also exakt dem Profil.

## Parameter, Standardwerte und Quellen

Grundlage sind die `Hopper-v4`-Profile des RL Baselines3 Zoo (Stand geprüft am
19.08.2026); nicht enthaltene Werte stammen aus den Voreinstellungen von
Stable-Baselines3 2.9.0.

| Parameter | PPO | SAC | TD3 |
| --- | --- | --- | --- |
| `total_timesteps` | 1.000.000 | 1.000.000 | 1.000.000 |
| `learning_rate` | 9,80828e-5 | 3e-4 | 1e-3 |
| `gamma` | 0,999 | 0,99 | 0,99 |
| `batch_size` | 32 | 256 | 256 |
| `n_steps` / `n_epochs` | 512 / 5 | – | – |
| `gae_lambda` / `clip_range` | 0,99 / 0,2 | – | – |
| `ent_coef` | 0,00229519 | `auto` | – |
| `vf_coef` / `max_grad_norm` | 0,835671 / 0,7 | – | – |
| `log_std_init` / `ortho_init` | −2,0 / aus | 0,0 / – | – |
| `buffer_size` / `learning_starts` | – | 1.000.000 / 10.000 | 1.000.000 / 10.000 |
| `tau` | – | 0,005 | 0,005 |
| `train_freq` / `gradient_steps` | – | 1 / 1 | 1 / 1 |
| Action Noise | – | keins | normal, σ = 0,1 |
| Hidden Layers (π und Q) | 256,256 | 256,256 | 256,256 |
| Aktivierung | ReLU | ReLU | ReLU |
| Normalisierung (Obs / Reward) | an / an | aus / aus | aus / aus |

Global außerhalb der Tabs: `Anzahl Verfahren` (Standard **4**),
`Eval-Intervall` **100.000** – ein Zehntel des Schrittbudgets, ein Standardlauf
liefert also zehn Stützstellen – und `Eval-Episoden M` **5**.

Das Trainingsbudget `N = 1.000.000` ist für alle drei Verfahren gleich und
zugleich genau der Wert, den alle drei Zoo-Profile nennen: Der Standardwert
weicht hier **nicht** vom Profil ab. Genau ein Wert tut das:

- **Hidden Layers `256,256` überall.** Der Zoo nennt für TD3 `400,300`; PPO und
  SAC liegen ohnehin bei `256,256`. Die Vereinheitlichung macht den Vergleich
  fair und kostet TD3 nichts an Qualität.

Alles bleibt in der UI frei änderbar. Alle übrigen getunten Hyperparameter sind
unverändert übernommen.

> **Laufzeit:** Ein Lauf über `1e6` Schritte dauert auf einem Laptop
> **Stunden** – bei mehreren Slots parallel entsprechend länger, mit
> eingeschalteter Animation zusätzlich. Für einen ersten Eindruck lohnt es
> sich, `Trainingsschritte N` auf etwa **100.000** und `Eval-Intervall` auf
> **10.000** zu setzen: Dann stehen die Verfahren in Minuten nebeneinander, die
> Kurven brechen allerdings früh ab. Wer das Budget verkleinert, verkleinert
> das Evaluationsintervall mit – sonst liefert der Lauf gar keine Stützstelle.
> Jeder Off-Policy-Slot belegt mit dem vollen Buffer rund 210 MB
> Arbeitsspeicher; bei vier Slots verkleinert man `buffer_size` besser.

Quellen: [Zoo PPO](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml),
[Zoo SAC](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml),
[Zoo TD3](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/td3.yml)

### Normalisierung

Das PPO-Profil verlangt `normalize: true`. Umgesetzt ist das mit `VecNormalize`.
Verbindlich gilt:

- Die laufenden Statistiken wachsen **nur im Training**. Deterministische
  Evaluation und Animation verwenden sie eingefroren.
- Graph, Summary und Evaluation zeigen immer den **unnormalisierten** Return –
  sonst wäre die Marke von 3800 bedeutungslos. Der `Monitor`-Wrapper sitzt dazu
  innerhalb der Normalisierung.
- Die Statistiken gehören zum Checkpoint. Ein ohne sie geladenes Modell
  verhielte sich anders als das gespeicherte.

Off-Policy-Verfahren normalisieren nicht per Voreinstellung: Ein Replay Buffer
speichert Beobachtungen, deren Normalisierungsstatistik sich weiter verschiebt.
Wählbar bleibt die Option trotzdem.

## Bedienablauf

1. **`Anzahl Verfahren`** wählen (2 bis 4, Standard 4: PPO, TD3, SAC und ein
   zweiter SAC-Slot, der beim Start eine abweichende Lernrate von `6e-4`
   bekommt – sonst wären die beiden SAC-Slots identisch. Diese Abweichung gilt
   nur für die Startbelegung; später hinzugefügte Slots starten mit den
   unveränderten Standardwerten ihres Algorithmus). Nicht aktive Slots
   verschwinden vollständig – Dropdown und Tab werden gar nicht erst erzeugt.
   Beim Verkleinern fragt die GUI nach, falls ein wegfallender Slot bereits
   einen Lernzustand besitzt; neue Slots starten mit den Standardwerten ihres
   Algorithmus, bestehende bleiben unberührt. Während eines Laufs ist das Feld
   gesperrt.
2. Je Slot einen Algorithmus wählen – gern mehrfach denselben, um
   Parametrisierungen zu vergleichen. Ein Dropdown-Wechsel lädt das Zoo-Profil
   und setzt **nur diesen** Slot zurück.
3. Im jeweiligen Tab die Parameter setzen. Der sichtbare Tab ist das aktive
   Verfahren (`Aktiv: Verfahren 2 – SAC`); alle Einzellauf-Buttons wirken
   darauf. Läuft bereits ein Einzellauf, bleibt dessen Ziel fixiert.
4. `Training starten / fortsetzen` trainiert das aktive Verfahren,
   `Vergleich starten / fortsetzen` alle aktiven Slots parallel. Ein erneut
   gestarteter, kompatibel konfigurierter Vergleich setzt nicht zurück, sondern
   hängt neue Messpunkte an.

Buttons zum manuellen Speichern und Laden gibt es nicht. Gesichert wird
automatisch der beste deterministische Evaluationsstand je Slot;
`Bestes Modell wiederherstellen` holt ihn zurück. Zum Checkpoint gehören je
nach Verfahren Policy, Value-Netz, beide Critics, Target-Netze, Optimizer,
Replay Buffer, der gelernte Temperaturparameter und die
Normalisierungsstatistiken. Ein Stand mit fremder Environment-Kennung – etwa
`Hopper-v4` oder `Walker2d-v5` – wird verständlich abgelehnt, ebenso ein Stand,
dessen Algorithmus nicht zum aktiven Slot passt.

## Interpretation der Ansichten

### Diagramme

Der Vergleichsgraph zeigt den explorativen Episoden-Return gegen die
Episodennummer. Die Slots werden allein über die **Farbe** unterschieden, jede
hervorgehobene Kurve ist **durchgezogen**:

| Slot | Farbe |
| --- | --- |
| `Verfahren 1` | Blau |
| `Verfahren 2` | Rot |
| `Verfahren 3` | Gelb |
| `Verfahren 4` | Grün |

Rohwerte erscheinen in derselben Farbe mit `10 %` Deckkraft; die kräftige Linie
(Strichstärke `1,2`) ist der gleitende Durchschnitt der letzten 20 Episoden. Die
Gelöst-Marke bei `3800` ist **weiß und gestrichelt** – Weiß gehört keinem Slot,
sodass eine Referenzlinie nie mit einer Datenlinie verwechselt werden kann.

Beide Graphen zeigen ausschließlich den explorativen Episoden-Return. Die
deterministischen Zwischenevaluationen sind **nicht** eingezeichnet – sie
liefen auf einer anderen Stützstellenzahl und lenkten von der Lernkurve ab;
ausgewiesen werden sie in der Summary. Eine Überschrift über den Achsen gibt es
nicht, der Platz gehört den Kurven, und die Y-Achse heißt schlicht `Return`.
Dass höhere Werte besser sind, sagen Legende, Summary und Bedienungsanleitung.

Ein Standardlauf erreicht `3800` nicht. Die Y-Achse folgt deshalb den Daten und
zwingt die Marke nicht in den sichtbaren Bereich; sonst wären alle Kurven unten
zusammengedrückt. Die Legende nennt den Wert, die Summary die Gelöst-Quote.

Kurvenpunkte entstehen nur für vollständig abgeschlossene Episoden. Endet das
Budget mitten in einer Episode, liegt der letzte Punkt deshalb vor dem
ausgeführten Schrittbudget; Summary und Status zeigen ausgeführte Schritte und
angefordertes Budget getrennt.

### Animation

`Animation zeigen` wirkt jederzeit, auch mitten in einem Lauf. Sie arbeitet auf
einer **Kopie der Policy**, die zu Beginn jeder sichtbaren Episode gezogen
wird; jede Episode zeigt also den Lernstand zu ihrem Beginn. Die
Trainingsschleife selbst rendert nicht.

Die Anzeigen liegen in einem Raster mit höchstens zwei Spalten und zwei Zeilen:

| Sichtbare Anzeigen | Raster |
| --- | --- |
| 1 | eine Anzeige über den gesamten Bereich |
| 2 | zwei nebeneinander |
| 3 | zwei oben, eine unten |
| 4 | zwei je Zeile und zwei je Spalte |

Jede Animation hat **über** ihrem Bild ein eigenes Auswahlfeld `Episode`, das
unabhängig von den anderen Anzeigen wirkt:

| Wert | Bedeutung |
| --- | --- |
| `aktuell` | der laufende Lernstand; die Beschriftung nennt die Nummer der zuletzt trainierten Episode – dieselbe Nummer wie auf der X-Achse der Graphen |
| `beste` | der Lernstand der bisher besten Episode (höchster explorativer Return), mit festem Seed, sodass die Wiederholung jedes Mal gleich aussieht |

Gesichert wird dafür je Slot **genau ein** zusätzlicher Lernstand: der der
besten Episode. Er entsteht unmittelbar nach dem Episodenende im Worker-Thread,
weil nur dort die Policy tatsächlich zu dieser Episode gehört. Ein Verlauf über
alle Episoden ist bewusst nicht vorgesehen – bei `1e6` Schritten entstehen
Tausende Episoden, deren Policy-Kopien Gigabytes belegten. `Neues Modell`
verwirft den gesicherten Stand mit.

Jede Anzeige trägt den Titel `Verfahren 3 – SAC`. Teilen sich mehrere Slots
einen Algorithmus, ergänzt der Titel wie die Legende des Vergleichsgraphen den
wichtigsten abweichenden Parameter: `Verfahren 3 – SAC (Lernrate α 0.0003)`.

**Unter jedem Bild steht nur eine Zeile**, kurz gehalten, weil die Rasterzellen
schmal sind: `E:   57  · S:  354/1000 · R:   2.700,0` – Episode, Schritt und
bisher kumulierter Return. Das Verfahren steht im Titel darüber. Läuft gerade
die beste statt der aktuellen Episode, markiert ein `*` hinter der
Episodennummer das.
Alle weiteren Messwerte – die drei Actions samt Moment in `N·m`, Höhe,
Rumpfwinkel, Gelenkwinkel, alle Geschwindigkeiten und die Zerlegung des letzten
Rewards – erscheinen erst, **wenn der Mauszeiger über dem Bild steht**. Sie
werden dann **neben** dem Bild eingeblendet, sodass Bild und Werte gleichzeitig
lesbar bleiben; das Bild selbst wird dabei weder verdeckt noch verschoben.
Wer die Werte unter dem Bild sucht, findet sie dort also bewusst nicht.

Die Bildrate ist einstellbar (1 bis 250, Standard **125 FPS** – die
environment-eigene Rate `1/dt`). Eine Episode über 1000 Schritte dauert damit
gut acht Sekunden. Die Workbench nennt sonst 1 bis 120; das läge hier unter dem
Standardwert und machte ihn selbst ungültig. Die eingestellte Rate ist eine
**Obergrenze**: Während eines Laufs konkurrieren Training und Rendern um
Rechenzeit, und der Effekt wächst mit der Zahl gleichzeitig sichtbarer
Verfahren. Ohne Animation läuft alles unverändert weiter.

### Summary

Die Vergleichs-Summary hat genau eine Ergebnisspalte je aktivem Slot. Sie zeigt
Algorithmus, Episoden, ausgeführte Schritte und Budget, Ø Return, die **beste
Episode mit Nummer und Return** (etwa `#137: 1.204,3`), Ø Länge sowie
Durchhalte-, Sturz- und Gelöst-Quote, dazu die letzte Evaluation je Slot mit
mittlerer Geschwindigkeit und zurückgelegter Strecke. Die deterministischen
Zwischenevaluationen stehen ausschließlich hier, nicht im Graphen. Darunter listet ein
Abschnitt genau die Parameter auf, in denen sich die Slots unterscheiden – je
Zeile ein Parameter mit dem Wert aller Slots. Verglichen werden nur Parameter,
die **alle** beteiligten Verfahren besitzen; verfahrenseigene stehen
vollständig im jeweiligen Tab.

Diagramm (PNG) und Summary (UTF-8-Text) lassen sich exportieren; die Dialoge
schlagen sprechende Dateinamen vor und überschreiben nichts unbemerkt. Beide
Schaltflächen stehen nebeneinander rechts in der Tableiste der Diagramme:
`PNG exportieren` speichert das gerade sichtbare Diagramm, `TXT exportieren`
die Summary daneben. Dort ist die Fläche ohnehin frei, sodass keine eigene
Kopfzeile Höhe kostet und weder Kurven noch Legende verdeckt werden. Die
Vergleichs-Summary trägt keine Überschrift; die Spaltenköpfe sagen bereits, was
verglichen wird.

## Grenzen

- **Auch das volle Profilbudget löst Hopper im Mittel nicht.** Der Benchmark
  des RL Baselines3 Zoo weist für `Hopper-v3` nach `1e6` Schritten aus:

  | Verfahren | mittlerer Return | Standardabweichung |
  | --- | --- | --- |
  | `PPO` | 2410,4 | 10,0 |
  | `SAC` | 2325,5 | 1129,7 |
  | `TD3` | 3606,4 | 4,0 |

  Die Marke von `3800` erreicht damit keines der drei Verfahren im Mittel;
  `TD3` kommt am nächsten heran, `SAC` streut extrem von Seed zu Seed. Ein
  sichtbar laufender Roboter entsteht trotzdem. Quelle:
  [benchmark.md](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md)
- Der Überlebensbonus von `+1` je Schritt macht die Kurven anfangs vor allem zu
  einer Kurve der Episodenlänge. Erst wenn ein Agent zuverlässig über mehrere
  Hundert Schritte kommt, trennt der Geschwindigkeitsanteil die Verfahren.
- Vier gleichzeitige Animationen kosten spürbar Rechenzeit und Speicher: vier
  Renderprozesse mit je eigener MuJoCo-Instanz.
- Mehrere Wiederholungen je Slot (Seed-Mittelung mit Unsicherheitsband) sind
  nicht implementiert; verglichen wird je Slot ein Lauf.
- Trainiert wird mit **einer** Environment-Instanz, damit Animation,
  Episodenbuchhaltung und schrittgenaue Zwischenevaluation eindeutig bleiben.

## Tests

```bash
python -m pytest tests -q
```

Geprüft werden unter anderem: Environment-Kennwerte (Action-Space,
Observation, Zeitlimit, `reward_threshold`, Bildrate, Framegröße, keine
Physik-Argumente), die Interpretation der Drehmomente mit `gear = 200`, die
Reward-Zerlegung aus `info`, die Unterscheidung von Sturz und Durchhalten ohne
Terminalstrafe, x-Position nur aus `info`, die Geschwindigkeitsclippung, die
SAC-Zielentropie `−3`, verzögerte TD3-Actor-Updates, das Verwerfen der Daten
bei PPO, Save/Load-Roundtrips samt Normalisierungsstatistiken, die Ablehnung
fremder Environment-Kennungen und Algorithmen, ein Vergleichslauf über vier
Slots sowie GUI-Layout, Slotverwaltung, Farbzuordnung, Linienstile und die
Trennung von Bildbeschriftung und Hover-Einblendung.

Tests, die MuJoCo brauchen, werden ohne das Paket übersprungen; der
GUI-Smoke-Test wird ohne Display übersprungen.

## Dateien

| Datei | Inhalt |
| --- | --- |
| `hopper_app.py` | Entry Point |
| `hopper_logic.py` | Environment-Factory, Konfiguration, Runner, Metriken, Checkpoints |
| `hopper_gui.py` | Tkinter-Oberfläche, Diagramme, Animation, Summary, Export |
| `hopper_render.py` | isolierter MuJoCo-Renderprozess |
| `tests/` | Logik- und GUI-Tests |
| `prompt.md` | projektspezifische Anforderungen |
