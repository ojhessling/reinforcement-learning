# HalfCheetah Workbench

Lokale Tkinter-Anwendung zum Konfigurieren, Trainieren, Beobachten, Evaluieren
und Vergleichen von **PPO**, **SAC** und **TD3** auf dem MuJoCo-Environment
`HalfCheetah-v5`. Bis zu **vier** Verfahren laufen gleichzeitig nebeneinander.

Grundlage sind [`../workbench.md`](../workbench.md) und [`prompt.md`](prompt.md).

## Ziel

Ein zweibeiniger Roboter im Seitenprofil soll so schnell wie möglich nach
rechts laufen.

**Die eine Besonderheit gegenüber allen Vorgängerprojekten:** HalfCheetah endet
nie vorzeitig. Es gibt keinen Sturz, kein Ziel und keinen terminalen Zustand –
`terminated` ist immer `False`. Jede Episode läuft exakt 1000 Schritte, und der
Return misst nur, wie weit der Roboter kommt, abzüglich der Steuerkosten.

Daraus folgt: Es gibt hier **keine Sturz-, Durchhalte- oder Zielquote**, keinen
Überlebensbonus und keinen Clipping-Hinweis zu den Geschwindigkeiten. An ihre
Stelle treten Vorwärtsquote, Tempo, Strecke und Steuerkosten.

## Installation und Start

```bash
conda env create -f ../environment.yml     # oder: conda env update -f ../environment.yml
conda activate rl-26-08
python halfcheetah_app.py
```

Keine neue Abhängigkeit: `HalfCheetah-v5` nutzt dasselbe MuJoCo wie
`13-hopper`. Fehlt es in einer bestehenden Umgebung:

```bash
pip install "gymnasium[mujoco]"    # installiert mujoco und imageio
```

`imageio` ist kein Zubehör: `gymnasium.envs.mujoco.mujoco_rendering` importiert
es, ohne das Paket scheitert bereits das Erzeugen des Environments.

### Rendering

`MUJOCO_GL` wird plattformabhängig gesetzt, **bevor** MuJoCo importiert wird:
macOS `cgl`, headless Linux `egl` (ersatzweise `osmesa`). Auf macOS ist `glfw`
bewusst nicht in Gebrauch – es meldet jeden Renderprozess beim Window Server
als Vordergrund-App an und erzeugt damit je Animation einen Dock-Eintrag.
Einzelheiten in Workbench 7.6.

## Environment

```python
env = gymnasium.make("HalfCheetah-v5", render_mode="rgb_array", width=480, height=480)
```

`forward_reward_weight`, `ctrl_cost_weight`, `reset_noise_scale` und
`exclude_current_positions_from_observation` bleiben unangetastet – sie zu
verstellen wäre Reward Shaping. `width`/`height` entsprechen den
Gymnasium-Voreinstellungen und betreffen nur die Darstellung. Die Kamera ist
eine `trackcom`-Kamera und folgt dem Roboter.

### Actions

`Box(-1.0, 1.0, (6,), float32)`. Anders als beim Hopper hat **jeder Motor eine
eigene Übersetzung** – derselbe Actionwert bedeutet je Gelenk ein anderes
Moment:

| Index | Gelenk (XML) | `gear` | Moment bei `aᵢ = 1` |
| --- | --- | --- | --- |
| `a₀` | hinterer Oberschenkel (`bthigh`) | 120 | 120 N·m |
| `a₁` | hinterer Unterschenkel (`bshin`) | 90 | 90 N·m |
| `a₂` | hinterer Fuß (`bfoot`) | 60 | 60 N·m |
| `a₃` | vorderer Oberschenkel (`fthigh`) | 120 | 120 N·m |
| `a₄` | vorderer Unterschenkel (`fshin`) | 60 | 60 N·m |
| `a₅` | vorderer Fuß (`ffoot`) | 30 | 30 N·m |

Vorzeichen = Drehrichtung, Betrag mal `gear` = Moment. `0` heißt „kein
Moment", nicht „Gelenk hält die Position". Werte außerhalb `[-1, 1]` clippt das
Environment. Ein Test vergleicht die Tabelle mit `model.actuator_gear`.

### Observation

17 Werte, `float64`:

| Index | Bedeutung | Einheit |
| --- | --- | --- |
| 0 | Höhe der Rumpfspitze | m |
| 1 | Neigungswinkel des Rumpfes | rad |
| 2–7 | Winkel der sechs Gelenke, Reihenfolge wie bei den Actions | rad |
| 8–9 | Geschwindigkeiten `vₓ`, `v_z` | m/s |
| 10 | Winkelgeschwindigkeit des Rumpfes | rad/s |
| 11–16 | Winkelgeschwindigkeiten der sechs Gelenke | rad/s |

- Die **x-Position** fehlt: Der Agent weiß nicht, wo er ist, nur wie schnell.
  Für Anzeige und Metriken liest die App sie aus `info["x_position"]`.
- Die Geschwindigkeiten sind **nicht geclippt** – anders als beim Hopper. Ein
  guter Agent erreicht `vₓ` deutlich über 10.

### Reward und Episodenende

```text
reward = forward_reward − ctrl_cost = 1,0 · vₓ − 0,1 · Σ aᵢ²
```

- `forward_reward = 1,0 · vₓ` mit `vₓ = Δx / dt`, `dt = 0,05 s`. Rückwärts →
  negativ.
- `ctrl_cost = 0,1 · Σ aᵢ²`, höchstens `0,6` je Schritt, also bis zu `−600` je
  Episode. Der Faktor ist **hundertmal so groß wie beim Hopper**: Wildes
  Zappeln kostet sichtbar Return, sparsamer zu steuern ist Teil des
  Lernfortschritts.
- **Kein Überlebensbonus** – Stillstehen bringt `0`.

Beide Anteile stehen in `info` als `reward_forward` und `reward_ctrl` (bereits
negativ) und werden von dort übernommen, nicht nachgerechnet.

Die Episode endet **ausschließlich** durch den `TimeLimit`-Wrapper nach 1000
Schritten (`truncated=True`). Bei Truncation wird **gebootstrapt** – bei einem
Environment ohne terminalen Zustand ist das die einzig richtige Behandlung
(Workbench 3.3); Stable-Baselines3 erledigt das über `TimeLimit.truncated`
selbst.

| Ausgang | Bedeutung |
| --- | --- |
| **Gelöst** | Return `≥ 4800` (offizieller `reward_threshold`) |
| **Vorwärts** | mittleres `vₓ` der Episode positiv |
| **Rückwärts** | mittleres `vₓ` negativ oder null |

Rückwärtslaufen ist kein Randfall: Ein untrainierter Agent kippt regelmäßig auf
den Rücken und rudert von dort rückwärts. Der Return ist nach unten **nicht**
beschränkt und liegt realistisch zwischen etwa `−600` und `+10.000`.

Quelle: [Gymnasium HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)

## Methoden

Verwendet werden die Implementierungen von Stable-Baselines3 unverändert.

**PPO** ist on-policy: Rollouts fester Länge, Vorteile per GAE, mehrere Epochen
auf denselben Daten mit geclipptem Surrogatziel; die Daten werden danach
verworfen.

```text
L = E[ min( r(θ)·Â , clip(r(θ), 1−ε, 1+ε)·Â ) ]   mit r(θ) = π_θ(a|s) / π_alt(a|s)
```

**TD3** ist off-policy mit deterministischem Actor, zwei Critics und dem
Minimum als Ziel; geclipptes Rauschen auf die Target-Action, Actor-Updates nur
alle `policy_delay` Schritte:

```text
ã = clip(π_target(s') + clip(N(0, σ_t), −c, +c), a_min, a_max)
y = r + γ·(1−done)·min( Q₁_target(s', ã), Q₂_target(s', ã) )
```

**SAC** ist off-policy mit stochastischem Actor und maximiert zusätzlich die
Entropie:

```text
y = r + γ·(1−done)·[ min(Q₁_target, Q₂_target) − α·log π(a'|s') ]
```

Bei `α = auto` wird die Temperatur gelernt; der SB3-Standard ist
`target_entropy = −dim(A)`, bei sechs Actions also **−6**.

Quellen: [PPO](https://arxiv.org/abs/1707.06347),
[GAE](https://arxiv.org/abs/1506.02438),
[TD3](https://arxiv.org/abs/1802.09477),
[SAC](https://arxiv.org/abs/1801.01290),
[SAC mit gelernter Temperatur](https://arxiv.org/abs/1812.05905),
[gSDE](https://arxiv.org/abs/2005.05719)

## Parameter, Standardwerte und Quellen

`HalfCheetah-v4`-Profile des RL Baselines3 Zoo, Stand geprüft am 20.08.2026;
nicht enthaltene Werte sind SB3-Defaults 2.9.0.

| Parameter | PPO | SAC | TD3 |
| --- | --- | --- | --- |
| `total_timesteps` | 1.000.000 | 1.000.000 | 1.000.000 |
| `learning_rate` | 2,0633e-5 | 3e-4 | 1e-3 |
| `gamma` | 0,98 | 0,99 | 0,99 |
| `batch_size` | 64 | 256 | 256 |
| `n_steps` / `n_epochs` | 512 / 20 | – | – |
| `gae_lambda` / `clip_range` | 0,92 / 0,1 | – | – |
| `ent_coef` | 0,000401762 | `auto` | – |
| `vf_coef` / `max_grad_norm` | 0,58096 / 0,8 | – | – |
| `log_std_init` / `ortho_init` | −2,0 / aus | 0,0 / – | – |
| `buffer_size` / `learning_starts` | – | 1.000.000 / 10.000 | 1.000.000 / 10.000 |
| `tau` | – | 0,005 | 0,005 |
| `train_freq` / `gradient_steps` | – | 1 / 1 | 1 / 1 |
| Action Noise | – | keins | normal, σ = 0,1 |
| Hidden Layers (π und Q) | 256,256 | 256,256 | 256,256 |
| Aktivierung | ReLU | ReLU | ReLU |
| Normalisierung (Obs / Reward) | an / an | aus / aus | aus / aus |

Global: `Anzahl Verfahren` **4** (V1 PPO, V2 TD3, V3 SAC, V4 SAC mit
`learning_rate = 6e-4`, damit sich die beiden SAC-Slots unterscheiden),
`Eval-Intervall` **100.000**, `Eval-Episoden M` **5**, `Glättung` **20**.

`N = 1.000.000` ist für alle drei Verfahren gleich **und** exakt der Zoo-Wert –
der Standardwert weicht hier nicht vom Profil ab. Genau ein Wert tut das:

- **Hidden Layers `256,256` überall.** Der Zoo nennt für TD3 `400,300`; PPO und
  SAC liegen ohnehin bei `256,256`. Die Vereinheitlichung macht den Vergleich
  fair und kostet TD3 nichts.

> **Laufzeit:** Ein Lauf über `1e6` Schritte dauert auf einem Laptop
> **Stunden** – bei vier Slots parallel entsprechend länger, mit Animation
> zusätzlich. Für einen ersten Eindruck `Trainingsschritte N` auf **100.000**
> und `Eval-Intervall` auf **10.000** setzen. Wer das Budget verkleinert, muss
> das Intervall mitverkleinern, sonst gibt es keine Stützstelle. Achtung: Weil
> jede Episode 1000 Schritte dauert, liefert ein Budget unter 1000 Schritten
> **gar keine** abgeschlossene Episode. Jeder Off-Policy-Slot belegt mit dem
> vollen Replay Buffer rund 300 MB.

### Normalisierung

Das PPO-Profil verlangt `normalize: true` – die 17 Werte sind sehr
unterschiedlich skaliert, die Geschwindigkeiten ungeclippt und der Return wird
fünfstellig. Umgesetzt mit `VecNormalize`: Die Statistiken wachsen nur im
Training, Evaluation und Animation nutzen sie eingefroren, Graph und Summary
zeigen immer den **unnormalisierten** Return, und die Statistiken gehören zum
Checkpoint. `SAC` und `TD3` normalisieren nicht per Voreinstellung, weil der
Replay Buffer Beobachtungen mit veraltender Statistik speichert.

## Bedienablauf

1. `Anzahl Verfahren` wählen (2 bis 4, Standard 4). Beim Verkleinern fragt die
   GUI nach, falls ein wegfallender Slot Daten hat.
2. Je Slot einen Algorithmus wählen – gern mehrfach denselben. Ein
   Dropdown-Wechsel lädt das Zoo-Profil und setzt **nur** diesen Slot zurück.
3. Im jeweiligen Tab die Parameter setzen. Der sichtbare Tab ist das aktive
   Verfahren; alle Einzellauf-Buttons wirken darauf.
4. `Training starten / fortsetzen` trainiert das aktive Verfahren,
   `Vergleich starten / fortsetzen` alle aktiven Slots parallel.

Buttons zum manuellen Speichern und Laden gibt es nicht: Gesichert wird
automatisch der beste deterministische Evaluationsstand je Slot,
`Bestes Modell wiederherstellen` holt ihn zurück. Ein Stand mit fremder
Environment-Kennung – etwa `HalfCheetah-v4` oder `Hopper-v5` – wird abgelehnt.

## Interpretation der Ansichten

### Diagramme

Beide Graphen zeigen den explorativen Episoden-Return gegen die Episodennummer.
Slotfarben: V1 blau, V2 rot, V3 gelb, V4 grün, alle Kurven durchgezogen; die
Gelöst-Marke bei `4800` ist weiß gestrichelt. Rohwerte liegen dezent in
derselben Farbe darunter.

`Glättung` rechts in der Tableiste stellt ein, über wie viele Episoden der
gleitende Durchschnitt mittelt – **global für alle Slots und beide Graphen**,
damit die Kurven vergleichbar bleiben. Standard 20, gültig 1 bis 500; bei 1
fällt die geglättete Kurve mit den Rohwerten zusammen. Die Änderung wirkt sofort, auch
mitten in einem Lauf, und verändert nur die Darstellung.

Da jede Episode exakt 1000 Schritte dauert, entspricht die Episodennummer auf
der X-Achse hier genau dem tausendfachen Schrittfortschritt.

### Animation

Die Anzeigen liegen in einem Raster mit höchstens zwei Spalten und zwei Zeilen.
Jede trägt den Titel `Verfahren 3 – SAC`; teilen sich Slots einen Algorithmus,
ergänzt der Titel wie die Legende den abweichenden Parameter.

Über jedem Bild steht ein Feld `Episode`: `aktuell` zeigt den laufenden
Lernstand mit der zuletzt trainierten Episodennummer, `beste` spielt den Stand
der besten Episode mit festem Seed erneut ab (ein `*` hinter der Nummer macht
das kenntlich). Jede Animation wählt unabhängig.

**Unter jedem Bild steht nur eine Zeile:** `E: 57 · S: 354/1000 · R: 2.700,0`.
Alle weiteren Messwerte – die sechs Actions mit ihrer **gelenkeigenen**
Übersetzung, Zustand und Reward-Zerlegung – erscheinen erst beim Überfahren mit
der Maus und werden dann **neben** dem Bild eingeblendet.

Die Bildrate ist einstellbar (1 bis 250, Standard **20 FPS** – die
environment-eigene Rate). Achtung: Bei 20 FPS dauert eine Episode **50
Sekunden**; zum Zuschauen lohnen sich 100 bis 200 FPS.

### Summary

Je aktivem Slot eine Spalte: Algorithmus, Episoden, ausgeführte Schritte und
Budget, Ø Return, beste Episode mit Nummer und Return, Ø Tempo, Ø Strecke,
Ø Steuerkosten, Vorwärtsquote und Gelöst-Quote. **Keine Zeile für die
Episodenlänge** – sie wäre immer 1000. Darunter listet ein Abschnitt genau die
Parameter, in denen sich die Slots unterscheiden.

`PNG exportieren` und `TXT exportieren` liegen nebeneinander rechts in der
Tableiste der Diagramme.

## Grenzen und Erwartung

Anders als beim Hopper ist die Gelöst-Marke hier erreichbar. Zoo-Benchmark nach
`1e6` Schritten:

| Verfahren | mittlerer Return | Standardabweichung |
| --- | --- | --- |
| `PPO` | 5819,1 | 663,5 |
| `SAC` | 9535,5 | 100,5 |
| `TD3` | 9655,7 | 969,9 |

Alle drei übertreffen die `4800`, `SAC` und `TD3` rund um das Doppelte. Der
große Abstand illustriert die Fairness-Regel aus Workbench 5.8: Bei gleichem
Schrittbudget lernen die Off-Policy-Verfahren aus jedem Übergang mehrfach, PPO
verwirft seine Daten nach jedem Update. Quelle:
[benchmark.md](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md)

Weitere Grenzen:

- In den ersten Episoden dominieren die Steuerkosten: Ein untrainierter Agent
  zappelt, kommt kaum vom Fleck und landet bei Returns um `−300`. Der Return
  steigt erst, wenn die Steuerung sparsamer und die Bewegung gerichtet wird.
- Vier gleichzeitige Animationen kosten spürbar Rechenzeit: vier
  Renderprozesse mit je eigener MuJoCo-Instanz.
- Mehrere Wiederholungen je Slot (Seed-Mittelung mit Unsicherheitsband) sind
  nicht implementiert; verglichen wird je Slot ein Lauf.
- Trainiert wird mit **einer** Environment-Instanz – so wie es auch das
  PPO-Zoo-Profil für dieses Environment vorsieht (`n_envs: 1`).

## Tests

```bash
python -m pytest tests -q
```

141 Tests. Geprüft werden unter anderem: Environment-Kennwerte (Spaces,
Zeitlimit, `reward_threshold`, Bildrate, `dt`, Framegröße, keine
Physik-Argumente), die sechs **unterschiedlichen** Übersetzungen gegen
`model.actuator_gear`, die Reward-Zerlegung aus `info` ohne Überlebensbonus,
die Steuerkostenformel, **dass das Environment nie terminiert und jede Episode
exakt 1000 Schritte dauert**, die fehlende Geschwindigkeits-Clippung, die
x-Position nur aus `info`, die SAC-Zielentropie `−6`, Save/Load-Roundtrips samt
Normalisierungsstatistiken, die Ablehnung fremder Environment-Kennungen, ein
Vergleichslauf über vier Slots sowie GUI-Layout, Slotverwaltung, Farbzuordnung,
Linienstile, Glättungsregler und die Trennung von Bildbeschriftung und
Hover-Einblendung.

Tests, die MuJoCo brauchen, werden ohne das Paket übersprungen; der
GUI-Smoke-Test ohne Display.

## Dateien

| Datei | Inhalt |
| --- | --- |
| `halfcheetah_app.py` | Entry Point |
| `halfcheetah_logic.py` | Environment-Factory, Konfiguration, Runner, Metriken, Checkpoints |
| `halfcheetah_gui.py` | Tkinter-Oberfläche, Diagramme, Animation, Summary, Export |
| `halfcheetah_render.py` | isolierter MuJoCo-Renderprozess |
| `tests/` | Logik- und GUI-Tests |
| `prompt.md` | projektspezifische Anforderungen |
