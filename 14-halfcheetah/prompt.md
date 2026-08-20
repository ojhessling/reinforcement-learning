# HalfCheetah – RL-Workbench

Projektordner: `Oliver/14-halfcheetah`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `halfcheetah`
Environment: `HalfCheetah-v5`

Dieser Prompt enthält nur die sieben Punkte aus Workbench-Abschnitt 1.1. Alles
Allgemeine steht dort und wird hier nicht wiederholt. Keine Abweichungen von
der Workbench.

## Ziel

Tkinter-Lernanwendung, die einen zweibeinigen Roboter im Seitenprofil so
schnell wie möglich nach rechts laufen lässt. Verfahren wie im Hopper-Projekt:
`PPO`, `SAC`, `TD3`.

**Die eine Besonderheit:** HalfCheetah endet nie vorzeitig – kein Sturz, kein
Ziel, kein terminaler Zustand. Jede Episode läuft exakt 1000 Schritte. Der
Return misst nur, wie weit der Roboter kommt, abzüglich der Steuerkosten.

Daraus folgt zwingend: **Sturzquote, Durchhaltequote und Zielquote entfallen**,
ebenso der Clipping-Hinweis zu den Geschwindigkeiten und der Überlebensbonus.
All das gibt es in diesem Environment nicht (Workbench 8.1).

## Startbelegung und globale Einstellungen

- `Anzahl Verfahren` = `4`: `V1 = PPO`, `V2 = TD3`, `V3 = SAC`, `V4 = SAC`.
  Damit steht beides da: Vergleich der drei Algorithmen und zweier
  Parametrisierungen desselben Verfahrens.
- Abweichender Startwert des zweiten SAC-Slots gemäß Workbench 6.2:
  `V4` startet mit `learning_rate = 6e-4` statt `3e-4`.
- `Eval-Intervall = 100.000` (ein Zehntel des Budgets, zehn Stützstellen),
  `Eval-Episoden M = 5`.

## Environment

```python
import gymnasium

env = gymnasium.make(
    "HalfCheetah-v5",
    render_mode="rgb_array",   # entfällt bei headless Evaluation
    width=480,
    height=480,
)
```

- `forward_reward_weight`, `ctrl_cost_weight`, `reset_noise_scale` und
  `exclude_current_positions_from_observation` werden **nicht** übergeben und
  nicht verändert – das wäre Reward Shaping.
- `width`/`height` entsprechen den Gymnasium-Voreinstellungen, werden aber
  mitgegeben, damit die Framegröße für Tests und Layout eindeutig ist. Die
  Kamera ist eine `trackcom`-Kamera und folgt dem Roboter.
- `v5` statt `v4`: gepflegte Fassung mit dokumentierter
  `observation_structure`. Die Zoo-Profile sind für `v4` hinterlegt; der
  Unterschied betrifft Detailkorrekturen, nicht die Größenordnung der
  Hyperparameter.

### Actions

`Box(-1.0, 1.0, (6,), float32)`. Anders als beim Hopper hat **jeder Motor eine
eigene Übersetzung** – derselbe Actionwert bedeutet je Gelenk ein anderes
Moment:

| Index | Gelenk (XML) | `gear` |
| --- | --- | --- |
| `a₀` | hinterer Oberschenkel (`bthigh`) | 120 |
| `a₁` | hinterer Unterschenkel (`bshin`) | 90 |
| `a₂` | hinterer Fuß (`bfoot`) | 60 |
| `a₃` | vorderer Oberschenkel (`fthigh`) | 120 |
| `a₄` | vorderer Unterschenkel (`fshin`) | 60 |
| `a₅` | vorderer Fuß (`ffoot`) | 30 |

Vorzeichen = Drehrichtung, Betrag mal `gear` = Moment in N·m. `0` heißt „kein
Moment", nicht „Gelenk hält die Position". Werte außerhalb `[-1, 1]` clippt das
Environment (`ctrlrange="-1 1"`).

### Observation

17 Werte, `float64`, `Box(-inf, inf, (17,))`:

| Index | Bedeutung | Einheit |
| --- | --- | --- |
| 0 | Höhe der Rumpfspitze (`rootz`) | m |
| 1 | Neigungswinkel des Rumpfes (`rooty`) | rad |
| 2–7 | Winkel der sechs Gelenke, Reihenfolge wie bei den Actions | rad |
| 8 | Geschwindigkeit `vₓ` der Rumpfspitze | m/s |
| 9 | Geschwindigkeit `v_z` der Rumpfspitze | m/s |
| 10 | Winkelgeschwindigkeit des Rumpfes | rad/s |
| 11–16 | Winkelgeschwindigkeiten der sechs Gelenke | rad/s |

- Die **x-Position** fehlt (`exclude_current_positions_from_observation=True`):
  Der Agent weiß nicht, wo er ist, nur wie schnell. Für Anzeige und Metriken
  steht sie in `info["x_position"]`.
- Die Geschwindigkeiten sind **nicht geclippt**. `qvel` kommt unverändert; ein
  guter Agent erreicht `vₓ` deutlich über 10. Als unbeschränkt beschriften.

### Reward und Episodenende

```text
reward = forward_reward − ctrl_cost = 1,0 · vₓ − 0,1 · Σ aᵢ²
```

- `forward_reward = 1,0 · vₓ` mit `vₓ = Δx / dt`, `dt = 0,05 s`. Rückwärts →
  **negativ**.
- `ctrl_cost = 0,1 · Σ aᵢ²`, höchstens `0,6` je Schritt, also höchstens rund
  `−600` je Episode. Der Faktor ist **hundertmal so groß wie beim Hopper**
  (`0,001`): Wildes Zappeln kostet sichtbar Return, und sparsamer zu steuern
  ist Teil des Lernfortschritts. In der Bedienungsanleitung erwähnen.
- **Kein Überlebensbonus** – Stillstehen bringt `0`.
- Beide Anteile stehen in `info` als `reward_forward` und `reward_ctrl`
  (bereits negativ) und werden von dort übernommen, nicht nachgerechnet.

Episodenende – ausdrücklich zu dokumentieren:

- `terminated` ist **immer** `False`. Der Roboter kann auf dem Rücken landen,
  sich überschlagen oder rückwärts laufen – die Episode läuft weiter.
- Beendet wird nur durch den `TimeLimit`-Wrapper nach `1000` Schritten
  (`truncated=True`).
- Jede Episode ist damit exakt gleich lang. Die Episodenlänge ist eine
  konstante Kennzahl und gehört nach Workbench 8.1 nicht in die Summary,
  sondern in einen Test.
- Bei Truncation wird **gebootstrapt** (Workbench 3.3).

### Erfolgsdefinitionen

- **Gelöst**: Return `≥ 4800` (offizieller `reward_threshold`).
- **Vorwärts**: mittleres `vₓ` der Episode positiv.
- **Rückwärts**: mittleres `vₓ` negativ oder null. Kein Randfall – ein
  untrainierter Agent kippt regelmäßig auf den Rücken und rudert rückwärts, was
  dauerhaft negative Returns liefert.

Beide Quoten sind komplementär, werden aber getrennt ausgewiesen, damit die
Summary ohne Kopfrechnen lesbar bleibt. Der Return ist nach unten **nicht**
beschränkt und liegt realistisch zwischen etwa `−600` und `+10.000`.

Quelle: [Gymnasium HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)

## Standardprofile

`HalfCheetah-v4`-Profile des RL Baselines3 Zoo, Stand geprüft am 20.08.2026;
nicht enthaltene Werte sind SB3-Defaults der Version aus `../environment.yml`.

`PPO` – vollständig getunter Zoo-Block:

- `N = 1.000.000`, `learning_rate = 2,0633e-5` konstant, `gamma = 0,98`
- `n_steps = 512`, `batch_size = 64`, `n_epochs = 20`
- `gae_lambda = 0,92`, `clip_range = 0,1`
- `ent_coef = 0,000401762`, `vf_coef = 0,58096`, `max_grad_norm = 0,8`
- `log_std_init = -2,0`, `ortho_init = aus`, Aktivierung `ReLU`
- Beobachtungen **und** Rewards normalisieren: an

`SAC` – der Zoo setzt nur `learning_starts`, alles Weitere SB3-Default:

- `N = 1.000.000`, `learning_starts = 10.000`, `buffer_size = 1.000.000`
- `learning_rate = 3e-4` konstant, `gamma = 0,99`, `tau = 0,005`
- `batch_size = 256`, `train_freq = 1`, `gradient_steps = 1`
- `ent_coef = auto`, `target_entropy = auto`, `target_update_interval = 1`
- `use_sde = aus`, Aktivierung `ReLU`

`TD3`:

- `N = 1.000.000`, `learning_rate = 1e-3` konstant, `gamma = 0,99`,
  `tau = 0,005`
- `learning_starts = 10.000`, `batch_size = 256`, `buffer_size = 1.000.000`
- `train_freq = 1`, `gradient_steps = 1`, Action Noise `normal` mit `σ = 0,1`
- übrige SB3-Defaults, insbesondere `policy_delay = 2`,
  `target_policy_noise = 0,2`, `target_noise_clip = 0,5`, Aktivierung `ReLU`

Bei sechs Actions ist die SAC-Zielentropie `-dim(A) = -6`.

**Normalisierung:** Das PPO-Profil verlangt `normalize: true` – die 17 Werte
sind sehr unterschiedlich skaliert, die Geschwindigkeiten ungeclippt und der
Return fünfstellig. `SAC` und `TD3` normalisieren nicht. Clip-Werte je `10,0`.

**Abweichung – genau eine:** Hidden Layers `256,256` für Actor und Critic. Der
Zoo nennt für PPO `pi=[256,256], vf=[256,256]`, für SAC den SB3-Default
`[256,256]`, für TD3 aber `[400,300]`. Die Vereinheitlichung macht den
Vergleich fair und kostet TD3 nichts. `N = 1.000.000` ist für alle drei gleich
und zugleich exakt der Zoo-Wert, weicht also **nicht** ab.

**Laufzeit:** `1e6` Schritte dauern auf einem Laptop Stunden, bei mehreren
Slots länger. Als Einstieg `100.000` Schritte mit `Eval-Intervall = 10.000`
nennen.

**Erwartung:** Anders als beim Hopper ist die Gelöst-Marke erreichbar. Zoo-
Benchmark nach `1e6` Schritten:

| Verfahren | Return | Std |
| --- | --- | --- |
| `PPO` | 5819,1 | 663,5 |
| `SAC` | 9535,5 | 100,5 |
| `TD3` | 9655,7 | 969,9 |

Alle drei übertreffen `4800`, `SAC` und `TD3` rund um das Doppelte. Der große
Abstand illustriert die Fairness-Regel aus Workbench 5.8.

Quellen: [benchmark.md](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md),
[ppo.yml](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml),
[sac.yml](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml),
[td3.yml](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/td3.yml)

## Metriken

Episoden-Return, mittleres Tempo `vₓ`, zurückgelegte Strecke, Steuerkosten je
Episode, Vorwärtsquote, Gelöst-Quote. Tempo und Strecke aus
`info["x_velocity"]` und `info["x_position"]`.

Vergleichs-Summary je Slot: Algorithmus · Episoden · Environment-Schritte
ausgeführt und angefordert · Ø Return · beste Episode mit Nummer und Return ·
Ø Tempo in m/s · Ø Strecke in m · Ø Steuerkosten je Episode · Vorwärtsquote ·
Gelöst-Quote. **Keine Zeile für die Episodenlänge** – sie wäre immer `1000`.

Evaluationsergebnis: dieselben Größen plus Standardabweichung des Returns sowie
letzte und beste deterministische Evaluation.

Referenzlinie der Graphen: `+4800`.

Da jede Episode exakt 1000 Schritte dauert, entspricht die Episodennummer auf
der X-Achse genau dem tausendfachen Schrittfortschritt – in der
Bedienungsanleitung erwähnen; an Workbench 8.4 ändert das nichts.

## Einblendung neben der Animation

- die sechs Rohwerte `a₀` bis `a₅` **und** ihre Bedeutung je Gelenk:
  Drehrichtung und Moment in N·m mit der jeweils **eigenen** Übersetzung
- Höhe und Neigungswinkel des Rumpfes
- die sechs Gelenkwinkel
- `vₓ`, `v_z` und Winkelgeschwindigkeit des Rumpfes – **ohne**
  Clipping-Hinweis
- Zerlegung des letzten Rewards in `reward_forward` und `reward_ctrl`
- Strecke aus `info["x_position"]`

Bei sechs Gelenken wird die Einblendung höher als beim Hopper: Zeilen kurz
halten, Zahlenbreiten fest, damit sie schmal bleibt und nicht wandert.

**Bildrate:** environment-eigen `20` FPS, eine Episode dauert damit **50
Sekunden**. Eine höhere Rate ist hier die Regel; die Bedienungsanleitung nennt
einen brauchbaren Startwert.

## Tests

Zusätzlich zu Workbench 5.9 und 9.2:

- Environment-Kennwerte: Action `Box(-1, 1, (6,))`, Observation `(17,)`
  `float64`, `max_episode_steps = 1000`, `reward_threshold = 4800`,
  `render_fps = 20`, `dt = 0,05`, Frame `480 × 480`; die Factory fordert genau
  diese Argumente an und übergibt **keinen** Physik- oder Reward-Parameter
- die sechs Übersetzungen stimmen mit `actuator_gear` überein und lauten
  `120, 90, 60, 120, 60, 30`; die Anzeige rechnet je Gelenk mit der eigenen
  Übersetzung, nicht mit einem gemeinsamen Faktor
- Drehmoment-Interpretation: Vorzeichen als Richtung, Betrag mal `gear` als
  Moment, Clipping außerhalb `[-1, 1]`
- Reward-Zerlegung: `reward_forward + reward_ctrl` aus `info` ergibt exakt den
  Reward; kein Überlebensbonus
- Steuerkosten `0,1 · Σ aᵢ²`, also `0,6` bei vollem Ausschlag
- **Das Environment terminiert nie**: über eine vollständige Episode ist
  `terminated` in jedem Schritt `False`, und sie endet nach exakt `1000`
  Schritten mit `truncated=True`. Der wichtigste Test des Projekts – an ihm
  hängt die gesamte Metrikwahl
- Vorwärts- und Rückwärtsquote komplementär, aus dem mittleren `vₓ`; eine
  rückwärts laufende Episode liefert negativen Return
- x-Position stammt aus `info["x_position"]`, nicht aus der Observation
- die Geschwindigkeiten werden **nicht** geclippt: ein Wert jenseits `±10`
  bleibt stehen
- SAC-Zielentropie bei `auto` genau `-6`
- Ablehnung eines Speicherstands mit Kennung `HalfCheetah-v4`, `Hopper-v5` oder
  `Walker2d-v5`

Bei kurzen Testläufen beachten: Weil nie vorzeitig beendet wird, liefert jede
abgeschlossene Episode 1000 Schritte – ein Budget darunter erzeugt **gar
keine** abgeschlossene Episode. Budgets entsprechend wählen und die Netze klein
halten.

## Abhängigkeiten

Keine neue Abhängigkeit: `HalfCheetah-v5` nutzt dasselbe MuJoCo wie
`13-hopper`, `mujoco` und `imageio` stehen bereits in `../environment.yml`.
Alle Dateien tragen das Präfix `halfcheetah_` bzw. `test_halfcheetah_`.

## Abnahme

Zusätzlich zu Workbench 9.3: `PPO`, `SAC` und `TD3` trainieren, evaluieren und
vergleichen mit den obigen Profilen; die sechs **unterschiedlichen**
Übersetzungen werden in Anzeige und Tests korrekt angewendet; die
17-dimensionale Observation erscheint mit Einheiten und ohne erfundenen
Clipping-Hinweis; nirgends taucht eine Sturz-, Durchhalte- oder Zielquote oder
ein Überlebensbonus auf, sondern Vorwärts- und Gelöst-Quote sowie Tempo,
Strecke und Steuerkosten; ein Test belegt, dass nie terminiert wird und jede
Episode exakt 1000 Schritte dauert; bei Truncation wird gebootstrapt; ein
Speicherstand fremder Environment-Kennung wird abgelehnt.
