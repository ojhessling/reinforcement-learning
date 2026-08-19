# Hopper – RL-Workbench

Projektordner: `Oliver/13-hopper`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `hopper`
Environment: `Hopper-v5`

Dieser Prompt enthält ausschließlich die projektspezifischen Anforderungen:
Verfahrensauswahl, Environment, Standardprofile, Metriken, Darstellung und die
environmentbezogenen Tests. Alles Allgemeine – Projektstruktur, GUI-Aufbau mit
Verfahrenswahl und Vergleichstabs, Steuerungsbuttons, Animationssteuerung,
Responsivität, Vergleichslogik, Export, Speichern und Laden, die Steckbriefe
der eingesetzten Verfahren samt ihrer UI-Parameter und die verfahrensbezogenen
Tests – steht in `../workbench.md` und wird hier bewusst nicht wiederholt.

Dieses Projekt enthält **keine Abweichungen** von der Workbench. Es ist das
erste Projekt, das die dort neu aufgenommenen Regeln zu bis zu vier
Verfahrensslots, zum Animationsraster, zur Hover-Einblendung und zur
Farbzuordnung der Slots vollständig umsetzt.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung, die einen einbeinigen Hüpfroboter
vorwärts springen lässt, ohne umzufallen. Verglichen werden drei Verfahren für
kontinuierliche Action-Spaces aus dem Verfahrenskatalog der Workbench:

1. `PPO`
2. `SAC`
3. `TD3`

Jeder Verfahrensslot kann frei eines dieser drei Verfahren aufnehmen, auch
mehrere Slots denselben Algorithmus. Fachlicher Kern, UI-Parameter, Prüfregeln
und Quellen der drei Verfahren stehen im Verfahrenskatalog; dieser Prompt legt
nur ihre environmentspezifischen Standardprofile fest.

Hopper ist das erste MuJoCo-Environment des Kurses. Es ist ein
Balance-Environment: Der Reward belohnt jeden Schritt, den der Roboter stehen
bleibt, zusätzlich zur Vorwärtsgeschwindigkeit. Eine Episode endet nicht mit
einer Strafe, sondern einfach damit, dass es keine weiteren Schritte und damit
keinen weiteren Reward mehr gibt.

## Verfahrensslots

`Anzahl Verfahren` ist in diesem Projekt standardmäßig `4`, mit der Belegung
`Verfahren 1 = PPO`, `Verfahren 2 = TD3`, `Verfahren 3 = SAC`,
`Verfahren 4 = SAC`. Damit steht nach dem Start beides ohne weiteres Zutun da:
der Vergleich der drei Algorithmen des Projekts und der zweier
Parametrisierungen desselben Verfahrens. Wählbar bleiben `2`, `3` und `4`.

Damit sich die beiden SAC-Slots überhaupt unterscheiden, bekommt
`Verfahren 4` beim Programmstart eine abweichende Lernrate von `6e-4`. Diese
Abweichung gilt **ausschließlich** für die Startbelegung; ein später über
`Anzahl Verfahren` hinzugefügter Slot startet gemäß Workbench mit den
unveränderten Standardwerten seines Algorithmus.

Die Farbzuordnung der Workbench gilt unverändert: `Verfahren 1` blau,
`Verfahren 2` rot, `Verfahren 3` gelb, `Verfahren 4` grün. Da das Projekt ein
dunkles Farbschema verwendet, werden helle, kräftige Töne gewählt, die sich
sowohl untereinander als auch gegen den dunklen Hintergrund deutlich abheben.
Die Gelöst-Linie ist weiß.

## Environment

Erzeuge das Environment zwingend mit:

```python
import gymnasium

env = gymnasium.make(
    "Hopper-v5",
    render_mode="rgb_array",   # entfällt bei headless Evaluation
    width=480,
    height=480,
)
```

Alle physikalischen Parameter bleiben auf ihren Voreinstellungen. Insbesondere
werden `forward_reward_weight`, `ctrl_cost_weight`, `healthy_reward`,
`terminate_when_unhealthy`, die drei `healthy_*_range`, `reset_noise_scale` und
`exclude_current_positions_from_observation` **nicht** übergeben und nicht
verändert; sie zu verstellen wäre Reward Shaping beziehungsweise eine Änderung
der Abbruchregeln und ist nach Workbench unzulässig.

`width` und `height` werden ausdrücklich mitgegeben, obwohl sie den
Gymnasium-Voreinstellungen entsprechen. Sie beeinflussen ausschließlich die
Bildgröße des Renderers, nicht die Physik, und machen die Framegröße für Tests
und Layout eindeutig.

Verwende `Hopper-v5`, nicht `Hopper-v4`: `v5` ist die aktuell gepflegte Fassung
mit dokumentierter `observation_structure` und vollständigem `info`-Dictionary.
Die Hyperparameterprofile des Zoo sind für `Hopper-v4` hinterlegt; der
Unterschied betrifft Detailkorrekturen der Umgebung, nicht die Größenordnung
der Hyperparameter. Sag das in der README.

### Actions

`Box(-1.0, 1.0, (3,), float32)` – Drehmomente für drei Gelenke:

| Index | Gelenk (XML-Name) |
| --- | --- |
| `a₀` | Hüfte, Rumpf gegen Oberschenkel (`thigh_joint`) |
| `a₁` | Knie, Oberschenkel gegen Unterschenkel (`leg_joint`) |
| `a₂` | Sprunggelenk, Unterschenkel gegen Fuß (`foot_joint`) |

Alle drei Motoren besitzen in `hopper.xml` die Übersetzung `gear = 200`. Das
anliegende Drehmoment ist damit `aᵢ · 200 N·m`; das Vorzeichen bestimmt die
Drehrichtung, der Betrag das Moment. `0` bedeutet „kein Moment", nicht „Gelenk
hält die Position". Werte außerhalb `[-1, 1]` werden vom Environment geclippt
(`ctrllimited="true"`, `ctrlrange="-1 1"`).

### Observation

11 Werte, `float64`, `Box(-inf, inf, (11,))`:

| Index | Bedeutung | Einheit |
| --- | --- | --- |
| 0 | Höhe des Rumpfes über dem Boden (`rootz`) | m |
| 1 | Neigungswinkel des Rumpfes (`rooty`) | rad |
| 2 | Winkel Hüftgelenk (`thigh_joint`) | rad |
| 3 | Winkel Kniegelenk (`leg_joint`) | rad |
| 4 | Winkel Sprunggelenk (`foot_joint`) | rad |
| 5 | Geschwindigkeit `vₓ` des Rumpfes | m/s |
| 6 | Geschwindigkeit `v_z` des Rumpfes | m/s |
| 7 | Winkelgeschwindigkeit des Rumpfes | rad/s |
| 8 | Winkelgeschwindigkeit Hüftgelenk | rad/s |
| 9 | Winkelgeschwindigkeit Kniegelenk | rad/s |
| 10 | Winkelgeschwindigkeit Sprunggelenk | rad/s |

Zwei Eigenheiten sind zu dokumentieren und in der Anzeige zu berücksichtigen:

- Die **x-Position** des Rumpfes ist wegen
  `exclude_current_positions_from_observation=True` nicht Teil der Observation.
  Der Agent weiß also nicht, wie weit er gekommen ist – nur, wie schnell er
  gerade ist. Für Anzeige und Metriken steht sie legitim in `info["x_position"]`
  zur Verfügung; auf Interna wie `env.unwrapped.data.qpos` wird **nicht**
  zugegriffen.
- Alle sechs Geschwindigkeiten sind im Environment auf `[-10, 10]` **geclippt**
  (`np.clip(self.data.qvel, -10, 10)`). Ein angezeigter Wert von genau `±10`
  bedeutet deshalb „mindestens so schnell", nicht „genau so schnell". Weise in
  der Bedienungsanleitung darauf hin.

Anders als bei BipedalWalker sind die Werte echte physikalische Größen in Meter
und Radiant; sie dürfen mit Einheit beschriftet werden.

### Reward und Episodenende

Der Reward setzt sich pro Schritt aus drei Anteilen zusammen:

```text
reward = healthy_reward + forward_reward − ctrl_cost
       = 1,0            + 1,0 · vₓ       − 0,001 · Σ aᵢ²
```

- `healthy_reward = 1,0` für jeden Schritt, in dem der Roboter „gesund" ist.
  Das ist der Überlebensbonus: Bloßes Stehenbleiben liefert bereits `+1` je
  Schritt und über eine volle Episode `+1000`.
- `forward_reward = 1,0 · vₓ` mit `vₓ = Δx / dt` und `dt = 0,008 s`. Belohnt
  wird die tatsächlich zurückgelegte Strecke pro Zeit, nicht die Absicht.
- `ctrl_cost = 0,001 · Σ aᵢ²`, also höchstens `0,003` pro Schritt. Diese Kosten
  sind gegenüber den beiden anderen Anteilen sehr klein und dienen nur dazu,
  unnötig ruckartige Steuerung leicht zu bestrafen.

Die drei Anteile stehen einzeln in `info` als `reward_survive`,
`reward_forward` und `reward_ctrl`. Nutze sie für die Anzeige, statt sie
nachzurechnen.

Die Episode endet:

- mit `terminated`, sobald der Roboter **ungesund** wird. Ungesund heißt: Höhe
  `z ≤ 0,7 m`, oder Rumpfwinkel außerhalb `(−0,2; 0,2) rad`, oder irgendein
  Zustandswert ab Index 2 außerhalb `(−100; 100)`, oder ein nicht endlicher
  Wert. Umgangssprachlich: der Roboter fällt um oder knickt weg.
- mit `truncated` nach `1000` Schritten durch den `TimeLimit`-Wrapper aus
  `gymnasium.make`.

Wichtig und ausdrücklich zu dokumentieren: Es gibt **weder einen Terminalbonus
noch eine Terminalstrafe**. Der Sturz kostet nichts weiter als alle Rewards der
Schritte, die nicht mehr stattfinden. Genau das ist die Lernaufgabe: Der Agent
maximiert den Return, indem er lange lebt und dabei schnell ist. `Hopper-v5`
selbst liefert immer `truncated=False`; das Zeitlimit setzt ausschließlich der
Wrapper.

Erfolgsdefinitionen für dieses Projekt:

- **Durchgehalten**: Episode endet mit `truncated`, der Roboter steht nach
  1000 Schritten noch. Das ist hier der gute Ausgang – anders als bei
  BipedalWalker bedeutet Zeitlimit nicht „zu langsam".
- **Sturz**: Episode endet mit `terminated`, der Roboter ist ungesund geworden.
- **Gelöst**: Episoden-Return `≥ 3800` (offizieller `reward_threshold` von
  `Hopper-v5`).

Durchhaltequote und Sturzquote sind zueinander komplementär; beide werden
trotzdem getrennt ausgewiesen, damit die Summary ohne Kopfrechnen lesbar
bleibt.

Höhere Werte sind besser. Der Return liegt realistisch zwischen etwa `+15`
(sofortiger Sturz) und `+4000`.

Quelle: [Gymnasium Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/)

## Normalisierung

Das PPO-Profil des RL Baselines3 Zoo verlangt für dieses Environment
`normalize: true`. Die elf Observationswerte sind sehr unterschiedlich skaliert
– Höhen um `1,25 m`, Winkel im Bogenmaß und Geschwindigkeiten bis `±10` –, und
der Return wächst über eine Episode auf mehrere Tausend. Die Gruppe
`Normalisierung` erscheint deshalb in jedem Verfahrenstab; ihre Regeln stehen
im Verfahrenskatalog der Workbench.

Standardwerte für dieses Environment: `PPO` normalisiert Beobachtungen und
Rewards, `SAC` und `TD3` nicht. Clip-Werte jeweils `10,0`.

## Standardprofile

Jedes Verfahren lädt beim Dropdown-Wechsel sein eigenes Profil in den
zugehörigen Tab. Grundlage sind die `Hopper-v4`-Profile des RL Baselines3 Zoo
(Stand geprüft am 19.08.2026); nicht enthaltene Werte stammen aus den
Voreinstellungen von Stable-Baselines3 in der Version aus `../environment.yml`.

`PPO` – der Zoo führt für Hopper einen vollständig getunten Block:

- Trainingsschritte `N = 1.000.000`
- `learning_rate = 9,80828e-5` konstant, `gamma = 0,999`
- `n_steps = 512`, `batch_size = 32`, `n_epochs = 5`
- `gae_lambda = 0,99`, `clip_range = 0,2`
- `ent_coef = 0,00229519`, `vf_coef = 0,835671`, `max_grad_norm = 0,7`
- `log_std_init = -2,0`, `ortho_init = aus`, Aktivierung `ReLU`
- Beobachtungen und Rewards normalisieren: an
- übrige Werte aus den SB3-Defaults, insbesondere
  `normalize_advantage = an`, `use_sde = aus`, Adam

`SAC` – der Zoo setzt für Hopper nur `learning_starts`; alles Weitere ist
SB3-Default:

- Trainingsschritte `N = 1.000.000`
- `learning_starts = 10.000`, `buffer_size = 1.000.000`
- `learning_rate = 3e-4` konstant, `gamma = 0,99`, `tau = 0,005`
- `batch_size = 256`, `train_freq = 1`, `gradient_steps = 1`
- `ent_coef = auto`, `target_entropy = auto`, `target_update_interval = 1`
- `use_sde = aus`, Aktivierung `ReLU`

`TD3`:

- Trainingsschritte `N = 1.000.000`
- `learning_rate = 1e-3` konstant, `gamma = 0,99`, `tau = 0,005`
- `learning_starts = 10.000`, `batch_size = 256`, `buffer_size = 1.000.000`
- `train_freq = 1`, `gradient_steps = 1`
- Action Noise `normal` mit `σ = 0,1`
- übrige Werte aus den SB3-Defaults, insbesondere `policy_delay = 2`,
  `target_policy_noise = 0,2`, `target_noise_clip = 0,5`, Aktivierung `ReLU`

Bei drei Actions ist die automatische SAC-Zielentropie
`target_entropy = -dim(A) = -3`.

Das Trainingsbudget `N = 1.000.000` ist für alle drei Verfahren identisch und
zugleich exakt der Wert, den alle drei Zoo-Profile nennen. Der Standardwert
weicht damit **nicht** vom Profil ab und erfüllt zugleich die Workbench-Regel,
dass alle Verfahren mit demselben Budget starten.

Genau **ein** Wert ist bewusst nicht aus den Profilen übernommen:

- **Hidden Layers `256,256` für Actor und Critic.** Der Zoo nennt für PPO
  `pi=[256,256], vf=[256,256]`, für SAC den SB3-Default `[256,256]` und für TD3
  `[400,300]`. Nur TD3 weicht ab; die Vereinheitlichung auf `256,256` macht den
  Vergleich fair und kostet TD3 nichts an Qualität.

Alles bleibt in der UI frei änderbar. Alle übrigen getunten Hyperparameter
werden unverändert übernommen.

Zur Laufzeit ist die README ausdrücklich ehrlich: Ein Lauf über `1e6` Schritte
dauert auf einem Laptop **Stunden**, bei mehreren Slots parallel entsprechend
länger, und mit eingeschalteter Animation zusätzlich. Nenne einen kleineren
Wert – etwa `100.000` – als Einstieg, mit dem sich die Verfahren in Minuten
nebeneinander beobachten lassen, auch wenn die Kurven dann früh abbrechen.

Ebenso ehrlich ist die README bei der Gelöst-Marke. Der Benchmark des RL
Baselines3 Zoo weist für `Hopper-v3` nach `1e6` Schritten aus:

| Verfahren | mittlerer Return | Standardabweichung |
| --- | --- | --- |
| `PPO` | 2410,4 | 10,0 |
| `SAC` | 2325,5 | 1129,7 |
| `TD3` | 3606,4 | 4,0 |

Auch mit dem vollen Profilbudget erreicht also keines der drei Verfahren im
Mittel die `3800`; `TD3` kommt am nächsten heran, `SAC` streut extrem. Sag das
in der README, statt einen Erfolg zu versprechen, den die Profile nicht
hergeben. Quelle:
[RL Baselines3 Zoo, benchmark.md](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md)

Zwei weitere Punkte sind ausdrücklich zu dokumentieren:

- Das PPO-Profil des Zoo verwendet für Hopper `n_envs = 1`. Anders als beim
  BipedalWalker-Projekt entsteht hier also **keine** zusätzliche Benachteiligung
  durch die Beschränkung auf eine Environment-Instanz: Der Rollout von 512
  Schritten je Update entspricht exakt dem Profil. Die grundsätzliche
  Bevorteilung der Off-Policy-Verfahren bei gleichem Schrittbudget bleibt
  bestehen und steht in der Workbench.
- Der Überlebensbonus von `+1` je Schritt macht die Kurven im Vergleichsgraphen
  anfangs vor allem zu einer Kurve der Episodenlänge. Erst wenn ein Agent
  zuverlässig über mehrere Hundert Schritte kommt, trennt der
  Geschwindigkeitsanteil die Verfahren. Erkläre das, sonst wirkt der frühe
  Kurvenverlauf beliebig.

Quellen: [RL Baselines3 Zoo, PPO](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml),
[SAC](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml),
[TD3](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/td3.yml)

## Evaluation

Standardwerte der globalen Evaluationseinstellungen:

- `Eval-Intervall = 100.000` Schritte, also ein Zehntel des Schrittbudgets. Ein
  Standardlauf liefert damit zehn Stützstellen.
- `Eval-Episoden M = 5`

Wer das Schrittbudget in der UI verkleinert, verkleinert das Intervall
sinnvollerweise mit – sonst liefert der Lauf gar keine Stützstelle.

Zeige im Evaluationsergebnis mindestens:

- mittleren Return und Standardabweichung
- mittlere Episodenlänge (Maximum 1000)
- Durchhaltequote (Anteil Episoden mit vollen 1000 Schritten)
- Sturzquote
- Gelöst-Quote (Anteil Episoden mit Return `≥ 3800`)
- mittlere Vorwärtsgeschwindigkeit und mittlere zurückgelegte Strecke aus
  `info["x_velocity"]` beziehungsweise `info["x_position"]`
- letzte und beste deterministische Evaluation

## Vergleich

Beide Graphen – Training wie Vergleich – zeigen ausschließlich den
explorativen Episoden-Return gegen die Episodennummer. Die deterministischen
Zwischenevaluationen werden **nicht** eingezeichnet; sie liefen auf einer
anderen Stützstellenzahl und lenkten von der Lernkurve ab. Ausgewiesen werden
sie ausschließlich in der Summary, wie es die Workbench zulässt.

Beide Graphen tragen **keine** Überschrift über den Achsen – der Tab-Name sagt
bereits, was zu sehen ist, und der Platz gehört den Kurven. Auch die
Exportschaltflächen bekommen keine eigene Kopfzeile: `PNG exportieren` und
`TXT exportieren` stehen kompakt nebeneinander rechts in der Tableiste der
Diagramme, wo die Fläche ohnehin frei ist. Der reservierte Bereich für die Legende richtet sich nach ihrer
tatsächlichen Breite; Labels wie `V3 – SAC (Lernrate α 0.0003)` sprengen jede
fest gewählte Reserve und würden sonst abgeschnitten. Die Y-Achse ist
schlicht mit `Return` beschriftet; dass höhere Werte besser sind und `3800` als
gelöst gilt, sagen Legende, Summary und Bedienungsanleitung.

Die hervorgehobenen Slotkurven haben die Strichstärke `1,2`, ebenso die
Referenzlinie. Die dezent hinterlegten Einzelepisoden verwenden dieselbe
Slotfarbe mit einer Deckkraft von `10 %`: gerade noch erkennbar, ohne den
gleitenden Durchschnitt zu überdecken.

Der Vergleichsgraph enthält eine weiße, gestrichelte Referenzlinie bei `+3800`.
Diese Marke liegt über den Werten, die die Profile im Mittel erreichen, und
früh im Lauf weit außerhalb des Datenbereichs. Die Y-Achse skaliert deshalb
nach den Daten und erzwingt die Referenzlinie nicht in den sichtbaren Bereich;
stattdessen nennt die Legende sie mit Wert, und die Summary weist die
Gelöst-Quote aus.

Die Vergleichs-Summary trägt keine Überschrift – die Spaltenköpfe sagen
bereits, was verglichen wird. Sie enthält je Slot diese Zeilen:

- Algorithmus
- Episoden
- Environment-Schritte, ausgeführt und angefordert
- durchschnittlicher Return
- beste Episode mit Nummer und Return, etwa `#137: 1.204,3`
- durchschnittliche Episodenlänge
- Durchhaltequote
- Sturzquote
- Gelöst-Quote
- beste deterministische Evaluation

## Darstellung

Zeige ausschließlich den offiziellen, von `env.render()` gelieferten
Gymnasium-RGB-Frame (480 × 480 Pixel). Erstelle keine eigene Grafik und öffne
kein separates Fenster.

Der Standardwert der Bildrate ist die environment-eigene Rate von 125 FPS
(`env.metadata["render_fps"]`, entspricht `1 / dt` mit `dt = 0,008 s`). Eine
volle Episode von 1000 Schritten dauert damit acht Sekunden. Gültig sind ganze
Zahlen von `1` bis `250`. Die Obergrenze der Workbench von 120 reicht hier
nicht: Sie läge unter dem Standardwert und machte ihn selbst ungültig. `250`
ist das Doppelte der environment-eigenen Rate und erlaubt zügiges Zuschauen. Weise in der
Bedienungsanleitung darauf hin, dass eine höhere Bildrate das Zuschauen
abkürzt, dass die Animation während eines Trainings- oder Vergleichslaufs
dessen Fortschritt verlangsamt und dass dieser Effekt mit der Zahl gleichzeitig
sichtbarer Verfahren wächst.

### Welche Episode gezeigt wird

Jede Animation besitzt über ihrem Bild ein eigenes Auswahlfeld `Episode` mit
genau zwei Werten; es wirkt unabhängig von den übrigen Anzeigen:

- `aktuell` (Standard): der laufende Lernstand. Die Beschriftung nennt die
  Nummer der zuletzt trainierten beziehungsweise verglichenen Episode – also
  dieselbe Nummer wie die X-Achse der Graphen. Der Seed bleibt offen, damit
  aufeinanderfolgende Episoden nicht identisch aussehen.
- `beste`: der Lernstand der bisher besten Episode, gemessen am explorativen
  Episoden-Return. Abgespielt wird mit dem konfigurierten Seed, damit die
  Wiederholung jedes Mal gleich aussieht.

Gesichert wird dafür je Verfahrensslot **genau ein** zusätzlicher Lernstand:
der der besten Episode. Er entsteht im Worker-Thread unmittelbar nach dem
Episodenende – nur dort gehört die Policy tatsächlich zu dieser Episode – als
losgelöste Kopie, die der weiterlaufende Optimizer nicht mehr verändert. Ein
Verlauf über alle Episoden ist ausdrücklich **nicht** vorgesehen: Bei `1e6`
Schritten entstehen Tausende Episoden, deren Kopien Gigabytes belegten. Ein
neues Modell verwirft den gesicherten Stand mit.

Ist `beste` gewählt, aber noch keine Episode abgeschlossen, läuft weiter der
aktuelle Stand, und die Statuszeile sagt das.

### Unter dem Bild

Gemäß Workbench steht unter jedem Animationsbild ausschließlich eine Zeile mit
Episode, Schritt, Verfahren und Return, zum Beispiel:

```text
E:   12  · S:  348/1000 · R:     412,7
```

Das Verfahren steht nicht in dieser Zeile, sondern im Titel der Anzeige:
`Verfahren 2 – SAC`. Teilen sich mehrere Slots einen Algorithmus, ergänzt der
Titel wie die Legende des Vergleichsgraphen den wichtigsten abweichenden
Parameter, etwa `Verfahren 3 – SAC (Lernrate α 0.0003)`.

Läuft gerade die beste statt der aktuellen Episode, markiert ein `*` hinter der
Episodennummer das; die Zeile behält dabei ihre Länge, damit die Bilder nicht
springen.

### Neben dem Bild, nur beim Überfahren

Alle weiteren Messwerte erscheinen ausschließlich, solange der Mauszeiger über
dem zugehörigen Bild steht, und werden neben dem Bild eingeblendet. Die
Einblendung enthält:

- die gewählte Action doppelt: die drei Rohwerte `a₀` bis `a₂` sowie ihre
  Bedeutung je Gelenk, also Drehrichtung und Moment in `N·m` (`aᵢ · 200`)
- Höhe des Rumpfes in Metern, dazu die Angabe, dass unterhalb von `0,70 m` die
  Episode endet
- Rumpfwinkel in Radiant, dazu der gesunde Bereich `±0,20`
- die drei Gelenkwinkel
- die sechs Geschwindigkeiten mit dem Hinweis auf die Clippung bei `±10`
- die Zerlegung des letzten Rewards in `reward_survive`, `reward_forward` und
  `reward_ctrl`
- zurückgelegte Strecke aus `info["x_position"]`

Die Einblendung nennt in ihrer Kopfzeile den Slot, zu dem sie gehört, und
verwendet dessen Farbe.

Projektspezifische Metriken sind Episoden-Return, Episodenlänge,
Durchhaltequote, Sturzquote, Gelöst-Quote, mittlere Vorwärtsgeschwindigkeit und
zurückgelegte Strecke. Legende, Summary und Hilfetexte erklären eindeutig, dass
höhere Werte besser sind und `3800` als gelöst gilt; die Achsenbeschriftung
bleibt dafür bewusst knapp.

## Rendering und MuJoCo

MuJoCo rendert nicht über SDL, sondern über einen eigenen OpenGL-Kontext. Für
den isolierten Renderprozess der Workbench gilt deshalb:

- Die Umgebungsvariable `MUJOCO_GL` wird **vor** dem ersten `import mujoco`
  beziehungsweise dem Import der Gymnasium-MuJoCo-Environments gesetzt. Auf
  macOS ist `cgl` zu wählen, auf headless Linux `egl`, ersatzweise `osmesa`.
  Die Anwendung wählt den Wert plattformabhängig und lässt eine bereits
  gesetzte Variable unverändert, damit der Benutzer sie überschreiben kann.
- `glfw` ist auf macOS ausdrücklich **nicht** zu verwenden, obwohl es
  funktioniert und Gymnasiums Standardliste es führt: `glfw.init()` meldet den
  Prozess beim Window Server als Vordergrund-App an. Bei vier Animationen
  entstünden so vier zusätzliche Programm- und Dock-Einträge neben der
  eigentlichen Anwendung. Der CGL-Kontext rendert rein offscreen und meldet
  sich gar nicht erst an. Nachprüfbar ist das mit `lsappinfo list`: Unter
  `glfw` taucht jeder Renderprozess dort als `type="Foreground"` auf, unter
  `cgl` keiner.
- Gymnasiums `MujocoRenderer` führt in `_ALL_RENDERERS` nur `glfw`, `egl` und
  `osmesa` und lehnt `cgl` sonst mit einer Fehlermeldung ab, obwohl MuJoCo den
  Kontext unter `mujoco.cgl` mitbringt. Der Renderprozess ergänzt den Eintrag
  deshalb, bevor er das Environment erzeugt. Ergänzt wird nur; die
  mitgelieferten Backends bleiben unangetastet.
- Der Renderprozess erzeugt kein sichtbares Fenster; `render_mode` ist immer
  `rgb_array`, nie `human`.
- Jede sichtbare Anzeige läuft in einem eigenen Prozess mit eigener
  Environment-Instanz. Bei `Anzahl Verfahren = 4` und eingeschalteter Animation
  laufen also vier Renderprozesse. Alle werden beim Schließen der Anwendung und
  beim Ausschalten der Animation zuverlässig beendet.
- Schlägt der Import von `mujoco` oder die Erzeugung des Grafikkontexts fehl,
  meldet die Anwendung das verständlich auf Deutsch mit dem Hinweis auf
  `pip install "gymnasium[mujoco]"` und auf `MUJOCO_GL`, statt mit einem Traceback
  abzustürzen. Training und Evaluation ohne Animation müssen in diesem Fall
  weiterhin möglich sein, sofern nur der Grafikkontext fehlt.

## Lernzustand sichern

Buttons zum manuellen Speichern und Laden gibt es nicht; gesichert wird
ausschließlich der automatische Checkpoint des besten Evaluationsergebnisses.
Umfang und Prüfungen stehen in der Workbench. Environmentspezifisch gehört zu
den Metadaten die Kennung `Hopper-v5`; ein Stand einer anderen
Environment-Kennung – etwa `Hopper-v4`, `Walker2d-v5` oder `HalfCheetah-v5` –
wird verständlich abgelehnt.

## Tests

Zusätzlich zu den allgemeinen und den verfahrensbezogenen Tests der Workbench:

- Environment-Kennwerte: Action-Space `Box(-1, 1, (3,))`, Observation `(11,)`
  vom Typ `float64`, `max_episode_steps = 1000`, `reward_threshold = 3800`,
  `render_fps = 125`, Framegröße `480 × 480`; die Factory fordert genau diese
  Argumente an und übergibt **keinen** der Physik- und Reward-Parameter
- Interpretation der drei Drehmomente: Vorzeichen als Richtung, Betrag mal
  `gear = 200` als Moment in `N·m`, Clipping außerhalb `[-1, 1]`
- Reward-Zerlegung: Die Summe aus `reward_survive`, `reward_forward` und
  `reward_ctrl` aus `info` ergibt den zurückgegebenen Reward
- Unterscheidung der Episodenausgänge: Sturz als `terminated` **ohne**
  Terminalstrafe, Durchhalten als `truncated` nach genau 1000 Schritten,
  gelöst ab `3800`; insbesondere, dass ein Sturz nicht als negativer Reward
  auftaucht und dass Durchhalten und Sturz komplementär gezählt werden
- die x-Position stammt aus `info["x_position"]` und nicht aus der Observation;
  die Observation enthält sie nicht
- die Geschwindigkeiten in der Anzeige sind die geclippten Werte aus der
  Observation
- `SAC`-Zielentropie ist bei `auto` genau `-3`
- Normalisierung: Statistiken wachsen im Training, bleiben in der Evaluation
  unverändert, der ausgewiesene Return ist unnormalisiert, ein
  Speichern/Laden-Zyklus stellt die Statistiken wieder her
- Ablehnung eines Speicherstands mit fremder Environment-Kennung

Halte kurze Testläufe klein: Eine Hopper-Episode dauert ohne Sturz 1000
Schritte, ein ungelernter Agent stürzt jedoch meist nach wenigen Dutzend
Schritten. Wähle Budgets, die zuverlässig mindestens eine abgeschlossene
Episode liefern.

Tests, die MuJoCo benötigen, werden übersprungen, wenn das Paket nicht
installiert ist – so wie der GUI-Smoke-Test ohne Display übersprungen wird. Die
reine Logik der Metriken, der Konfiguration und der Slot-Verwaltung wird ohne
MuJoCo getestet.

## Abhängigkeiten

Dieses Projekt führt als erstes **neue** Abhängigkeiten ein: `mujoco` und
`imageio`. Beide zusammen bilden Gymnasiums Extra `gymnasium[mujoco]`;
`imageio` wird von `gymnasium.envs.mujoco.mujoco_rendering` importiert, sodass
ohne es bereits das Erzeugen des Environments scheitert. Gymnasiums
MuJoCo-Environments verwenden die offiziellen Python-Bindings von DeepMind; das
ältere `mujoco-py` wird nicht verwendet und ist für `Hopper-v5` auch nicht
unterstützt. Trage beide Pakete in `../environment.yml` in den `pip`-Abschnitt
und in die `requirements.txt` dieses Projekts ein. Box2D bleibt in
`environment.yml`, weil die früheren Projekte es weiterhin brauchen; für Hopper
wird es nicht benötigt.

Die README nennt den Installationsschritt ausdrücklich und weist darauf hin,
dass MuJoCo-Binärräder für macOS, Linux und Windows vorliegen und kein
Compiler nötig ist.

## Abnahme

Zusätzlich zu allen Abnahmekriterien aus `../workbench.md` gilt: Fertig, wenn
`PPO`, `SAC` und `TD3` mit den obigen Profilen trainieren, deterministisch
evaluiert und fair verglichen werden; zwei, drei oder vier Verfahrensslots
gleichzeitig laufen und ihre Animationen dabei im Raster mit höchstens zwei
Spalten und zwei Zeilen erscheinen; unter jedem Bild ausschließlich Episode,
Schritt, Verfahren und Return stehen und die übrigen Messwerte nur beim
Überfahren neben dem Bild erscheinen, ohne es zu verdecken oder zu verschieben;
der Vergleichsgraph durchgezogene Kurven in Blau, Rot, Gelb und Grün sowie eine
weiße, gestrichelte Gelöst-Linie bei `3800` zeigt; der dreidimensionale
Action-Space in Anzeige und Tests korrekt als Drehmomente mit `gear = 200`
interpretiert wird;
die elfdimensionale Observation mit Einheiten und dem Hinweis auf die
Geschwindigkeitsclippung dargestellt wird; Durchhalte-, Sturz- und Gelöst-Quote
getrennt ausgewiesen werden und die Summary je Slot die beste Episode mit
Nummer und Return nennt; jede Animation unabhängig zwischen dem aktuellen und
dem besten Lernstand umschalten kann und dabei die echte Trainings-Episodennummer
anzeigt; die Normalisierung im Training lernt, in der
Evaluation eingefroren bleibt, den ausgewiesenen Return nicht verfälscht und
vollständig gespeichert wird; die Renderprozesse MuJoCo headless mit korrekt
gesetztem `MUJOCO_GL` betreiben und beim Schließen zuverlässig enden; und ein
Speicherstand einer fremden Environment-Kennung abgelehnt wird.
