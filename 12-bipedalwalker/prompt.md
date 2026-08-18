# BipedalWalker – RL-Workbench

Projektordner: `Oliver/12-bipedalwalker`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `bipedalwalker`
Environment: `BipedalWalker-v3`

Dieser Prompt enthält ausschließlich die projektspezifischen Anforderungen:
Verfahrensauswahl, Environment, Standardprofile, Metriken, Darstellung und die
environmentbezogenen Tests. Alles Allgemeine – Projektstruktur, GUI-Aufbau mit
Verfahrenswahl und Vergleichstabs, Steuerungsbuttons, Animationssteuerung,
Responsivität, Vergleichslogik, Export, Speichern und Laden, die Steckbriefe
der eingesetzten Verfahren samt ihrer UI-Parameter und die verfahrensbezogenen
Tests – steht in `../workbench.md` und wird hier bewusst nicht wiederholt.

Dieses Projekt enthält **keine Abweichungen** von der Workbench.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung, die einen zweibeinigen Roboter über
unebenes Gelände laufen lässt. Verglichen werden drei Verfahren für
kontinuierliche Action-Spaces aus dem Verfahrenskatalog der Workbench:

1. `PPO`
2. `TD3`
3. `SAC`

Jeder der beiden Verfahrensslots kann frei eines dieser drei Verfahren
aufnehmen, auch beide denselben Algorithmus. Fachlicher Kern, UI-Parameter,
Prüfregeln und Quellen der drei Verfahren stehen im Verfahrenskatalog; dieser
Prompt legt nur ihre environmentspezifischen Standardprofile fest.

BipedalWalker ist deutlich schwerer als die Vorgänger-Environments: Der Agent
steuert vier Gelenke gleichzeitig, muss dabei das Gleichgewicht halten und
erhält Fortschritt nur über eine kontinuierliche Shaping-Belohnung. Ein
sichtbar gehender Roboter entsteht nicht in wenigen Minuten.

## Environment

Erzeuge das Environment zwingend mit:

```python
import gymnasium

env = gymnasium.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")
```

`hardcore=False` wird ausdrücklich mitgegeben, obwohl es dem Standard
entspricht: Die Hardcore-Variante mit Stufen, Gruben und Hindernissen
(`BipedalWalkerHardcore-v3`) gehört **nicht** zum Projekt und braucht ein um
Größenordnungen höheres Trainingsbudget.

### Actions

`Box(-1.0, 1.0, (4,), float32)` – Drehmomente für vier Motoren:

| Index | Gelenk |
| --- | --- |
| `a₀` | Hüfte Bein 1 |
| `a₁` | Knie Bein 1 |
| `a₂` | Hüfte Bein 2 |
| `a₃` | Knie Bein 2 |

Das Vorzeichen bestimmt die Drehrichtung, der Betrag das maximal anliegende
Drehmoment als Anteil von `MOTORS_TORQUE = 80`. Ein Wert von `0` bedeutet also
nicht „Gelenk hält die Position", sondern „kein Moment".

### Observation

24 Werte, `float32`, mit den in der Umgebung deklarierten Grenzen:

| Index | Bedeutung | Bereich |
| --- | --- | --- |
| 1 | Rumpfwinkel | `[-π; π]` |
| 2 | Rumpf-Winkelgeschwindigkeit, normiert | `[-5; 5]` |
| 3 | Geschwindigkeit `vₓ`, normiert | `[-5; 5]` |
| 4 | Geschwindigkeit `v_y`, normiert | `[-5; 5]` |
| 5 | Hüftwinkel Bein 1 | `[-π; π]` |
| 6 | Hüftgeschwindigkeit Bein 1 | `[-5; 5]` |
| 7 | Kniewinkel Bein 1 (mit Offset `+1`) | `[-π; π]` |
| 8 | Kniegeschwindigkeit Bein 1 | `[-5; 5]` |
| 9 | Bodenkontakt Bein 1 | `0` oder `1` |
| 10–14 | dieselben fünf Werte für Bein 2 | wie oben |
| 15–24 | zehn Lidar-Messwerte | `[-1; 1]` |

Die Geschwindigkeiten sind im Environment bereits skaliert
(`2·ω/FPS` beziehungsweise `0,3·v·(Viewport/SCALE)/FPS`), die Kniewinkel
zusätzlich um `+1` verschoben. Behandle diese Werte in der Anzeige deshalb als
**normierte Größen** und rechne sie nicht in physikalische Einheiten zurück;
eine saubere Umrechnung existiert hier nicht. Beschrifte sie entsprechend.

Die zehn Lidar-Werte geben den Anteil der Strahllänge (`LIDAR_RANGE ≈ 5,33`)
bis zum Bodentreffer an: kleinere Werte bedeuten näheren Boden vor dem Roboter.
Sie sind die einzige Wahrnehmung des Geländes.

Die absolute Position des Rumpfes steht **nicht** in der Observation. Greife
für Anzeige oder Metriken nicht auf Interna wie `env.unwrapped.hull.position`
zu; die zurückgelegte Strecke wird nicht ausgewiesen.

### Reward und Episodenende

Der Reward setzt sich pro Schritt zusammen aus:

- der Differenz zweier Shaping-Terme. Der Shaping-Term belohnt Vorwärtskommen
  (`130 · x / SCALE`) und bestraft Schräglage des Rumpfes
  (`−5 · |Rumpfwinkel|`). Die Normierung ist so gewählt, dass die vollständige
  Strecke in Summe rund `+300` ergibt.
- Drehmomentkosten `−0,00035 · 80 · |aᵢ|` je Motor und Schritt, also bis zu
  `−0,112` pro Schritt bei vollem Moment an allen vier Gelenken. Über eine
  ganze Episode summiert sich das auf grob `−50`.

Die Episode endet:

- mit `terminated` und Reward **exakt `−100`**, wenn der Rumpf den Boden
  berührt oder der Roboter hinter den Startpunkt zurückfällt (Sturz),
- mit `terminated` ohne Bonus, sobald der Roboter das Ende der Strecke
  erreicht,
- mit `truncated` nach `1600` Schritten durch den `TimeLimit`-Wrapper aus
  `gymnasium.make`.

`BipedalWalker` selbst liefert immer `truncated=False`; das Zeitlimit setzt
ausschließlich der Wrapper. Es gibt **keinen positiven Terminalbonus**: Das
Erreichen des Ziels beendet die Episode kommentarlos.

Erfolgsdefinitionen für dieses Projekt:

- **Ziel erreicht**: Episode endet mit `terminated` und der letzte Reward liegt
  über `−50`, der Roboter ist also nicht gestürzt.
- **Sturz**: Episode endet mit `terminated` und dem Terminalreward `−100`.
- **Zeitlimit**: Episode endet mit `truncated`; der Roboter steht noch, war
  aber zu langsam.
- **Gelöst**: Episoden-Return `≥ 300` (offizieller `reward_threshold`).

Alle vier Quoten werden getrennt ausgewiesen. Die Unterscheidung ist hier
wichtiger als bei einem Environment mit klarem Terminalbonus, weil ein
vorsichtiger Agent das Zeitlimit zuverlässig erreicht, ohne je zu stürzen oder
voranzukommen – die Sturzquote allein würde ihn gut aussehen lassen.

Höhere Werte sind besser; der Episoden-Return liegt realistisch zwischen etwa
`−130` und `+320`.

Quelle: [Gymnasium Bipedal Walker](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)

## Normalisierung

Das PPO-Profil des RL Baselines3 Zoo verlangt für dieses Environment
`normalize: true`. Ohne Normalisierung der 24 sehr unterschiedlich skalierten
Observationswerte lernt PPO hier kaum. Die Gruppe `Normalisierung` erscheint
deshalb in jedem Verfahrenstab; ihre Regeln stehen im Verfahrenskatalog der
Workbench.

Standardwerte für dieses Environment: `PPO` normalisiert Beobachtungen und
Rewards, `TD3` und `SAC` nicht. Clip-Werte jeweils `10,0`.

## Standardprofile

Jedes Verfahren lädt beim Dropdown-Wechsel sein eigenes Profil in den
zugehörigen Tab. Grundlage sind die getunten `BipedalWalker-v3`-Profile des RL
Baselines3 Zoo (Stand geprüft am 18.08.2026); nicht enthaltene Werte stammen
aus den Voreinstellungen von Stable-Baselines3 in der Version aus
`../environment.yml`.

`PPO`:

- Trainingsschritte `N = 100.000`
- `learning_rate = 3e-4` konstant, `gamma = 0,999`
- `n_steps = 2048`, `batch_size = 64`, `n_epochs = 10`
- `gae_lambda = 0,95`, `clip_range = 0,18`, `ent_coef = 0,0`
- Beobachtungen und Rewards normalisieren: an
- übrige Werte aus den SB3-Defaults, insbesondere Aktivierung `Tanh`,
  `vf_coef = 0,5`, `max_grad_norm = 0,5`, Adam mit `eps = 1e-5`

`TD3`:

- Trainingsschritte `N = 100.000`
- `learning_rate = 1e-3` konstant, `gamma = 0,98`
- `buffer_size = 200.000`, `learning_starts = 10.000`
- `train_freq = 1`, `gradient_steps = 1`
- Action Noise `normal` mit `σ = 0,1`
- übrige Werte aus den SB3-Defaults, insbesondere `batch_size = 256`,
  `tau = 0,005`, `policy_delay = 2`, `target_policy_noise = 0,2`,
  `target_noise_clip = 0,5`, Aktivierung `ReLU`

`SAC`:

- Trainingsschritte `N = 100.000`
- `learning_rate = 7,3e-4` konstant
- `buffer_size = 300.000`, `learning_starts = 10.000`, `batch_size = 256`
- `gamma = 0,98`, `tau = 0,02`
- `train_freq = 64`, `gradient_steps = 64`
- `ent_coef = auto`, `use_sde = ja`, `log_std_init = -3`
- übrige Werte aus den SB3-Defaults, insbesondere `target_update_interval = 1`,
  `target_entropy = auto`, Aktivierung `ReLU`

Bei vier Actions ist die automatische SAC-Zielentropie
`target_entropy = -dim(A) = -4`.

Zwei Werte sind bewusst **nicht** aus den Profilen übernommen, sondern für alle
drei Verfahren vereinheitlicht:

- **Trainingsbudget `N = 100.000` Schritte.** Die Profile nennen `5e6` (PPO),
  `1e6` (TD3) und `5e5` (SAC). Diese Budgets sind interaktiv nicht abwartbar,
  und unterschiedliche Budgets machen einen Vergleich zweier Slots ohne Zutun
  unfair.
- **Hidden Layers `64,64` für Actor und Critic.** Die Profile nennen für TD3
  und SAC `400,300`; solche Netze kosten je Schritt deutlich mehr Rechenzeit.

Beides bleibt in der UI frei änderbar. Alle übrigen getunten Hyperparameter
werden unverändert übernommen. Sag in der README klar, dass BipedalWalker mit
diesen Startwerten **nicht** gelöst wird: Ein sicher gehender Roboter braucht
die Profilbudgets und größere Netze. Die Startwerte dienen dazu, die Verfahren
in überschaubarer Zeit nebeneinander zu sehen.

Drei weitere Punkte sind ausdrücklich zu dokumentieren:

- Das PPO-Profil verwendet `n_envs = 32`. Dieses Projekt trainiert mit **einer**
  Environment-Instanz, damit Animation, Episodenbuchhaltung und schrittgenaue
  Zwischenevaluation eindeutig bleiben. Der effektive Rollout je Update beträgt
  dadurch 2048 statt 65.536 Schritte. Zusammen mit dem ohnehin höchsten Budget
  heißt das: PPO ist auf einer einzelnen Environment für dieses Projekt kein
  realistischer Sieger. Sag das in der README deutlich, statt es durch ein
  geschöntes Profil zu verdecken.
- `SAC` trainiert mit `train_freq = 64` und `gradient_steps = 64` in Schüben:
  64 gesammelte Schritte, dann 64 Gradientenschritte. Das ist deutlich
  rechenintensiver je Environment-Schritt als bei einfacheren Environments und
  der Grund für das vergleichsweise kleine Schrittbudget. Erkläre den
  Zusammenhang.
- Ändert der Benutzer eines der beiden Budgets, weist die GUI gemäß Workbench
  vor dem Vergleichsstart auf die Abweichung hin.

Quellen: [RL Baselines3 Zoo, PPO](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml),
[TD3](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/td3.yml),
[SAC](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml)

## Evaluation

Zeige im Evaluationsergebnis mindestens:

- mittleren Return und Standardabweichung
- mittlere Episodenlänge (Maximum 1600)
- Zielquote (Anteil Episoden, die das Streckenende erreichen)
- Gelöst-Quote (Anteil Episoden mit Return `≥ 300`)
- Sturzquote
- Zeitlimitquote
- letzte und beste deterministische Evaluation

## Vergleich

Der Vergleichsgraph zeigt den explorativen Episoden-Return gegen die
Episodennummer und enthält eine Referenzlinie bei `+300`.

Die Vergleichs-Summary enthält je Slot diese Zeilen:

- Algorithmus
- Episoden
- Environment-Schritte, ausgeführt und angefordert
- durchschnittlicher Return
- Zielquote
- Gelöst-Quote
- Sturzquote
- Zeitlimitquote
- beste deterministische Evaluation

## Darstellung

Zeige ausschließlich den offiziellen, von `env.render()` gelieferten
Gymnasium-RGB-Frame (600 × 400 Pixel). Erstelle keine eigene Grafik und öffne
kein separates Pygame-Fenster.

Der Standardwert der Bildrate ist die environment-eigene Rate von 50 FPS
(`env.metadata["render_fps"]`). Weise in der Bedienungsanleitung darauf hin,
dass eine Episode bis zu 1600 Schritte dauert – bei 50 FPS gut eine halbe
Minute – und dass eine höhere Bildrate das Zuschauen abkürzt, während die
Animation während eines Trainings- oder Vergleichslaufs dessen Fortschritt
verlangsamt.

Zeige neben der Animation:

- Rumpfwinkel und normierte Rumpf-Winkelgeschwindigkeit
- normierte Geschwindigkeiten `vₓ` und `v_y`
- je Bein Hüft- und Kniewinkel, deren Geschwindigkeiten und den Bodenkontakt
  als verständliche Ja/Nein-Anzeige
- die zehn Lidar-Werte kompakt in einer Zeile, mit dem Hinweis, dass kleinere
  Werte näheren Boden bedeuten
- die gewählte Action doppelt: die vier Rohwerte `a₀` bis `a₃` sowie ihre
  Bedeutung je Gelenk, also Drehrichtung und Moment in Prozent
- aktuellen Episodenschritt und bisher kumulierten Return
- welchem Verfahrensslot die laufende Episode gehört

Projektspezifische Metriken sind Episoden-Return, Episodenlänge, Zielquote,
Gelöst-Quote, Sturzquote und Zeitlimitquote. Achsen, Summary und Hilfetexte
erklären eindeutig, dass höhere Werte besser sind und `300` als gelöst gilt.

## Lernzustand sichern

Buttons zum manuellen Speichern und Laden gibt es nicht; gesichert wird
ausschließlich der automatische Checkpoint des besten Evaluationsergebnisses.
Umfang und Prüfungen stehen in der Workbench. Environmentspezifisch gehört zu
den Metadaten die Kennung `BipedalWalker-v3` samt `hardcore=False`; ein Stand
der Hardcore-Variante wird verständlich abgelehnt.

## Tests

Zusätzlich zu den allgemeinen und den verfahrensbezogenen Tests der Workbench:

- Environment-Kennwerte: Action-Space `Box(-1, 1, (4,))`, Observation `(24,)`,
  `max_episode_steps = 1600`, `reward_threshold = 300`, `hardcore` bleibt
  `False`, die Factory fordert genau diese Argumente an
- Interpretation der vier Drehmomente: Vorzeichen als Richtung, Betrag als
  Momentanteil, Clipping außerhalb `[-1, 1]`
- Unterscheidung der vier Episodenausgänge: Ziel erreicht, Sturz mit
  Terminalreward `−100`, Zeitlimit, gelöst ab `300`; insbesondere, dass ein
  Zeitlimit weder als Ziel noch als Sturz zählt
- die Lidar-Werte werden vollständig und unverändert angezeigt
- `SAC`-Zielentropie ist bei `auto` genau `-4`
- Normalisierung: Statistiken wachsen im Training, bleiben in der Evaluation
  unverändert, der ausgewiesene Return ist unnormalisiert, ein
  Speichern/Laden-Zyklus stellt die Statistiken wieder her

Halte kurze Testläufe klein: BipedalWalker-Episoden laufen ohne Sturz bis zu
1600 Schritte. Wähle Budgets, die zuverlässig mindestens eine abgeschlossene
Episode liefern, ohne die Testlaufzeit unnötig zu verlängern.

## Abhängigkeiten

Dieses Projekt führt **keine** neue Abhängigkeit ein. `BipedalWalker-v3` nutzt
dasselbe Box2D wie die LunarLander-Projekte; `swig`, `box2d`, Gymnasium,
PyTorch und Stable-Baselines3 stehen bereits in `../environment.yml`. Die
`requirements.txt` listet dieselben direkten Abhängigkeiten. Die README nennt
den Box2D-Installationsschritt ausdrücklich, da das Projekt ohne Box2D nicht
startet.

## Abnahme

Zusätzlich zu allen Abnahmekriterien aus `../workbench.md` gilt: Fertig, wenn
`PPO`, `TD3` und `SAC` mit den obigen Profilen trainieren, deterministisch
evaluiert und fair verglichen werden; der vierdimensionale Action-Space in
Anzeige und Tests korrekt als Drehmomente interpretiert wird; die
24-dimensionale Observation einschließlich der zehn Lidar-Werte verständlich
und ohne erfundene Einheitenumrechnung dargestellt wird; Ziel-, Gelöst-,
Sturz- und Zeitlimitquote getrennt ausgewiesen werden; die Normalisierung im
Training lernt, in der Evaluation eingefroren bleibt, den ausgewiesenen Return
nicht verfälscht und vollständig gespeichert wird; die Animation sich auch
während Training und Vergleich zuschalten und in der Bildrate einstellen lässt;
und ein Speicherstand der Hardcore-Variante abgelehnt wird.
