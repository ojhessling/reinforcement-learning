# BipedalWalker Workbench

Lokale Tkinter-Lernanwendung, die **PPO**, **TD3** und **SAC** auf Gymnasiums
`BipedalWalker-v3` trainiert, deterministisch evaluiert und paarweise
vergleicht. Zwei gleichrangige Verfahrensslots stehen sich gegenüber:
wahlweise zwei verschiedene Verfahren oder zweimal dasselbe mit anderen
Parametern.

Grundlage sind `../workbench.md` und `prompt.md` in diesem Ordner.

## Installation und Start

Das Projekt nutzt die gemeinsame Kursumgebung `../environment.yml` und führt
**keine** neue Abhängigkeit ein. `BipedalWalker-v3` benötigt wie die
LunarLander-Projekte **Box2D**; ohne dieses Paket startet die App nicht:

```bash
pip install swig
pip install "gymnasium[box2d]"
```

`swig` muss vor `box2d` installiert werden, weil das Box2D-Paket es zum Bauen
benötigt. Danach:

```bash
cd Oliver/12-bipedalwalker
python bipedalwalker_app.py
```

## Environment

Das Environment wird ausschließlich so erzeugt:

```python
import gymnasium

env = gymnasium.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")
```

`hardcore=False` steht ausdrücklich da, obwohl es dem Standard entspricht: Die
Hardcore-Variante mit Stufen, Gruben und Hindernissen gehört nicht zum Projekt
und bräuchte ein um Größenordnungen höheres Budget.

**Actions** – `Box(-1, 1, (4,), float32)`, Drehmomente für vier Motoren:

| Index | Gelenk |
| --- | --- |
| `a₀` | Hüfte Bein 1 |
| `a₁` | Knie Bein 1 |
| `a₂` | Hüfte Bein 2 |
| `a₃` | Knie Bein 2 |

Das Vorzeichen bestimmt die Drehrichtung, der Betrag das maximal anliegende
Drehmoment als Anteil von `MOTORS_TORQUE = 80`. Ein Wert von `0` heißt also
nicht „Gelenk hält die Position", sondern „kein Moment".

**Observation** – 24 Werte: Rumpfwinkel und -winkelgeschwindigkeit, `vₓ` und
`v_y`, je Bein Hüft- und Kniewinkel samt Geschwindigkeiten und Bodenkontakt,
dazu zehn Lidar-Messwerte. Die Lidar-Werte geben den Anteil der Strahllänge bis
zum Bodentreffer an – kleinere Werte bedeuten näheren Boden. Sie sind die
einzige Wahrnehmung des Geländes.

Die Geschwindigkeiten sind im Environment bereits skaliert und die Kniewinkel
um `+1` verschoben. Die Anzeige behandelt sie deshalb als **normierte Größen**
und markiert sie mit `*`; eine saubere Umrechnung in physikalische Einheiten
existiert nicht. Nur der Rumpfwinkel ist ein echter Winkel und erscheint
zusätzlich in Grad. Die absolute Position des Rumpfes steht nicht in der
Observation und wird deshalb auch nicht angezeigt.

**Reward** je Schritt: die Differenz zweier Shaping-Terme, die Vorwärtskommen
belohnen (`130 · x / SCALE`) und Schräglage bestrafen (`−5 · |Rumpfwinkel|`),
abzüglich Drehmomentkosten von `0,00035 · 80 · |aᵢ|` je Motor – bis zu `−0,112`
pro Schritt, über eine Episode grob `−50`. Die Normierung ist so gewählt, dass
die vollständige Strecke rund `+300` ergibt.

Vier Ausgänge werden getrennt ausgewiesen:

| Ausgang | Erkennung |
| --- | --- |
| **Ziel erreicht** | `terminated`, letzter Reward über `−50` |
| **Sturz** | `terminated` mit dem Terminalreward `−100` |
| **Zeitlimit** | `truncated` nach 1600 Schritten |
| **Gelöst** | Episoden-Return `≥ 300` (offizieller `reward_threshold`) |

Die Unterscheidung ist hier wichtiger als bei einem Environment mit klarem
Terminalbonus: Ein vorsichtiger Agent erreicht das Zeitlimit zuverlässig, ohne
je zu stürzen oder voranzukommen – eine reine Sturzquote ließe ihn gut
aussehen.

Quelle: [Gymnasium Bipedal Walker](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)

## Methoden

Alle drei Verfahren stammen unverändert aus Stable-Baselines3; ihre fachlichen
Steckbriefe und vollständigen Parameterlisten stehen im Verfahrenskatalog der
Workbench. Kurz:

- **PPO** ist on-policy: Rollouts fester Länge, Vorteile über GAE, ein
  geclipptes Surrogatziel über mehrere Epochen. Kein Replay Buffer.
- **TD3** ist off-policy mit deterministischem Actor, zwei Critics, verzögerten
  Policy-Updates und Target Policy Smoothing. Ohne explizites Action Noise
  exploriert es nicht.
- **SAC** ist off-policy mit stochastischem Actor und einer Entropie, deren
  Temperatur mitgelernt wird; die Zielentropie ist hier `-dim(A) = -4`.

## Normalisierung

Das PPO-Profil des Zoo verlangt für dieses Environment `normalize: true` – ohne
Normalisierung der 24 sehr unterschiedlich skalierten Werte lernt PPO kaum. Die
Gruppe `Normalisierung` steht deshalb in jedem Verfahrenstab: Beobachtungen und
Rewards getrennt schaltbar, dazu die beiden Clip-Werte (Standard `10,0`).
Umgesetzt ist sie mit `VecNormalize`.

- Die laufenden Statistiken wachsen **nur im Training**. Evaluation und
  Animation verwenden dieselben Statistiken eingefroren.
- Graph, Summary und Evaluation zeigen immer den **unnormalisierten** Return –
  sonst wäre die Referenzlinie bei `+300` bedeutungslos. Der `Monitor`-Wrapper
  sitzt dafür innerhalb der Normalisierung, und der Terminalreward `−100` wird
  über `get_original_reward()` gelesen.
- Die Statistiken gehören zum Checkpoint. Ein Stand ohne sie wird abgelehnt.

Standard: PPO normalisiert Beobachtungen und Rewards, TD3 und SAC nicht. Grund:
Ein Replay Buffer speichert Beobachtungen, deren Normalisierungsstatistik sich
weiter verschiebt, sodass alte Einträge nicht mehr zur aktuellen Normierung
passen. Wählbar bleibt die Option für alle drei.

## Standardprofile und Parameter

Ein Dropdown-Wechsel lädt das Profil des neu gewählten Verfahrens in den
zugehörigen Tab und setzt nur diesen Slot zurück. Grundlage sind die getunten
`BipedalWalker-v3`-Profile des RL Baselines3 Zoo (Stand geprüft am 18.08.2026);
alles Übrige stammt aus den Voreinstellungen von Stable-Baselines3 2.9.0.

| | PPO | TD3 | SAC |
| --- | --- | --- | --- |
| Trainingsschritte `N` | 100.000 | 100.000 | 100.000 |
| Lernrate `α` | 3e-4 | 1e-3 | 7,3e-4 |
| Batch-Größe `B` | 64 | 256 | 256 |
| Diskontfaktor `γ` | 0,999 | 0,98 | 0,98 |
| Hidden Layers | 64,64 | 64,64 | 64,64 |
| Aktivierung | Tanh | ReLU | ReLU |
| Normalisierung | Obs + Reward | aus | aus |
| Verfahrenseigen | `n_steps=2048`, `n_epochs=10`, `λ_GAE=0,95`, `ε_clip=0,18` | Buffer 200.000, `t₀=10.000`, Action Noise normal `σ=0,1` | Buffer 300.000, `t₀=10.000`, `τ=0,02`, `train_freq=64`, `gradient_steps=64`, gSDE, `log σ₀=-3` |

**Budget und Netzgröße sind bewusst vereinheitlicht.** Die Zoo-Profile nennen
5.000.000 (PPO), 1.000.000 (TD3) und 500.000 (SAC) Schritte sowie `400,300` als
Netzgröße für TD3 und SAC. Diese Budgets sind interaktiv nicht abwartbar, und
unterschiedliche Budgets machten den Vergleich zweier Slots ohne Zutun unfair;
die großen Netze kosten je Schritt viel Rechenzeit. Beides ist in der UI frei
änderbar, alle übrigen getunten Hyperparameter stammen unverändert aus den
Profilen.

Drei weitere Punkte:

- Das PPO-Profil verwendet `n_envs = 32`. Diese App trainiert mit **einer**
  Environment-Instanz, damit Animation, Episodenbuchhaltung und schrittgenaue
  Zwischenevaluation eindeutig bleiben. Der effektive Rollout je Update beträgt
  dadurch 2048 statt 65.536 Schritte. Zusammen mit dem ohnehin höchsten
  Profilbudget heißt das: PPO ist hier kein realistischer Sieger.
- SAC trainiert mit `train_freq = 64` und `gradient_steps = 64` in Schüben – 64
  gesammelte Schritte, dann 64 Gradientenschritte. Das ist je
  Environment-Schritt deutlich rechenintensiver als bei den anderen beiden und
  der Grund für das kleinste Profilbudget.
- Weichen die Budgets der beiden Slots voneinander ab, weil du eines geändert
  hast, fragt die GUI vor dem Vergleich nach; verboten ist es nicht.

**Exploration** entsteht aus der Policy selbst beziehungsweise aus explizitem
Action Noise. ε-greedy-Parameter gibt es nicht.

Quellen: [Zoo PPO](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml),
[Zoo TD3](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/td3.yml),
[Zoo SAC](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml)

## Bedienung und Ansichten

1. Oben für `Verfahren 1` und `Verfahren 2` je einen Algorithmus wählen – gern
   zweimal denselben.
2. Im zugehörigen Tab die Parameter setzen. Jeder Tab enthält vollständig und
   unabhängig die Parameter seines Slots; Parameter, die das gewählte Verfahren
   nicht kennt, erscheinen gar nicht.
3. Der sichtbare Tab ist das **aktive Verfahren**; über den Buttons steht
   dauerhaft `Aktiv: Verfahren 1 – PPO`. Während eines Laufs bleibt das Ziel
   fixiert.

Global bleiben nur `Eval-Episoden M` und `Eval-Intervall`, damit beide Slots
dieselben Stützstellen liefern.

| Button | Wirkung |
| --- | --- |
| Training starten / fortsetzen | trainiert das aktive Verfahren |
| Stoppen | bricht den laufenden Lauf ab |
| Deterministisch evaluieren | evaluiert das aktive Verfahren ohne Exploration |
| Sichtbare Episode abspielen | zeigt eine Episode des aktiven Verfahrens |
| Vergleich starten / fortsetzen | trainiert **beide** Slots parallel |
| Bestes Modell wiederherstellen | holt den besten Checkpoint des aktiven Slots |
| Neues Modell | verwirft den Lernzustand des aktiven Slots |

Neben der Animation stehen alle 24 Beobachtungswerte – Rumpf, Geschwindigkeiten,
beide Beine mit Bodenkontakt und die zehn Lidar-Werte –, die vier Rohactions und
ihre Bedeutung je Gelenk (Richtung und Moment in Prozent), Episodenschritt,
kumulierter Return und der laufende Slot.

Die Animation hat einen Schalter, `Animation zeigen`, und ein Feld
`Bildrate (FPS)` – Standard 50 FPS, die Bildrate des Environments, gültig 1 bis
120. Beide wirken **jederzeit**, auch mitten in einem Lauf. Eingeschaltet siehst
du dem Lernen live zu: beim Vergleich für beide Verfahren gleichzeitig,
untereinander. Jede Episode spielt eine Kopie der Policy von ihrem Beginn, damit
Anzeige und Training nie gleichzeitig auf dasselbe Netz zugreifen. Das kostet
Rechenzeit; zum reinen Trainieren schaltest du die Animation besser aus.

Eine Episode dauert bis zu 1600 Schritte – bei 50 FPS gut eine halbe Minute. Ein
Roboter, der stehen bleibt statt zu gehen, läuft immer in dieses Limit; eine
höhere Bildrate kürzt das Zuschauen ab.

Unten links liegen die Diagramme, rechts die Summary. `Training` zeigt das
aktive Verfahren mit gleitendem Mittel über 20 Episoden und getrennt davon die
deterministischen Evaluationen; `Vergleich` beide Slots als `V1 – …` und
`V2 – …`, unterschieden durch Farbe und Linienstil. Beide Graphen tragen eine
Referenzlinie bei `+300`. Die Vergleichs-Summary hat genau zwei Ergebnisspalten
und listet darunter die Parameter, in denen sich die Konfigurationen
unterscheiden. Diagramm (PNG) und Summary (UTF-8-Text) sind exportierbar.

**Fairness:** PPO sieht bei gleichem Schrittbudget systematisch schlechter aus
als TD3 und SAC – die Off-Policy-Verfahren lernen aus jedem gespeicherten
Übergang mehrfach, PPO verwirft seine Daten nach jedem Update. Bei
BipedalWalker fällt der Unterschied größer aus als bei einfacheren
Environments.

## Lernzustand sichern

Buttons zum manuellen Speichern und Laden gibt es nicht. Im eingestellten
Schrittintervall wird automatisch deterministisch evaluiert; der beste Stand
wird je Slot getrennt als vollständiger Checkpoint gesichert und ist über
`Bestes Modell wiederherstellen` zurückholbar.

| | PPO | TD3 und SAC |
| --- | --- | --- |
| Modell | Policy, Value-Netz, Optimizer | Actor, Critics, Target-Netze, Optimizer (SAC zusätzlich die gelernte Temperatur) |
| Replay Buffer | entfällt – PPO ist on-policy | gehört dazu |
| Normalisierung | Statistiken, wenn aktiv | dito |
| Metadaten | Format-Version, Environment samt `hardcore`, Algorithmus, vollständige Konfiguration | dito |

Ein Stand, der nicht zum Verfahren des Slots passt oder dem ein Bestandteil
fehlt, wird verständlich abgelehnt. Die Checkpoints liegen in einem temporären
Verzeichnis und verschwinden mit dem Schließen der App: Ein Lernstand überlebt
die Sitzung nicht.

## Tests und Grenzen

```bash
python -m unittest discover -s tests -v
```

Die Tests prüfen die Zoo-Profile je Verfahren, dass `model_kwargs()` nur
Argumente verwendet, die der jeweilige Algorithmus kennt, den
vierdimensionalen Action Space samt Drehrichtung, Momentanteil und Clipping,
alle 24 Beobachtungswerte inklusive vollständiger Lidar-Anzeige, die
unveränderten Environment-Argumente samt `hardcore=False`, die Unterscheidung
von Ziel, Sturz und Zeitlimit (ein Sturz endet nachweislich mit exakt `−100`),
TD3s verzögerte Actor-Updates, SACs automatische Entropie-Anpassung mit
Zielentropie `−4`, PPOs Clip-Schedule und die Regel „Batch teilt Rollout",
kurze Trainings- und Evaluationsläufe aller drei Verfahren, die unveränderte
Evaluation, Save/Load/Fortsetzen je Verfahren, die Unabhängigkeit zweier Slots
mit demselben Algorithmus, den Renderer im eigenen Prozess, Export und
Startlayout in beiden Verfahrenstabs. Eigene Tests decken die Normalisierung
ab: Statistiken wachsen im Training, bleiben in der Evaluation unverändert, der
ausgewiesene Return bleibt unnormalisiert, und ein Speichern/Laden-Zyklus
stellt die Statistiken wieder her. Zwei GUI-Vergleichsläufe prüfen den echten
Weg über Worker-Threads und Ereignis-Queue, einer davon mit Live-Animation für
beide Verfahren.

**Grenzen.** Mit den Startwerten wird BipedalWalker **nicht** gelöst. Ein
sicher gehender Roboter braucht die Profilbudgets (Millionen Schritte) und
größere Netze; die Startwerte dienen dazu, die Verfahren in überschaubarer Zeit
nebeneinander zu sehen. Rechne mit typischen Zwischenzuständen: Ein
untrainierter Roboter stürzt nach rund 40 bis 130 Schritten, und ein häufiges
lokales Optimum ist Stehenbleiben – die Episode endet dann erst am Zeitlimit
mit einem Return nahe null. Genau dafür gibt es die getrennte Zeitlimitquote.
Wiederholungsläufe mit Mittelwert und Streuband sind nicht implementiert, jeder
Slot trainiert einen Seed.
