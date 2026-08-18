# LunarLander Policy-Gradient Workbench

Lokale Tkinter-Lernanwendung, die **PPO**, **TD3** und **SAC** auf dem
kontinuierlichen `LunarLander-v3` von Gymnasium trainiert, deterministisch
evaluiert und paarweise vergleicht. Die Anwendung stellt zwei gleichrangige
Verfahrensslots gegenüber: Slot 1 gegen Slot 2, wahlweise zwei verschiedene
Verfahren oder zweimal dasselbe Verfahren mit anderen Parametern.

Grundlage sind `../workbench.md` und `prompt.md` in diesem Ordner.

## Installation und Start

Das Projekt nutzt die gemeinsame Kursumgebung `../environment.yml` und führt
**keine** neue Abhängigkeit ein. `LunarLander-v3` benötigt allerdings **Box2D**;
ohne dieses Paket startet die App nicht:

```bash
pip install swig
pip install "gymnasium[box2d]"
```

`swig` muss vor `box2d` installiert werden, weil das Box2D-Paket es zum Bauen
benötigt. Danach:

```bash
cd Oliver/11-lunarlander_policy_gradient
python lunarlander_pg_app.py
```

## Environment und Reward

Das Environment wird ausschließlich so erzeugt:

```python
import gymnasium

env = gymnasium.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
```

Das ist gleichwertig zur registrierten Kennung `LunarLanderContinuous-v3`. Alle
weiteren Konstruktorargumente bleiben auf ihren Standardwerten
(`gravity=-10.0`, `enable_wind=False`, `wind_power=15.0`,
`turbulence_power=1.5`); Dynamik, Startverteilung, Reward und Abbruchregeln
werden nicht verändert.

**Warum kontinuierlich?** TD3 und SAC arbeiten ausschließlich mit
kontinuierlichen Action-Spaces. PPO beherrscht beides und läuft hier ebenfalls
kontinuierlich, damit alle drei Verfahren dieselbe Aufgabe lösen und fair
vergleichbar bleiben. Die diskrete Variante ist Gegenstand von
`Oliver/10-lunarlander`.

**Actions** – `Box(-1, 1, (2,), float32)`:

| Wert | Bedeutung |
| --- | --- |
| `a₀ ≤ 0` | Haupttriebwerk aus |
| `a₀ > 0` | Haupttriebwerk mit 50 % bis 100 % Schub |
| `a₁ < -0,5` | linkes Steuertriebwerk, 50 % bis 100 % |
| `a₁ > 0,5` | rechtes Steuertriebwerk, 50 % bis 100 % |
| sonst | Steuertriebwerke aus |

**Observation** (8 Werte): Position `x`, `y`, Geschwindigkeit `vₓ`, `v_y`,
Winkel `θ`, Winkelgeschwindigkeit `θ̇` sowie Bodenkontakt beider Beine. Die
Winkelgeschwindigkeit liegt in Einheiten von `0,4 rad/s` vor und wird für die
Anzeige mit `2,5` multipliziert; der Winkel wird in Grad umgerechnet.

**Reward** je Schritt: eine Differenz zweier Shaping-Terme für Annäherung an
die Plattform, geringere Geschwindigkeit, geringere Schräglage und Beinkontakt
(`+10` je Bein), abzüglich `0,3` je Frame bei vollem Haupttriebwerksschub und
`0,03` je Frame bei vollem Steuertriebwerksschub. Anders als in der diskreten
Variante ist dieser Treibstoffabzug **stufenlos**: Er ist proportional zum
tatsächlichen Schub, sanfteres Gasgeben kostet also weniger. Die Episode endet
mit `-100` bei Absturz und `+100`, sobald der Lander zur Ruhe kommt; nach 1000
Schritten wird sie trunkiert.

Höhere Werte sind besser. Der Episoden-Return liegt realistisch zwischen etwa
`-400` und `+320`; Gymnasium nennt `200` als `reward_threshold`.

Zwei Erfolgsmaße werden getrennt ausgewiesen, weil eine sichere Landung mit
hohem Treibstoffverbrauch die Lösungsschwelle verfehlen kann:

- **Landequote**: Episode endet mit `terminated` und ruhendem Lander (`+100`)
- **Gelöst-Quote**: Episoden-Return `≥ 200`

Quelle: [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

## Methoden

Alle drei Verfahren stammen unverändert aus Stable-Baselines3; es werden keine
eigenen Algorithmusvarianten gebaut. Die eigene fachliche Arbeit liegt in
Konfiguration, Runnern, Metriken, Vergleich, GUI und Tests.

**PPO** ist on-policy. Es sammelt Rollouts fester Länge `n_steps`, schätzt
Vorteile mit Generalized Advantage Estimation und optimiert über `n_epochs`
Durchläufe ein geclipptes Surrogatziel:

```text
L = E[ min( r(θ)·Â , clip(r(θ), 1-ε, 1+ε)·Â ) ]   mit r(θ) = π_θ(a|s) / π_alt(a|s)
```

Die gesammelten Daten werden nach dem Update verworfen; einen Replay Buffer
gibt es nicht.

**TD3** ist off-policy mit deterministischem Actor. Es lernt zwei Critics und
bildet das Ziel aus deren Minimum, um Überschätzung zu dämpfen. Auf die
Target-Action kommt geclipptes Rauschen (Target Policy Smoothing), und Actor
sowie Target-Netze werden nur alle `policy_delay` Updates aktualisiert:

```text
ã = clip(π_target(s') + clip(N(0, σ_t), -c, +c), -1, +1)
y = r + γ·(1-done)·min( Q₁_target(s', ã), Q₂_target(s', ã) )
```

Weil der Actor deterministisch ist, exploriert TD3 ohne explizites Action Noise
überhaupt nicht. Die Konfiguration lehnt `Action Noise = keins` für TD3 daher
mit einer verständlichen Meldung ab.

**SAC** ist off-policy mit stochastischem Actor. Es maximiert zusätzlich zur
erwarteten Rendite die Entropie der Policy, gewichtet mit einer Temperatur `α`:

```text
y = r + γ·(1-done)·[ min(Q₁_target, Q₂_target) - α·log π(a'|s') ]
```

Bei `Entropie α = auto` wird `α` selbst gelernt, sodass die mittlere Entropie
einer Zielentropie folgt; Stable-Baselines3 verwendet als Standard
`target_entropy = -dim(A)`, hier also `-2`.

**Einordnung:** Im engeren Sinn ist nur PPO ein Policy-Gradient-Verfahren. TD3
und SAC sind Off-Policy-Actor-Critic-Verfahren; gemeinsam ist allen dreien,
dass sie eine Policy direkt parametrisieren und über Gradienten verbessern.

Quellen: [PPO](https://arxiv.org/abs/1707.06347),
[GAE](https://arxiv.org/abs/1506.02438),
[TD3](https://arxiv.org/abs/1802.09477),
[SAC](https://arxiv.org/abs/1801.01290),
[SAC mit gelernter Temperatur](https://arxiv.org/abs/1812.05905),
[gSDE](https://arxiv.org/abs/2005.05719)

## Standardprofile und Parameter

Jedes Verfahren bringt sein eigenes Profil mit. Ein Wechsel im Dropdown lädt
die Werte des neu gewählten Verfahrens in den zugehörigen Tab und setzt
ausschließlich den Lernzustand dieses Slots zurück. Grundlage sind die
getunten `LunarLanderContinuous-v3`-Profile des RL Baselines3 Zoo (Stand
geprüft am 18.08.2026); alles Übrige stammt aus den Voreinstellungen von
Stable-Baselines3 2.9.0.

| | PPO | TD3 | SAC |
| --- | --- | --- | --- |
| Trainingsschritte `N` | 100.000 | 100.000 | 100.000 |
| Lernrate `α` | 3e-4 konstant | 1e-3 konstant | 7,3e-4 linear fallend |
| Batch-Größe `B` | 64 | 256 | 256 |
| Diskontfaktor `γ` | 0,999 | 0,98 | 0,99 |
| Hidden Layers | 64,64 | 64,64 | 64,64 |
| Aktivierung | Tanh | ReLU | ReLU |
| Verfahrenseigen | `n_steps=1024`, `n_epochs=4`, `λ_GAE=0,98`, `c_ent=0,01` | Buffer 200.000, `t₀=10.000`, Action Noise normal `σ=0,1` | Buffer 1.000.000, `t₀=10.000`, `τ=0,01`, `α=auto` |

**Budget und Netzgröße sind bewusst vereinheitlicht.** Die Zoo-Profile nennen
1.000.000 (PPO), 300.000 (TD3) und 500.000 (SAC) Schritte sowie `400,300` als
Netzgröße für TD3 und SAC. Diese Budgets sind interaktiv nicht abwartbar, und
unterschiedliche Budgets machen einen Vergleich zweier Slots ohne Zutun unfair;
die großen Netze kosten je Schritt viel Rechenzeit und lohnen sich erst bei
entsprechendem Budget. Beides ist in der UI frei änderbar – für belastbare
Ergebnisse braucht es mehr Schritte und größere Netze. Alle übrigen getunten
Hyperparameter stammen unverändert aus den Profilen.

Zwei weitere bewusste Abweichungen:

- Das PPO-Profil des Zoo verwendet `n_envs = 16`. Diese App trainiert wie alle
  Vorgängerprojekte mit **einer** Environment-Instanz, damit Animation,
  Episodenbuchhaltung und schrittgenaue Zwischenevaluation eindeutig bleiben.
  Der effektive Rollout je Update beträgt dadurch 1024 statt 16.384 Schritte:
  PPO macht je Environment-Schritt mehr, aber kleinere Updates und kommt
  langsamer voran als im Zoo-Profil.
- Weichen die Budgets der beiden Slots voneinander ab, weil du eines geändert
  hast, fragt die GUI vor dem Vergleich nach; verboten ist es nicht.

Alle Netz-Hyperparameter sind in der UI änderbar: Actor- und Critic-Hidden-
Layers getrennt, Aktivierung, Optimizer samt `eps` und Weight Decay, Lernrate
mit konstantem oder linear fallendem Verlauf, Batch-Größe, Gradient Clipping
(PPO) sowie Target-Update und Gradientenschritte (TD3, SAC). Technische
Optionen wie `verbose`, `device` oder `tensorboard_log` gehören nicht in die
UI.

**Exploration** entsteht in allen drei Verfahren aus der Policy selbst
beziehungsweise aus explizitem Action Noise. ε-greedy-Parameter gibt es in
diesem Projekt nicht.

Quellen: [Zoo PPO](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml),
[Zoo TD3](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/td3.yml),
[Zoo SAC](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml)

## Bedienung und Ansichten

1. Oben im Bedienpanel für `Verfahren 1` und `Verfahren 2` je einen Algorithmus
   wählen – gern zweimal denselben.
2. Im zugehörigen Tab die Parameter setzen. Jeder Tab enthält vollständig und
   unabhängig die Parameter seines Slots, einschließlich Trainingsbudget und
   Seed. Parameter, die das gewählte Verfahren nicht kennt, erscheinen gar
   nicht.
3. Der sichtbare Tab ist das **aktive Verfahren**; über den Buttons steht
   dauerhaft `Aktiv: Verfahren 1 – PPO`. Alle Einzellauf-Buttons wirken darauf.
   Während eines laufenden Laufs bleibt das Ziel fixiert; ein Tabwechsel zeigt
   dann nur andere Parameter an.

Global – also außerhalb der Tabs – bleiben nur `Eval-Episoden M` und
`Eval-Intervall`, damit beide Slots dieselben Stützstellen liefern.

Die Buttons in der dritten Spalte:

| Button | Wirkung |
| --- | --- |
| Training starten / fortsetzen | trainiert das aktive Verfahren |
| Stoppen | bricht den laufenden Lauf ab |
| Deterministisch evaluieren | evaluiert das aktive Verfahren ohne Exploration |
| Sichtbare Episode abspielen | zeigt eine Episode des aktiven Verfahrens |
| Vergleich starten / fortsetzen | trainiert **beide** Slots parallel |
| Bestes Modell wiederherstellen | lädt den besten Checkpoint des aktiven Slots |
| Neues Modell | verwirft den Lernzustand des aktiven Slots |

Neben der Animation stehen Position, Geschwindigkeit, Winkel in Grad,
Winkelgeschwindigkeit in `rad/s`, Bodenkontakt beider Beine, Episodenschritt,
kumulierter Return und der laufende Slot. Die Action wird doppelt angezeigt:
als Rohwerte `a₀`/`a₁` und als Triebwerksbedeutung, also Haupttriebwerk `aus`
oder Schub in Prozent und Steuertriebwerk `aus`, `links` oder `rechts`.

Die Animation hat genau einen Schalter, `Animation zeigen`, und ein Feld
`Bildrate (FPS)` – Standard sind 50 FPS, die Bildrate des Environments, gültig
sind 1 bis 120. Beide gelten global und lassen sich **jederzeit** ändern, auch
mitten in einem laufenden Training oder Vergleich.

Eingeschaltet siehst du dem Lernen live zu: Es laufen fortlaufend Episoden mit
dem aktuellen Lernstand – beim Einzeltraining für dessen Slot, beim Vergleich
für **beide Verfahren gleichzeitig**, untereinander, jedes mit eigenem Bild und
eigener Messwertanzeige. Die Bilder sind dann kleiner; mehr Platz bekommst du
über den Splitter oder ein größeres Fenster.

Jede Episode spielt eine Kopie der Policy von ihrem Beginn: Anzeige und
Training greifen so nie gleichzeitig auf dasselbe Netz zu. Das kostet
Rechenzeit und verlangsamt den Lauf; die eingestellte Bildrate ist eine
Obergrenze, die während eines Laufs nicht immer erreicht wird. Zum reinen
Trainieren schaltest du die Animation besser aus.

Gerendert wird ausschließlich der offizielle Gymnasium-RGB-Frame (600 × 400) in
einem isolierten Prozess, damit SDL und Tk unter macOS nicht im selben Prozess
landen. Der zweite Renderprozess entsteht erst, wenn wirklich zwei Episoden
gleichzeitig laufen.

Unten links liegen die Diagramme, rechts die Summary:

- **Training** zeigt die Episodenwerte des aktiven Slots dezent, den gleitenden
  Mittelwert über 20 Episoden kräftig und getrennt davon die deterministischen
  Evaluationen. Eine Referenzlinie markiert `+200`.
- **Vergleich** zeigt beide Slots als `V1 – …` und `V2 – …`, unterschieden
  durch Farbe und Linienstil. Verwenden beide Slots denselben Algorithmus,
  nennt das Label zusätzlich den wichtigsten abweichenden Parameter.
- Die **Vergleichs-Summary** hat genau zwei Ergebnisspalten und listet darunter
  die Parameter, in denen sich die beiden Konfigurationen unterscheiden. Ohne
  diesen Block wäre ein Vergleich zweier Parametrisierungen desselben
  Algorithmus nicht interpretierbar. Verglichen werden dabei nur Parameter, die
  beide Verfahren besitzen; verfahrenseigene Parameter stehen vollständig im
  jeweiligen Tab.

Diagramm (PNG) und Summary (UTF-8-Text) sind exportierbar; beide teilen
denselben Dateinamensstamm, solange sich der dargestellte Stand nicht ändert.

**Fairness beim Vergleich:** PPO sieht bei gleichem Schrittbudget systematisch
schlechter aus als TD3 und SAC. Die beiden Off-Policy-Verfahren lernen aus
jedem gespeicherten Übergang mehrfach, PPO verwirft seine Daten nach jedem
Update. Das ist kein Messfehler, sondern eine Eigenschaft der Verfahrensklassen.

**Vergleich und Einzeltraining sind getrennt.** Der Vergleich benutzt eigene
Modelle je Slot und verändert das sichtbare Einzelexperiment nicht. Ein erneut
gestarteter, kompatibel konfigurierter Vergleich setzt die Vergleichsmodelle
nicht zurück, sondern setzt ihr Training fort.

## Lernzustand sichern

Buttons zum manuellen Speichern und Laden gibt es nicht. Im eingestellten
Schrittintervall wird automatisch und deterministisch evaluiert; der beste
Stand wird je Slot getrennt als vollständiger Checkpoint gesichert und ist über
`Bestes Modell wiederherstellen` zurückholbar – wiederhergestellt wird der
komplette Lernzustand, nicht nur das Netz.

Gesichert wird genau das, was das jeweilige Verfahren besitzt:

| | PPO | TD3 und SAC |
| --- | --- | --- |
| Modell | Policy, Value-Netz, Optimizer | Actor, Critics, Target-Netze, Optimizer (SAC zusätzlich die gelernte Temperatur) |
| Replay Buffer | entfällt – PPO ist on-policy | gehört dazu |
| Metadaten | Format-Version, Environment samt `continuous`, Algorithmus, vollständige Konfiguration | dito |

Die Checkpoints liegen in einem temporären Verzeichnis und verschwinden mit dem
Schließen der App: Ein Lernstand überlebt die Sitzung nicht.

## Tests und Grenzen

```bash
python -m unittest discover -s tests -v
```

Die Tests prüfen die Zoo-Profile je Verfahren, dass `model_kwargs()` nur
Argumente verwendet, die der jeweilige Algorithmus kennt, den kontinuierlichen
Action Space samt Triebwerkszuordnung und Clipping, die unveränderten
Environment-Argumente, die Unterscheidung von Landung, Absturz und Zeitlimit,
TD3s verzögerte Actor-Updates (`policy_delay`), SACs automatische
Entropie-Anpassung und Zielentropie `-dim(A)`, PPOs Clip-Schedule und die Regel
„Batch teilt Rollout", kurze Trainings- und Evaluationsläufe aller drei
Verfahren, die unveränderte Evaluation, die automatische Evaluation im
konfigurierten Intervall, Save/Load/Fortsetzen je Verfahren inklusive Replay
Buffer nur dort, wo es einen gibt, die Ablehnung inkompatibler Dateien und
falscher Verfahren, die Unabhängigkeit zweier Slots mit demselben Algorithmus,
den Renderer im eigenen Prozess, Export von Diagramm und Summary sowie das
Startlayout in beiden Verfahrenstabs für alle drei Algorithmen. Zwei
GUI-Vergleichsläufe prüfen zusätzlich den echten Weg über Worker-Threads und
Ereignis-Queue: einer ohne Animation, einer mit zugeschalteter Live-Animation,
der belegt, dass während des Laufs beide Verfahren angezeigt werden und die
Animation dabei auf einer Kopie der Policy arbeitet. Ein weiterer Test schaltet
die Animation mitten im Vergleich an und wieder aus. Dazu kommen Tests für
Standardwert, Grenzen und Fehlermeldung der Bildrate.

Grenzen: Kurze Läufe lösen LunarLander nicht; für belastbare Ergebnisse ist das
volle Budget nötig, und PPO braucht davon deutlich mehr als TD3 und SAC.
Paralleltraining beider Slots beansprucht entsprechend CPU. Das Projekt
unterstützt genau die kontinuierliche Variante und genau diese drei Verfahren;
Wiederholungsläufe mit Mittelwert und Streuband sind nicht implementiert, jeder
Slot trainiert einen Seed.
