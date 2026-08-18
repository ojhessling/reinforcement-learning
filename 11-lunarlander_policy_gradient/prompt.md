# LunarLander Policy Gradient – RL-Workbench

Projektordner: `Oliver/11-lunarlander_policy_gradient`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `lunarlander_pg`
Environment: `LunarLander-v3` in der kontinuierlichen Variante

Dieser Prompt enthält nur die projektspezifischen Anforderungen. Alles
Allgemeine – Projektstruktur, GUI-Aufbau mit Verfahrenswahl und
Vergleichstabs, Steuerungsbuttons, Responsivität, Vergleichslogik, Export,
Tests und Abnahme – steht ausschließlich in `../workbench.md` und wird hier
bewusst nicht wiederholt.

Dieses Projekt enthält **keine Abweichungen** von der Workbench. Insbesondere
gelten ohne Einschränkung:

- Genau zwei gleichrangige Verfahrensslots mit den Dropdowns `Verfahren 1` und
  `Verfahren 2` sowie zwei gleich aufgebauten Parametertabs.
- Die Animation hat genau einen Schalter, der jederzeit wirkt – auch mitten in
  einem Lauf –, und ein Feld für die Bildrate (siehe `Darstellung`).
- Längere Trainings- und Vergleichsläufe werden im sichtbaren, konfigurierbaren
  Schrittintervall automatisch headless deterministisch evaluiert; Graph und
  Summary werden live mitgeschrieben.
- Diagramm und Summary sind exportierbar.

Der Dateipräfix lautet `lunarlander_pg`, nicht `lunarlander`, weil
`Oliver/10-lunarlander` bereits Module und Testdateien mit diesem Namen
enthält und gleichnamige Testdateien ohne Paketkontext einen `pytest`-Lauf vom
Repository-Root brechen.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung für Gymnasiums kontinuierliches
`LunarLander-v3`. Verglichen werden drei Verfahren für kontinuierliche
Action-Spaces aus Stable-Baselines3:

1. `PPO` – On-Policy, geclipptes Surrogatziel, GAE, kein Replay Buffer
2. `TD3` – Off-Policy, deterministischer Actor, zwei Critics, verzögerte
   Policy-Updates, Target Policy Smoothing, explizites Action Noise
3. `SAC` – Off-Policy, stochastischer Actor, Entropieregularisierung mit
   automatisch angepasstem Temperaturparameter, zwei Critics

Jeder der beiden Verfahrensslots kann frei eines dieser drei Verfahren
aufnehmen, auch beide denselben Algorithmus. Damit sind zwei Fragestellungen
möglich: der Vergleich zweier Verfahren und der Vergleich zweier
Parametrisierungen desselben Verfahrens.

Anders als in `Oliver/09-acrobot` und `Oliver/10-lunarlander` werden **keine
eigenen Algorithmusvarianten** gebaut. Verwende die
Stable-Baselines3-Implementierungen von `PPO`, `TD3` und `SAC` unverändert. Die
eigene fachliche Arbeit liegt in Konfiguration, Runnern, Metriken, Vergleich,
GUI und Tests. Ergänze keine eigenen Netz-, Buffer- oder Loss-Bausteine.

Ordnername und Projektname nennen `policy gradient`, weil alle drei Verfahren
eine Policy direkt parametrisieren und über Gradienten verbessern. Im engeren
Sinn ist nur `PPO` ein Policy-Gradient-Verfahren; `TD3` und `SAC` sind
Off-Policy-Actor-Critic-Verfahren. Ordne das in der README kurz ein.

## Environment

Erzeuge das Environment zwingend mit:

```python
import gymnasium

env = gymnasium.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
```

Diese Schreibweise ist der registrierten Kennung `LunarLanderContinuous-v3`
gleichwertig und macht den Unterschied zum Vorgängerprojekt sichtbar. Eine
gemeinsame Factory erzeugt getrennte Instanzen für Training, deterministische
Evaluation und sichtbare Animation. Verwende für rein headless Evaluationen
kein Rendering, wenn dadurch dieselbe unveränderte Environment-Spezifikation
erhalten bleibt. Environment-Instanzen werden weder gleichzeitig noch
threadübergreifend geteilt.

Verändere Dynamik, Startzustandsverteilung, Reward oder Abbruchregeln nicht.
Belasse alle weiteren Konstruktorargumente auf ihren Standardwerten:
`gravity=-10.0`, `enable_wind=False`, `wind_power=15.0`,
`turbulence_power=1.5`. Die diskrete Variante gehört nicht zum Projekt.

Der kontinuierliche Action Space ist zwingend, weil `TD3` und `SAC`
ausschließlich mit kontinuierlichen Actions arbeiten. `PPO` beherrscht beides
und läuft hier ebenfalls kontinuierlich, damit alle drei Verfahren dieselbe
Aufgabe lösen und fair vergleichbar bleiben.

Actions (`Box(-1.0, 1.0, (2,), float32)`):

- `a₀` Haupttriebwerk: Werte `≤ 0` schalten es aus. Werte `> 0` regeln den
  Schub linear von 50 % bis 100 %.
- `a₁` Steuertriebwerke: Werte `< -0,5` zünden das linke, Werte `> 0,5` das
  rechte Triebwerk; dazwischen bleiben beide aus. Der Betrag regelt den Schub
  von 50 % bis 100 %.

Observation (8-dimensional, `float32`), identisch zum Vorgängerprojekt:

1. Position `x` relativ zur Landeplattform, Bereich `[-2,5; 2,5]`
2. Position `y` relativ zur Landeplattform, Bereich `[-2,5; 2,5]`
3. Geschwindigkeit `vₓ`, Bereich `[-10; 10]`
4. Geschwindigkeit `v_y`, Bereich `[-10; 10]`
5. Winkel `θ` des Landers, Bereich `[-2π; 2π]`
6. Winkelgeschwindigkeit `θ̇`, Bereich `[-10; 10]`
7. Bodenkontakt linkes Bein, `0` oder `1`
8. Bodenkontakt rechtes Bein, `0` oder `1`

Die Landeplattform liegt immer bei `(0, 0)`. Die Einheiten der Observation sind
normalisiert und nicht durchgängig SI-konform; die Winkelgeschwindigkeit ist in
Einheiten von `0,4 rad/s` angegeben und muss für eine Anzeige in `rad/s` mit
`2,5` multipliziert werden. Rechne Winkel für die Anzeige in Grad um.

Der Reward setzt sich pro Schritt zusammen aus:

- einer Differenz zweier Shaping-Terme, die Annäherung an die Plattform,
  geringere Geschwindigkeit, geringere Schräglage sowie Beinkontakt
  (`+10` je Bein) belohnen
- `-0,3` je Frame bei vollem Haupttriebwerksschub, anteilig zum tatsächlichen
  Schub zwischen 50 % und 100 %
- `-0,03` je Frame bei vollem Steuertriebwerksschub, ebenfalls anteilig

Der Treibstoffabzug ist damit anders als in der diskreten Variante stufenlos:
Sanfteres Gasgeben kostet weniger. Zusätzlich endet die Episode mit `-100` bei
Absturz oder Verlassen des sichtbaren Bereichs (`|x| ≥ 1`) und mit `+100`,
sobald der Lander zur Ruhe kommt. Nach 1000 Schritten wird die Episode
trunkiert. Dieses Shaping ist Teil des Original-Environments; eigenes Reward
Shaping ist nicht zulässig.

Höhere Werte sind besser, der Episoden-Return liegt realistisch etwa zwischen
`-400` und `+320`. Gymnasium nennt auch für die kontinuierliche Variante `200`
als `reward_threshold`.

Erfolgsdefinition für dieses Projekt:

- **Landung**: Episode endet mit `terminated` und ruhendem Lander (`+100`),
  also weder Absturz noch Zeitlimit.
- **Gelöst**: Episoden-Return `≥ 200`.

Beide Quoten werden getrennt ausgewiesen, weil eine sichere Landung mit hohem
Treibstoffverbrauch die Lösungsschwelle verfehlen kann.

Quelle: [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

## Gemeinsame Grundlagen

Behandle `terminated` und `truncated` entsprechend den Regeln der Workbench
konsistent über alle drei Verfahren. Beachte, dass `LunarLander` selbst immer
`truncated=False` liefert und das Zeitlimit ausschließlich vom
`TimeLimit`-Wrapper aus `gymnasium.make` gesetzt wird. Stable-Baselines3
bootstrappt bei Truncation über `infos["TimeLimit.truncated"]`; dokumentiere
dieses Verhalten und teste es.

Exploration entsteht in allen drei Verfahren aus der Policy selbst
beziehungsweise aus explizitem Action Noise. Es gibt in diesem Projekt daher
**keine** ε-greedy-Parameter wie `exploration_fraction` oder
`exploration_final_eps`.

In **beiden** Verfahrenstabs einstellbar:

- `total_timesteps`
- `learning_rate` mit wählbarem Verlauf: `konstant` oder `linear fallend`
  (Stable-Baselines3 akzeptiert dafür eine Callable-Schedule)
- `batch_size`
- `gamma`
- `seed`
- Hidden Layers für Actor und Critic getrennt, Aktivierungsfunktion, Optimizer
  sowie Optimizer-`eps` und `weight_decay`

Global außerhalb der Tabs, weil beide Läufe dieselben Stützstellen brauchen:

- Intervall der deterministischen Zwischenevaluation in Environment-Schritten
- Anzahl der Evaluationsepisoden je Zwischenevaluation

Technische Optionen wie `verbose`, `tensorboard_log`, `device`,
`policy`-Kennung und `_init_setup_model` gehören nicht in die UI.

## PPO

On-Policy-Verfahren. Es sammelt Rollouts fester Länge, schätzt Vorteile mit
Generalized Advantage Estimation und optimiert ein geclipptes Surrogatziel über
mehrere Epochen auf denselben Daten. Ein Replay Buffer existiert nicht; die
gesammelten Daten werden nach dem Update verworfen.

Zusätzlich in diesem Tab:

- `n_steps` – Rollout-Länge je Update
- `n_epochs` – Optimierungsdurchläufe je Rollout
- `gae_lambda` – GAE-Glättung
- `clip_range` – Clipping des Policy-Ratios
- `clip_range_vf` – optionales Clipping des Value-Ziels, leer bedeutet aus
- `normalize_advantage`
- `ent_coef` – Entropiebonus
- `vf_coef` – Gewicht des Value-Loss
- `max_grad_norm` – Gradient Clipping
- `target_kl` – optionaler früher Abbruch, leer bedeutet aus
- `use_sde` und `sde_sample_freq` – generalized State-Dependent Exploration
- `log_std_init` – initiale Streuung der Gauß-Policy

Validierung: `batch_size` muss `n_steps` teilen, sonst verwirft
Stable-Baselines3 Daten und warnt zur Laufzeit. Weise darauf vor dem Start
verständlich hin.

Quellen: [Schulman et al., Proximal Policy Optimization](https://arxiv.org/abs/1707.06347),
[Schulman et al., Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

## TD3

Off-Policy-Verfahren mit deterministischem Actor. Es lernt zwei Critics und
verwendet für das Ziel deren Minimum, um Überschätzung zu dämpfen. Die Policy
und die Target-Netze werden nur alle `policy_delay` Updates aktualisiert. Auf
die Target-Action wird geclipptes Rauschen addiert (Target Policy Smoothing).
Weil der Actor deterministisch ist, braucht TD3 zwingend explizites Action
Noise für die Exploration.

Zusätzlich in diesem Tab:

- `buffer_size`
- `learning_starts`
- `tau` – Soft-Update der Target-Netze
- `train_freq` und `gradient_steps`
- `policy_delay`
- `target_policy_noise` und `target_noise_clip`
- Action-Noise-Typ `keins`, `normal` oder `Ornstein-Uhlenbeck` sowie dessen `σ`

Quelle: [Fujimoto et al., Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)

## SAC

Off-Policy-Verfahren mit stochastischem Actor. Es maximiert nicht nur den
erwarteten Return, sondern zusätzlich die Entropie der Policy, gewichtet mit
einer Temperatur `α`. Bei `ent_coef = auto` wird `α` selbst gelernt, sodass die
mittlere Entropie der Policy einer Zielentropie folgt; Stable-Baselines3
verwendet als Standard `target_entropy = -dim(A)`, hier also `-2`. Auch SAC
nutzt zwei Critics und deren Minimum.

Zusätzlich in diesem Tab:

- `buffer_size`
- `learning_starts`
- `tau`
- `train_freq` und `gradient_steps`
- `ent_coef` als `auto` oder fester Zahlenwert, bei `auto` zusätzlich der
  initiale Wert von `α`
- `target_entropy` als `auto` oder fester Zahlenwert
- `target_update_interval`
- `use_sde` und `sde_sample_freq`
- optionales Action Noise mit `σ`

Quellen: [Haarnoja et al., Soft Actor-Critic](https://arxiv.org/abs/1801.01290),
[Haarnoja et al., Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905),
[Raffin et al., Smooth Exploration for Robotic RL (gSDE)](https://arxiv.org/abs/2005.05719)

## Standardprofile

Jedes Verfahren hat sein eigenes Standardprofil. Ein Wechsel des Algorithmus im
Dropdown lädt die Werte des neu gewählten Verfahrens in den zugehörigen Tab.
Grundlage sind die getunten `LunarLanderContinuous-v3`-Profile des RL
Baselines3 Zoo (Stand geprüft am 18.08.2026); nicht im Profil enthaltene Werte
stammen aus den Voreinstellungen von Stable-Baselines3 in der Version aus
`../environment.yml`.

`PPO`:

- Trainingsschritte `N = 100.000`
- `n_steps = 1024`, `batch_size = 64`
- `gae_lambda = 0,98`, `gamma = 0,999`
- `n_epochs = 4`, `ent_coef = 0,01`
- übrige Werte aus den SB3-Defaults, insbesondere `learning_rate = 3e-4`,
  `clip_range = 0,2`, `vf_coef = 0,5`, `max_grad_norm = 0,5`, Aktivierung
  `Tanh`, Adam mit `eps = 1e-5`

`TD3`:

- Trainingsschritte `N = 100.000`
- `learning_rate = 1e-3`, `gamma = 0,98`
- `buffer_size = 200.000`, `learning_starts = 10.000`
- `train_freq = 1`, `gradient_steps = 1`
- Action Noise `normal` mit `σ = 0,1`
- übrige Werte aus den SB3-Defaults, insbesondere `batch_size = 256`,
  `tau = 0,005`, `policy_delay = 2`, `target_policy_noise = 0,2`,
  `target_noise_clip = 0,5`, Aktivierung `ReLU`

`SAC`:

- Trainingsschritte `N = 100.000`
- `learning_rate = 7,3e-4`, linear fallend
- `buffer_size = 1.000.000`, `learning_starts = 10.000`
- `batch_size = 256`, `gamma = 0,99`, `tau = 0,01`
- `train_freq = 1`, `gradient_steps = 1`, `ent_coef = auto`
- übrige Werte aus den SB3-Defaults, insbesondere `target_update_interval = 1`,
  `target_entropy = auto`, Aktivierung `ReLU`

Zwei Werte sind bewusst **nicht** aus den Profilen übernommen, sondern für alle
drei Verfahren vereinheitlicht:

- **Trainingsbudget `N = 100.000` Schritte.** Die Profile nennen `1e6` (PPO),
  `3e5` (TD3) und `5e5` (SAC). Diese Budgets sind interaktiv nicht abwartbar,
  und unterschiedliche Budgets machen einen Vergleich zweier Slots ohne Zutun
  unfair.
- **Hidden Layers `64,64` für Actor und Critic.** Die Profile nennen für TD3
  und SAC `400,300`. Solche Netze kosten je Schritt deutlich mehr Rechenzeit
  und lohnen sich erst bei entsprechend großem Budget.

Beides bleibt in der UI frei änderbar. Alle übrigen getunten Hyperparameter
werden unverändert übernommen. Nenne die Profilwerte in der README und weise
darauf hin, dass belastbare Ergebnisse mehr Schritte und größere Netze
brauchen.

Zwei weitere Punkte sind ausdrücklich zu dokumentieren:

- Das PPO-Profil des Zoo verwendet `n_envs = 16`. Dieses Projekt trainiert wie
  alle Vorgängerprojekte mit **einer** Environment-Instanz, damit Animation,
  Episodenbuchhaltung und schrittgenaue Zwischenevaluation eindeutig bleiben.
  Der effektive Rollout je Update ist dadurch `1024` statt `16.384` Schritte.
  Erkläre in der README, dass PPO deshalb je Environment-Schritt langsamer
  vorankommt als im Zoo-Profil.
- Ändert der Benutzer eines der beiden Budgets, weist die GUI gemäß Workbench
  vor dem Vergleichsstart auf die Abweichung hin; verboten ist sie nicht.

Quellen: [RL Baselines3 Zoo, PPO](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml),
[TD3](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/td3.yml),
[SAC](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml)

## Evaluation

Deterministisch evaluiert wird mit `predict(..., deterministic=True)`: PPO und
SAC verwenden dann den Mittelwert ihrer Policy-Verteilung, TD3 ist ohnehin
deterministisch und erhält kein Action Noise. Zeige im Evaluationsergebnis
mindestens:

- mittleren Return und Standardabweichung
- mittlere Episodenlänge
- Landequote (Anteil sicher gelandeter Episoden)
- Gelöst-Quote (Anteil Episoden mit Return `≥ 200`)
- Absturzquote
- letzte und beste deterministische Evaluation

## Vergleich

Verglichen werden immer `Verfahren 1` und `Verfahren 2` gemäß Workbench. Der
Vergleichsgraph zeigt den explorativen Episoden-Return gegen die
Episodennummer und enthält eine Referenzlinie bei `+200`.

Die Vergleichs-Summary enthält je Slot diese Zeilen:

- Algorithmus
- Episoden
- Environment-Schritte, ausgeführt und angefordert
- durchschnittlicher Return
- Landequote
- Gelöst-Quote
- beste deterministische Evaluation

Darunter folgt der von der Workbench geforderte Abschnitt mit den Parametern,
in denen sich die beiden Konfigurationen unterscheiden.

Erkläre in Bedienungsanleitung und README, dass ein Vergleich von PPO gegen
TD3 oder SAC bei gleichem Schrittbudget systematisch zugunsten der
Off-Policy-Verfahren ausfällt: Diese lernen aus jedem gespeicherten Übergang
mehrfach, PPO verwirft seine Daten nach jedem Update. Das ist kein Fehler der
Messung, sondern eine Eigenschaft der Verfahrensklassen.

## Darstellung

Zeige ausschließlich den offiziellen, von `env.render()` gelieferten
Gymnasium-RGB-Frame (600 × 400 Pixel). Erstelle keine eigene
LunarLander-Grafik und öffne kein separates Pygame-Fenster. Beachte die
macOS-Prozesstrennung aus der Workbench.

Standardwert der Bildrate ist die environment-eigene Rate von 50 FPS
(`env.metadata["render_fps"]`), also rund 20 ms je Frame; einstellbar ist sie
gemäß Workbench über das Feld `Bildrate (FPS)`. Eingeschaltet läuft die
Animation auch während eines Laufs: beim Einzeltraining für dessen Slot, beim
Vergleich für **beide Verfahren gleichzeitig**.

Zeige neben der Animation:

- Position `x` und `y`
- Geschwindigkeit `vₓ` und `v_y`
- Winkel `θ` in Grad und Winkelgeschwindigkeit `θ̇` in `rad/s`
  (Observationswert × 2,5)
- Bodenkontakt beider Beine als verständliche Ja/Nein-Anzeige
- die gewählte Action doppelt: die beiden Rohwerte `a₀` und `a₁` sowie ihre
  Bedeutung, also Haupttriebwerk `aus` oder Schub in Prozent und
  Steuertriebwerk `aus`, `links` oder `rechts` mit Schub in Prozent
- aktuellen Episodenschritt
- bisher kumulierten Return
- welchem Verfahrensslot die laufende Episode gehört

Projektspezifische Metriken sind Episoden-Return, Episodenlänge, Landequote,
Gelöst-Quote und Absturzquote. Achsen, Summary und Hilfetexte erklären
eindeutig, dass höhere Werte besser sind und `200` als gelöst gilt.

## Speichern und Laden

Es gibt **keine** Buttons zum manuellen Speichern und Laden. Der Lernzustand
wird ausschließlich über den automatischen Checkpoint des besten
Evaluationsergebnisses gesichert – je Slot getrennt – und mit `Bestes Modell
wiederherstellen` zurückgeholt.

Gesichert wird genau der Zustand, den das Verfahren besitzt: `PPO` Policy,
Value-Netz und Optimizer, `TD3` und `SAC` zusätzlich den Replay Buffer, `SAC`
außerdem den gelernten Temperaturparameter. Verwende dafür `model.save()`,
`Algorithmus.load()` sowie `save_replay_buffer()`/`load_replay_buffer()` von
Stable-Baselines3; eigene Replay-Buffer-Wrapper gibt es nicht, die SB3-Helfer
greifen deshalb unverändert. Zum Checkpoint gehören außerdem Algorithmus,
vollständige Slot-Konfiguration, Environment-Kennung samt `continuous=True` und
eine Format-Version. Ein Stand, der nicht zum Verfahren des Slots passt oder
dem ein Bestandteil fehlt, wird verständlich abgelehnt, ohne den aktiven
Zustand zu verändern.

## Tests

Zusätzlich zu den Testanforderungen der Workbench:

- Interpretation des kontinuierlichen Action Space: Grenzen von `a₀` und `a₁`,
  Zuordnung zu Haupt- und Steuertriebwerk, korrekte Prozentanzeige
- `PPO`: Wirkung von `clip_range`, `gae_lambda` und `n_epochs`; kein Replay
  Buffer vorhanden; Fortsetzen des Trainings ohne Rücksetzen des Schrittzählers
- `TD3`: verzögerte Policy-Updates gemäß `policy_delay`, Clipping des
  Target-Rauschens, Action Noise wirkt im Training und nicht in der Evaluation
- `SAC`: automatische Entropieanpassung verändert `α`, Zielentropie entspricht
  `-dim(A) = -2` bei `target_entropy = auto`
- Save-/Load-Roundtrip je Verfahren, mit Replay Buffer für `TD3` und `SAC` und
  ohne für `PPO`
- Ablehnung einer Modelldatei, deren Algorithmus nicht zum aktiven Slot passt

## Abhängigkeiten

Dieses Projekt führt **keine** neue Abhängigkeit ein. `swig`, `box2d`,
Gymnasium, PyTorch und Stable-Baselines3 stehen bereits in
`../environment.yml`; `PPO`, `TD3` und `SAC` sind Bestandteil von
Stable-Baselines3. Übernimm die `requirements.txt` inhaltlich aus
`Oliver/10-lunarlander` und erweitere sie nicht. Die README nennt den
Box2D-Installationsschritt trotzdem ausdrücklich, da das Projekt ohne Box2D
nicht startet.

## Abnahme

Zusätzlich zu allen Abnahmekriterien aus `../workbench.md` gilt: Fertig, wenn
`PPO`, `TD3` und `SAC` korrekt konfiguriert trainieren, deterministisch
evaluiert und fair verglichen werden; beide Verfahrensslots frei und
unabhängig belegbar sind, auch zweimal mit demselben Algorithmus; ein
Dropdown-Wechsel die passenden Profilwerte lädt und nur diesen Slot
zurücksetzt; algorithmusspezifische Parameter nur im jeweiligen Tab erscheinen;
keine ε-greedy-Parameter existieren; der kontinuierliche Action Space in
Anzeige und Tests korrekt interpretiert wird; Lande- und Gelöst-Quote getrennt
ausgewiesen werden; gespeicherte Zustände je Verfahren vollständig
wiederherstellbar sind; die Summary die Unterschiede zwischen beiden
Konfigurationen ausweist; alle Verfahren mit demselben Budget und derselben
Netzgröße starten; und die offizielle Gymnasium-Animation eingebettet ist, sich
jederzeit ein- und ausschalten sowie in der Bildrate einstellen lässt und im
Vergleich beide Verfahren gleichzeitig zeigt.
