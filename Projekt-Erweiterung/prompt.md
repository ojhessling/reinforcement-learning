# Humanoid – Erweiterung: moderne Off-Policy-Verfahren

Projektordner: `Oliver/Projekt-Erweiterung`

## Grundlage

Es gelten **unverändert**:

- die verbindlichen Regeln aus `../workbench.md`,
- der Projekt-Prompt `../Projekt/prompt.md` mit allem, was dort über
  Environment, Metriken, Budgetgrenzen, Einblendung, Oberfläche und Abnahme
  steht.

Beide Dateien werden für diese Erweiterung **nicht geändert**. Dieser Prompt
nennt ausschließlich, was *anders* ist — alles Ungenannte bleibt, wie es im
Projekt geregelt ist.

Projektname: `humanoid_extended`
Environment: `Humanoid-v5`, identisch konfiguriert wie im Projekt

## Ziel

Der Projektbericht endet mit der Feststellung, dass SAC bei 300.000 Schritten
mit Abstand gewinnt, die Figur aber nicht läuft. Diese Erweiterung stellt die
Anschlussfrage:

> **Holen neuere Off-Policy-Verfahren bei genau demselben Budget mehr heraus
> als SAC?**

Verglichen werden drei Verfahren am selben Environment, mit demselben
Schrittbudget wie im Projekt:

1. `SAC` — Referenz, exakt die Konfiguration aus dem Projekt
2. `CrossQ` — SAC ohne Target-Netze, dafür mit Batch Normalization
3. `TQC` — SAC mit verteilungsbasiertem Critic (Quantile) und abgeschnittenen
   Überschätzungen

Beide neuen Verfahren stammen aus `sb3-contrib` (Version 2.9.0, im Environment
vorhanden) und sind damit **keine Eigenbauten**, sondern Referenzimplementierungen.

## Abgrenzung

**`BRO` ist ausdrücklich nicht Teil dieser Erweiterung.** Das Verfahren existiert
weder in Stable-Baselines3 noch in `sb3-contrib`; die Originalimplementierung ist
in JAX geschrieben. Ein Nachbau wäre ein eigenes Projekt und würde hier nur eine
Näherung liefern, die man nicht „BRO" nennen dürfte. Wird der Nachbau später
angegangen, gehört er in einen eigenen Ordner mit eigenem Prompt.

## Unterschiede zum Projekt

### 1 Eigenständige Kopie der Anwendung

Die vier Module werden aus `../Projekt` **kopiert** und umbenannt:

| Projekt | Erweiterung |
| --- | --- |
| `humanoid_app.py` | `humanoid_extended_app.py` |
| `humanoid_logic.py` | `humanoid_extended_logic.py` |
| `humanoid_gui.py` | `humanoid_extended_gui.py` |
| `humanoid_render.py` | `humanoid_extended_render.py` |
| `tests/test_humanoid_*.py` | `tests/test_humanoid_extended_*.py` |

Kein Import über Ordnergrenzen hinweg. Grund: Die Kursabgabe liegt in
`../Projekt` und darf durch diese Erweiterung unter keinen Umständen berührt
werden — auch nicht durch eine geänderte gemeinsame Datei. Die Duplikation ist
der Preis dafür und wird bewusst bezahlt.

### 2 Verfahrenskatalog: zwei Ergänzungen

Weil `../workbench.md` nicht angefasst wird, führt **dieser** Prompt die beiden
neuen Verfahren, in derselben Tiefe wie Workbench 5.2 bis 5.4.

#### CrossQ

Kernidee: keine Target-Netze. Stattdessen laufen aktueller und nächster Zustand
gemeinsam durch den Critic (der „Cross"-Batch), und Batch Normalization
stabilisiert, was sonst das Target-Netz stabilisiert. Das spart einen
vollständigen Netzdurchlauf je Update und beschleunigt das Lernen pro Schritt.

| Parameter | Wert | Quelle |
| --- | --- | --- |
| `learning_rate` | `1e-3` | sb3-contrib-Default |
| `batch_size` | `256` | sb3-contrib-Default |
| `gamma` | `0.99` | sb3-contrib-Default |
| Actor-Netz | `256, 256` | sb3-contrib-Default |
| **Critic-Netz** | **`1024, 1024`** | sb3-contrib-Default — Teil des Verfahrens |
| `batch_norm` | `True` | Kern des Verfahrens |
| `batch_norm_momentum` | `0.01` | sb3-contrib-Default |
| `batch_norm_eps` | `0.001` | sb3-contrib-Default |
| `renorm_warmup_steps` | `100000` | sb3-contrib-Default |
| `policy_delay` | `3` | sb3-contrib-Default |
| `n_critics` | `2` | sb3-contrib-Default |
| `ent_coef`, `target_entropy` | `auto` | wie SAC |
| `learning_starts` | `10000` | **abweichend vom Default (100)**, damit alle drei Verfahren gleich starten |

**CrossQ besitzt weder `tau` noch `target_update_interval`** — es gibt keine
Target-Netze. Die Oberfläche darf diese Felder für CrossQ deshalb **nicht**
anzeigen (Workbench-Regel: kein Parameter, den das Verfahren nicht hat).

#### TQC

Kernidee: Der Critic schätzt nicht einen Wert, sondern eine Verteilung über
`n_quantiles` Quantile. Von jedem Critic werden die obersten Quantile
weggeworfen — das dämpft die Überschätzung gezielter als das Minimum zweier
Critics bei TD3 und SAC.

| Parameter | Wert | Quelle |
| --- | --- | --- |
| `learning_rate` | `3e-4` | sb3-contrib-Default, wie SAC |
| `batch_size` | `256` | sb3-contrib-Default |
| `gamma`, `tau` | `0.99`, `0.005` | wie SAC |
| Netz | `256, 256` | sb3-contrib-Default, wie SAC |
| `n_quantiles` | `25` | sb3-contrib-Default |
| `n_critics` | `2` | sb3-contrib-Default |
| `top_quantiles_to_drop_per_net` | `2` | sb3-contrib-Default |
| `ent_coef`, `target_entropy` | `auto` | wie SAC |
| `learning_starts` | `10000` | **abweichend vom Default (100)**, siehe oben |

Neue Eingabefelder nur für TQC: `n_quantiles` (Quantile `N_q`) und
`top_quantiles_to_drop_per_net` (verworfene Quantile `k`).

### 3 Architekturen werden **nicht** vereinheitlicht

Das Projekt hat TD3s Netz von `400,300` auf `256,256` vereinheitlicht, damit der
Vergleich nicht an der Netzgröße hängt. **Hier gilt das Gegenteil:** CrossQs
breiter Critic mit `1024,1024` und Batch Normalization *ist* das Verfahren —
wer ihn auf `256,256` stutzt, vergleicht nicht mehr CrossQ. Verglichen werden
also drei Verfahren mitsamt ihrer vorgesehenen Architektur, und der Bericht
benennt das ausdrücklich als bewussten Unterschied zur Vorgehensweise in 2.2.2
des Projektberichts.

Gleich gehalten werden dagegen alle Größen, die das **Budget** betreffen:
Schrittzahl, `learning_starts`, `batch_size`, `gamma`, `buffer_size`, Seed.

### 4 Ein Durchgang statt zwei

Gefahren wird **ein** Durchgang mit `seed = 0`, nicht zwei wie im Projekt.

Das ist eine bewusste Einschränkung aus Zeitgründen und hat eine Folge, die im
Ergebnis stehen muss: Nach den Erfahrungen aus dem Projekt — TD3 kippte
zwischen zwei Seeds von 172 auf 643, und die beiden kleineren Lernraten
tauschten die Plätze — **darf aus einem einzelnen Durchgang keine Rangfolge
zwischen ähnlich starken Verfahren behauptet werden**. Belastbar sind nur
Abstände, die deutlich größer sind als die im Projekt gemessene Seed-Streuung
von SAC (Faktor 1,5).

### 5 Kein eigener Bericht

Diese Erweiterung erzeugt **keinen** eigenen Bericht und ist **nicht** Teil der
Kursabgabe. Das Ergebnis wird als **Ausblick** in den Projektbericht
`../Projekt/KLR-339-2026-08-Hessling_Oliver.md` aufgenommen, in einem eigenen
Abschnitt am Ende von Teil 2.

**Wichtig für die Ablage:** Die Plots, die im Bericht erscheinen, müssen nach
Vorgabe A4 **direkt in `../Projekt/` liegen**. Sie werden also aus diesem Ordner
dorthin kopiert; der Code bleibt hier.

## Experiment

Ein Durchgang, drei Slots gleichzeitig, identisch zum Projekt außer beim
Verfahren:

| | Wert |
| --- | --- |
| `Anzahl Verfahren` | `3` — `V1 = SAC`, `V2 = CrossQ`, `V3 = TQC` |
| `Trainingsschritte N` | `300.000` |
| `Episoden E` | `0` |
| `Seed s` | `0` |
| `buffer_size` | `500.000` |
| `Glättung` | `50` |
| Animation | alle Slots `inaktiv` |

Die Schrittzahl entspricht exakt der des Verfahrensvergleichs im Projekt. Damit
sind die SAC-Zahlen aus 2.2.3 unmittelbar als vierter Vergleichswert lesbar —
und die Referenz-SAC-Läufe dieses Durchgangs zeigen zugleich, wie stark der
Unterschied zwischen zwei nominell gleichen SAC-Läufen ausfällt (siehe 2.3.4
des Projektberichts, „Der Seed macht Läufe vergleichbar, nicht identisch").

**Laufzeit, geschätzt:** SAC schafft allein rund 50 Schritte/s. CrossQ rechnet
mit `1024`-breiten Critics und Batch Normalization je Update deutlich mehr, TQC
mit 2 × 25 Quantilen ebenfalls; zu dritt parallel ist mit **5 bis 7 Stunden** zu
rechnen. Vor dem Start wird die tatsächliche Geschwindigkeit an den ersten
20.000 Schritten gemessen und die Schätzung im Ergebnis korrigiert.

### Dateien

| Datei | Inhalt |
| --- | --- |
| `erw-vergleich-seed0.png` | alle drei Verfahren gemeinsam |
| `erw-sac-seed0.png` | nur SAC |
| `erw-crossq-seed0.png` | nur CrossQ |
| `erw-tqc-seed0.png` | nur TQC |
| `erw-summary-seed0.txt` | Kennzahlen und vollständige Konfiguration |

Alle fünf werden nach `../Projekt/` kopiert, sobald sie im Bericht erscheinen.

## Tests

Zusätzlich zu Workbench 5.10 und 9.2 sowie den Tests des Projekts:

- `CrossQ` besitzt **keine** Target-Netze: Weder `tau` noch
  `target_update_interval` werden an den Konstruktor übergeben, und die
  Oberfläche zeigt beide Felder für CrossQ nicht an
- die Batch-Normalization-Parameter erreichen die Policy unverändert
- `TQC` erhält `n_quantiles` und `top_quantiles_to_drop_per_net`; die
  Oberfläche zeigt beide nur für TQC
- `k < N_q` wird validiert: Es dürfen nie alle Quantile verworfen werden
- die Netzgrößen der drei Verfahren werden **nicht** vereinheitlicht — ein Test
  hält fest, dass CrossQ mit `1024,1024` und SAC mit `256,256` läuft
- alle drei Verfahren akzeptieren dasselbe Budget, denselben Seed und
  denselben Buffer und liefern die im Projekt definierten Metriken

## Abhängigkeiten

Neu: `sb3-contrib` (bereits im Environment, Version 2.9.0). In
`requirements.txt` dieses Ordners aufnehmen, `../environment.yml` bleibt
unberührt.

## Abnahme

Zusätzlich zu Workbench 9.3 und der Abnahme des Projekts:

- die Kursabgabe in `../Projekt` ist unverändert — kein Modul, kein Bild, kein
  Text dort wurde durch diese Erweiterung angefasst, mit der einzigen Ausnahme
  des Ausblick-Abschnitts im Bericht und der fünf dorthin kopierten Dateien
- `CrossQ` und `TQC` erscheinen in der Oberfläche mit genau den Parametern, die
  sie besitzen, und ohne die, die sie nicht besitzen
- der Vergleich läuft mit identischem Budget, Seed und Buffer für alle drei
- das Ergebnis benennt ausdrücklich, dass ein einzelner Durchgang keine
  Rangfolge zwischen ähnlich starken Verfahren belegt
