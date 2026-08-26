# Humanoid – Erweiterung: SAC gegen CrossQ und TQC

Seitenprojekt zur Kursabgabe in `../Projekt`. Dieselbe Workbench, dasselbe
Environment, dasselbe Budget — aber drei Verfahren, die sich **allein im
Critic** unterscheiden.

> Die Kursabgabe in `../Projekt` bleibt davon unberührt. Diese Erweiterung hat
> eigene Module (`humanoid_extended_*.py`), eigene Tests und ein eigenes
> `requirements.txt`. Geteilt wird nichts.

## Die Frage

Der Projektbericht endet damit, dass SAC bei 300.000 Schritten deutlich gewinnt
und die Figur trotzdem nicht läuft. Hier steht die Anschlussfrage: **Holen
neuere Off-Policy-Verfahren bei genau demselben Budget mehr heraus?**

## Die drei Verfahren

| | Critic | Target-Netze | Besonderheit |
| --- | --- | --- | --- |
| `SAC` | Minimum zweier Critics | ja | Referenz, exakt die Konfiguration des Projekts |
| `CrossQ` | zwei Critics, `1024,1024` | **nein** | Batch Normalization ersetzt das Target-Netz |
| `TQC` | 2 × 25 Quantile | ja | verwirft je Critic die `k` obersten Quantile |

**CrossQ besitzt weder `Soft-Update τ` noch `Target-Intervall C`.** Beide Felder
erscheinen für dieses Verfahren nicht in der Oberfläche und werden dem
Konstruktor nicht übergeben — er kennt sie nicht. Ein Test hält das fest, ein
zweiter prüft am erzeugten Modell, dass tatsächlich kein `critic_target`
existiert.

**Die Netzgrößen werden bewusst nicht vereinheitlicht.** Im Projekt wurde TD3s
Netz auf `256,256` gestutzt, damit der Vergleich nicht an der Größe hängt. Hier
gilt das Gegenteil: CrossQs breiter Critic *ist* das Verfahren. Gleich gehalten
wird stattdessen alles, was das **Budget** betrifft — Schritte, Lernstart,
Batch, Buffer, Diskont und Seed.

## Parameter

Alle Werte stammen aus den Voreinstellungen von `sb3-contrib` 2.9.0, mit einer
Ausnahme: `learning_starts` steht bei allen drei Verfahren auf `10.000` statt
auf `100`, damit sie gleich starten. Ein Zoo-Profil für `Humanoid` gibt es für
CrossQ und TQC nicht; deshalb führt dieses Projekt für sie auch **kein**
Profilbudget — eine erfundene Zahl wäre eine Behauptung.

| | CrossQ | TQC |
| --- | --- | --- |
| Lernrate | `1e-3` | `3e-4` |
| Actor / Critic | `256,256` / `1024,1024` | `256,256` / `256,256` |
| Eigene Felder | `BN-Mom.` 0,01 · `Warmlauf` 100.000 · `Delay d` 3 | `Quantile` 25 · `Verworfen` 2 |

## Start

```bash
conda activate rl-26-08
python humanoid_extended_app.py
python -m pytest tests -q
```

Die Startbelegung ist der Vergleich: `V1 = SAC`, `V2 = CrossQ`, `V3 = TQC`.

## Messlauf

Ein Durchgang, `seed = 0`, `N = 300.000`, `Episoden E = 0`, Glättung 50 — also
exakt das Budget des Verfahrensvergleichs aus dem Projekt, damit die dortigen
SAC-Zahlen unmittelbar danebenstehen können.

**Ein Durchgang genügt für keine Rangfolge zwischen ähnlich starken Verfahren.**
Im Projekt kippte TD3 zwischen zwei Seeds von 172 auf 643, und zwei Lernraten
tauschten die Plätze. Belastbar sind hier nur Abstände, die deutlich größer sind
als die dort gemessene Seed-Streuung von SAC (Faktor 1,5).

Ergebnisse gehen als **Ausblick** in den Projektbericht; die Plots werden dafür
nach `../Projekt/` kopiert, weil die Abgabe alle Bilder direkt im Abgabeordner
verlangt.

## Dateien

| Datei | Inhalt |
| --- | --- |
| `prompt.md` | was diese Erweiterung anders macht als das Projekt |
| `humanoid_extended_logic.py` | Environment, Konfiguration, die fünf Verfahren, Metriken |
| `humanoid_extended_gui.py` | Oberfläche, Diagramme, Animation |
| `humanoid_extended_render.py` | isolierter MuJoCo-Renderprozess |
| `humanoid_extended_app.py` | Einstiegspunkt |
| `tests/` | Testsuite, darunter acht Tests eigens für CrossQ und TQC |

Alles Weitere — Environment, Metriken, Budgetregeln, Einblendung, Bedienung —
steht unverändert in `../Projekt/README.md` und `../workbench.md`.
