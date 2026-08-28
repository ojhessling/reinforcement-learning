# Humanoid – RL-Workbench (Abschlussprojekt)

Tkinter-Lernanwendung, die eine dreidimensionale humanoide Figur von 42 kg
aufrecht halten und vorwärts laufen lassen soll. Drei Verfahren im direkten
Vergleich: **PPO**, **TD3** und **SAC**.

Abschlussprojekt des Kurses D21195UYS (AlfaTraining, August 2026). Die
zugeteilte Aufgabe lautet `Humanoid-v5` mit genau diesen drei Verfahren.

> **Erwartungshaltung vorweg:** Die Messläufe dieser Arbeit fahren rund
> 300.000 Schritte – weit **unter** den Zoo-Profilwerten (PPO 10 Mio., TD3/SAC
> je 2 Mio.). Die Figur wird damit **nicht laufen**. Erwartbar ist, dass sie sich zunehmend länger
> aufrecht hält und schwankend vorwärts kommt. Warum das so gewählt ist, steht
> unter [Budget und Laufzeit](#budget-und-laufzeit).

## Installation und Start

```bash
conda env create -f ../environment.yml     # oder: pip install -r requirements.txt
conda activate rl-26-08
python humanoid_app.py
```

Tests:

```bash
python -m pytest tests -q
```

## Environment

`Humanoid-v5` wird **unverändert** verwendet – kein Reward Shaping, keine
geänderten Abbruchregeln, keine veränderten Observationsschalter. Übergeben
werden nur `render_mode`, `width` und `height`.

| | Wert |
| --- | --- |
| Observation | `Box(-inf, inf, (348,), float64)` |
| Action | `Box(-0.4, 0.4, (17,), float32)` |
| Episodenlänge | höchstens 1000 Schritte |
| Schrittdauer | 0,015 s (`frame_skip = 5`) |
| Bildrate | 67 FPS (environment-eigen) – eine volle Episode dauert 15 s Echtzeit |
| `reward_threshold` | **keiner** |

### Actions

Der Wertebereich ist **±0,4**, nicht ±1 wie bei Hopper, HalfCheetah und
Walker2d. Jede Anzeige liest die Grenze aus dem Space.

Das Moment in Newtonmetern ist `aᵢ · gearᵢ`; die Übersetzungen unterscheiden
sich stark:

| Gruppe | Aktuatoren | `gear` | max. Moment |
| --- | --- | --- | --- |
| Rumpf | `abdomen_y/z/x` | 100 | 40 N·m |
| Hüfte seitlich/drehen | `hip_x`, `hip_z` | 100 | 40 N·m |
| Hüfte vor/zurück | `hip_y` | 300 | **120 N·m** |
| Knie | `knee` | 200 | 80 N·m |
| Arme | Schultern, Ellbogen | 25 | 10 N·m |

Die Beine sind bis zu zwölfmal kräftiger übersetzt als die Arme.

### Observation

| Bereich | Größe | Inhalt |
| --- | --- | --- |
| `0` | 1 | Rumpfhöhe `z` |
| `1:5` | 4 | Rumpforientierung (Quaternion) |
| `5:22` | 17 | Gelenkwinkel |
| `22:45` | 23 | Geschwindigkeiten (3 linear, 3 Winkel, 17 Gelenke) |
| `45:175` | 130 | `cinert` – Trägheitstensoren, 13 Körper × 10 |
| `175:253` | 78 | `cvel` – Körpergeschwindigkeiten |
| `253:270` | 17 | `qfrc_actuator` – Aktuatorkräfte |
| `270:348` | 78 | `cfrc_ext` – externe Kontaktkräfte |

x- und y-Position sind ausgeschlossen und stehen in `info`. Die
Geschwindigkeiten sind **nicht** geclippt (anders als bei Walker2d).

> **Stolperstelle:** Die Reihenfolge der 17 Gelenkwinkel in der Observation ist
> **nicht** die der Aktuatoren – `abdomen_y` und `abdomen_z` sind vertauscht.
> Die Einblendung mappt deshalb über `ACTUATOR_TO_JOINT`; ein Test prüft beide
> Reihenfolgen gegen das MuJoCo-Modell.

### Reward

```text
reward = healthy_reward + forward_reward − ctrl_cost   − contact_cost
       = 5,0            + 1,25 · vₓ      − 0,1 · Σaᵢ²  − 5e−7 · Σcfrc_ext²
```

Der **Überlebensbonus von 5,0 je Schritt dominiert alles andere**: Eine Figur,
die 1000 Schritte lang nur steht, sammelt 5000 Return. Das ist der Schlüssel
zum Verständnis der Skala – und der Grund für die Zielmarke unten.

Die Steuerkosten erreichen höchstens `0,1 · 17 · 0,4² = 0,272` je Schritt, die
Kontaktkosten sind nach oben auf 10 gedeckelt. Alle vier Anteile werden aus
`info` übernommen und nicht nachgerechnet.

### Episodenende

- **Sturz** (`terminated`): Die Rumpfhöhe verlässt den Bereich (1,0; 2,0) m.
  Eine Winkelbedingung gibt es **nicht** – die Figur darf beliebig verdreht
  sein, solange die Höhe stimmt.
- **Durchgehalten** (`truncated`): 1000 Schritte erreicht.

Weder Terminalbonus noch Terminalstrafe. Ein Sturz kostet nur die Rewards der
Schritte, die nicht mehr stattfinden – bei 5,0 je Schritt aber ein sehr
scharfes Signal.

### Zielmarke 5000 – keine Gelöst-Schwelle

`Humanoid-v5` führt **keinen** offiziellen `reward_threshold`. Die Marke von
`5000` ist **projektintern** gesetzt und entspricht `1000 Schritte ×
healthy_reward 5,0`: genau dem Return einer Figur, die eine volle Episode nicht
stürzt, ohne sich vorwärts zu bewegen. Sie trennt damit „bleibt aufrecht" von
„stürzt" und heißt überall **Zielquote**, nie Gelöst-Quote.

Die Referenzlinie im Diagramm ist weiß und gestrichelt. Die Y-Achse folgt den
**Daten**, nicht der Marke: Läge die Marke weit über allem Erreichten, drückte
sie sonst alle Kurven in einen Bruchteil der Bildhöhe.

## Verfahren

| | PPO | TD3 | SAC |
| --- | --- | --- | --- |
| Klasse | on-policy | off-policy | off-policy |
| Replay Buffer | nein | ja | ja |
| Policy | stochastisch | deterministisch | stochastisch |
| Exploration | Streuung der Gauß-Policy | Action Noise | Entropie |

**PPO** sammelt einen Rollout fester Länge, schätzt Vorteile über GAE und
optimiert ein geclipptes Ziel über mehrere Epochen. Die Daten werden danach
verworfen.

**TD3** lernt zwei Critics und nutzt das Minimum beider (gegen
Überschätzung), verzögert die Actor-Updates und glättet das Target mit
Rauschen. Der Actor ist deterministisch und exploriert **nur** über Action
Noise – deshalb lehnt die Anwendung `Action Noise = keins` für TD3 ab.

**SAC** maximiert Return **und** Entropie. Der Temperaturparameter α wird bei
`auto` selbst gelernt; die Zielentropie ist `−dim(A)`, hier also **−17**.

### Standardprofile

`Humanoid-v4`-Profile des RL Baselines3 Zoo (geprüft am 24.08.2026); für `v5`
sind keine hinterlegt.

| Parameter | PPO | TD3 | SAC |
| --- | --- | --- | --- |
| Lernrate | 3,57e−5 | 1e−3 | 3e−4 |
| γ | **0,95** | 0,99 | 0,99 |
| Batch | 256 | 256 | 256 |
| Netz | 256,256 | 256,256 ¹ | 256,256 |
| Normalisierung | **ja** | nein | nein |
| Profilbudget | 10 Mio. | 2 Mio. | 2 Mio. |

¹ Das TD3-Profil nennt `400,300`. Vereinheitlicht auf `256,256`, damit der
Vergleich nicht schon an der Netzgröße hängt.

`gamma = 0.95` bei PPO ist auffällig niedrig und `normalize: true`
verpflichtend – beides stammt aus dem Profil und wurde nicht „korrigiert".

### Was die Profile laut Referenz erreichen

Benchmark des Zoo bei **2 Mio.** Schritten:

| Verfahren | Return |
| --- | --- |
| SAC | 6232,3 ± 279,9 |
| TD3 | 5566,7 ± 14,5 |
| PPO | kein Humanoid-Eintrag |

Beide Werte liegen über der Zielmarke 5000. Mit rund 300.000 Schritten sind
sie **nicht** erreichbar; sie dienen der Einordnung.

## Budget und Laufzeit

Die Anwendung führt **zwei** Budgetgrenzen je Slot. Der Lauf endet an der
zuerst erreichten; die Summary weist unter *Ende durch* aus, welche es war.

| Feld | Standard | Bedeutung |
| --- | --- | --- |
| `Trainingsschritte N` | 100.000 | Environment-Schritte |
| `Episoden E` | 1000 | abgeschlossene Episoden, `0` = unbegrenzt |

Die beiden Standardwerte liegen bewusst in derselben Größenordnung: 1000
Episoden entsprechen bei SAC gemessen rund 80.000 Schritten. Stünde bei `N` das
Budget der Messläufe (300.000), griffe immer die Episodengrenze und die
Schrittzahl wäre reine Dekoration.

**Für die Messläufe** wird `E = 0` gesetzt und das Schrittbudget vorgegeben –
nur so bekommen alle Verfahren exakt dieselbe Datenmenge. Der
Verfahrensvergleich (Bericht 2.2) lief mit `N = 300.000`, die Parameterstudie
(2.3) ebenfalls – ihr erster Durchgang wurde bei rund 300.400 Schritten von
Hand gestoppt. Dazu kommt eine Gegenprobe mit PPO über 1,5 Mio Schritte
(Bericht 2.2.5). Innerhalb eines Vergleichs ist das Budget immer gleich.

Warum beides: Eine Humanoid-Episode endet beim Sturz. Untrainiert fällt die
Figur nach im Mittel **24,5 Schritten**; mit dem Lernfortschritt werden
Episoden länger. Eine Episodenzahl entspricht deshalb keiner festen
Schrittzahl. Gemessen an SAC:

| nach … Schritten | Episoden | Ø Länge | Ø Return |
| --- | --- | --- | --- |
| 60.000 | 802 | 93 | 461 |

Hochgerechnet sind 1000 Episoden nach rund **80.000 Schritten** erreicht – 16 %
des Schrittbudgets. Für einen **fairen Vergleich** ist das Schrittbudget die
richtige Grenze: Das Verfahren, das besser lernt, hat längere Episoden und
bekäme bei gleicher Episodenzahl mehr Trainingsdaten.

**Laufzeit bei 500.000 Schritten** auf der Referenzmaschine (6 Kerne, davon 2
Performance-Kerne):

| Verfahren | Schritte/s | Laufzeit |
| --- | --- | --- |
| PPO | 481 | ~17 min |
| TD3 | ~50 | ~2,8 h |
| SAC | ~50 | ~2,8 h |

### Threads

Das Feld `Threads (PyTorch)` (Standard `4`) steht neben `Anzahl Verfahren` –
und zwar mit Absicht. Es bestimmt, über wie viele Kerne PyTorch **eine
einzelne** Matrixmultiplikation verteilt, nicht wie viele Verfahren gleichzeitig
laufen. Weil ein Vergleich seine Slots als Threads **eines** Prozesses ausführt,
teilen sich alle denselben Pool: Die Einstellung gilt für die ganze Anwendung,
nicht je Lauf.

| Threads | SAC Schritte/s |
| --- | --- |
| Standard (6) | 16 |
| 2 | 46 |
| **4** | **49,5** |

Faktor 3. Ursache: Von den 6 Kernen sind nur 2 Performance-Kerne. Nimmt PyTorch
alle sechs, warten die schnellen Threads an jedem Synchronisationspunkt auf die
langsamen. Der Wert wird beim Start eines Laufs übernommen; eine Änderung
während eines laufenden Trainings wirkt nicht mehr auf dieses.

### Speicher

Der Replay Buffer ist auf `500.000` statt `1e6` verkleinert und puffert
Beobachtungen als `float32`:

| `buffer_size` | dtype | nur Observations |
| --- | --- | --- |
| `1e6` | `float64` | 5,19 GB |
| `500.000` | `float32` | **1,30 GB** |

Die Verkleinerung ist für die Messläufe verhaltensneutral: Bei höchstens
300.000 Schritten wird nie ein Übergang verdrängt. Erst jenseits von 500.000
Schritten – etwa in einem langen Zusatzlauf – vergisst der Buffer die ältesten
Übergänge; dort ist es eine echte Abweichung vom Profil und gehört benannt. `float32` ist folgenlos, weil SB3 beim Sampeln ohnehin
dorthin wandelt. `optimize_memory_usage=True` wäre unzulässig: Es verträgt sich
nicht mit `handle_timeout_termination`, und Humanoid trunkiert.

## Bedienung

Links der **Konfigurator** (Verfahrenswahl, Parameter, globale Einstellungen),
rechts die **Animation**, unten **Diagramme** und **Summary**.

Vier Schaltflächen, mehr braucht der Ablauf nicht:

1. `Training starten / fortsetzen` trainiert den aktiven Slot.
2. `Vergleich starten / fortsetzen` trainiert alle aktiven Slots gleichzeitig.
3. `Stoppen` bricht den laufenden Vorgang ab; bereits gelaufene Episoden
   bleiben erhalten.
4. `Zurücksetzen` verwirft den Lernzustand des aktiven Slots.

**Ein zweiter Druck setzt nichts zurück**, sondern hängt erneut das volle
Budget an – Kurve und Summary wachsen weiter. Der Fortschrittsbalken zählt in
Episoden; steht `Episoden E` auf `0`, zählt er Schritte.

Liegt das Schrittbudget über der Startbelegung von 100.000, fragt die
Anwendung vor dem Start nach und nennt die erwartete Dauer – gerechnet mit den
gemessenen Geschwindigkeiten des langsamsten Slots. Ein Fehlgriff kostet hier
keine Sekunden, sondern Stunden.

Oben rechts öffnet `Bedienungsanleitung` ein eigenes Fenster mit der
vollständigen Erklärung von Ablauf, Budget, Environment, Verfahren, Diagrammen,
Summary und Animation.

Neben jedem Verfahren im Konfigurator steht unter der Überschrift **Animation**
ein Feld:

| Wahl | Wirkung |
| --- | --- |
| `akt. Ep.` | der laufende Lernstand |
| `beste Ep.` | die bisher beste Episode, **exakt nachgespielt** |
| `beste Pol.` | deren Lernstand, deterministisch und mit festem Seed |
| `inaktiv` | diese Anzeige entfällt; die übrigen bekommen ihren Platz |

`beste Ep.` und `beste Pol.` beantworten verschiedene Fragen: die eine „was ist
damals passiert", die andere „wie gut ist dieser Stand ohne das Glück
explorativer Züge". Ihr Abstand misst, wie viel des Spitzenwerts Zufall war —
bei SAC gemessen 433,5 gegen 395,8.

Einen globalen Schalter „Animation zeigen" gibt es nicht – alle Felder auf
`inaktiv` zu stellen ist dasselbe, und zwei Bedienelemente für dieselbe Sache
könnten einander nur widersprechen.

Während eines Laufs zeigt die Statuszeile durchgehend, welche Verfahren mit
welchem Budget laufen. Meldungen der Animation überschreiben sie nicht.

**`beste Ep.` spielt die Episode wirklich nach.** Aufgezeichnet werden der
Simulatorzustand zu ihrem Beginn und jede ausgeführte Action; die Wiedergabe
setzt den Zustand und spielt die Actions ab. Bewegung und Return sind damit
**identisch** mit der Trainingsepisode.

Die naheliegende Alternative – die gespeicherte Policy deterministisch laufen
lassen – wurde verworfen: Sie erreicht gemessen nur 80 bis 91 % des Returns.
Die beste Episode ist das Maximum über Hunderte Episoden und verdankt ihren
Wert zum Teil glücklichen Explorationszügen und ihrem Startzustand. Dieselbe
SAC-Policy schwankt über fünf Startzustände zwischen 328 und 374, während die
Episode selbst 433 erreichte.

Die Werte neben der Animation erscheinen erst beim **Überfahren mit der Maus** –
unter dem Bild steht nur Episode, Schritt und Return.

### Diagramme

| Tab | Inhalt |
| --- | --- |
| **Training** | der aktive Slot aus Einzelläufen |
| **Vergleich** | alle aktiven Slots gemeinsam, je Slot eine Farbe |
| **Einzelverfahren** | genau **ein** Slot, wählbar – auch nach einem Vergleichslauf |

Die Anzeigen ordnen sich nach der Form des Bereichs: Gewählt wird die
Aufteilung, die das größte Bild ergibt. Quadratische Frames sind in einem
breiten, flachen Bereich höhenbegrenzt – drei Anzeigen nebeneinander sind dort
deutlich größer als zwei über zwei.

Je Slot eine feste Farbe (blau, rot, gelb, grün), Rohkurve dünn und blass, der
gleitende Durchschnitt kräftig. Die Fensterbreite stellt `Glättung` ein – global
für alle Slots und alle Graphen, damit die Kurven vergleichbar bleiben; eine
Änderung wirkt sofort, auch während eines Laufs. Dieselbe Zahl bestimmt, über
wie viele Episoden die Summary mittelt.

Beide Kurven sind auf 2.000 Punkte gedeckelt. Ohne diese Grenze wüchse die
Zeichenzeit linear mit der Episodenzahl; gemessen wird weiter vollständig.

**Export:** `PNG exportieren` sichert das sichtbare Diagramm, `TXT exportieren`
die Summary, `Je Verfahren PNG` schreibt in einer Aktion **je aktivem Slot eine
eigene Datei**. Letzteres ist für Berichte gedacht, die einen Plot je Verfahren
oder je Parameterausprägung verlangen.

### Slots und Vergleich

Bis zu vier gleichrangige Slots: verschiedene Algorithmen, verschiedene
Parametrisierungen desselben Algorithmus oder eine Mischung. Belegen mehrere
Slots denselben Algorithmus, nennt die Legende zusätzlich den abweichenden
Parameter. Die Summary führt einen Block mit genau den Parametern, in denen
sich die Slots unterscheiden.

Alle Slots erhalten identische Environment-Konfigurationen; Budget und Seed
stammen aus dem jeweiligen Tab.

### Summary

Oben Umfang und Ausgang des Laufs – Episoden, Schritte, beide Budgetgrenzen und
welche den Lauf beendet hat –, darunter, abgetrennt durch eine fett gesetzte
Zwischenüberschrift, alle Mittelwerte über die **letzten** Episoden im
Glättungsfenster. Die Zeilen darunter tragen kein `Ø` mehr; die Überschrift sagt
es bereits für alle. Über den ganzen Lauf gemittelt hinge jede
Kennzahl noch am untrainierten Anfang; gerade der Fortschritt verschwände im
Mittel. `Beste Episode` zählt dagegen über alle Episoden.

Der Block **Unterschiede** erscheint nur, wenn ein Algorithmus mehrere Slots
belegt – also bei einer Parameterstudie. Bei lauter verschiedenen Verfahren
sagt die Kopfzeile bereits alles, und der Block würde die Tabelle nur aus dem
sichtbaren Bereich schieben.

Eine automatische Zwischenevaluation gibt es nicht: Sie beantwortet dieselbe
Frage wie der Mittelwert über die letzten Episoden, kostet aber Rechenzeit.
`HumanoidWorkbench.evaluate()` bleibt für eine explorationsfreie Messung von
Hand im Logikmodul.

## Bekannte Grenzen

- **Die Figur läuft mit diesem Budget nicht.** Das ist kein Fehler, sondern
  die Folge eines Budgets unter dem Profilwert. Gemessen erreicht SAC nach
  300.000 Schritten im Mittel 2.154 und 3.269 Return (zwei Durchgänge) – die
  Figur hält sich zunehmend länger aufrecht, statt zu laufen. In der besten
  gemessenen Konfiguration bleibt sie in 68 % der Episoden die vollen 1000
  Schritte oben, kommt dabei aber nur rund 2 m weit.
- **PPO tritt benachteiligt an – geprüft.** 300.000 Schritte sind 3 % seines
  Profilbudgets, bei TD3 und SAC je 15 %. Die Gegenprobe mit 1,5 Mio Schritten
  (ebenfalls 15 %, rund 50 Minuten je Lauf) bringt PPO um 40 % nach vorn und
  lässt es trotzdem beim Vierfachen unter SAC. Bericht 2.2.5.
- Die Animation kostet spürbar Rechenzeit; die eingestellte Bildrate ist eine
  Obergrenze. Der Standard von **20 FPS** liegt bewusst unter der
  environment-eigenen Rate von 67 – das ergibt rund dreifache Zeitlupe, weil
  ein stürzender Humanoid in Echtzeit kaum zu verfolgen ist.
- **Lernzustände werden nicht auf Platte gesichert.** Ohne automatische
  Evaluation gibt es keinen Auslöser, der sagen könnte, welcher Stand der beste
  ist; Speichern und Laden bleiben als Funktionen des Logikmoduls verfügbar,
  haben aber kein Bedienelement. Beim Schließen der Anwendung ist der
  Lernfortschritt weg – Diagramme und Summary vorher exportieren.
- Mehrere Verfahren gleichzeitig zu trainieren bringt auf einer Maschine mit
  zwei Performance-Kernen wenig Zeitgewinn.
- **Der Seed macht Läufe vergleichbar, nicht identisch.** Ein Vergleich führt
  seine Slots als Threads eines Prozesses aus; alle teilen sich denselben
  Zufallsstrom von PyTorch, und wie die Threads verschachtelt werden, ist von
  Lauf zu Lauf verschieden. Zwei Läufe derselben Konfiguration mit demselben
  Seed lieferten gemessen 2.136 gegen 1.947 Episoden. Für exakte
  Wiederholbarkeit müssten die Slots eigene Prozesse mit je einem Thread sein.
- Von den 348 Observationswerten zeigt die Einblendung eine begründete Auswahl;
  `cinert`, `cvel` und `cfrc_ext` (286 Werte) erscheinen nur verdichtet.

## Dateien

| Datei | Inhalt |
| --- | --- |
| `humanoid_app.py` | Einstiegspunkt |
| `humanoid_logic.py` | Environment, Konfiguration, Verfahren, Metriken |
| `humanoid_gui.py` | Tkinter-Oberfläche, Diagramme, Animation |
| `humanoid_render.py` | isolierter MuJoCo-Renderprozess |
| `prompt.md` | Projektvorgaben und Vorgabenabgleich |
| `tests/` | Testsuite |
| `KLR-339-2026-08-Hessling_Oliver.md`, `.pdf` | Bericht der Kursabgabe, Quelle und Satz |
| `praesentation.md`, `.html`, `praesentation-notizen.md` | Folien, Vortragsfassung und Sprechtext |
| `diagramme/` | 25 Reward-Plots aller Messläufe |
| `kennzahlen/` | Summary je Lauf: Kennzahlen und vollständige Konfiguration |
| `screenshots/` | vier Bilder der Anwendung |
| `videos/` | sieben Videos in Echtzeit und fünf Standbildstreifen |
| `werkzeuge/` | Skripte für PDF- und HTML-Erzeugung, nicht Teil der Anwendung |
| `requirements.txt`, `conftest.py` | Abhängigkeiten und Testkonfiguration |
| `exports/` | Rohexporte der Anwendung, nicht Teil der Abgabe |
