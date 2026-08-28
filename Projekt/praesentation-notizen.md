# Notizen · 20 Folien · 10–12 Minuten

Stichpunkte als Gedächtnisstütze, kein Sprechtext. Ausführlich steht alles im
Bericht. Videos starten automatisch, sobald ihre Folie erscheint.

---

**1 · Titel** · 40 s
- Video läuft: in 15 Sekunden fällt sie immer wieder von vorn
- 42 kg, 17 Gelenke, 3D
- Ansage: diese 15 Sekunden merken, kommen am Schluss wieder

**2 · Warum Humanoid schwer ist** · 35 s
- größte Umgebung der Reihe: 348 Werte statt 11
- 17 Gelenke gleichzeitig stellen, 67-mal je Sekunde
- 10¹⁷ ist nur eine Veranschaulichung — in Wahrheit stufenlos
- 3D heißt: kippt auch zur Seite, nicht nur nach vorn

**3 · Die Reward-Formel** · 60 s · **Kernfolie**
- vier Anteile, einer erschlägt alle: 5,0 fürs Nichtumfallen
- 1000 Schritte stehen = 5000; laufen bei 1 m/s = 1,25 je Schritt
- Zielmarke 5000 selbst gesetzt, Humanoid hat keine offizielle Schwelle
- Ansage: darauf kommen wir dreimal zurück

**4 · Die Workbench** · 35 s
- links einstellen, rechts zusehen, unten messen
- Parameter je Verfahren gefiltert — PPO hat keinen Replay Buffer
- abweichender Parameter wandert automatisch in Titel und Legende

**5 · Was die Anwendung zeigt** · 30 s
- unter dem Bild nur drei Zahlen, Rest beim Überfahren
- Reward einzeln nach seinen vier Anteilen aufgeschlüsselt
- 286 der 348 Werte erklären kein Verhalten → verdichtet als Quadratsumme

**6 · Versuchsaufbau** · 40 s
- Zoo-Parameter unverändert, auch die ungewöhnlichen (γ 0,95 bei PPO)
- 300.000 Schritte, Episodengrenze aus
- warum Schritte: Episode endet beim Sturz → besseres Verfahren bekäme mehr Daten
- zwei Durchgänge — warum, kommt gleich

**7 · Der Vergleich** · 30 s
- dünn = jede Episode, kräftig = Durchschnitt über 20
- gelb SAC löst sich ab Episode 1500, blau und rot bleiben unten
- Kurven enden verschieden → SAC braucht weniger Episoden, weil seine länger sind

**8 · Die Zahlen** · 40 s
- SAC 2.712 gegen 492 und 407 → Faktor 5,5
- schlechterer SAC-Lauf schlägt besseren PPO-Lauf noch um Faktor 4
- nur SAC erreicht die Zielmarke
- zu Platz 2 sage ich nichts — warum, kommt jetzt

**9 · Dieselben Parameter, zwei Bilder** · 60 s · **Höhepunkt 1**
- identische Konfiguration, nur anderer Zufallsstart
- links flach bei 172 über 7.673 Episoden, rechts steigt auf 643
- ein Durchgang hätte „TD3 lernt nicht" ergeben — falsch
- deshalb Platz 2 offen: 85 Punkte Abstand, 471 Punkte Eigenstreuung

**10 · Warum SAC vorn liegt** · 35 s
- off-policy: jeder Übergang mehrfach; PPO wirft weg
- SAC regelt Erkundung über Entropie selbst
- TD3-Verdacht: Lernrate 1e−3, dreimal SAC — prüfe ich später nach

**11 · War der Vergleich fair?** · 45 s
- eigener Einwand: PPO bei 3 % seines Budgets, andere bei 15 %
- PPO rechnet 10× schneller → Gegenprobe kostet nur 50 min
- 1,5 Mio Schritte = 15 %: +40 %, bleibt Faktor 4 zurück
- Rangfolge ist kein Budget-Artefakt

**12 · Wer rennt, verliert** · 50 s · **Höhepunkt 2**
- PPO wird schneller statt standfester, 1,60 m/s
- Durchhaltequote bleibt 0 % — fällt in jeder Episode
- zurück zur Formel: 2,0 vorwärts gegen 5,0 überleben
- 112 Schritte rennen < 651 Schritte stehen

**13 · Parameterstudie** · 40 s
- bestes Verfahren = SAC, Kriterium vorab festgelegt
- Lernrate: wirkt auf zwei der drei Bewertungskriterien, sauberer Dreischritt
- große Ausprägung ist genau TD3s 1e−3 → prüft meine eigene These

**14 · Das Ergebnis** · 50 s
- 1e−3 verliert klar, erreicht die Marke nie → TD3-These belegt
- oben dreht sich die Reihenfolge mit dem Seed → aus Return nicht entscheidbar
- entschieden über Streuung: 208 gegen 1.315 Punkte, sechsmal verlässlicher
- Zoo-Empfehlung bestätigt sich als verlässlichster, nicht höchster Return

**15 · Stehen oder Gehen** · 45 s
- 1e−4 hält länger durch, erreicht die Marke aber seltener
- Rechnung: 1000 Schritte = genau 5000, darüber nur mit Bewegung
- kleine Lernrate lernt Stehen, Profilwert lernt Gehen
- der schönste Befund der Arbeit, war nicht erwartet

**16 · Lernverlauf in vier Videos** · 40 s
- alle vier gleichzeitig, je 1000 Schritte in 15 s Echtzeit
- links wackelt sie, rechts geht sie
- Return 5.134 → 7.470

**17 · Und mit mehr Budget?** · 60 s · **Höhepunkt 3**
- nach der Abgabe weitergelaufen: 7 Mio Schritte, 21 Stunden
- Kurve: bis Episode 4.000 wächst die Länge (stehen lernen), dann Return (gehen)
- 6.718 im Mittel, 1,95 m/s, 28,6 m, 92 % durchgehalten
- Zoo-Benchmark 6.232 übertroffen
- dieselben 15 Sekunden wie Folie 1 — damals immer wieder Stürze, jetzt ein Durchlauf
- Vorbehalt: ein Seed, anderes Budget, gehört in den Ausblick

**18 · Was ich mitnehme** · 40 s
- SAC gewinnt bei knappem Budget klar — Faktor 5,5
- Zoo-Werte sind gut gewählt: Profilwert schlägt beide Nachbarn, über Verlässlichkeit
- ein Lauf ist kein Ergebnis; gleicher Seed ≠ gleicher Lauf
- Reward-Formel entscheidet, ob stehen oder laufen gelernt wird — und in welcher Reihenfolge
- letzter Satz: erst aufrecht bleiben, dann vorwärts — das ist keine Wahl des Verfahrens, sondern der Formel

**19 · Ausblick: neuere Verfahren** · 45 s
- Seitenprojekt: SAC gegen CrossQ und TQC, gleiche Workbench, 300.000 Schritte
- CrossQ bei gleicher Datenmenge klar vorn — und es läuft: 1,06 m/s, 14 m
- ABER: braucht dreifache Rechenzeit je Schritt
- zwei Bedeutungen von fair: nach Zeit ist CrossQ Letzter, nach Daten Erster
- Antwort hängt davon ab, welche Ressource knapp ist — Roboter vs. Rechner
- ein Seed, nicht Teil der bewerteten Arbeit

**20 · Danke** · 20 s
- Video: CrossQ nach 600.000 Schritten, joggt **aufrecht**
- Ø 6.988 — mehr als SAC nach 7 Mio (6.718), bei einem Zwölftel der Daten
- SAC läuft gekrümmt, CrossQ aufrecht: gleiche Formel, verschiedene Lösungen
- Bericht, 25 Plots, Kennzahlen je Lauf, 7 Videos, 210 Tests

---

## Wenn Fragen kommen

**Warum haben Sie TD3 nicht mit kleinerer Lernrate nachgefahren?**
Aufgabe verlangt die empfohlenen Parameter. Dass TD3 damit unzuverlässig ist,
ist ein Ergebnis. Die These prüfe ich stattdessen an SAC — dort ist alles andere
konstant, das ist der sauberere Beleg.

**Warum nur 300.000 Schritte?**
Zeit. Voller Profillauf wären 11 h je Off-Policy-Verfahren. Lieber Budget senken
und zwei Seeds fahren als einen langen Lauf ohne Absicherung.

**Ist die Figur bei 300.000 gelaufen?**
Nein, sie lernt stehen. Zielmarke 5000 entspricht genau dem Stehenbleiben.
Gelaufen ist sie erst im Ausblick bei 7 Mio.

**Warum ist SAC besser als TD3, beide sind off-policy?**
Stochastischer Actor mit gelernter Entropie gegen deterministischen Actor mit
festem Rauschen (σ 0,1). Bei 17 Dimensionen ist festes Rauschen zu wenig, um aus
einem schlechten Start herauszukommen.

**Warum streut SAC innerhalb eines Laufs so stark?**
Kehrseite des schnellen Lernens: Der Agent probiert mehr aus, einzelne Episoden
misslingen deutlich. Die Richtung ist in beiden Durchgängen identisch.

**Woher kommt die Zielmarke 5000?**
Selbst gesetzt: 1000 Schritte × 5,0 Überlebensbonus. Humanoid-v5 führt keinen
offiziellen reward_threshold — ich nenne sie deshalb nie „gelöst".

**Warum zwei Seeds und nicht fünf?**
Rechenzeit. Zwei verhindern falsche Aussagen, entscheiden aber nicht jede Frage —
das steht auch so im Bericht. Fünf bis zehn wären rund 20 Stunden.

**Wieso hat TD3 im Vergleich ein anderes Netz als im Zoo-Profil?**
Profil nennt 400,300. Vereinheitlicht auf 256,256, damit der Vergleich nicht an
der Netzgröße hängt. Steht als Abweichung im Bericht.

**Warum ist der Replay Buffer kleiner als im Profil?**
8 GB RAM. 1 Mio Übergänge wären 5,2 GB allein für Beobachtungen. Bei 300.000
Schritten verdrängt der kleinere Buffer nie etwas — beim 7-Mio-Lauf schon, das
ist dort vermerkt.

**Ist PPO für Humanoid ungeeignet?**
Das belegen meine Daten nicht. Belegt ist: Bei 15 % seines Arbeitspunktes liegt
es Faktor 4 zurück und optimiert Tempo statt Überleben.

**Wie lange lief alles zusammen?**
Verfahrensvergleich rund 7 h, Parameterstudie rund 11 h, PPO-Gegenprobe 1,7 h,
der lange Lauf 21 h.

**Warum X-Achse Episoden und nicht Schritte?**
Die Aufgabe verlangt „Reward über die Episoden". Die Schrittzahl steht in der
Summary, und die unterschiedlichen Endpunkte der Kurven sind selbst eine Aussage.

**Wie stellen Sie sicher, dass die Zahlen stimmen?**
Jede Zahl stammt aus einer exportierten Summary, die neben den Kennzahlen die
vollständige Konfiguration des Laufs enthält. 210 Tests prüfen unter anderem die
Reward-Zerlegung und den Wertebereich ±0,4 gegen das MuJoCo-Modell.

**Was war der größte Fehler unterwegs?**
Nach Durchgang 1 stand im Entwurf „TD3 lernt nicht". Der zweite Seed hat das
widerlegt, bevor es in den Bericht kam.

**Warum laufen die Verfahren gleichzeitig?**
Vorgabe 1.2d. Jeder Slot als eigener Thread mit eigenem Modell und eigener
Environment — und genau daher kommt auch der geteilte Zufallsstrom.

**Was würden Sie mit mehr Zeit machen?**
Fünf Seeds je Verfahren, um Platz 2 zu entscheiden, und PPO über sein volles
Profilbudget von 10 Mio Schritten.

**Warum kein Reward Shaping?**
Hätte den Vergleich mit dem Zoo-Benchmark unmöglich gemacht und die
Aufgabenstellung verlangt die Umgebung unverändert.

**Was ist gSDE und warum aus?**
Zustandsabhängiges Explorationsrauschen. Die Zoo-Profile für Humanoid setzen es
nicht, also bleibt es aus — das Feld ist in der Oberfläche vorhanden.
