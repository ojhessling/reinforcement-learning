# Sprechnotizen zur Präsentation

**Gesamtdauer 10–12 Minuten · 17 Folien.** Die Zeitangabe je Folie ist ein
Richtwert; die drei Videofolien sind die Stellen, an denen man Luft holen darf.

Vor dem Start: die drei Videodateien in QuickTime öffnen und in den Hintergrund
legen, damit sie mit einem Klick laufen. In der PDF-Fassung liegt auf jeder
Videofolie ein Standbild — das Video wird daneben abgespielt.

---

## 1 · Titel — mit Video (45 s)

> **Video abspielen: `animation-sturz.mp4`** — laufen lassen, während man spricht.

„Das hier ist eine humanoide Figur: 42 Kilogramm, 17 Gelenke, dreidimensional.
Und das ist der Normalzustand am Anfang jedes Trainings — sie fällt nach etwa
25 Schritten um.

Meine Aufgabe war, ihr mit drei Verfahren beizubringen, aufrecht zu bleiben und
vorwärts zu gehen: PPO, TD3 und SAC. Ich zeige Ihnen in den nächsten zehn
Minuten, welches gewinnt, warum das schnellste Verfahren das schlechteste ist,
und zweimal, wie ich mich beinahe selbst getäuscht hätte."

---

## 2 · Warum Humanoid anders ist (40 s)

„Zur Einordnung, weil Sie die Umgebung schon von den anderen Vorträgen kennen:
Humanoid ist in dieser Reihe die größte Umgebung. 348 Beobachtungswerte gegen
11 beim Hopper, 17 Gelenke statt drei, und zum ersten Mal drei Dimensionen —
die Figur kann also auch **seitlich** wegkippen.

Zwei Fallen stecken im Detail. Erstens: Der Wertebereich der Actions ist
**±0,4**, nicht ±1 wie in allen Vorgängerprojekten — wer das als ±1 annimmt,
skaliert falsch. Zweitens: Die Übersetzung ist je Gelenk verschieden, von 25 an
den Armen bis 300 an der Hüfte. Derselbe Zahlenwert bedeutet an der Hüfte das
Zwölffache an Moment."

---

## 3 · Die Reward-Formel (60 s) — **Kernfolie**

„Diese Folie erklärt jedes Ergebnis, das noch kommt. Der Reward hat vier
Anteile, und einer davon erschlägt die anderen: **fünf Punkte für jeden Schritt,
den die Figur nicht umfällt.**

Rechnen wir das durch: 1000 Schritte einfach nur stehen bleiben ergibt 5000
Punkte. Vorwärtsgehen bringt 1,25 pro Meter und Sekunde — bei einem Meter pro
Sekunde also 1,25 Punkte je Schritt gegen 5 Punkte fürs Nichtumfallen.

**Aufrechtbleiben zahlt viermal besser als Laufen.** Merken Sie sich das, wir
kommen darauf zurück.

Deshalb ist meine Zielmarke 5000: Das ist genau der Return einer Figur, die eine
volle Episode durchhält. Humanoid hat keine offizielle Gelöst-Schwelle — das ist
eine Marke, die ich für diese Arbeit gesetzt habe, und ich nenne sie bewusst
nicht ‚gelöst'."

---

## 4 · Die Workbench (40 s)

„Kurz zum Werkzeug: Links der Konfigurator — zwei bis vier Verfahren
gleichzeitig, jeder Parameter des jeweiligen Algorithmus einzeln einstellbar.
Rechts läuft die Animation, für jedes Verfahren eine eigene, jede in einem
eigenen Prozess. Unten die Reward-Plots und eine Kennzahlentabelle.

Alle ausgewählten Verfahren trainieren **gleichzeitig**. Wenn derselbe
Algorithmus mehrere Slots belegt, erkennt die Oberfläche das und schreibt den
abweichenden Parameter automatisch in Titel und Legende — das brauchen wir
später bei der Parameterstudie."

---

## 5 · Was die Anwendung zeigt (35 s)

„Unter jedem Bild stehen nur drei Zahlen: Episode, Schritt, Return. Alles
Weitere erscheint erst, wenn man mit der Maus über das Bild fährt — dann
**neben** dem Bild, damit man beides sieht: alle 17 Actions mit ihrem Moment in
Newtonmetern, Rumpfhöhe, Neigung, die Gelenkwinkel und die vier Reward-Anteile
einzeln.

348 Werte kann man nicht anzeigen. 286 davon sind Trägheitstensoren und
Kontaktkräfte, die kein Verhalten erklären — die fasse ich zu einer Quadratsumme
zusammen.

Und eine Kleinigkeit, auf die ich stolz bin: ‚beste Episode' rechnet die Policy
nicht nach, sondern spielt die Episode **exakt** nach — aufgezeichnet werden
Simulatorzustand und jede Action. Der Return stimmt deshalb auf die
Nachkommastelle mit dem Training überein."

---

## 6 · Versuchsaufbau (45 s)

„Die Hyperparameter kommen unverändert aus dem RL Baselines3 Zoo — die Aufgabe
verlangt die empfohlenen Werte, also habe ich auch die ungewöhnlichen
übernommen, etwa ein Gamma von 0,95 bei PPO.

Jedes Verfahren bekommt exakt 300.000 Schritte, die Episodengrenze steht auf
unbegrenzt. Das ist wichtig: Eine Humanoid-Episode endet beim Sturz, sie wird
mit dem Lernfortschritt also **länger**. Wer besser lernt, sammelt in 1000
Episoden dreimal so viele Schritte wie ein schlechtes Verfahren — dieselbe
Episodenzahl wäre also gerade **nicht** fair.

Und jede Konfiguration läuft zweimal, mit Zufallsstart 0 und 1. Warum das nicht
Vorsicht, sondern Notwendigkeit war, sehen Sie gleich."

---

## 7 · Der Vergleich: die Bilder (35 s)

„Hier sind beide Durchgänge. Dünn und blass der Reward jeder einzelnen Episode,
kräftig der gleitende Durchschnitt über 20 Episoden, eine Farbe je Verfahren.

Die Aussage ist in beiden Bildern dieselbe: Die gelbe Kurve — SAC — löst sich ab
etwa Episode 1500 nach oben ab. Blau ist PPO, rot ist TD3, und beide bleiben
unten.

Eine Beobachtung nebenbei: Die Kurven enden bei verschiedenen Episodennummern,
obwohl alle gleich viele Schritte hatten. SAC braucht weniger Episoden, weil
seine Episoden länger sind — das ist die Kernaussage in einem Bild."

---

## 8 · Der Vergleich: die Zahlen (45 s)

„In Zahlen: SAC kommt im Mittel beider Durchgänge auf 2.712, PPO auf 492, TD3
auf 407. Das ist **Faktor 5,5**.

Der Abstand ist so groß, dass Zufall als Erklärung ausscheidet: Selbst der
schlechtere der beiden SAC-Läufe schlägt den besseren PPO-Lauf noch um das
Vierfache.

SAC ist auch das einzige Verfahren, das die Zielmarke überhaupt erreicht — die
beste Episode liegt bei 5.102, die Figur bleibt dort die vollen 1000 Schritte
oben. Im zweiten Durchgang schafft sie das in 40 Prozent der letzten Episoden.

Zwischen PPO und TD3 sage ich **nichts** — und warum, sehen Sie jetzt."

---

## 9 · Die Überraschung: TD3 (60 s) — **Dramaturgischer Höhepunkt 1**

„Das sind zwei Läufe von TD3. Gleicher Algorithmus, exakt gleiche Parameter,
gleiches Budget. Der einzige Unterschied ist der Zufallsstart.

Links: über 7.673 Episoden eine flache Linie bei 172. Das sieht nicht nach
langsamem Lernen aus, sondern nach ‚lernt überhaupt nichts'.

Rechts: dieselbe Konfiguration steigt durchgehend auf 643 und schafft eine beste
Episode von 1.595.

Hätte ich nur den ersten Durchgang gefahren — und die Aufgabe verlangt nur
einen —, stünde in meinem Bericht der Satz ‚TD3 lernt mit den empfohlenen
Parametern nicht'. Der wäre schlicht falsch. Die richtige Aussage lautet: **TD3
lernt manchmal, und ob es klappt, entscheidet sich früh.**

Deshalb sage ich zu Platz 2 nichts: Der Abstand zwischen PPO und TD3 beträgt 85
Punkte — TD3 schwankt zwischen seinen eigenen zwei Läufen um 471."

---

## 10 · Warum SAC vorn liegt (40 s)

„Die Erklärung liegt in der Verfahrensklasse. SAC ist off-policy: Jeder
gespeicherte Übergang wird immer wieder zum Lernen benutzt. PPO wirft seine
Daten nach jedem Update weg — bei 300.000 Schritten ist das teuer. Dazu kommt
die gelernte Entropie: SAC regelt selbst, wie viel es noch ausprobiert.

TD3 hat denselben Buffer-Vorteil, nutzt ihn aber nur in einem von zwei Läufen.
Auffällig ist seine Lernrate: 1e−3, dreimal so hoch wie bei SAC. Bei 348
Eingabewerten ist das mein Hauptverdächtiger — und den prüfe ich später
tatsächlich nach."

---

## 11 · Gegenprobe (50 s)

„Bevor ich eine Rangfolge behaupte, muss ich einen Einwand ausräumen — meinen
eigenen. Die Zoo-Profile sind für unterschiedlich lange Läufe getunt: PPO für 10
Millionen Schritte, TD3 und SAC für je zwei. Meine 300.000 Schritte sind bei PPO
also nur drei Prozent seines Arbeitspunktes, bei den anderen 15.

Der Vergleich könnte also unfair sein. Zum Glück rechnet PPO zehnmal schneller,
die Gegenprobe kostet daher nur 50 Minuten: PPO mit 1,5 Millionen Schritten,
ebenfalls 15 Prozent — und ebenfalls zwei Durchgänge.

Ergebnis: PPO legt um 40 Prozent zu, von 492 auf 687. Und bleibt damit beim
**Vierfachen** unter SAC. Die Rangfolge ist also kein Artefakt des Budgets."

---

## 12 · Wer rennt, verliert (50 s) — **Dramaturgischer Höhepunkt 2**

„Viel interessanter ist, **wohin** PPO die fünffache Datenmenge steckt. Nicht
ins Aufrechtbleiben: Die Episodenlänge wächst von 89 auf 110 Schritte, und die
Figur fällt weiterhin in **jeder einzelnen** Episode um.

Was wächst, ist das Tempo — auf 1,60 Meter pro Sekunde. Das ist die schnellste
Figur in dieser ganzen Arbeit, siebenmal schneller als SAC.

Und jetzt erinnern Sie sich an die Formel von vorhin: Bei 1,60 m/s bringt die
Vorwärtsbewegung zwei Punkte pro Schritt. Das Überleben bringt fünf. **Wer 112
Schritte rennt, sammelt weniger als wer 685 Schritte steht.**

PPO optimiert also fleißig — nur den falschen Teil der Formel. Wenn Sie den
Läufen zusehen, sieht PPO am besten aus und ist es am wenigsten."

---

## 13 · Parameterstudie (50 s)

„Zum zweiten Teil der Aufgabe: einen sensitiven Parameter am besten Verfahren
in drei Ausprägungen. Ich habe SAC genommen — nach einem Kriterium, das ich
**vor** den Läufen festgelegt habe — und die Lernrate, mit Faktor drei nach oben
und unten um den Profilwert.

Drei Gründe für die Lernrate: Sie wirkt auf zwei der drei Bewertungskriterien,
sie erlaubt einen sauberen Dreischritt, und der große Wert ist genau die 1e−3,
die bei TD3 im Verdacht steht. Ich prüfe damit also nebenbei meine eigene These
aus dem ersten Teil.

Das Ergebnis unten: Die große Lernrate verliert klar — Faktor drei Rückstand,
und sie erreicht die Zielmarke in keiner einzigen Episode. **Meine TD3-These ist
damit belegt**, innerhalb desselben Verfahrens, bei identischem Seed und
Budget."

---

## 14 · Was wirklich entschieden hat (55 s)

„Zwischen den beiden kleineren Lernraten wird es interessant — und ehrlich
gesagt unbequem. Der Profilwert gewinnt Durchgang eins deutlich und verliert
Durchgang zwei knapp. Der Abstand ist kleiner als die Streuung, die eine einzige
Konfiguration zwischen zwei Zufallsstarts zeigt. Aus dem Return allein ist hier
**nichts** zu entscheiden, und das schreibe ich auch so.

Entschieden hat etwas anderes: Die beiden Läufe mit dem Profilwert liegen 208
Punkte auseinander, die mit der kleinen Lernrate 1.315. **Sechsmal
reproduzierbarer.** Der empfohlene Wert bestätigt sich also nicht als höchster,
sondern als **verlässlichster** Return.

Und dann dieser Befund, den ich nicht erwartet hatte: Die kleine Lernrate hält
**länger** durch — 68 gegen 44 Prozent — erreicht die Zielmarke aber
**seltener**. Der Grund steckt wieder in der Formel: 1000 Schritte aufrecht
ergeben genau 5000. Wer darüber will, muss sich zusätzlich bewegen. **Die kleine
Lernrate hat sicheres Stehen gelernt, der Profilwert vorsichtiges Gehen.**"

---

## 15 · Der Durchbruch (45 s) — **Video**

> **Video abspielen: `animation-beste-episode.mp4`** — 15 Sekunden, Echtzeit.

„Und so sieht es aus, wenn es klappt. Diese Episode läuft die vollen 1000
Schritte und endet mit einem Return von 5.133 — die Zielmarke ist gefallen.

Das Video läuft in **Echtzeit**: Eine Humanoid-Episode dauert mit 1000 Schritten
genau 15 Sekunden Simulationszeit, und genau so lange ist der Clip.

Vier Standbilder dazu auf der Folie, mit dem Return an jedem Punkt — Sie sehen,
wie er gleichmäßig steigt: rund fünf Punkte pro Schritt, praktisch reiner
Überlebensbonus. Die Figur läuft nicht. Sie bleibt stehen. Und genau das ist bei
dieser Reward-Formel die richtige Antwort."

---

## 16 · Was ich mitnehme (50 s)

„Drei Dinge.

**Erstens: Ein Lauf ist kein Ergebnis.** Zweimal in dieser Arbeit hätte ein
einzelner Durchgang zu einer falschen Aussage geführt — bei TD3 und bei der
Lernrate. Zwei Durchgänge verhindern falsche Aussagen; sie entscheiden aber
nicht jede Frage.

**Zweitens: Gleicher Seed heißt nicht gleicher Lauf.** Ich habe dieselbe
Konfiguration zweimal mit demselben Zufallsstart gefahren und 2.136 gegen 1.947
Episoden bekommen. Der Grund: Die Slots laufen als Threads eines Prozesses und
teilen sich denselben Zufallsstrom von PyTorch. Der Seed macht Läufe
**vergleichbar**, nicht identisch.

**Drittens: Die Reward-Formel erklärt jedes Ergebnis.** Wer sie gelesen hat,
sagt vorher, warum das schnellste Verfahren das schlechteste ist.

Und was fehlt: Budget. Der offizielle Benchmark erreicht mit SAC nach zwei
Millionen Schritten 6.232. Ich stehe nach 300.000 bei 2.712. Was ich zeige, ist
die Frühphase — der Übergang von ‚fällt sofort um' zu ‚bleibt eine Weile
stehen'."

---

## 17 · Danke (20 s)

„Im Abgabeordner liegen der Bericht, 19 Reward-Plots, die Kennzahlen jedes
einzelnen Laufs, die Videos und der Quellcode mit 210 Tests. Vielen Dank —
Fragen gern."

---

## Wenn Fragen kommen

**„Warum haben Sie TD3 nicht mit kleinerer Lernrate nachgefahren?"**
Die Aufgabe verlangt die empfohlenen Parameter. Dass TD3 damit unzuverlässig
ist, ist ein Ergebnis, kein Fehler. Die Parameterstudie prüft die These
stattdessen an SAC — dort ist alles andere konstant, das ist der sauberere
Beleg.

**„Warum nur 300.000 Schritte?"**
Zeit. Ein voller Profillauf wären 11 Stunden je Off-Policy-Verfahren, bei sechs
Läufen im Vergleich und sechs in der Studie. Ich habe das Budget lieber gesenkt
und **zwei Seeds** gefahren, als einen einzelnen langen Lauf ohne Absicherung.

**„Ist die Figur damit gelaufen?"**
Nein, und das steht auch so im Bericht. Sie hält sich zunehmend länger aufrecht.
Die Zielmarke von 5000 entspricht genau dem Stehenbleiben — sie zu erreichen
heißt noch nicht laufen.

**„Wie groß ist der Aufwand für die Anwendung?"**
Rund 4.700 Zeilen für Logik, Oberfläche und Renderprozess, dazu 2.800 Zeilen
Tests — 210 Stück, die unter anderem die Reward-Zerlegung, den Wertebereich
±0,4 und die Zuordnung der Gelenkwinkel gegen das MuJoCo-Modell prüfen.
