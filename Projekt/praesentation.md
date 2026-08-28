<!--
Foliensatz. Trenner ist eine Zeile mit ---
HTML-Fassung erzeugen: python werkzeuge/praesentation_html.py
-->

# Eine Figur, die stehen bleiben soll

## `Humanoid-v5` mit PPO, TD3 und SAC

<video src="videos/animation-1-fruehphase-ep563-579.mp4" autoplay loop muted playsinline></video>

**Immer wieder von vorn** · Oliver Hessling · Kurs D21195UYS

---

# Warum Humanoid schwer ist

| | Hopper | HalfCheetah | Walker2d | **Humanoid** |
| --- | --- | --- | --- | --- |
| Beobachtungswerte | 11 | 17 | 17 | **348** |
| Gelenke | 3 | 6 | 6 | **17** |
| Raum | 2D | 2D | 2D | **3D** |

- **348 Werte hinein, 17 Gelenkbefehle hinaus** — 67-mal je Sekunde
- schon bei nur zehn Stufen je Gelenk wären das **10¹⁷ Möglichkeiten** je Schritt
- 42 kg, die in **drei** Richtungen kippen — Hopper und HalfCheetah nur in einer Ebene

---

# Die Formel, die alles erklärt

```text
Reward =    5,0      +  1,25 · vₓ   −  0,1 · Σaᵢ²  −  5e−7 · Σcfrc²
         (Überleben)    (vorwärts)     (Steuerung)     (Aufprall)
```

### 1000 Schritte nur stehen = **5000 Return**

- Überleben zahlt **viermal** besser als Laufen bei 1 m/s
- Zielmarke **5000** = 1000 Schritte aufrecht, projektintern gesetzt
- `Humanoid-v5` hat **keine** offizielle Gelöst-Schwelle

---

# Die Workbench

![Konfigurator](screenshots/screenshot-konfigurator.png)

- **Konfigurator** links: 2 bis 4 Verfahren, jeder Parameter einzeln · **ausführender Teil** rechts: Animation je Slot
- alle Slots trainieren **gleichzeitig**, Legende nennt automatisch den **abweichenden Parameter**

---

# Was die Anwendung zeigt

![Anzeige mit Einblendung](screenshots/screenshot-anzeige.png)

- unter dem Bild nur **Episode · Schritt · Return** — alles Weitere erscheint beim Überfahren **neben** dem Bild
- 17 Actions mit Moment, Rumpfhöhe, Neigung, Gelenkwinkel, alle vier **Reward-Anteile einzeln**
- von 348 Werten eine **begründete Auswahl**: 286 davon erklären kein Verhalten

---

# Versuchsaufbau

| | Wert |
| --- | --- |
| Hyperparameter | **RL Baselines3 Zoo**, unverändert |
| Budget | **300.000 Schritte** je Verfahren |
| Episodengrenze | **aus** (`E = 0`) |
| Durchgänge | **zwei**, Zufallsstart 0 und 1 |

> Schritte statt Episoden: Wer besser lernt, hat längere Episoden — und bekäme
> bei gleicher Episodenzahl **mehr** Daten.

---

# Der Vergleich

![Durchgang 1](diagramme/2-2-vergleich-seed0.png)

![Durchgang 2](diagramme/2-2-vergleich-seed1.png)

**SAC** gelb · **PPO** blau · **TD3** rot — zwei Durchgänge, dasselbe Muster

---

# Die Zahlen

| Ø der letzten 20 Episoden | PPO | TD3 | **SAC** |
| --- | --- | --- | --- |
| Return, Mittel beider Durchgänge | 491,9 | 407,3 | **2.711,6** |
| Beste Einzelepisode | 1.313 | 1.595 | **5.102** |
| Ø Episodenlänge | 89 / 96 | 39 / 134 | **434 / 651** |
| 1000 Schritte durchgehalten | 0 % | 0 % | **15 / 40 %** |

### SAC gewinnt mit **Faktor 5,5**

Der schlechtere SAC-Lauf schlägt den besseren PPO-Lauf noch um das Vierfache.

---

# Dieselben Parameter. Zwei Bilder.

![TD3 Durchgang 1](diagramme/2-2-td3-seed0.png)

**Zufallsstart 0** — 7.673 Episoden flach bei 172

![TD3 Durchgang 2](diagramme/2-2-td3-seed1.png)

**Zufallsstart 1** — steigt auf 643, beste Episode 1.595

### Aus einem Durchgang wäre eine falsche Aussage geworden

---

# Warum SAC vorn liegt

- **SAC** off-policy: jeder Übergang wird mehrfach benutzt, Entropie regelt die Erkundung selbst
- **PPO** verwirft seine Daten nach jedem Update — teuer bei 300.000 Schritten
- **TD3** hat denselben Buffer, aber Lernrate `1e−3` — dreimal SAC
- Platz 2 bleibt **offen**: 85 Punkte Abstand gegen 471 Punkte Eigenstreuung von TD3

---

# War der Vergleich fair?

**Einwand:** 300.000 Schritte sind bei PPO nur 3 % seines Profilbudgets, bei TD3 und SAC 15 %.

![PPO mit 1,5 Mio Schritten](diagramme/2-2-ppo-1500k-vergleich.png)

| PPO | 300.000 | 1,5 Mio = **15 %** |
| --- | --- | --- |
| Ø Return | 491,9 | **686,6** (+40 %) |
| 1000 Schritte durchgehalten | 0 % | **0 %** |

---

# Wer rennt, verliert

| | Tempo | Episodenlänge | Ø Return |
| --- | --- | --- | --- |
| PPO nach 1,5 Mio | **1,60 m/s** | 112 | 730 |
| SAC nach 300.000 | 0,22 m/s | **651** | **3.269** |

- PPO ist **siebenmal schneller** — und trotzdem letzter
- bei 1,60 m/s: **2,0** Punkte vorwärts gegen **5,0** fürs Überleben
- PPO optimiert fleißig den **falschen** Teil der Formel

<div class="vier">
<figure>
<video src="videos/animation-7-ppo-ep11317-return1502.mp4" autoplay loop muted playsinline></video>
<figcaption><strong>PPO</strong> nach 1 Mio Schritten — stolpert nach vorn, Sturz bei Schritt 267</figcaption>
</figure>
<figure>
<video src="videos/animation-2-ep2194-return5134.mp4" autoplay loop muted playsinline></video>
<figcaption><strong>SAC</strong> nach 1 Mio Schritten — bleibt die vollen 1000 Schritte oben</figcaption>
</figure>
</div>

---

# Parameterstudie: die Lernrate von SAC

![Durchgang 1](diagramme/2-3-vergleich-seed0.png)

![Durchgang 2](diagramme/2-3-vergleich-seed1.png)

| | kleiner | **empfohlen** | größer |
| --- | --- | --- | --- |
| Lernrate | `1e−4` | **`3e−4`** | `1e−3` |

Bestes Verfahren aus 2.2, drei Ausprägungen, je zwei Durchgänge

---

# Das Ergebnis

| Ø der letzten 50 Episoden | `1e−4` | **`3e−4`** | `1e−3` |
| --- | --- | --- | --- |
| Return, Mittel | 3.064 | **3.475** | 1.172 |
| Zielmarke erreicht | 0 / 12 % | **6 / 34 %** | **nie** |
| Streuung zwischen den Durchgängen | **43 %** | **6 %** | 28 % |

- gegen `1e−3` ist der Abstand **eindeutig** — Faktor 3, in beiden Durchgängen
- gegen `1e−4` entscheidet **nicht** der Return: Der dreht sich mit dem Seed
- der Profilwert gewinnt über **Verlässlichkeit** — sechsmal reproduzierbarer

---

# Stehen oder Gehen?

| Durchgang 2 | `1e−4` | **`3e−4`** |
| --- | --- | --- |
| 1000 Schritte durchgehalten | **68 %** | 44 % |
| Zielmarke erreicht | 12 % | **34 %** |
| Strecke vorwärts | 1,9 m | **2,5 m** |

### 1000 Schritte aufrecht ergeben *genau* 5000

Wer die Marke überschreitet, hat sich **bewegt**.
Die kleine Lernrate lernt **sicheres Stehen**, der Profilwert **vorsichtiges Gehen**.

---

# Der Lernverlauf in vier Videos

<div class="vier">
<figure>
<video src="videos/animation-2-ep2194-return5134.mp4" autoplay loop muted playsinline></video>
<figcaption>~1 Mio Schritte<br><strong>5.134</strong></figcaption>
</figure>
<figure>
<video src="videos/animation-3-ep4091-return6372.mp4" autoplay loop muted playsinline></video>
<figcaption>~2 Mio Schritte<br><strong>6.372</strong></figcaption>
</figure>
<figure>
<video src="videos/animation-4-ep5302-return7186.mp4" autoplay loop muted playsinline></video>
<figcaption>~3 Mio Schritte<br><strong>7.186</strong></figcaption>
</figure>
<figure>
<video src="videos/animation-5-ep8083-return7470.mp4" autoplay loop muted playsinline></video>
<figcaption>7 Mio Schritte<br><strong>7.470</strong></figcaption>
</figure>
</div>

Alle vier: **je 1000 Schritte in 15 Sekunden Echtzeit**

---

# Und mit mehr Budget?

![7 Millionen Schritte](diagramme/2-4-lauf-7mio.png)

<div class="zwei">
<div>

| | 300.000 | **7 Mio** |
| --- | --- | --- |
| Ø Return | 2.154 / 3.269 | **6.718** |
| Tempo | 0,21 m/s | **1,95 m/s** |
| Strecke | 1,5 m | **28,6 m** |
| durchgehalten | 15 / 40 % | **92 %** |

**Die Figur läuft** · Zoo-Benchmark 6.232 **übertroffen**

Dieselben 15 Sekunden wie am Anfang: damals fällt sie immer wieder, jetzt ein Durchlauf

</div>
<div>

<video src="videos/animation-5-ep8083-return7470.mp4" autoplay loop muted playsinline></video>

</div>
</div>

---

# Was ich mitnehme

- **SAC gewinnt** — bei knappem Budget mit Faktor 5,5 vor PPO und TD3
- **Die empfohlenen Werte sind gut gewählt.** Der Profilwert der Lernrate schlägt beide Nachbarn — nicht im Return, sondern in der Verlässlichkeit
- **Ein Lauf ist kein Ergebnis**, und gleicher Seed heißt nicht gleicher Lauf
- **Die Reward-Formel entscheidet alles.** Sie bestimmt, ob die Figur stehen oder laufen lernt — und in welcher Reihenfolge

---

# Ausblick: neuere Verfahren, gleiches Budget

![Gleiche Schrittzahl](diagramme/2-5-vergleich-gleiche-schritte.png)

**300.000 Schritte** für alle drei — Seitenprojekt mit `CrossQ` und `TQC`

<div class="zwei">
<div>

| bei 300.000 | SAC | **CrossQ** |
| --- | --- | --- |
| Ø Return | 2.893 | **4.435** |
| Zielmarke erreicht | 14 % | **64 %** |
| Tempo | 0,21 m/s | **1,06 m/s** |
| Strecke | 2,1 m | **14,0 m** |

</div>
<div>

- CrossQ **läuft** schon nach 300.000 Schritten
- **aber**: dreifache Rechenzeit je Schritt
- nach Zeit gemessen ist es Letzter, nach Daten Erster

</div>
</div>

---

# Danke

<video src="videos/animation-6-crossq-ep1931-return7618.mp4" autoplay loop muted playsinline></video>

**CrossQ nach 600.000 Schritten** — Ø 6.988, aufrecht joggend.
Mehr als SAC nach 7 Millionen, bei einem Zwölftel der Daten.

## Fragen?
