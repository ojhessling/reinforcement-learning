# CartPole – RL-Workbench

Projektordner: `Oliver/07-cartpole`

## Grundlage

Berücksichtige die verbindlichen Regeln aus `../workbench.md`.

Projektname: `cartpole`
Environment: `CartPole-v1`

Dieser Prompt enthält nur die CartPole-spezifischen Anforderungen.

## Ziel

Erstelle eine lokale Tkinter-Lernanwendung für Gymnasiums `CartPole-v1`.
Verwende `DQN` aus `stable_baselines3` und eine darauf aufbauende
Double-DQN-Variante (`DDQN`), jeweils mit `MlpPolicy`.

## Environment

Erzeuge das Environment zwingend mit:

```python
import gymnasium

env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
```

Verwende eine gemeinsame Factory, die für Training, Evaluation und sichtbare
Animation jeweils eine eigene Environment-Instanz mit diesem Aufruf erzeugt.
Environment-Instanzen werden weder gleichzeitig noch threadübergreifend
verwendet.

Verändere Dynamik, Startzustände, Rewards und Abbruchbedingungen nicht.

Actions:

- `0`: Wagen nach links beschleunigen
- `1`: Wagen nach rechts beschleunigen

Observation:

1. Wagenposition
2. Wagengeschwindigkeit
3. Stangenwinkel
4. Winkelgeschwindigkeit der Stange

Pro Schritt wird der originale Reward `+1` vergeben. Eine Episode endet bei
einem Stangenwinkel außerhalb von ungefähr `±12°`, einer Wagenposition
außerhalb von `±2,4` oder durch das Zeitlimit von 500 Schritten. Ein Erfolg
ist eine ausschließlich durch das Zeitlimit beendete Episode.

## DQN

Erzeuge das Modell zwingend über:

```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    env,
    # Werte aus der UI
)
```

Nutze für DQN die unveränderte Implementierung von Stable-Baselines3. DDQN
verwendet dieselbe SB3-Infrastruktur und unterscheidet sich ausschließlich in
der Target-Berechnung: Das Online-Netz wählt die nächste Action, das
Target-Netz bewertet sie. Dueling DQN und Prioritized Experience Replay gehören
nicht zum Projekt.

Folgende DQN-Parameter müssen in der UI einstellbar sein:

- `total_timesteps`
- `learning_rate`
- `buffer_size`
- `learning_starts`
- `batch_size`
- `tau`
- `gamma`
- `train_freq`
- `gradient_steps`
- `target_update_interval`
- `exploration_fraction`
- `exploration_initial_eps`
- `exploration_final_eps`
- `max_grad_norm`
- `seed`

Zusätzlich müssen `net_arch`, Aktivierungsfunktion, Optimizer und dessen
relevante Parameter gemäß `../workbench.md` einstellbar sein. Technische
Optionen wie `verbose`, `tensorboard_log`, `device` und `_init_setup_model`
gehören nicht in die UI. Als Ausgangswerte dienen die DQN-Standardwerte der
installierten Stable-Baselines3-Version. Die README nennt diese Version und
verlinkt die zugehörige DQN-Dokumentation.

Training verwendet `model.learn(total_timesteps=...)`. Evaluation verwendet
`model.predict(observation, deterministic=True)` und darf weder `learn()` noch
andere Lernupdates ausführen.

Binde Fortschritt, Abbruchprüfung und Live-Metriken über eine von
`stable_baselines3.common.callbacks.BaseCallback` abgeleitete Callback-Klasse
an. Der Callback übergibt Daten thread-sicher an die GUI und beendet das
Training kontrolliert, wenn `Stoppen` angefordert wurde.

## Evaluation

Die Anzahl der Evaluations-Episoden ist in der UI einstellbar. Jede Evaluation
verwendet eine eigene Environment-Instanz und einen reproduzierbar abgeleiteten
Seed. Zeige mindestens:

- mittleren Episoden-Reward
- Standardabweichung des Episoden-Rewards
- mittlere Episodenlänge
- Erfolgsrate als Anteil der Episoden, die das Zeitlimit von 500 Schritten
  erreichen

Evaluation verändert weder Modellparameter noch Replay Buffer oder
Trainingsstatistiken.

## Vergleich

DQN und DDQN sind einzeln für den Vergleich auswählbar. Ein Vergleich erfordert
mindestens zwei ausgewählte Algorithmen, verwendet identische Hyperparameter
und fair abgeleitete Seeds und verändert das sichtbare Modell nicht.

Der Vergleichsgraph wird ab dem ersten abgeschlossenen Episodenergebnis
angezeigt und während aller Läufe fortlaufend aktualisiert. Er zeigt Reward
gegen Environment-Schritte, pro Lauf eine transparente Rohkurve sowie je
Algorithmus den Mittelwert. Ab zwei Wiederholungen kommt die Standardabweichung
als Unsicherheitsband hinzu. Der Graph darf nicht bis zum Ende aller Läufe leer
bleiben.

## Speichern und Laden

Gespeicherte Modelle müssen später weitertrainiert werden können. Speichere und
lade deshalb neben dem SB3-Modell auch den Replay Buffer mit den dafür
vorgesehenen Stable-Baselines3-Methoden. Stelle nach dem Laden die zugehörige
Environment-Instanz her und prüfe Modell, Metadaten und Replay Buffer vor der
Übernahme auf Kompatibilität.

## Darstellung

Zeige ausschließlich den von `env.render()` gelieferten offiziellen
Gymnasium-RGB-Frame in der Tkinter-Oberfläche. Erstelle keine eigene
CartPole-Grafik und öffne kein separates Pygame-Fenster.

Zeige neben der Animation die vier aktuellen Observation-Werte mit Einheiten:

- Wagenposition in Metern
- Wagengeschwindigkeit in Metern pro Sekunde
- Stangenwinkel in Grad
- Winkelgeschwindigkeit in Grad pro Sekunde

CartPole-spezifische Metriken sind Episoden-Reward beziehungsweise
Episodenlänge, Erfolgsrate, Stangenwinkel und Wagenposition.

## Abhängigkeiten

Dokumentiere `stable-baselines3` und `gymnasium` in `requirements.txt` und in
der zentralen `../environment.yml`. Verwende kompatible, gemeinsam installierbare
Versionen.

## Abnahme

Fertig, wenn DQN und DDQN korrekt trainiert und deterministisch evaluiert
werden, alle relevanten DQN- und Netzwerkparameter in der UI änderbar sind,
Training kontrolliert abgebrochen werden kann, der auswählbare Vergleichsgraph
bereits während der Läufe aktuelle Ergebnisse zeigt, erfolgreiche Episoden bis
zum Zeitlimit erkennbar sind, Modelle samt Replay Buffer weitertrainierbar
geladen werden, die offizielle Gymnasium-Grafik eingebettet ist und die
allgemeinen Tests und Abnahmekriterien aus `../workbench.md` erfüllt sind.
