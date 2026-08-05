# Modellbasiertes Gridworld Lab

Die lokale Tkinter-Anwendung demonstriert `Value Iteration` und
`Q-Value Iteration` in einem deterministischen Gridworld. Beide Verfahren
verwenden das vollständige Modell des Environments. Epsilon beeinflusst nur
die sichtbaren Agenten-Rollouts, niemals die Bellman-Updates.

## Start

Vom Repository-Hauptordner aus:

```bash
conda activate rl-26-08
cd Oliver/02-gridworld_model_based
python gridworld_app.py
```

Falls das Environment noch nicht aktualisiert wurde:

```bash
conda env update --name rl-26-08 --file ../../environment.yml --prune
```

## Sinnvolle Reihenfolge in der App

1. Grid und Parameter einstellen und anwenden.
2. Mit `Planner single sweep` einzelne Bellman-Updates beobachten.
3. Mit `Planner bis Konvergenz` die Policy fertig berechnen.
4. Den Agenten greedy oder mit Epsilon durch das Grid laufen lassen.
5. Im Tab `Methodenvergleich` beide Verfahren isoliert vergleichen.
6. Value- oder Q-Tabelle öffnen und bei Bedarf als CSV exportieren.

Die Koordinaten werden überall als `(row, column)` angegeben. Die Actions sind
`Up=0`, `Down=1`, `Left=2` und `Right=3`.

Direkt nach dem Start stehen alle Werte definitionsgemäß auf `0`. Ein
Agenten-Rollout verändert diese Werte nicht. Erst `Planner single sweep` oder
`Planner bis Konvergenz` berechnet die Value- und Q-Tabellen.

## Tests

```bash
conda run --name rl-26-08 python -m unittest discover -s Oliver/02-gridworld_model_based/tests -v
```

Die Tests benötigen keine geöffnete GUI und prüfen Environment, synchrone
Planner-Sweeps, Konvergenz, Rollouts, Vergleich und Tabellendaten.
