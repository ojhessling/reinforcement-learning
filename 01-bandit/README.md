# Multi-Armed Bandit

Eine lokale Tkinter-Anwendung, die Exploration und Exploitation anhand von drei
Bernoulli-Banditen und fünf Agentenstrategien visualisiert.

## Installation

Im Ordner `Oliver/bandit`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Start

```bash
python bandit_app.py
```

Mit den drei Bandit-Buttons können einzelne Pulls manuell ausprobiert werden.
Diese Pulls verändern die Environment-Statistik, aber nicht das Wissen des
Agenten. `Agent Single Step` führt einen Lernschritt aus; `Agent Run N Loops`
führt die eingestellte Anzahl Schritte aus.

Im Tab `Strategien vergleichen` können beliebige Strategien ausgewählt und
über mehrere reproduzierbare Wiederholungen verglichen werden. Das Diagramm
zeigt den mittleren kumulativen Reward, optionale 95-%-Konfidenzintervalle und
den optimalen Erwartungswert. Die Tabelle ergänzt End-Reward, Reward pro Pull,
Auswahlanteil des besten Banditen und Regret.

## Tests

```bash
python -m unittest discover -s tests
```
