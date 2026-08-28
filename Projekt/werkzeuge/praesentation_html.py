"""Foliensatz aus Markdown nach HTML.

Erzeugt eine einzelne, in sich geschlossene HTML-Datei: eine Folie je
Bildschirm, Navigation über Pfeiltasten, und — der eigentliche Grund für HTML
statt PDF — **eingebettete Videos**. Sie starten, sobald ihre Folie sichtbar
wird, und halten an, wenn sie es nicht mehr ist.

    python werkzeuge/praesentation_html.py
"""
import pathlib
import sys

import markdown

WURZEL = pathlib.Path(__file__).resolve().parent.parent
QUELLE = WURZEL / "praesentation.md"
ZIEL = WURZEL / "praesentation.html"

STIL = """
* { box-sizing: border-box; }
body { margin: 0; background: #05070d; color: #e2e8f0; overflow: hidden;
       font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; }
.folie { display: none; position: absolute; inset: 0; padding: 3.2vh 4vw 5vh;
         flex-direction: column; justify-content: flex-start; }
/* Alle Überschriften auf derselben Höhe: Der Inhalt beginnt oben, und was
   darunter wachsen darf, füllt den Rest. Bei zentriertem Inhalt sprang die
   Überschrift von Folie zu Folie. */
.folie > h1:first-child { flex: 0 0 auto; min-height: 7vh; }
.folie.aktiv { display: flex; }
h1 { font-size: 3.6vh; margin: 0 0 1.6vh; color: #fff; line-height: 1.15; }
h2 { font-size: 2.6vh; margin: 0 0 1.6vh; color: #93c5fd; font-weight: 600; }
h3 { font-size: 2.4vh; margin: 1.8vh 0 0.8vh; color: #fbbf24; }
p, li { font-size: 2.05vh; line-height: 1.5; margin: 0.5vh 0; }
ul { margin: 0.8vh 0; padding-left: 3vh; }
li { margin: 0.9vh 0; }
strong { color: #fbbf24; }
h1 strong, h3 strong, figcaption strong { color: #fbbf24; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 0.94em;
       background: #16203a; padding: 0.1em 0.35em; border-radius: 3px; color: #bfdbfe; }
pre { background: #0f172a; border: 1px solid #1e293b; border-radius: 6px;
      padding: 1.6vh 2vh; overflow-x: auto; }
pre code { background: none; font-size: 1.9vh; line-height: 1.5; color: #e2e8f0; }
blockquote { margin: 1.4vh 0; padding: 0.6vh 0 0.6vh 2vh; border-left: 3px solid #fbbf24;
             color: #cbd5e1; font-size: 1.95vh; }
table { border-collapse: collapse; width: 100%; margin: 1.2vh 0; font-size: 1.95vh; }
th, td { border: 1px solid #1e293b; padding: 0.7vh 1.1vh; text-align: left; }
th { background: #111c33; color: #cbd5e1; font-weight: 600; }
td strong, th strong { color: #fbbf24; }
tr:nth-child(even) td { background: #0a1122; }
img, video { max-width: 100%; object-fit: contain; display: block;
            margin: 0.8vh auto; border-radius: 4px; }
video { background: #000; }

/* Abbildungen nehmen den Platz, den Text und Tabellen übrig lassen: Der Absatz,
   der nur ein Bild oder Video enthält, wächst in die freie Höhe, das Medium
   füllt ihn aus. Ohne das blieb auf textarmen Folien ein Drittel leer. */
.folie > p:has(> img:only-child), .folie > p:has(> video:only-child),
.zwei > div > p:has(> img:only-child), .zwei > div > p:has(> video:only-child) {
  flex: 1 1 auto; min-height: 0; display: flex; align-items: center;
  justify-content: center; margin: 0.6vh 0; }
/* `width` statt nur `max-width`: Ohne das bleiben die Diagramme bei ihren
   nativen 975 px stehen, weil CSS Bilder nie über die Originalgröße hinaus
   vergrößert. Die Deckelung bei 1500 px hält die Skalierung unter 1,6-fach,
   darüber würde die Beschriftung der Achsen weich. */
.folie > p > img, .folie > p > video,
.zwei > div > p > img, .zwei > div > p > video {
  width: 100%; max-width: 1500px; height: 100%; max-height: 100%;
  object-fit: contain; margin: 0; }
.folie > video { max-height: 62vh; }

.zwei { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5vw;
        align-items: center; flex: 1 1 auto; min-height: 0; }
.zwei > div { display: flex; flex-direction: column; justify-content: center;
              min-height: 0; max-height: 100%; }
.vier { display: grid; grid-template-columns: repeat(2, 1fr); grid-auto-rows: 1fr;
        gap: 1vh 2vw; margin: 1vh 0; flex: 1 1 auto; min-height: 0; }
.vier figure { margin: 0; display: flex; flex-direction: column; min-height: 0; }
.vier video { flex: 1 1 auto; min-height: 0; width: 100%; height: 100%;
              max-height: 100%; margin: 0; object-fit: contain; }
figcaption { font-size: 1.7vh; color: #94a3b8; text-align: center; margin-top: 0.6vh;
             line-height: 1.35; }
.nummer { position: absolute; right: 2.5vw; bottom: 1.8vh; font-size: 1.6vh;
          color: #475569; }
"""

SKRIPT = """
const folien = [...document.querySelectorAll('.folie')];
let aktuell = 0;
function zeige(n) {
  aktuell = Math.max(0, Math.min(folien.length - 1, n));
  folien.forEach((f, i) => {
    const an = i === aktuell;
    f.classList.toggle('aktiv', an);
    f.querySelectorAll('video').forEach(v => {
      if (an) { v.currentTime = 0; v.play().catch(() => {}); } else { v.pause(); }
    });
  });
  location.hash = aktuell + 1;
}
document.addEventListener('keydown', e => {
  if (['ArrowRight', 'PageDown', ' ', 'Enter'].includes(e.key)) zeige(aktuell + 1);
  if (['ArrowLeft', 'PageUp', 'Backspace'].includes(e.key)) zeige(aktuell - 1);
  if (e.key === 'Home') zeige(0);
  if (e.key === 'End') zeige(folien.length - 1);
});
document.addEventListener('click', e => {
  if (e.target.closest('video')) return;          // Klick auf Video steuert das Video
  zeige(aktuell + (e.clientX > window.innerWidth / 2 ? 1 : -1));
});
zeige(parseInt(location.hash.slice(1)) - 1 || 0);
"""


def bauen() -> pathlib.Path:
    text = QUELLE.read_text()
    # Führenden HTML-Kommentar entfernen, er ist ein Hinweis an den Autor.
    if text.lstrip().startswith("<!--"):
        text = text.split("-->", 1)[1]
    # `markdown="1"`, damit Markdown auch innerhalb der Layout-Container gilt.
    text = text.replace('<div class="zwei">', '<div class="zwei" markdown="1">')
    text = text.replace("<div>", '<div markdown="1">')

    folien = []
    for nummer, block in enumerate(text.split("\n---\n"), start=1):
        block = block.strip()
        if not block:
            continue
        inhalt = markdown.markdown(
            block, extensions=["tables", "fenced_code", "md_in_html", "attr_list"]
        )
        folien.append(
            f'<section class="folie">{inhalt}'
            f'<div class="nummer">{nummer}</div></section>'
        )

    seite = (
        '<!doctype html><html lang="de"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Humanoid-v5 – PPO, TD3 und SAC</title>"
        f"<style>{STIL}</style></head><body>"
        + "".join(folien)
        + f"<script>{SKRIPT}</script></body></html>"
    )
    ZIEL.write_text(seite)
    return ZIEL


if __name__ == "__main__":
    ziel = bauen()
    print(f"{ziel.name}: {len(ziel.read_text().splitlines())} Zeilen, "
          f"{ziel.stat().st_size / 1024:.0f} kB")
    sys.exit(0)
