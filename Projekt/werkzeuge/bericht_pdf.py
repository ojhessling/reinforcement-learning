"""Bericht aus Markdown nach PDF.

Markdown -> HTML mit einem Druck-Stylesheet für A4, gedruckt von Chrome im
Headless-Betrieb. Kein LaTeX, kein pandoc.

    python werkzeuge/bericht_pdf.py KLR-339-2026-08-Hessling_Oliver.md \
                                    KLR-339-2026-08-Hessling_Oliver.pdf
"""
import pathlib, subprocess, sys, markdown

quelle = pathlib.Path(sys.argv[1])
ziel = pathlib.Path(sys.argv[2])
text = quelle.read_text()

# python-markdown verarbeitet Markdown in HTML-Blöcken nur mit `markdown="1"`.
# Das wird hier beim Konvertieren ergänzt, damit die Quelle in jedem anderen
# Renderer (GitHub, VS Code, Marp) unverändert funktioniert.
text = text.replace('<div align="center">', '<div align="center" markdown="1">')

koerper = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "md_in_html", "toc"],
    extension_configs={"md_in_html": {}},
)

stil = """
@page { size: A4; margin: 18mm 16mm 16mm 16mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
       font-size: 10.5pt; line-height: 1.45; color: #16181d; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; }
h2 { font-size: 15pt; margin: 20pt 0 6pt 0; border-bottom: 1px solid #c9ced6;
     padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 12.5pt; margin: 15pt 0 5pt 0; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }
h5 { font-size: 10.5pt; margin: 10pt 0 3pt 0; color: #3a4149; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
img { max-width: 100%; height: auto; page-break-inside: avoid; }
div[align="center"] { text-align: center; page-break-inside: avoid; margin: 10pt 0; }
div[align="center"] em { display: block; font-size: 9pt; color: #4a525c;
                         margin-top: 4pt; text-align: left; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8pt 0;
        page-break-inside: avoid; }
th, td { border: 1px solid #c9ced6; padding: 3pt 5pt; text-align: left;
         vertical-align: top; }
th { background: #eef1f5; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 9pt;
       background: #f2f4f7; padding: 0 2px; border-radius: 2px; }
pre { background: #f2f4f7; padding: 6pt 8pt; border-radius: 3px; font-size: 9pt;
      overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #c9ced6; margin: 8pt 0; padding: 2pt 0 2pt 10pt;
             color: #3a4149; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #d6dae0; margin: 14pt 0; }
a { color: #16181d; text-decoration: none; }
"""

seite = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<title>{quelle.stem}</title><style>{stil}</style></head><body>{koerper}</body></html>"""

html = ziel.with_suffix(".html")
html.write_text(seite)
subprocess.run([
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "--headless", "--disable-gpu", "--no-pdf-header-footer",
    f"--print-to-pdf={ziel.resolve()}", html.resolve().as_uri(),
], check=True, capture_output=True)
print(f"{ziel} erzeugt: {ziel.stat().st_size/1_000_000:.1f} MB")
