#!/usr/bin/env python3
import json
import random
import threading
import urllib.request
import tkinter as tk
from datetime import datetime
from tkinter import ttk

OPEN_METEO_URL = (
    'https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.405'
    '&current_weather=true&timezone=Europe%2FBerlin'
)

GERMAN_WEEKDAYS = [
    'Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag'
]

GERMAN_MONTHS = [
    'Januar', 'Februar', 'März', 'April', 'Mai', 'Juni',
    'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember'
]

QUOTES = [
    'Stärke wächst im Augenblick des Handelns.',
    'Jeder Schritt bringt dich näher zum Ziel.',
    'Heute ist dein Tag – mach etwas Großartiges daraus.',
    'Kleine Fortschritte sind auch Fortschritte.',
    'Du bist näher dran, als du denkst.'
]


def fetch_weather_text():
    try:
        request = urllib.request.Request(
            OPEN_METEO_URL,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; OliverApp/1.0)'}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read().decode('utf-8')
            data = json.loads(payload)
            current = data.get('current_weather', {})
            temperature = current.get('temperature')
            windspeed = current.get('windspeed')
            if temperature is None or windspeed is None:
                raise ValueError('Unvollständige Wetterdaten')
            return f'{temperature:.1f} °C · Wind {windspeed:.1f} km/h'
    except Exception as err:
        return f'Wetterdaten konnten nicht geladen werden ({err})'


def format_date(now):
    weekday = GERMAN_WEEKDAYS[now.weekday()]
    month = GERMAN_MONTHS[now.month - 1]
    return f'{weekday}, {now.day:02d}. {month} {now.year}'


def format_time(now):
    return now.strftime('%H:%M:%S')


class OliverApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Oliver Wetter App')
        self.root.geometry('460x430')
        self.root.resizable(False, False)

        self.style = ttk.Style(self.root)
        self.style.theme_use('default')
        self.style.configure('TLabel', font=('Segoe UI', 11))
        self.style.configure('Header.TLabel', font=('Segoe UI', 18, 'bold'))
        self.style.configure('Quote.TLabel', font=('Segoe UI', 10, 'italic'), foreground='#333333')
        self.style.configure('TButton', font=('Segoe UI', 11))

        self.quote_var = tk.StringVar(value=random.choice(QUOTES))
        self.date_var = tk.StringVar(value=format_date(datetime.now()))
        self.time_var = tk.StringVar(value=format_time(datetime.now()))
        self.weather_var = tk.StringVar(value='Wetter wird geladen...')

        self._build_ui()
        self._start_clock()
        self._load_weather_async()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text='Hallo Oliver!', style='Header.TLabel').grid(row=0, column=0, columnspan=2, sticky='w')
        self._build_image(frame)

        ttk.Label(frame, textvariable=self.date_var).grid(row=2, column=0, columnspan=2, sticky='w', pady=(8, 0))
        ttk.Label(frame, textvariable=self.time_var).grid(row=3, column=0, columnspan=2, sticky='w')

        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky='ew', pady=12)

        ttk.Label(frame, text='🌤 Wetter in Berlin:').grid(row=5, column=0, sticky='w')
        ttk.Label(frame, textvariable=self.weather_var, wraplength=400).grid(row=6, column=0, columnspan=2, sticky='w')

        ttk.Label(frame, text='💬 Zufälliges Zitat:').grid(row=7, column=0, sticky='w', pady=(12, 0))
        ttk.Label(frame, textvariable=self.quote_var, style='Quote.TLabel', wraplength=400).grid(row=8, column=0, columnspan=2, sticky='w')

        refresh_button = ttk.Button(frame, text='Aktualisieren', command=self.refresh)
        refresh_button.grid(row=9, column=0, columnspan=2, pady=16)

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _build_image(self, parent):
        canvas = tk.Canvas(parent, width=420, height=130, bg='#87CEEB', highlightthickness=0)
        canvas.grid(row=1, column=0, columnspan=2, pady=(8, 8))

        canvas.create_rectangle(0, 0, 420, 130, fill='#87CEEB', width=0)
        canvas.create_oval(320, 10, 390, 80, fill='#FFD24D', outline='')
        canvas.create_oval(45, 20, 110, 60, fill='white', outline='')
        canvas.create_oval(75, 10, 140, 55, fill='white', outline='')
        canvas.create_oval(105, 25, 165, 60, fill='white', outline='')
        canvas.create_oval(170, 35, 235, 70, fill='white', outline='')
        canvas.create_oval(205, 25, 265, 65, fill='white', outline='')
        canvas.create_oval(235, 35, 295, 70, fill='white', outline='')
        canvas.create_arc(-120, 50, 280, 250, start=0, extent=180, fill='#70A02E', outline='')
        canvas.create_arc(180, 70, 520, 270, start=0, extent=180, fill='#4F8A18', outline='')
        canvas.create_rectangle(0, 92, 420, 130, fill='#3B7C17', width=0)
        canvas.create_text(15, 110, anchor='w', text='Scene2', fill='white', font=('Segoe UI', 10, 'bold'))

    def _start_clock(self):
        now = datetime.now()
        self.time_var.set(format_time(now))
        self.date_var.set(format_date(now))
        self.root.after(1000, self._start_clock)

    def _load_weather_async(self):
        thread = threading.Thread(target=self._fetch_weather, daemon=True)
        thread.start()

    def _fetch_weather(self):
        weather_text = fetch_weather_text()
        self.root.after(0, lambda: self.weather_var.set(weather_text))

    def refresh(self):
        self.quote_var.set(random.choice(QUOTES))
        self.weather_var.set('Wetter wird geladen...')
        self._load_weather_async()


def main():
    root = tk.Tk()
    OliverApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
