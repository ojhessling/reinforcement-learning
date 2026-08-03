#!/usr/bin/env python3
import json
import random
import urllib.request
from datetime import datetime

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
            headers={'User-Agent': 'Mozilla/5.0 (compatible; OliverCLI/1.0)'}
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


def main():
    now = datetime.now()
    quote = random.choice(QUOTES)
    weather = fetch_weather_text()

    print('Hallo Oliver!')
    print('============================')
    print(quote)
    print()
    print(f'Datum: {format_date(now)}')
    print(f'Uhrzeit: {format_time(now)}')
    print(f'Wetter in Berlin: {weather}')
    print()
    print('Dieses Programm läuft lokal im Terminal und gibt keine HTML-Seite aus.')


if __name__ == '__main__':
    main()
