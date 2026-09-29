# Music Typo 🎵

> Intelligent music classifier and `.m3u8` playlist generator that analyzes songs by streaming directly into the **middle of each track**.

(!ss)[Screenshot.png]
---

## Features

- Classifies music into 28 categories, from Arabic Oud and Classical to Rock, EDM, and Hip-Hop.
- Detects musical modes, keys, and vocal languages.
- Generates `.m3u8` playlists by genre, language, and mode, plus CSV and JSON reports.
- Includes a modern desktop GUI and CLI for easy use.

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Launch GUI
```bash
python main.py
```
*(or `python gui.py`)*

### 3. Command Line Interface (CLI)
```bash
# Scan a folder:
python cli.py -i "D:/My Music" -o "./Playlists"

# Classify specific files:
python cli.py -f "song1.mp3" "song2.wav" -o "./Playlists"

# Custom mid-song duration (e.g. 15s) and file organizing:
python cli.py -i "D:/My Music" -d 15 --organize copy
```
---

## 👤 Author

- **GitHub**: [@DangerousAngel](https://github.com/DangerousAngel)

