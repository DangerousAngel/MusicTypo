# Music Typo 🎵

> Intelligent music classifier and `.m3u8` playlist generator that analyzes songs by streaming directly into the **middle of each track**.
---

## ✨ Features

- **Mid-Song Smart Seek**: Reads only a 15–30s excerpt from the middle of the track ($\text{duration}/2$), skipping intros/outros for fast, accurate classification.
- **Multi-Type Classification**: Detects **Rock**, **Beat**, **Piano**, **Melody**, **Epic**, **Vocal**, **Ambient**, and **Classical** using DSP timbre & acoustic features + ID3 metadata.
- **Select by Folder or Files**: Process an entire directory or select specific individual audio files.
- **Instant Playlists**: Produces standard `.m3u8` playlists for each genre, a master playlist, plus CSV & JSON reports.
- **Cyber-Dark Modern UI**: Built with CustomTkinter, featuring real-time progress, waveform excerpt slider, and double-click to play. Also includes a rich CLI.

---

## 🚀 Quick Start

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
by [**ME**](https://github.com/DangerousAngel)  
---

## 👤 Author

- **GitHub**: [@DangerousAngel](https://github.com/DangerousAngel)
- **Project**: [MusicTypo](https://github.com/DangerousAngel/MusicTypo)
