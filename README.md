# Music Typo 🎵

> Intelligent music classifier and `.m3u8` playlist generator that analyzes songs by streaming directly into the **middle of each track**.
---

## ✨ Features

- **Mid-Song Smart Seek**: Reads only a 15–30s excerpt from the middle of the track ($\text{duration}/2$), skipping intros/outros for fast, accurate classification.
- **28 Nuanced Musical Categories**:
  - **Acoustic & Classical**: Piano, Acoustic, Classical, Melody, Choral-Opera, World-Folk
  - **Electronic & Modern**: Electronic, House-Techno, Synthwave, Lo-Fi, Beat, Trap, Hip-Hop
  - **Rock & Energy**: Rock, Metal, Punk-Alternative, Epic
  - **Groove & Soul**: Jazz, Blues, Soul-Funk, R&B, Reggae
  - **Regional & World**: Oud (Middle Eastern / Arabic / Maqam), Latin (Reggaeton, Salsa, Flamenco), Country, Pop, Vocal
- **Advanced Mode & Key Detection**: Detects musical modes (Major, Minor, Dorian, Phrygian / Maqam Hijaz, Lydian, Mixolydian, Harmonic Minor) and exact pitch centers.
- **Multilingual Vocal Language Identification**: Identifies vocal language (Arabic, English, Spanish, French, German, Italian, Portuguese, Russian, Turkish, Persian, Korean, Japanese, Hindi, or Instrumental) using acoustic vocal formant analysis, script detection, and metadata tag inspection.
- **Smart Categorized Playlists**: Produces standard UTF-8 `.m3u8` playlists by Category, by Vocal Language, by Musical Mode, and a unified Master playlist, plus CSV & JSON reports.
- **Cyber-Dark Modern Desktop GUI**: Built with CustomTkinter featuring real-time search, category/mode/language filter menus, progress tracking, and double-click to play. Also includes a rich CLI.

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
---

## 👤 Author

- **GitHub**: [@DangerousAngel](https://github.com/DangerousAngel)

