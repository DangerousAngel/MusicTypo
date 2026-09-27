"""
Metadata Reader Module - Extracts tags and embedded information using Mutagen.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import mutagen
from mutagen.easyid3 import EasyID3


@dataclass
class TrackMetadata:
    title: str
    artist: str
    album: str
    genre: str
    year: Optional[str] = None
    bpm: Optional[float] = None
    language: Optional[str] = None
    lyrics: Optional[str] = None
    comment: Optional[str] = None
    raw_tags: Optional[dict] = None


def extract_metadata(file_path: Path | str) -> TrackMetadata:
    """
    Extracts tags (title, artist, album, genre, bpm) from an audio file.
    Falls back gracefully to file name if tags are absent.
    """
    path = Path(file_path)
    default_title = path.stem
    default_artist = "Unknown Artist"
    default_album = "Unknown Album"
    default_genre = ""
    bpm_val: Optional[float] = None
    year_val: Optional[str] = None
    lang_val: Optional[str] = None
    lyrics_val: Optional[str] = None
    comment_val: Optional[str] = None
    raw_dict = {}

    try:
        audio = mutagen.File(str(path))
        if audio is not None and audio.tags:
            # Common tag extraction
            tags = audio.tags

            # Title
            for k in ["title", "TIT2", "\xa9nam"]:
                if k in tags:
                    val = tags[k]
                    default_title = str(val[0] if isinstance(val, list) else val).strip()
                    break

            # Artist
            for k in ["artist", "TPE1", "\xa9ART"]:
                if k in tags:
                    val = tags[k]
                    default_artist = str(val[0] if isinstance(val, list) else val).strip()
                    break

            # Album
            for k in ["album", "TALB", "\xa9alb"]:
                if k in tags:
                    val = tags[k]
                    default_album = str(val[0] if isinstance(val, list) else val).strip()
                    break

            # Genre
            for k in ["genre", "TCON", "\xa9gen"]:
                if k in tags:
                    val = tags[k]
                    default_genre = str(val[0] if isinstance(val, list) else val).strip()
                    break

            # BPM
            for k in ["bpm", "TBPM", "tmpo"]:
                if k in tags:
                    val = tags[k]
                    try:
                        str_val = str(val[0] if isinstance(val, list) else val).strip()
                        bpm_val = float(str_val.split(".")[0])
                    except (ValueError, TypeError):
                        pass
                    break

            # Year / Date
            for k in ["date", "year", "TDRC", "TYER", "\xa9day"]:
                if k in tags:
                    val = tags[k]
                    year_val = str(val[0] if isinstance(val, list) else val).strip()[:4]
                    break

            # Language (TLAN in ID3, language in Vorbis/MP4)
            for k in ["language", "TLAN", "\xa9lan", "lang"]:
                if k in tags:
                    val = tags[k]
                    lang_val = str(val[0] if isinstance(val, list) else val).strip()
                    break

            # Lyrics (USLT / SYLT in ID3, lyrics in Vorbis, \xa9lyr in MP4)
            for k in ["lyrics", "USLT", "SYLT", "unsyncedlyrics", "\xa9lyr"]:
                if k in tags:
                    val = tags[k]
                    lyrics_val = str(val[0] if isinstance(val, list) else val).strip()
                    break

            # Comments
            for k in ["comment", "COMM", "\xa9cmt", "description"]:
                if k in tags:
                    val = tags[k]
                    comment_val = str(val[0] if isinstance(val, list) else val).strip()
                    break

            raw_dict = {str(k): str(v) for k, v in tags.items()}

    except Exception:
        # If tag reading fails, use default fallback
        pass

    return TrackMetadata(
        title=default_title or path.stem,
        artist=default_artist,
        album=default_album,
        genre=default_genre,
        year=year_val,
        bpm=bpm_val,
        language=lang_val,
        lyrics=lyrics_val,
        comment=comment_val,
        raw_tags=raw_dict
    )
