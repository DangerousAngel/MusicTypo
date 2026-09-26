"""
Audio Loader Module - Stream-seeks directly to the middle of the song.
Extracts a snippet around the midpoint (duration/2) without reading the entire file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf
import librosa
from mutagen import File as MutagenFile


@dataclass
class AudioSnippet:
    file_path: Path
    audio: np.ndarray          # 1D mono float32
    sample_rate: int           # Target sampling rate (e.g. 22050 Hz)
    total_duration: float      # Total length of song in seconds
    snippet_offset: float      # Start timestamp (seconds) where snippet was extracted
    snippet_duration: float    # Duration of snippet extracted
    channels: int              # Original channel count
    original_sr: int           # Original sample rate


def get_audio_duration(file_path: Path | str) -> float:
    """Quickly extracts total duration in seconds from header without reading audio data."""
    file_path = Path(file_path)
    # Method 1: SoundFile header
    try:
        with sf.SoundFile(str(file_path)) as f:
            return float(len(f)) / float(f.samplerate)
    except Exception:
        pass

    # Method 2: Mutagen header
    try:
        mf = MutagenFile(str(file_path))
        if mf is not None and mf.info and getattr(mf.info, "length", 0) > 0:
            return float(mf.info.length)
    except Exception:
        pass

    # Method 3: Librosa duration
    try:
        return float(librosa.get_duration(path=str(file_path)))
    except Exception:
        return 0.0


def load_middle_snippet(
    file_path: Path | str,
    snippet_duration: float = 20.0,
    target_sr: int = 22050
) -> AudioSnippet:
    """
    Directly seeks to the exact middle of the audio file and loads a snippet.
    
    Args:
        file_path: Path to the audio file.
        snippet_duration: Duration of the excerpt in seconds (default: 20s).
        target_sr: Target sample rate for downstream DSP processing.
        
    Returns:
        AudioSnippet with mono waveform and metadata.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Primary method: Fast SoundFile seeking
    try:
        with sf.SoundFile(str(path)) as f:
            file_sr = f.samplerate
            channels = f.channels
            total_frames = len(f)
            total_duration = total_frames / file_sr

            frames_to_read = int(snippet_duration * file_sr)

            if total_duration <= snippet_duration:
                # Song is shorter than requested snippet, read whole track
                start_frame = 0
                snippet_offset = 0.0
                actual_frames = total_frames
            else:
                mid_frame = total_frames // 2
                start_frame = max(0, mid_frame - (frames_to_read // 2))
                snippet_offset = start_frame / file_sr
                actual_frames = min(frames_to_read, total_frames - start_frame)

            f.seek(start_frame)
            data = f.read(frames=actual_frames, dtype="float32", always_2d=True)

            # Convert multi-channel to mono
            mono_audio = np.mean(data, axis=1)

            # Resample if needed
            if file_sr != target_sr:
                audio_resampled = librosa.resample(mono_audio, orig_sr=file_sr, target_sr=target_sr)
            else:
                audio_resampled = mono_audio

            actual_snippet_sec = len(audio_resampled) / target_sr

            return AudioSnippet(
                file_path=path,
                audio=audio_resampled,
                sample_rate=target_sr,
                total_duration=total_duration,
                snippet_offset=snippet_offset,
                snippet_duration=actual_snippet_sec,
                channels=channels,
                original_sr=file_sr,
            )
    except Exception as e_sf:
        # Fallback method: Librosa stream with offset/duration
        try:
            total_dur = get_audio_duration(path)
            if total_dur > 0 and total_dur > snippet_duration:
                snippet_offset = max(0.0, (total_dur / 2.0) - (snippet_duration / 2.0))
            else:
                snippet_offset = 0.0

            y, sr = librosa.load(
                str(path),
                sr=target_sr,
                offset=snippet_offset,
                duration=snippet_duration,
                mono=True
            )

            actual_snippet_sec = len(y) / sr
            actual_total = total_dur if total_dur > 0 else actual_snippet_sec

            return AudioSnippet(
                file_path=path,
                audio=y,
                sample_rate=target_sr,
                total_duration=actual_total,
                snippet_offset=snippet_offset,
                snippet_duration=actual_snippet_sec,
                channels=2,
                original_sr=target_sr,
            )
        except Exception as e_lib:
            raise RuntimeError(
                f"Failed to load audio snippet from {path.name}. "
                f"SoundFile error: {e_sf}. Librosa error: {e_lib}"
            )
