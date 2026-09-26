"""
Synthetic Test Audio Generator
Creates synthetic music tracks for testing audio classification, mid-song seek,
and playlist generation without needing external audio files.
"""

from pathlib import Path
import numpy as np
import soundfile as sf


def generate_kick(sr: int, duration: float = 0.2) -> np.ndarray:
    """Generates an electronic bass drum / kick."""
    t = np.linspace(0, duration, int(sr * duration), False)
    # Pitch drop from 150Hz to 40Hz
    freq = np.linspace(150, 40, len(t))
    env = np.exp(-t * 25)
    return np.sin(2 * np.pi * freq * t) * env


def generate_snare(sr: int, duration: float = 0.2) -> np.ndarray:
    """Generates a snare hit (noise + tone)."""
    t = np.linspace(0, duration, int(sr * duration), False)
    noise = np.random.uniform(-1, 1, len(t))
    tone = np.sin(2 * np.pi * 180 * t)
    env = np.exp(-t * 20)
    return (noise * 0.7 + tone * 0.3) * env


def generate_piano_chord(freqs: list, sr: int, duration: float = 3.0) -> np.ndarray:
    """Generates a piano-like decaying chord with harmonics."""
    t = np.linspace(0, duration, int(sr * duration), False)
    chord = np.zeros_like(t)
    for f in freqs:
        # fundamental + harmonics
        harmonics = [
            (f * 1.0, 1.0, 1.5),
            (f * 2.0, 0.5, 2.5),
            (f * 3.0, 0.25, 3.5),
            (f * 4.0, 0.12, 4.5),
        ]
        for h_freq, amp, decay in harmonics:
            env = np.exp(-t * decay)
            chord += amp * np.sin(2 * np.pi * h_freq * t) * env

    # Sharp initial attack
    attack_samples = int(sr * 0.01)
    attack = np.linspace(0, 1, attack_samples)
    chord[:attack_samples] *= attack
    return chord


def generate_beat_track(output_path: Path, duration: float = 35.0, sr: int = 22050):
    """Generates a track dominated by heavy electronic drum rhythm (EDM/Beat)."""
    samples = int(duration * sr)
    audio = np.zeros(samples, dtype=np.float32)

    bpm = 128
    beat_interval = int(sr * (60.0 / bpm))
    kick = generate_kick(sr)
    snare = generate_snare(sr)

    # 4/4 Beat pattern
    for i in range(0, samples - len(kick), beat_interval):
        beat_num = (i // beat_interval) % 4
        # Kick on 1 and 3
        if beat_num in (0, 2):
            audio[i : i + len(kick)] += kick * 0.8
        # Snare on 2 and 4
        if beat_num in (1, 3):
            audio[i : i + len(snare)] += snare * 0.6
        # Hi-hat on off-beats
        hihat_pos = i + beat_interval // 2
        if hihat_pos + int(sr * 0.05) < samples:
            hh_t = np.linspace(0, 0.05, int(sr * 0.05))
            hh = np.random.uniform(-1, 1, len(hh_t)) * np.exp(-hh_t * 80) * 0.3
            audio[hihat_pos : hihat_pos + len(hh)] += hh

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.9
    sf.write(str(output_path), audio, sr)


def generate_piano_track(output_path: Path, duration: float = 35.0, sr: int = 22050):
    """Generates a melodic piano solo with clean decaying chords."""
    samples = int(duration * sr)
    audio = np.zeros(samples, dtype=np.float32)

    # Chord progression: Am (A3, C4, E4), F (F3, A3, C4), C (C3, E3, G3), G (G3, B3, D4)
    progression = [
        [220.0, 261.63, 329.63],
        [174.61, 220.0, 261.63],
        [130.81, 164.81, 196.0],
        [196.0, 246.94, 293.66],
    ]

    chord_interval = int(sr * 3.0)
    idx = 0
    pos = int(sr * 1.0)
    while pos + chord_interval < samples:
        chord = progression[idx % len(progression)]
        chord_wave = generate_piano_chord(chord, sr, duration=3.5)
        clip_len = min(len(chord_wave), samples - pos)
        audio[pos : pos + clip_len] += chord_wave[:clip_len] * 0.6
        pos += chord_interval
        idx += 1

    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.85
    sf.write(str(output_path), audio, sr)


def generate_vocal_track(output_path: Path, duration: float = 35.0, sr: int = 22050):
    """Generates a track with strong vocal formant resonance and vibrato in 300-3400Hz."""
    t = np.linspace(0, duration, int(duration * sr), False)
    vocal = np.zeros_like(t)

    # Singing note progression with 5Hz vibrato
    notes = [440.0, 493.88, 523.25, 587.33, 659.25]
    note_dur = 2.0
    for idx, f0 in enumerate(notes):
        start = int(idx * note_dur * sr)
        if start >= len(t):
            break
        end = min(len(t), int((idx + 1) * note_dur * sr))
        t_sub = t[start:end] - t[start]
        # 5 Hz vibrato
        vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t_sub)
        f_curr = f0 * vibrato
        # Formants at 800Hz, 1200Hz, 2600Hz
        sig = (
            np.sin(2 * np.pi * f_curr * t_sub) * 0.5 +
            np.sin(2 * np.pi * 2 * f_curr * t_sub) * 0.35 +
            np.sin(2 * np.pi * 3 * f_curr * t_sub) * 0.25
        )
        # Envelope
        env = np.sin(np.pi * t_sub / note_dur)
        vocal[start:end] += sig * env

    # Loop notes to fill duration
    repeats = int(np.ceil(duration / (len(notes) * note_dur)))
    full_audio = np.tile(vocal[: int(len(notes) * note_dur * sr)], repeats)[: len(t)]
    full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-6) * 0.85
    sf.write(str(output_path), full_audio, sr)


def generate_rock_track(output_path: Path, duration: float = 35.0, sr: int = 22050):
    """Generates distorted guitar power chord riff with fast tempo and clipping saturation."""
    t = np.linspace(0, duration, int(duration * sr), False)
    # Power chord E5 (82.4Hz + 123.5Hz + 164.8Hz)
    raw = (
        np.sin(2 * np.pi * 82.4 * t) * 1.5 +
        np.sin(2 * np.pi * 123.5 * t) * 1.2 +
        np.sin(2 * np.pi * 164.8 * t) * 1.0
    )
    # Add rhythm pulses at 140 BPM
    bpm = 140
    pulse = np.abs(np.sin(np.pi * (bpm / 60.0) * t)) ** 4
    raw = raw * pulse

    # Heavy overdrive distortion (soft-clipping)
    distorted = np.tanh(raw * 5.0)

    # Add cymbal crash / high frequency grit
    noise = np.random.uniform(-0.15, 0.15, len(t))
    rock_mix = distorted * 0.8 + noise * 0.2
    rock_mix = rock_mix / (np.max(np.abs(rock_mix)) + 1e-6) * 0.9
    sf.write(str(output_path), rock_mix, sr)


def generate_epic_track(output_path: Path, duration: float = 35.0, sr: int = 22050):
    """Generates an orchestral swelling crescendo with soaring dynamic range and sub-bass."""
    t = np.linspace(0, duration, int(duration * sr), False)
    # Build crescendo: quiet start -> massive orchestral climax
    crescendo_env = (t / duration) ** 2.5

    # Sub-bass + Brass low root (55Hz) + Soaring strings (880Hz, 1320Hz)
    sub = np.sin(2 * np.pi * 55.0 * t) * 0.6
    brass = np.sin(2 * np.pi * 110.0 * t) * 0.4 + np.sin(2 * np.pi * 220.0 * t) * 0.3
    strings = np.sin(2 * np.pi * 880.0 * t) * 0.3 + np.sin(2 * np.pi * 1320.0 * t) * 0.2

    epic_wave = (sub + brass + strings) * crescendo_env
    # Add dynamic timpani hit at the climax
    hit_pos = int(sr * 25.0)
    if hit_pos + int(sr * 1.5) < len(t):
        timp = np.sin(2 * np.pi * 45.0 * np.linspace(0, 1.5, int(sr * 1.5))) * np.exp(-np.linspace(0, 1.5, int(sr * 1.5)) * 4)
        epic_wave[hit_pos : hit_pos + len(timp)] += timp * 1.2

    epic_wave = epic_wave / (np.max(np.abs(epic_wave)) + 1e-6) * 0.95
    sf.write(str(output_path), epic_wave, sr)


def generate_all_samples(target_dir: Path | str) -> list[Path]:
    """Generates a complete suite of synthetic music tracks for testing."""
    folder = Path(target_dir)
    folder.mkdir(parents=True, exist_ok=True)

    tracks = [
        ("Synthetic_Beat_Track.wav", generate_beat_track),
        ("Synthetic_Piano_Solo.wav", generate_piano_track),
        ("Synthetic_Vocal_Melody.wav", generate_vocal_track),
        ("Synthetic_Rock_Riff.wav", generate_rock_track),
        ("Synthetic_Epic_Crescendo.wav", generate_epic_track),
    ]

    generated = []
    for filename, gen_fn in tracks:
        out_path = folder / filename
        gen_fn(out_path)
        generated.append(out_path)

    return generated


if __name__ == "__main__":
    test_dir = Path(__file__).parent / "sample_audio"
    print(f"Generating test audio files in: {test_dir}")
    files = generate_all_samples(test_dir)
    for f in files:
        print(f"  Created: {f.name}")
