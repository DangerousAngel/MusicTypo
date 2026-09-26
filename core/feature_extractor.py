"""
Feature Extractor Module - Digital Signal Processing (DSP) on mid-song audio excerpts.
Computes acoustic indicators for genre, tempo, timbre, dynamics, and instrumentation.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from dataclasses import dataclass
import numpy as np
import librosa


@dataclass
class AcousticFeatures:
    bpm: float
    beat_strength: float
    percussive_ratio: float
    harmonic_ratio: float
    spectral_centroid: float
    spectral_flatness: float
    spectral_rolloff: float
    rms_mean: float
    rms_std: float
    dynamic_crest: float
    vocal_band_ratio: float
    chroma_salience: float
    onset_rate: float
    zero_crossing_rate: float


def extract_features(audio: np.ndarray, sample_rate: int = 22050) -> AcousticFeatures:
    """
    Extracts acoustic features from a mid-song audio excerpt.
    Optimized for fast computation (~100-250ms for a 20s clip).
    """
    # Guard against silence or empty clips
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-5:
        return AcousticFeatures(
            bpm=0.0,
            beat_strength=0.0,
            percussive_ratio=0.0,
            harmonic_ratio=0.0,
            spectral_centroid=0.0,
            spectral_flatness=0.0,
            spectral_rolloff=0.0,
            rms_mean=0.0,
            rms_std=0.0,
            dynamic_crest=0.0,
            vocal_band_ratio=0.0,
            chroma_salience=0.0,
            onset_rate=0.0,
            zero_crossing_rate=0.0
        )

    # 1. Harmonic-Percussive Separation (HPSS)
    try:
        y_harm, y_perc = librosa.effects.hpss(audio)
        harm_energy = float(np.sum(y_harm ** 2))
        perc_energy = float(np.sum(y_perc ** 2))
        total_energy = harm_energy + perc_energy + 1e-9
        percussive_ratio = perc_energy / total_energy
        harmonic_ratio = harm_energy / total_energy
    except Exception:
        percussive_ratio = 0.5
        harmonic_ratio = 0.5

    # 2. Beat, Tempo, and Onsets
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        beat_strength = float(np.mean(onset_env))
        
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate, onset_envelope=onset_env)
        if isinstance(tempo, (list, np.ndarray)):
            bpm = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            bpm = float(tempo)
        bpm = round(bpm, 1)

        # Onset rate (transients per second)
        duration_sec = len(audio) / sample_rate
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
        )
        onset_rate = float(len(peaks)) / max(1.0, duration_sec)
    except Exception:
        bpm = 0.0
        beat_strength = 0.0
        onset_rate = 0.0

    # 3. Spectral Features
    try:
        sc = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_centroid = float(np.mean(sc))
    except Exception:
        spectral_centroid = 0.0

    try:
        sf = librosa.feature.spectral_flatness(y=audio)
        spectral_flatness = float(np.mean(sf))
    except Exception:
        spectral_flatness = 0.0

    try:
        sro = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.85)
        spectral_rolloff = float(np.mean(sro))
    except Exception:
        spectral_rolloff = 0.0

    # 4. Dynamics & RMS
    try:
        rms_series = librosa.feature.rms(y=audio)[0]
        rms_mean = float(np.mean(rms_series))
        rms_std = float(np.std(rms_series))
        peak_amp = float(np.max(np.abs(audio)))
        dynamic_crest = peak_amp / (rms_mean + 1e-6)
    except Exception:
        rms_mean = 0.0
        rms_std = 0.0
        dynamic_crest = 0.0

    # 5. Vocal Formant Band Energy (300 Hz - 3400 Hz)
    try:
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        vocal_mask = (freqs >= 300) & (freqs <= 3400)
        vocal_energy = np.sum(S[vocal_mask, :] ** 2)
        total_spec_energy = np.sum(S ** 2) + 1e-9
        vocal_band_ratio = float(vocal_energy / total_spec_energy)
    except Exception:
        vocal_band_ratio = 0.0

    # 6. Chroma Salience (musical pitch clarity)
    try:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_salience = float(np.mean(np.max(chroma, axis=0) - np.mean(chroma, axis=0)))
    except Exception:
        chroma_salience = 0.0

    # 7. Zero Crossing Rate (noise / distortion / percussion edge)
    try:
        zcr = librosa.feature.zero_crossing_rate(audio)
        zero_crossing_rate = float(np.mean(zcr))
    except Exception:
        zero_crossing_rate = 0.0

    return AcousticFeatures(
        bpm=bpm,
        beat_strength=beat_strength,
        percussive_ratio=percussive_ratio,
        harmonic_ratio=harmonic_ratio,
        spectral_centroid=spectral_centroid,
        spectral_flatness=spectral_flatness,
        spectral_rolloff=spectral_rolloff,
        rms_mean=rms_mean,
        rms_std=rms_std,
        dynamic_crest=dynamic_crest,
        vocal_band_ratio=vocal_band_ratio,
        chroma_salience=chroma_salience,
        onset_rate=onset_rate,
        zero_crossing_rate=zero_crossing_rate
    )
