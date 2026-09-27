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
    
    # New fields
    mode_major_score: float
    detected_key: str
    spectral_contrast_mean: float
    mfcc_mean: list[float]
    spectral_bandwidth: float
    low_freq_ratio: float
    mid_freq_ratio: float
    high_freq_ratio: float
    tempo_stability: float
    spectral_flux: float
    vocal_presence_score: float
    offbeat_ratio: float
    detected_mode: str = "Major"
    detected_mode_type: str = "Major"
    mode_confidence: float = 0.5


def extract_features(audio: np.ndarray, sample_rate: int = 22050) -> AcousticFeatures:
    """
    Extracts acoustic features from a mid-song audio excerpt.
    Optimized for fast computation (~100-250ms for a 20s clip).
    """
    # Guard against silence or empty clips
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-5:
        return AcousticFeatures(
            bpm=0.0, beat_strength=0.0, percussive_ratio=0.0, harmonic_ratio=0.0,
            spectral_centroid=0.0, spectral_flatness=0.0, spectral_rolloff=0.0,
            rms_mean=0.0, rms_std=0.0, dynamic_crest=0.0, vocal_band_ratio=0.0,
            chroma_salience=0.0, onset_rate=0.0, zero_crossing_rate=0.0,
            mode_major_score=0.0, detected_key="C Major", spectral_contrast_mean=0.0,
            mfcc_mean=[0.0]*13, spectral_bandwidth=0.0, low_freq_ratio=0.0,
            mid_freq_ratio=0.0, high_freq_ratio=0.0, tempo_stability=0.0,
            spectral_flux=0.0, vocal_presence_score=0.0, offbeat_ratio=0.0,
            detected_mode="Major", detected_mode_type="Major", mode_confidence=0.0
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
        y_harm, y_perc = audio, audio
        percussive_ratio = 0.5
        harmonic_ratio = 0.5

    # 2. Beat, Tempo, and Onsets
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        beat_strength = float(np.mean(onset_env))
        
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate, onset_envelope=onset_env)
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
        
        # Tempo stability
        if len(beat_frames) > 2:
            beat_intervals = np.diff(beat_frames)
            tempo_stability = float(np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-9))
        else:
            tempo_stability = 0.0
            
        # Offbeat ratio
        if len(beat_frames) > 1:
            on_beat_energy = 0.0
            off_beat_energy = 0.0
            for i in range(len(beat_frames) - 1):
                start = beat_frames[i]
                end = beat_frames[i+1]
                mid = (start + end) // 2
                on_beat_energy += np.sum(onset_env[start:start + (end-start)//4])
                off_beat_energy += np.sum(onset_env[mid - (end-start)//8 : mid + (end-start)//8])
            total_beat_energy = on_beat_energy + off_beat_energy + 1e-9
            offbeat_ratio = float(off_beat_energy / total_beat_energy)
        else:
            offbeat_ratio = 0.0
            
    except Exception:
        bpm = 0.0
        beat_strength = 0.0
        onset_rate = 0.0
        tempo_stability = 0.0
        offbeat_ratio = 0.0

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
        
    try:
        sbw = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        spectral_bandwidth = float(np.mean(sbw))
    except Exception:
        spectral_bandwidth = 0.0
        
    try:
        sc_mean = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spectral_contrast_mean = float(np.mean(sc_mean))
    except Exception:
        spectral_contrast_mean = 0.0

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

    # 5. Vocal Formant Band Energy (300 Hz - 3400 Hz) & Freq ratios
    try:
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        
        vocal_mask = (freqs >= 300) & (freqs <= 3400)
        vocal_energy = np.sum(S[vocal_mask, :] ** 2)
        total_spec_energy = np.sum(S ** 2) + 1e-9
        vocal_band_ratio = float(vocal_energy / total_spec_energy)
        
        low_mask = freqs < 300
        mid_mask = (freqs >= 300) & (freqs <= 2000)
        high_mask = freqs > 5000
        
        low_freq_ratio = float(np.sum(S[low_mask, :] ** 2) / total_spec_energy)
        mid_freq_ratio = float(np.sum(S[mid_mask, :] ** 2) / total_spec_energy)
        high_freq_ratio = float(np.sum(S[high_mask, :] ** 2) / total_spec_energy)
        
        # Vocal presence score
        S_harm = np.abs(librosa.stft(y_harm, n_fft=2048, hop_length=512))
        vocal_harm_energy = np.sum(S_harm[vocal_mask, :] ** 2)
        total_harm_energy = np.sum(S_harm ** 2) + 1e-9
        vocal_presence_score = float(vocal_harm_energy / total_harm_energy)
        
        # Spectral flux
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        spectral_flux = float(np.mean(np.maximum(0, np.diff(S_db, axis=1))))
        
    except Exception:
        vocal_band_ratio = 0.0
        low_freq_ratio = 0.0
        mid_freq_ratio = 0.0
        high_freq_ratio = 0.0
        vocal_presence_score = 0.0
        spectral_flux = 0.0

    # 6. Chroma Salience & Modal Key Detection
    try:
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_salience = float(np.mean(np.max(chroma, axis=0) - np.mean(chroma, axis=0)))

        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean_centered = chroma_mean - np.mean(chroma_mean)
        chroma_std = np.std(chroma_mean)
        if chroma_std < 1e-6:
            chroma_mean_centered = np.ones(12)

        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Krumhansl-Schmuckler profiles & modal variants
        maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        dorian_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 4.30, 3.34, 2.10])
        # Phrygian / Hijaz (Flamenco / Middle Eastern / Arabic Maqam / Oud / Metal)
        phrygian_profile = np.array([6.33, 4.80, 2.50, 4.80, 3.50, 4.20, 2.30, 4.80, 3.80, 2.50, 3.50, 2.20])
        lydian_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 2.52, 4.20, 5.19, 2.39, 3.66, 2.29, 2.88])
        mixo_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 3.80, 2.20])
        harm_min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 2.00, 4.50])

        all_profiles = {
            "Major": (maj_profile, "Major"),
            "Minor": (min_profile, "Minor"),
            "Dorian": (dorian_profile, "Minor"),
            "Phrygian": (phrygian_profile, "Minor"),
            "Lydian": (lydian_profile, "Major"),
            "Mixolydian": (mixo_profile, "Major"),
            "Harmonic Minor": (harm_min_profile, "Minor"),
        }

        # 1. Base Major vs Minor calculation for mode_major_score
        maj_centered = maj_profile - np.mean(maj_profile)
        min_centered = min_profile - np.mean(min_profile)
        maj_corrs = [np.corrcoef(chroma_mean_centered, np.roll(maj_centered, i))[0, 1] for i in range(12)]
        min_corrs = [np.corrcoef(chroma_mean_centered, np.roll(min_centered, i))[0, 1] for i in range(12)]

        best_maj_idx = int(np.nanargmax(maj_corrs))
        best_min_idx = int(np.nanargmax(min_corrs))
        best_maj_corr = float(maj_corrs[best_maj_idx]) if not np.isnan(maj_corrs[best_maj_idx]) else 0.0
        best_min_corr = float(min_corrs[best_min_idx]) if not np.isnan(min_corrs[best_min_idx]) else 0.0

        mode_major_score = float(np.clip(best_maj_corr - best_min_corr, -1.0, 1.0))

        # 2. Detailed Modal classification
        best_overall_corr = -2.0
        best_mode_name = "Major" if best_maj_corr >= best_min_corr else "Minor"
        best_mode_type = "Major" if best_maj_corr >= best_min_corr else "Minor"
        best_root_idx = best_maj_idx if best_maj_corr >= best_min_corr else best_min_idx

        for mode_name, (prof, m_type) in all_profiles.items():
            p_centered = prof - np.mean(prof)
            for i in range(12):
                c = np.corrcoef(chroma_mean_centered, np.roll(p_centered, i))[0, 1]
                if not np.isnan(c) and c > best_overall_corr:
                    best_overall_corr = float(c)
                    best_mode_name = mode_name
                    best_mode_type = m_type
                    best_root_idx = i

        detected_mode = best_mode_name
        detected_mode_type = best_mode_type
        mode_confidence = float(np.clip((best_overall_corr + 1.0) / 2.0, 0.0, 1.0))
        root_note = pitch_classes[best_root_idx]
        detected_key = f"{root_note} {best_mode_name}"

    except Exception:
        chroma_salience = 0.0
        mode_major_score = 0.0
        detected_key = "C Major"
        detected_mode = "Major"
        detected_mode_type = "Major"
        mode_confidence = 0.5

    # 7. Zero Crossing Rate (noise / distortion / percussion edge)
    try:
        zcr = librosa.feature.zero_crossing_rate(audio)
        zero_crossing_rate = float(np.mean(zcr))
    except Exception:
        zero_crossing_rate = 0.0
        
    # 8. MFCC
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
    except Exception:
        mfcc_mean = [0.0] * 13

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
        zero_crossing_rate=zero_crossing_rate,
        mode_major_score=mode_major_score,
        detected_key=detected_key,
        spectral_contrast_mean=spectral_contrast_mean,
        mfcc_mean=mfcc_mean,
        spectral_bandwidth=spectral_bandwidth,
        low_freq_ratio=low_freq_ratio,
        mid_freq_ratio=mid_freq_ratio,
        high_freq_ratio=high_freq_ratio,
        tempo_stability=tempo_stability,
        spectral_flux=spectral_flux,
        vocal_presence_score=vocal_presence_score,
        offbeat_ratio=offbeat_ratio,
        detected_mode=detected_mode,
        detected_mode_type=detected_mode_type,
        mode_confidence=mode_confidence
    )
