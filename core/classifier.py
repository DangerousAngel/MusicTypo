"""
Music Classifier Module - Analyzes audio features and metadata to classify tracks into
Rock, Beat, Piano, Melody, Epic, Vocal, Ambient, Classical, etc.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

from core.feature_extractor import AcousticFeatures
from core.metadata_reader import TrackMetadata


# Standardized Target Categories
CATEGORIES = [
    "Rock",
    "Beat",
    "Piano",
    "Melody",
    "Epic",
    "Vocal",
    "Ambient",
    "Classical"
]

GENRE_KEYWORDS = {
    "Rock": ["rock", "metal", "punk", "grunge", "alternative", "hard rock", "guitar", "indie rock"],
    "Beat": ["beat", "beats", "hip hop", "hip-hop", "rap", "edm", "dance", "techno", "house", "trap", "drill", "drum and bass", "dubstep"],
    "Piano": ["piano", "keys", "pianist", "nocturne", "chopin", "beethoven", "acoustic piano"],
    "Melody": ["melody", "melodic", "ballad", "lyrical", "theme", "lullaby", "love song", "acoustic"],
    "Epic": ["epic", "cinematic", "trailer", "soundtrack", "heroic", "dramatic", "score", "blockbuster", "two steps from hell", "zimmer"],
    "Vocal": ["vocal", "vocals", "acapella", "singing", "choir", "choral", "voice", "song", "pop"],
    "Ambient": ["ambient", "chill", "relax", "meditation", "drone", "space", "atmospheric", "new age", "lofi", "lo-fi"],
    "Classical": ["classical", "baroque", "symphony", "concerto", "orchestra", "philharmonic", "sonata", "waltz"]
}


@dataclass
class ClassificationResult:
    file_path: Path
    title: str
    artist: str
    primary_type: str
    confidence: float
    secondary_tags: List[str]
    category_scores: Dict[str, float]
    bpm: float
    total_duration: float
    snippet_offset: float
    metadata: TrackMetadata
    features: AcousticFeatures


class MusicClassifier:
    """
    Combines acoustic DSP features (from the mid-song snippet)
    with track metadata to classify songs.
    """

    def __init__(self, secondary_threshold: float = 0.40):
        self.secondary_threshold = secondary_threshold

    def _match_metadata_keywords(self, meta: TrackMetadata) -> Dict[str, float]:
        """Calculates genre/title prior scores based on embedded tags."""
        text_corpus = f"{meta.title} {meta.artist} {meta.album} {meta.genre}".lower()
        prior_scores = {cat: 0.0 for cat in CATEGORIES}

        for cat, keywords in GENRE_KEYWORDS.items():
            for kw in keywords:
                # Match full words or tokens
                if re.search(r'\b' + re.escape(kw) + r'\b', text_corpus):
                    prior_scores[cat] += 0.35
                    break

        # Normalize tag priors between 0.0 and 0.5 max
        return {cat: min(0.50, score) for cat, score in prior_scores.items()}

    def _compute_dsp_scores(self, feat: AcousticFeatures) -> Dict[str, float]:
        """Calculates acoustic style scores directly from DSP features."""
        scores = {}

        # Non-drum harmonic purity: how melodic/harmonic vs percussive
        harmonic_dom = max(0.0, feat.harmonic_ratio - feat.percussive_ratio)
        percussive_dom = max(0.0, feat.percussive_ratio - feat.harmonic_ratio)

        # 1. ROCK: Heavy distortion/saturation (high flatness & ZCR), fast rhythm, driving energy
        sf_rock = min(1.0, feat.spectral_flatness / 0.02)
        zcr_rock = min(1.0, feat.zero_crossing_rate / 0.06)
        energy_rock = min(1.0, feat.rms_mean / 0.10)
        tempo_rock = 1.0 if (100 <= feat.bpm <= 185) else 0.5
        # Must have noticeable noise/distortion or high ZCR to be rock
        distortion_factor = (sf_rock * 0.6 + zcr_rock * 0.4)
        scores["Rock"] = float(np_clip(
            (0.50 * distortion_factor + 0.25 * energy_rock + 0.25 * feat.percussive_ratio) * tempo_rock
        ))

        # 2. BEAT: Distinct drum transients, high percussive ratio, clear tempo, strong beat envelope
        perc_beat = min(1.0, feat.percussive_ratio / 0.35)
        strength_beat = min(1.0, feat.beat_strength / 1.2)
        onset_beat = min(1.0, feat.onset_rate / 2.5)
        tempo_beat = 1.0 if (70 <= feat.bpm <= 165) else 0.4
        scores["Beat"] = float(np_clip(
            (0.40 * perc_beat + 0.30 * strength_beat + 0.20 * onset_beat + 0.10 * tempo_beat)
        ))

        # 3. PIANO: Clean harmonic decay, sharp percussive hammer strike, high chroma salience, low noise
        harm_piano = feat.harmonic_ratio
        purity_piano = max(0.0, 1.0 - (feat.spectral_flatness / 0.008))
        chroma_piano = min(1.0, feat.chroma_salience / 0.30)
        # Moderate onset rate (piano chords/keystrokes)
        onset_piano = 1.0 if (0.3 <= feat.onset_rate <= 3.5) else 0.4
        centroid_piano = 1.0 if (300 <= feat.spectral_centroid <= 2500) else 0.5
        # Penalize heavy distortion
        piano_score = (0.35 * harm_piano + 0.25 * purity_piano + 0.25 * chroma_piano + 0.15 * centroid_piano) * onset_piano
        if feat.spectral_flatness > 0.015:
            piano_score *= 0.3
        scores["Piano"] = float(np_clip(piano_score))

        # 4. MELODY: Strong harmonic dominance, low distortion, expressive pitch clarity
        harm_melody = min(1.0, feat.harmonic_ratio / 0.70)
        chroma_melody = min(1.0, feat.chroma_salience / 0.25)
        clean_melody = max(0.0, 1.0 - (feat.spectral_flatness / 0.01))
        # High percussive drums reduce pure melody score
        drum_penalty = max(0.2, 1.0 - feat.percussive_ratio)
        scores["Melody"] = float(np_clip(
            (0.45 * harm_melody + 0.35 * chroma_melody + 0.20 * clean_melody) * drum_penalty
        ))

        # 5. EPIC: Massive dynamic crest factor, huge dynamic swelling/crescendo (rms_std), wide spectrum
        crest_epic = min(1.0, max(0.0, (feat.dynamic_crest - 3.0) / 4.0))
        std_epic = min(1.0, feat.rms_std / 0.04)
        rolloff_epic = min(1.0, feat.spectral_rolloff / 3500.0)
        epic_score = (0.45 * crest_epic + 0.35 * std_epic + 0.20 * rolloff_epic)
        scores["Epic"] = float(np_clip(epic_score))

        # 6. VOCAL: Vocal formant band (300Hz-3400Hz) dominance, harmonicity, human voice centroid
        vocal_band = min(1.0, max(0.0, (feat.vocal_band_ratio - 0.45) / 0.35))
        harm_vocal = feat.harmonic_ratio
        centroid_vocal = 1.0 if (600 <= feat.spectral_centroid <= 2800) else 0.5
        vocal_score = (0.50 * vocal_band + 0.30 * harm_vocal + 0.20 * centroid_vocal)
        scores["Vocal"] = float(np_clip(vocal_score))

        # 7. AMBIENT: Low onset rate, low beat transients, smooth harmonic envelope
        if feat.onset_rate > 2.0 or feat.beat_strength > 1.2:
            ambient_score = 0.1
        else:
            low_onset_amb = max(0.0, 1.0 - (feat.onset_rate / 1.0))
            low_beat_amb = max(0.0, 1.0 - (feat.beat_strength / 0.6))
            ambient_score = 0.45 * low_onset_amb + 0.35 * low_beat_amb + 0.20 * feat.harmonic_ratio
        scores["Ambient"] = float(np_clip(ambient_score))

        # 8. CLASSICAL: Acoustic harmony, uncompressed dynamics, clean harmonic spectrum
        clean_class = max(0.0, 1.0 - (feat.spectral_flatness / 0.01))
        dynamics_class = min(1.0, feat.dynamic_crest / 5.0)
        class_score = (0.40 * feat.harmonic_ratio + 0.35 * clean_class + 0.25 * dynamics_class)
        if feat.percussive_ratio > 0.40:
            class_score *= 0.3
        scores["Classical"] = float(np_clip(class_score))

        return scores

    def classify(
        self,
        file_path: Path | str,
        features: AcousticFeatures,
        metadata: TrackMetadata,
        total_duration: float,
        snippet_offset: float
    ) -> ClassificationResult:
        """
        Calculates unified classification scores, determining the primary category
        and relevant secondary tags.
        """
        path = Path(file_path)
        dsp_scores = self._compute_dsp_scores(features)
        meta_priors = self._match_metadata_keywords(metadata)

        # Merge DSP + Metadata
        final_scores = {}
        for cat in CATEGORIES:
            dsp = dsp_scores.get(cat, 0.0)
            prior = meta_priors.get(cat, 0.0)
            # Weighted merge: 75% DSP analysis + 25% Metadata prior (if present)
            score = (dsp * 0.75) + (prior * 0.50)
            final_scores[cat] = round(float(np_clip(score)), 3)

        # Determine Primary Category
        sorted_cats = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        primary_type, confidence = sorted_cats[0]

        # Determine Secondary Tags (other categories above threshold)
        secondary_tags = [
            cat for cat, score in sorted_cats[1:]
            if score >= self.secondary_threshold and score >= (confidence * 0.70)
        ]

        # Use metadata title/artist if clean
        title = metadata.title or path.stem
        artist = metadata.artist or "Unknown Artist"

        return ClassificationResult(
            file_path=path,
            title=title,
            artist=artist,
            primary_type=primary_type,
            confidence=confidence,
            secondary_tags=secondary_tags,
            category_scores=final_scores,
            bpm=features.bpm,
            total_duration=total_duration,
            snippet_offset=snippet_offset,
            metadata=metadata,
            features=features
        )


def np_clip(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, val))
