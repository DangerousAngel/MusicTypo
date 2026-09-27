"""
Music Classifier Module - Analyzes audio features and metadata to classify tracks into
various genres, instruments, and styles (28 distinct categories).
Supports full multi-modal Key/Mode detection and multilingual Vocal Language detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

from core.feature_extractor import AcousticFeatures
from core.metadata_reader import TrackMetadata


# Standardized Target Categories (Expanded 28 categories)
CATEGORIES = [
    "Rock", "Metal", "Beat", "Hip-Hop", "Trap", "Electronic",
    "House-Techno", "Synthwave", "Lo-Fi", "Piano", "Acoustic",
    "Melody", "Epic", "Classical", "Vocal", "Choral-Opera",
    "Ambient", "Jazz", "Blues", "Soul-Funk", "R&B",
    "Oud", "Reggae", "Latin", "Pop", "Country", "Punk-Alternative", "World-Folk"
]

GENRE_KEYWORDS = {
    "Rock": [
        "rock", "hard rock", "classic rock", "grunge", "alternative rock", "guitar rock",
        "indie rock", "stoner rock", "garage rock", "psychedelic rock"
    ],
    "Metal": [
        "metal", "heavy metal", "death metal", "black metal", "thrash", "metalcore",
        "doom metal", "power metal", "djent", "nu metal", "sludge"
    ],
    "Beat": [
        "beat", "beats", "drum and bass", "dnb", "breakbeat", "drumstep", "jungle",
        "percussion", "drum solo", "drums"
    ],
    "Hip-Hop": [
        "hip hop", "hip-hop", "hiphop", "rap", "boom bap", "grime", "conscious rap",
        "underground rap", "freestyle", "mc"
    ],
    "Trap": [
        "trap", "drill", "uk drill", "ny drill", "808", "808s", "sub bass", "trap beat",
        "trap rap", "chiraq", "metro boomin", "travis scott", "future", "carti"
    ],
    "Electronic": [
        "electronic", "edm", "electro", "electronica", "dance", "club", "rave", "dj",
        "remix", "future bass", "dubstep", "glitch", "idm", "breakcore", "electropop"
    ],
    "House-Techno": [
        "house", "techno", "deep house", "tech house", "minimal techno", "acid house",
        "progressive house", "electro house", "four on the floor", "club mix", "ibiza",
        "berlin techno", "detroit techno"
    ],
    "Synthwave": [
        "synthwave", "retrowave", "outrun", "darksynth", "chillwave", "vaporwave",
        "cyberpunk", "80s synth", "analog synth", "kavinsky", "synth pop"
    ],
    "Lo-Fi": [
        "lofi", "lo-fi", "chillhop", "lo fi", "study beats", "sleep beats", "dusty beats",
        "vinyl crackle", "relaxing beats", "jazzhop", "chilledcow", "lofi hip hop"
    ],
    "Piano": [
        "piano", "keys", "pianist", "nocturne", "chopin", "beethoven", "acoustic piano",
        "grand piano", "piano solo", "solo piano", "liszt", "debussy"
    ],
    "Acoustic": [
        "acoustic", "unplugged", "fingerstyle", "acoustic guitar", "folk guitar",
        "campfire", "stripped", "acoustic version", "live acoustic", "ukulele", "harp"
    ],
    "Melody": [
        "melody", "melodic", "ballad", "lyrical", "theme", "lullaby", "love song",
        "sweet", "sentimental", "romantic", "slow song", "emotional", "serenade"
    ],
    "Epic": [
        "epic", "cinematic", "trailer", "soundtrack", "heroic", "dramatic", "score",
        "blockbuster", "two steps from hell", "zimmer", "orchestral epic", "anthem", "battle"
    ],
    "Classical": [
        "classical", "baroque", "symphony", "concerto", "orchestra", "philharmonic",
        "sonata", "waltz", "chamber", "mozart", "bach", "brahms", "tchaikovsky", "vivaldi"
    ],
    "Vocal": [
        "vocal", "vocals", "acapella", "a cappella", "singing", "voice", "song",
        "solo vocal", "lead vocal", "vocalist"
    ],
    "Choral-Opera": [
        "choir", "choral", "opera", "operatic", "soprano", "tenor", "baritone", "aria",
        "requiem", "gregorian", "sacred", "liturgical", "vocal ensemble", "polyphony", "hymn"
    ],
    "Ambient": [
        "ambient", "chill", "relax", "meditation", "drone", "space", "atmospheric",
        "new age", "soundscape", "meditative", "healing", "calm", "sleep", "binaural"
    ],
    "Jazz": [
        "jazz", "swing", "bebop", "bossa nova", "fusion", "smooth jazz", "big band",
        "miles davis", "coltrane", "jazz trio", "cool jazz", "hard bop"
    ],
    "Blues": [
        "blues", "delta blues", "chicago blues", "electric blues", "12-bar", "bb king",
        "muddy waters", "clapton", "stevie ray", "slide guitar", "harmonica blues"
    ],
    "Soul-Funk": [
        "soul", "funk", "motown", "groove", "slap bass", "james brown", "aretha",
        "stevie wonder", "earth wind and fire", "funky", "horn section", "disco", "disco funk"
    ],
    "R&B": [
        "r&b", "rnb", "contemporary r&b", "neo soul", "slow jam", "urban", "quiet storm"
    ],
    "Oud": [
        "oud", "عود", "oriental", "arabic", "maqam", "middle eastern", "turkish", "persian",
        "ney", "qanun", "kanun", "darbuka", "riq", "dabke", "tarab", "taksim", "taqsim",
        "تقاسيم", "طرب", "مقام", "شرقي", "بياتي", "حجاز", "راست", "سيكا", "صبا", "عجم",
        "نهوند", "كرد", "كلثوم", "فيروز", "عبد الحليم", "محمد عبده", "فريد الأطرش",
        "خليجي", "شامي", "عراقي", "مغربي", "mahraganat", "shaabi", "khaliji", "tarabish"
    ],
    "Reggae": [
        "reggae", "ska", "dub", "dancehall", "rocksteady", "roots reggae", "bob marley", "rastafari"
    ],
    "Latin": [
        "latin", "latino", "reggaeton", "salsa", "bachata", "cumbia", "merengue", "flamenco",
        "tango", "dembow", "bad bunny", "rosalia", "urbano latino", "latin pop"
    ],
    "Pop": [
        "pop", "k-pop", "j-pop", "indie pop", "synth pop", "electro pop", "top 40",
        "chart", "radio hit", "mainstream"
    ],
    "Country": [
        "country", "bluegrass", "nashville", "western", "americana", "pedal steel",
        "honky tonk", "twang", "banjo", "cowboy", "southern rock"
    ],
    "Punk-Alternative": [
        "punk", "pop punk", "hardcore punk", "grunge", "post-punk", "emo", "skate punk",
        "riot", "anarchy", "nirvana", "green day"
    ],
    "World-Folk": [
        "world", "folk", "celtic", "irish", "sitar", "traditional", "ethnic", "tribal",
        "indian classical", "balkan", "flamenco folk", "bouzouki", "bagpipes", "afrobeat"
    ]
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
    mode: str
    key: str
    vocal_language: str


class MusicClassifier:
    """
    Combines acoustic DSP features (from the mid-song snippet)
    with track metadata to classify songs.
    """

    def __init__(self, secondary_threshold: float = 0.40):
        self.secondary_threshold = secondary_threshold

    def _match_metadata_keywords(self, meta: TrackMetadata) -> Dict[str, float]:
        """Calculates genre/title prior scores based on embedded tags."""
        text_corpus = f"{meta.title} {meta.artist} {meta.album} {meta.genre} {meta.comment or ''}".lower()
        clean_corpus = re.sub(r'[_\\-]', ' ', text_corpus)
        prior_scores = {cat: 0.0 for cat in CATEGORIES}

        for cat, keywords in GENRE_KEYWORDS.items():
            for kw in keywords:
                # Match full words or tokens
                if re.search(r'\b' + re.escape(kw) + r'\b', clean_corpus):
                    prior_scores[cat] += 0.35
                    break

        # Normalize tag priors between 0.0 and 0.50 max
        return {cat: min(0.50, score) for cat, score in prior_scores.items()}

    def _detect_vocal_language(self, feat: AcousticFeatures, meta: TrackMetadata, primary_type: str = "") -> str:
        """
        High-precision vocal presence & language identification:
        1. Checks acoustic vocal presence score & vocal formant band.
        2. Inspects embedded metadata tags (TLAN, language, lyrics, comments).
        3. Detects Unicode scripts (Arabic, Cyrillic, CJK, Devanagari, Greek, Hebrew, etc.).
        4. Analyzes title, artist, album, genre, and lyrics keywords across multiple languages.
        """
        vocal_score = getattr(feat, 'vocal_presence_score', 0.0)
        vocal_band = getattr(feat, 'vocal_band_ratio', 0.0)
        has_lyrics = bool(meta.lyrics and len(meta.lyrics.strip()) > 10)

        # Check explicit ID3 language tag first if present
        if meta.language:
            lang_code = meta.language.strip().lower()
            lang_map = {
                "ara": "Arabic", "ar": "Arabic", "arabic": "Arabic",
                "eng": "English", "en": "English", "english": "English",
                "spa": "Spanish", "es": "Spanish", "spanish": "Spanish",
                "fra": "French", "fre": "French", "fr": "French", "french": "French",
                "deu": "German", "ger": "German", "de": "German", "german": "German",
                "ita": "Italian", "it": "Italian", "italian": "Italian",
                "por": "Portuguese", "pt": "Portuguese", "portuguese": "Portuguese",
                "rus": "Russian", "ru": "Russian", "russian": "Russian",
                "tur": "Turkish", "tr": "Turkish", "turkish": "Turkish",
                "fas": "Persian", "per": "Persian", "fa": "Persian", "persian": "Persian", "farsi": "Persian",
                "kor": "Korean", "ko": "Korean", "korean": "Korean",
                "jpn": "Japanese", "ja": "Japanese", "japanese": "Japanese",
                "zho": "Chinese", "chi": "Chinese", "zh": "Chinese", "chinese": "Chinese",
                "hin": "Hindi", "hi": "Hindi", "hindi": "Hindi",
                "ell": "Greek", "el": "Greek", "greek": "Greek",
                "heb": "Hebrew", "he": "Hebrew", "hebrew": "Hebrew"
            }
            for code, name in lang_map.items():
                if lang_code == code or lang_code.startswith(f"{code}-") or lang_code.startswith(f"{code}_"):
                    if vocal_score >= 0.25 or has_lyrics:
                        return name

        # If not enough vocal evidence and no lyrics, mark Instrumental
        if vocal_score < 0.28 and vocal_band < 0.38 and not has_lyrics:
            return "Instrumental"

        corpus_raw = f"{meta.title} {meta.artist} {meta.album} {meta.genre} {meta.lyrics or ''} {meta.comment or ''}".lower()
        corpus = re.sub(r'[_\\-]', ' ', corpus_raw)

        # Unicode Script Detection
        # Arabic script (\u0600-\u06FF, \u0750-\u077F, \u08A0-\u08FF)
        if re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', corpus):
            if any(c in corpus for c in ['گ', 'چ', 'پ', 'ژ']):
                return 'Persian'
            if any(c in corpus for c in ['ٹ', 'ڈ', 'ڑ', 'ں', 'ے']):
                return 'Urdu'
            return 'Arabic'

        # Cyrillic script (\u0400-\u04FF)
        if re.search(r'[\u0400-\u04FF]', corpus):
            return 'Russian'

        # Hangul Korean script (\uAC00-\uD7AF, \u1100-\u11FF)
        if re.search(r'[\uAC00-\uD7AF\u1100-\u11FF]', corpus):
            return 'Korean'

        # Japanese Hiragana/Katakana (\u3040-\u309F, \u30A0-\u30FF)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', corpus):
            return 'Japanese'

        # Hanzi Chinese (\u4E00-\u9FFF)
        if re.search(r'[\u4E00-\u9FFF]', corpus):
            return 'Chinese'

        # Devanagari Hindi script (\u0900-\u097F)
        if re.search(r'[\u0900-\u097F]', corpus):
            return 'Hindi'

        # Greek (\u0370-\u03FF)
        if re.search(r'[\u0370-\u03FF]', corpus):
            return 'Greek'

        # Hebrew (\u0590-\u05FF)
        if re.search(r'[\u0590-\u05FF]', corpus):
            return 'Hebrew'

        # Lexical keyword and token matching
        arabic_keywords = [
            "عربي", "arabic", "عود", "مقام", "طرب", "شرقي", "habibi", "fairuz",
            "om kalthoum", "amr diab", "cheb", "rai", "dabke", "khaliji", "tarab", "mahraganat",
            "aghani", "yalla"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in arabic_keywords):
            return 'Arabic'

        spanish_keywords = [
            "español", "espanol", "spanish", "latino", "reggaeton", "bachata", "cancion", "canción",
            "corazon", "corazón", "amor", "vida", "noche", "despacito", "ella", "para ti",
            "te quiero", "por favor", "hermano", "cumbia", "salsa", "flamenco"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in spanish_keywords):
            return 'Spanish'

        french_keywords = [
            "français", "francais", "french", "chanson", "amour", "la vie", "nuit",
            "je t'aime", "avec", "dans", "pour", "cœur", "femme", "monde", "café",
            "stromae", "piaf", "indila"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in french_keywords):
            return 'French'

        german_keywords = [
            "deutsch", "german", "liebe", "nacht", "herz", "mond", "zeit", "immer",
            "leben", "rammstein", "schlager", "deutschrap"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in german_keywords):
            return 'German'

        italian_keywords = [
            "italiano", "italian", "canzone", "amore", "notte", "bella", "cuore",
            "vita", "senza", "sanremo", "bocelli", "ciao"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in italian_keywords):
            return 'Italian'

        portuguese_keywords = [
            "português", "portugues", "portuguese", "coração", "você", "canção",
            "bossa nova", "samba", "fado", "sertanejo", "obrigado"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in portuguese_keywords):
            return 'Portuguese'

        turkish_keywords = [
            "türkçe", "turkce", "turkish", "türk", "turk", "aşk", "ask", "gece",
            "şarkı", "sarki", "sevdim", "tarkan"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in turkish_keywords):
            return 'Turkish'

        persian_keywords = [
            "persian", "farsi", "iranian", "shajarian", "googoosh"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in persian_keywords):
            return 'Persian'

        korean_keywords = [
            "korean", "k-pop", "kpop", "bts", "blackpink", "kdrama", "sarang", "oppa"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in korean_keywords):
            return 'Korean'

        japanese_keywords = [
            "japanese", "j-pop", "jpop", "anime", "vocaloid", "hatsune miku", "yoasobi",
            "watashi", "anata", "kimi", "ost"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in japanese_keywords):
            return 'Japanese'

        hindi_keywords = [
            "hindi", "bollywood", "desi", "arijit", "pyar", "ishq", "dil"
        ]
        if any(re.search(r'\b' + re.escape(w) + r'\b', corpus) for w in hindi_keywords):
            return 'Hindi'

        # Category contextual cues
        if primary_type == "Oud":
            return 'Arabic'

        if primary_type == "Latin":
            return 'Spanish'

        # Default to English if vocal presence is recognized
        if vocal_score >= 0.28 or vocal_band >= 0.38 or has_lyrics:
            return 'English'

        return 'Instrumental'

    def _compute_dsp_scores(self, feat: AcousticFeatures) -> Dict[str, float]:
        """Calculates acoustic style scores directly from DSP features."""
        scores = {}

        # 1. ROCK
        sf_rock = min(1.0, feat.spectral_flatness / 0.02)
        zcr_rock = min(1.0, feat.zero_crossing_rate / 0.06)
        energy_rock = min(1.0, feat.rms_mean / 0.10)
        tempo_rock = 1.0 if (100 <= feat.bpm <= 185) else 0.5
        distortion_factor = (sf_rock * 0.6 + zcr_rock * 0.4)
        scores["Rock"] = float(np_clip(
            (0.50 * distortion_factor + 0.25 * energy_rock + 0.25 * feat.percussive_ratio) * tempo_rock
        ))

        # 2. METAL
        sf_metal = min(1.0, feat.spectral_flatness / 0.03)
        zcr_metal = min(1.0, feat.zero_crossing_rate / 0.08)
        energy_metal = min(1.0, feat.rms_mean / 0.15)
        tempo_metal = 1.0 if (120 <= feat.bpm <= 220) else 0.5
        metal_dist = (sf_metal * 0.5 + zcr_metal * 0.5)
        metal_score = (0.40 * metal_dist + 0.40 * energy_metal + 0.20 * feat.percussive_ratio) * tempo_metal
        if feat.spectral_flatness <= 0.01:
            metal_score *= 0.2
        scores["Metal"] = float(np_clip(metal_score))

        # 3. BEAT
        perc_beat = min(1.0, feat.percussive_ratio / 0.35)
        strength_beat = min(1.0, feat.beat_strength / 1.2)
        onset_beat = min(1.0, feat.onset_rate / 2.5)
        tempo_beat = 1.0 if (70 <= feat.bpm <= 165) else 0.4
        scores["Beat"] = float(np_clip(
            (0.40 * perc_beat + 0.30 * strength_beat + 0.20 * onset_beat + 0.10 * tempo_beat)
        ))

        # 4. HIP-HOP
        bass_hh = min(1.0, getattr(feat, 'low_freq_ratio', 0.0) / 0.3)
        perc_hh = min(1.0, feat.percussive_ratio / 0.3)
        tempo_hh = 1.0 if (60 <= feat.bpm <= 110) else 0.5
        vocal_hh = min(1.0, getattr(feat, 'vocal_presence_score', 0.0))
        stab_hh = max(0.0, 1.0 - getattr(feat, 'tempo_stability', 1.0))
        scores["Hip-Hop"] = float(np_clip(
            (0.35 * bass_hh + 0.20 * perc_hh + 0.20 * vocal_hh + 0.15 * stab_hh) * tempo_hh
        ))

        # 5. TRAP
        bass_trap = min(1.0, getattr(feat, 'low_freq_ratio', 0.0) / 0.35)
        onset_trap = min(1.0, feat.onset_rate / 3.0)
        tempo_trap = 1.0 if (125 <= feat.bpm <= 165 or 60 <= feat.bpm <= 85) else 0.5
        flux_trap = min(1.0, getattr(feat, 'spectral_flux', 0.0) / 1.4)
        scores["Trap"] = float(np_clip(
            (0.40 * bass_trap + 0.30 * onset_trap + 0.20 * flux_trap + 0.10 * feat.percussive_ratio) * tempo_trap
        ))

        # 6. ELECTRONIC
        stab_elec = max(0.0, 1.0 - getattr(feat, 'tempo_stability', 1.0))
        flux_elec = min(1.0, getattr(feat, 'spectral_flux', 0.0) / 1.5)
        high_elec = min(1.0, getattr(feat, 'high_freq_ratio', 0.0) / 0.2)
        tempo_elec = 1.0 if (110 <= feat.bpm <= 180) else 0.5
        contrast_elec = max(0.0, 1.0 - (getattr(feat, 'spectral_contrast_mean', 0.0) / 20.0))
        scores["Electronic"] = float(np_clip(
            (0.30 * stab_elec + 0.25 * flux_elec + 0.25 * high_elec + 0.20 * contrast_elec) * tempo_elec
        ))

        # 7. HOUSE-TECHNO
        tempo_ht = 1.0 if (118 <= feat.bpm <= 135) else 0.4
        stab_ht = max(0.0, 1.0 - getattr(feat, 'tempo_stability', 1.0) * 1.5)
        perc_ht = min(1.0, feat.percussive_ratio / 0.35)
        beat_ht = min(1.0, feat.beat_strength / 1.2)
        scores["House-Techno"] = float(np_clip(
            (0.35 * beat_ht + 0.30 * stab_ht + 0.25 * perc_ht + 0.10 * (1.0 - getattr(feat, 'offbeat_ratio', 0.0))) * tempo_ht
        ))

        # 8. SYNTHWAVE
        tempo_sw = 1.0 if (95 <= feat.bpm <= 135) else 0.5
        harm_sw = feat.harmonic_ratio
        high_sw = min(1.0, getattr(feat, 'high_freq_ratio', 0.0) / 0.22)
        flux_sw = min(1.0, getattr(feat, 'spectral_flux', 0.0) / 1.2)
        scores["Synthwave"] = float(np_clip(
            (0.35 * harm_sw + 0.25 * high_sw + 0.25 * flux_sw + 0.15 * min(1.0, feat.onset_rate / 2.5)) * tempo_sw
        ))

        # 9. LO-FI
        tempo_lofi = 1.0 if (65 <= feat.bpm <= 92) else 0.4
        warm_low_high = max(0.0, 1.0 - (getattr(feat, 'high_freq_ratio', 0.0) / 0.12))
        harm_lofi = min(1.0, feat.harmonic_ratio / 0.6)
        mid_lofi = min(1.0, getattr(feat, 'mid_freq_ratio', 0.0) / 0.45)
        scores["Lo-Fi"] = float(np_clip(
            (0.35 * warm_low_high + 0.30 * harm_lofi + 0.25 * mid_lofi + 0.10 * feat.percussive_ratio) * tempo_lofi
        ))

        # 10. PIANO
        harm_piano = feat.harmonic_ratio
        purity_piano = max(0.0, 1.0 - (feat.spectral_flatness / 0.008))
        chroma_piano = min(1.0, feat.chroma_salience / 0.30)
        onset_piano = 1.0 if (0.3 <= feat.onset_rate <= 3.5) else 0.4
        centroid_piano = 1.0 if (300 <= feat.spectral_centroid <= 2500) else 0.5
        piano_score = (0.35 * harm_piano + 0.25 * purity_piano + 0.25 * chroma_piano + 0.15 * centroid_piano) * onset_piano
        if feat.spectral_flatness > 0.015:
            piano_score *= 0.3
        scores["Piano"] = float(np_clip(piano_score))

        # 11. ACOUSTIC
        clean_ac = max(0.0, 1.0 - (feat.spectral_flatness / 0.012))
        mid_ac = min(1.0, getattr(feat, 'mid_freq_ratio', 0.0) / 0.5)
        dyn_ac = min(1.0, feat.dynamic_crest / 4.5)
        harm_ac = feat.harmonic_ratio
        scores["Acoustic"] = float(np_clip(
            (0.35 * harm_ac + 0.25 * clean_ac + 0.25 * mid_ac + 0.15 * dyn_ac) * (1.0 - 0.3 * feat.percussive_ratio)
        ))

        # 12. MELODY
        harm_melody = min(1.0, feat.harmonic_ratio / 0.70)
        chroma_melody = min(1.0, feat.chroma_salience / 0.25)
        clean_melody = max(0.0, 1.0 - (feat.spectral_flatness / 0.01))
        drum_penalty = max(0.2, 1.0 - feat.percussive_ratio)
        scores["Melody"] = float(np_clip(
            (0.45 * harm_melody + 0.35 * chroma_melody + 0.20 * clean_melody) * drum_penalty
        ))

        # 13. EPIC
        crest_epic = min(1.0, max(0.0, (feat.dynamic_crest - 3.0) / 4.0))
        std_epic = min(1.0, feat.rms_std / 0.04)
        rolloff_epic = min(1.0, feat.spectral_rolloff / 3500.0)
        epic_score = (0.45 * crest_epic + 0.35 * std_epic + 0.20 * rolloff_epic)
        scores["Epic"] = float(np_clip(epic_score))

        # 14. CLASSICAL
        clean_class = max(0.0, 1.0 - (feat.spectral_flatness / 0.01))
        dynamics_class = min(1.0, feat.dynamic_crest / 5.0)
        class_score = (0.40 * feat.harmonic_ratio + 0.35 * clean_class + 0.25 * dynamics_class)
        if feat.percussive_ratio > 0.40:
            class_score *= 0.3
        scores["Classical"] = float(np_clip(class_score))

        # 15. VOCAL
        vocal_band = min(1.0, max(0.0, (feat.vocal_band_ratio - 0.45) / 0.35))
        harm_vocal = feat.harmonic_ratio
        centroid_vocal = 1.0 if (600 <= feat.spectral_centroid <= 2800) else 0.5
        vocal_score = (0.50 * vocal_band + 0.30 * harm_vocal + 0.20 * centroid_vocal)
        vocal_score = vocal_score * 0.7 + getattr(feat, 'vocal_presence_score', 0.0) * 0.3
        scores["Vocal"] = float(np_clip(vocal_score))

        # 16. CHORAL-OPERA
        vocal_co = getattr(feat, 'vocal_presence_score', 0.0)
        harm_co = feat.harmonic_ratio
        dyn_co = min(1.0, feat.dynamic_crest / 4.5)
        clean_co = max(0.0, 1.0 - (feat.spectral_flatness / 0.012))
        co_score = (0.35 * vocal_co + 0.30 * harm_co + 0.20 * dyn_co + 0.15 * clean_co)
        if feat.percussive_ratio > 0.35:
            co_score *= 0.4
        scores["Choral-Opera"] = float(np_clip(co_score))

        # 17. AMBIENT
        if feat.onset_rate > 2.0 or feat.beat_strength > 1.2:
            ambient_score = 0.1
        else:
            low_onset_amb = max(0.0, 1.0 - (feat.onset_rate / 1.0))
            low_beat_amb = max(0.0, 1.0 - (feat.beat_strength / 0.6))
            ambient_score = 0.45 * low_onset_amb + 0.35 * low_beat_amb + 0.20 * feat.harmonic_ratio
        scores["Ambient"] = float(np_clip(ambient_score))

        # 18. JAZZ
        contrast_jazz = min(1.0, getattr(feat, 'spectral_contrast_mean', 0.0) / 25.0)
        mid_jazz = min(1.0, getattr(feat, 'mid_freq_ratio', 0.0) / 0.5)
        tempo_jazz = 1.0 if (80 <= feat.bpm <= 180) else 0.5
        stab_jazz = min(1.0, getattr(feat, 'tempo_stability', 0.0))
        harm_jazz = feat.harmonic_ratio
        scores["Jazz"] = float(np_clip(
            (0.30 * contrast_jazz + 0.25 * mid_jazz + 0.20 * harm_jazz + 0.25 * stab_jazz) * tempo_jazz
        ))

        # 19. BLUES
        chroma_blues = min(1.0, feat.chroma_salience / 0.28)
        tempo_blues = 1.0 if (70 <= feat.bpm <= 135) else (0.2 if feat.bpm == 0 else 0.6)
        mid_blues = min(1.0, getattr(feat, 'mid_freq_ratio', 0.0) / 0.45)
        harm_blues = feat.harmonic_ratio
        scores["Blues"] = float(np_clip(
            (0.35 * chroma_blues + 0.25 * mid_blues + 0.25 * harm_blues + 0.15 * (1.0 - feat.spectral_flatness / 0.03)) * tempo_blues
        ))

        # 20. SOUL-FUNK
        offbeat_funk = min(1.0, getattr(feat, 'offbeat_ratio', 0.0) / 0.35)
        bass_funk = min(1.0, getattr(feat, 'low_freq_ratio', 0.0) / 0.25)
        perc_funk = min(1.0, feat.percussive_ratio / 0.35)
        tempo_funk = 1.0 if (85 <= feat.bpm <= 128) else (0.2 if feat.bpm == 0 else 0.5)
        scores["Soul-Funk"] = float(np_clip(
            (0.35 * offbeat_funk + 0.25 * bass_funk + 0.25 * perc_funk + 0.15 * feat.harmonic_ratio) * tempo_funk
        ))

        # 21. R&B
        vocal_rb = min(1.0, getattr(feat, 'vocal_presence_score', 0.0))
        harm_rb = feat.harmonic_ratio
        mid_rb = min(1.0, getattr(feat, 'mid_freq_ratio', 0.0) / 0.4)
        bass_rb = min(1.0, getattr(feat, 'low_freq_ratio', 0.0) / 0.2)
        tempo_rb = 1.0 if (70 <= feat.bpm <= 120) else (0.2 if feat.bpm == 0 else 0.5)
        scores["R&B"] = float(np_clip(
            (0.30 * vocal_rb + 0.20 * harm_rb + 0.25 * mid_rb + 0.25 * bass_rb) * tempo_rb
        ))

        # 22. OUD
        mode_oud = 1.0 if (getattr(feat, 'mode_major_score', 0.0) < 0 or getattr(feat, 'detected_mode', '') in ('Phrygian', 'Dorian', 'Minor', 'Harmonic Minor')) else 0.5
        mid_oud = min(1.0, getattr(feat, 'mid_freq_ratio', 0.0) / 0.5)
        clean_oud = max(0.0, 1.0 - (feat.spectral_flatness / 0.02))
        harm_oud = feat.harmonic_ratio
        high_oud_penalty = max(0.0, 1.0 - (getattr(feat, 'high_freq_ratio', 0.0) / 0.2))
        chroma_oud = min(1.0, feat.chroma_salience / 0.3)
        stab_oud = min(1.0, getattr(feat, 'tempo_stability', 0.0))
        tempo_oud = 1.0 if (75 <= feat.bpm <= 145) else 0.6
        scores["Oud"] = float(np_clip(
            (0.20 * mode_oud + 0.20 * mid_oud + 0.15 * clean_oud + 0.15 * harm_oud + 0.10 * high_oud_penalty + 0.10 * chroma_oud + 0.10 * stab_oud) * tempo_oud
        ))

        # 23. REGGAE
        offbeat_reggae = min(1.0, getattr(feat, 'offbeat_ratio', 0.0) / 0.4)
        bass_reggae = min(1.0, getattr(feat, 'low_freq_ratio', 0.0) / 0.3)
        onset_reggae = max(0.0, 1.0 - (feat.onset_rate / 4.0))
        tempo_reggae = 1.0 if (60 <= feat.bpm <= 100) else (0.2 if feat.bpm == 0 else 0.5)
        scores["Reggae"] = float(np_clip(
            (0.50 * offbeat_reggae + 0.30 * bass_reggae + 0.20 * onset_reggae) * tempo_reggae
        ))

        # 24. LATIN
        perc_latin = min(1.0, feat.percussive_ratio / 0.35)
        onset_latin = min(1.0, feat.onset_rate / 2.8)
        tempo_latin = 1.0 if (85 <= feat.bpm <= 135) else (0.2 if feat.bpm == 0 else 0.5)
        contrast_latin = min(1.0, getattr(feat, 'spectral_contrast_mean', 0.0) / 22.0)
        scores["Latin"] = float(np_clip(
            (0.35 * perc_latin + 0.30 * onset_latin + 0.20 * contrast_latin + 0.15 * feat.harmonic_ratio) * tempo_latin
        ))

        # 25. POP
        vocal_pop = min(1.0, getattr(feat, 'vocal_presence_score', 0.0))
        major_pop = 1.0 if getattr(feat, 'mode_major_score', 0.0) > 0 else 0.5
        tempo_pop = 1.0 if (90 <= feat.bpm <= 140) else (0.2 if feat.bpm == 0 else 0.6)
        clean_pop = max(0.0, 1.0 - (feat.spectral_flatness / 0.03))
        scores["Pop"] = float(np_clip(
            (0.40 * vocal_pop + 0.30 * major_pop + 0.30 * clean_pop) * tempo_pop
        ))

        # 26. COUNTRY
        major_country = 1.0 if getattr(feat, 'mode_major_score', 0.0) > 0 else 0.5
        tempo_country = 1.0 if (85 <= feat.bpm <= 135) else (0.2 if feat.bpm == 0 else 0.6)
        mid_country = min(1.0, getattr(feat, 'mid_freq_ratio', 0.0) / 0.45)
        clean_country = max(0.0, 1.0 - (feat.spectral_flatness / 0.025))
        scores["Country"] = float(np_clip(
            (0.30 * major_country + 0.30 * mid_country + 0.20 * clean_country + 0.20 * feat.harmonic_ratio) * tempo_country
        ))

        # 27. PUNK-ALTERNATIVE
        sf_punk = min(1.0, feat.spectral_flatness / 0.025)
        zcr_punk = min(1.0, feat.zero_crossing_rate / 0.07)
        tempo_punk = 1.0 if (135 <= feat.bpm <= 210) else (0.2 if feat.bpm == 0 else 0.5)
        perc_punk = min(1.0, feat.percussive_ratio / 0.35)
        scores["Punk-Alternative"] = float(np_clip(
            (0.40 * (sf_punk * 0.5 + zcr_punk * 0.5) + 0.35 * perc_punk + 0.25 * min(1.0, feat.rms_mean / 0.12)) * tempo_punk
        ))

        # 28. WORLD-FOLK
        chroma_wf = min(1.0, feat.chroma_salience / 0.25)
        clean_wf = max(0.0, 1.0 - (feat.spectral_flatness / 0.02))
        dyn_wf = min(1.0, feat.dynamic_crest / 4.0)
        scores["World-Folk"] = float(np_clip(
            (0.35 * chroma_wf + 0.30 * clean_wf + 0.20 * feat.harmonic_ratio + 0.15 * dyn_wf)
        ))

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

        mode_str = getattr(features, 'detected_mode_type', 'Major' if getattr(features, 'mode_major_score', 0.0) >= 0 else 'Minor')
        vocal_language = self._detect_vocal_language(features, metadata, primary_type)

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
            features=features,
            mode=mode_str,
            key=getattr(features, 'detected_key', 'Unknown'),
            vocal_language=vocal_language
        )


def np_clip(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, val))
