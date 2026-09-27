"""
Unit and Integration Tests for Music Typo (Expanded)
Tests mid-song seek, DSP feature extraction, 28-category classifier,
modal Key/Mode detection, multilingual vocal language detection, and playlist generation.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np

from core.audio_loader import load_middle_snippet, get_audio_duration
from core.metadata_reader import extract_metadata, TrackMetadata
from core.feature_extractor import extract_features, AcousticFeatures
from core.classifier import MusicClassifier, CATEGORIES
from core.playlist_generator import PlaylistGenerator
from core.orchestrator import MusicTypoPipeline
from tests.generate_test_audio import generate_all_samples


class TestMusicTypo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent / "sample_audio"
        # Regenerate samples including new types
        generate_all_samples(cls.test_dir)
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="musictypo_test_"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_mid_song_seek_and_load(self):
        """Verify that audio loader strictly reads the middle excerpt of the song."""
        beat_file = self.test_dir / "Synthetic_Beat_Track.wav"
        total_duration = get_audio_duration(beat_file)
        self.assertAlmostEqual(total_duration, 35.0, delta=1.0)

        # Request 15 seconds from the middle
        snippet = load_middle_snippet(beat_file, snippet_duration=15.0)

        # Midpoint of 35s is 17.5s; 15s window should start at ~10.0s (17.5 - 7.5)
        self.assertAlmostEqual(snippet.snippet_offset, 10.0, delta=1.5)
        self.assertAlmostEqual(snippet.snippet_duration, 15.0, delta=0.5)
        self.assertEqual(len(snippet.audio.shape), 1)  # Mono 1D array
        self.assertGreater(len(snippet.audio), 0)

    def test_02_feature_extraction_expanded(self):
        """Verify expanded acoustic features including mode, key, new spectral fields."""
        piano_file = self.test_dir / "Synthetic_Piano_Solo.wav"
        snippet = load_middle_snippet(piano_file, snippet_duration=20.0)
        feats = extract_features(snippet.audio, snippet.sample_rate)

        # Check all original fields exist and are valid
        self.assertGreater(feats.harmonic_ratio, 0.0)
        self.assertGreater(feats.percussive_ratio, 0.0)
        self.assertGreater(feats.dynamic_crest, 0.0)

        # Check new fields exist
        self.assertIsInstance(feats.mode_major_score, float)
        self.assertIsInstance(feats.detected_key, str)
        self.assertIsInstance(feats.detected_mode, str)
        self.assertIn(feats.detected_mode_type, ("Major", "Minor"))
        self.assertIsInstance(feats.mode_confidence, float)
        self.assertIsInstance(feats.mfcc_mean, list)
        self.assertEqual(len(feats.mfcc_mean), 13)
        self.assertIsInstance(feats.spectral_bandwidth, float)
        self.assertIsInstance(feats.low_freq_ratio, float)
        self.assertIsInstance(feats.mid_freq_ratio, float)
        self.assertIsInstance(feats.high_freq_ratio, float)
        self.assertIsInstance(feats.tempo_stability, float)
        self.assertIsInstance(feats.spectral_flux, float)
        self.assertIsInstance(feats.vocal_presence_score, float)
        self.assertIsInstance(feats.offbeat_ratio, float)

        # Frequency ratios should roughly sum to ~1.0
        freq_sum = feats.low_freq_ratio + feats.mid_freq_ratio + feats.high_freq_ratio
        self.assertAlmostEqual(freq_sum, 1.0, delta=0.25)

    def test_03_28_categories_exist(self):
        """Verify all 28 categories are defined, including Oud and Electronic."""
        expected = [
            "Rock", "Metal", "Beat", "Hip-Hop", "Trap", "Electronic",
            "House-Techno", "Synthwave", "Lo-Fi", "Piano", "Acoustic",
            "Melody", "Epic", "Classical", "Vocal", "Choral-Opera",
            "Ambient", "Jazz", "Blues", "Soul-Funk", "R&B",
            "Oud", "Reggae", "Latin", "Pop", "Country", "Punk-Alternative", "World-Folk"
        ]
        self.assertEqual(CATEGORIES, expected)
        self.assertIn("Oud", CATEGORIES)
        self.assertIn("Electronic", CATEGORIES)

    def test_04_classification_with_mode_key_language(self):
        """Verify that classification produces mode, key, and language fields."""
        classifier = MusicClassifier()

        for wav_file in sorted(self.test_dir.glob("*.wav")):
            snippet = load_middle_snippet(wav_file, snippet_duration=15.0)
            meta = extract_metadata(wav_file)
            feats = extract_features(snippet.audio, snippet.sample_rate)
            res = classifier.classify(wav_file, feats, meta, snippet.total_duration, snippet.snippet_offset)

            # Result should have primary type from the 28 categories
            self.assertIn(res.primary_type, CATEGORIES, f"{wav_file.name}: {res.primary_type}")

            # Mode and Key must be valid strings
            self.assertIn(res.mode, ("Major", "Minor"), f"{wav_file.name}: mode={res.mode}")
            self.assertIsInstance(res.key, str)
            self.assertTrue(len(res.key) > 0, f"{wav_file.name}: empty key")

            # Language must be set
            self.assertIsInstance(res.vocal_language, str)
            self.assertTrue(len(res.vocal_language) > 0, f"{wav_file.name}: empty language")

    def test_05_classification_type_matching(self):
        """Verify that synthetic profiles match their expected musical styles."""
        classifier = MusicClassifier()

        samples = {
            "Synthetic_Beat_Track.wav": ["Beat", "Hip-Hop", "Electronic", "House-Techno", "Trap"],
            "Synthetic_Piano_Solo.wav": ["Piano", "Melody", "Classical", "Acoustic"],
            "Synthetic_Rock_Riff.wav": ["Rock", "Metal", "Beat", "Punk-Alternative"],
            "Synthetic_Epic_Crescendo.wav": ["Epic", "Classical", "Ambient", "Choral-Opera"],
            "Synthetic_Vocal_Melody.wav": ["Vocal", "Melody", "Pop", "Choral-Opera"],
            "Synthetic_Oud_Track.wav": ["Oud", "World-Folk", "Melody", "Acoustic"],
            "Synthetic_Electronic_Track.wav": ["Electronic", "House-Techno", "Synthwave", "Beat"],
            "Synthetic_Jazz_Track.wav": ["Jazz", "Blues", "Soul-Funk", "R&B"],
        }

        for filename, expected_candidates in samples.items():
            filepath = self.test_dir / filename
            if not filepath.exists():
                continue

            snippet = load_middle_snippet(filepath, snippet_duration=20.0)
            meta = extract_metadata(filepath)
            feats = extract_features(snippet.audio, snippet.sample_rate)
            res = classifier.classify(filepath, feats, meta, snippet.total_duration, snippet.snippet_offset)

            all_tags = [res.primary_type] + res.secondary_tags
            matched = any(tag in expected_candidates for tag in all_tags)
            self.assertTrue(
                matched,
                f"Track {filename} was classified as '{res.primary_type}' "
                f"(tags: {all_tags}), expected one of {expected_candidates}"
            )

    def test_06_playlist_generation(self):
        """Verify that .m3u8 playlists and summary reports are created and valid."""
        out_playlist_dir = self.temp_dir / "playlists"
        pipeline = MusicTypoPipeline(snippet_duration=15.0)

        results, stats, playlists = pipeline.process_library(
            input_dir=self.test_dir,
            output_dir=out_playlist_dir,
            recursive=False
        )

        self.assertGreaterEqual(stats.processed, 5)
        self.assertEqual(stats.errors, 0)
        self.assertIn("Master", playlists)
        self.assertTrue(playlists["Master"].exists())

        # Check Master playlist content
        with open(playlists["Master"], "r", encoding="utf-8") as f:
            content = f.read()
            self.assertTrue(content.startswith("#EXTM3U"))
            self.assertIn("#EXTINF:", content)

        # Check CSV and JSON reports exist
        csv_file = out_playlist_dir / "classification_report.csv"
        json_file = out_playlist_dir / "summary.json"
        self.assertTrue(csv_file.exists())
        self.assertTrue(json_file.exists())

        # Verify JSON report has new fields
        import json
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertGreater(len(data), 0)
            first = data[0]
            self.assertIn("mode", first)
            self.assertIn("key", first)
            self.assertIn("vocal_language", first)

        # Verify Language and Mode playlists exist
        lang_pls = [k for k in playlists.keys() if k.startswith("Language_")]
        mode_pls = [k for k in playlists.keys() if k.startswith("Mode_")]
        self.assertGreater(len(lang_pls), 0)
        self.assertGreater(len(mode_pls), 0)

    def test_07_vocal_language_detection(self):
        """Verify vocal language detection across scripts, keywords, and metadata."""
        classifier = MusicClassifier()
        dummy_feat_vocal = AcousticFeatures(
            bpm=120.0, beat_strength=0.5, percussive_ratio=0.3, harmonic_ratio=0.7,
            spectral_centroid=1500.0, spectral_flatness=0.005, spectral_rolloff=2500.0,
            rms_mean=0.08, rms_std=0.02, dynamic_crest=3.0, vocal_band_ratio=0.60,
            chroma_salience=0.2, onset_rate=1.5, zero_crossing_rate=0.03,
            mode_major_score=0.5, detected_key="C Major", spectral_contrast_mean=15.0,
            mfcc_mean=[0.0]*13, spectral_bandwidth=1500.0, low_freq_ratio=0.2,
            mid_freq_ratio=0.5, high_freq_ratio=0.1, tempo_stability=0.2,
            spectral_flux=0.5, vocal_presence_score=0.75, offbeat_ratio=0.1
        )
        dummy_feat_inst = AcousticFeatures(
            bpm=120.0, beat_strength=0.5, percussive_ratio=0.3, harmonic_ratio=0.7,
            spectral_centroid=1500.0, spectral_flatness=0.005, spectral_rolloff=2500.0,
            rms_mean=0.08, rms_std=0.02, dynamic_crest=3.0, vocal_band_ratio=0.10,
            chroma_salience=0.2, onset_rate=1.5, zero_crossing_rate=0.03,
            mode_major_score=0.5, detected_key="C Major", spectral_contrast_mean=15.0,
            mfcc_mean=[0.0]*13, spectral_bandwidth=1500.0, low_freq_ratio=0.2,
            mid_freq_ratio=0.5, high_freq_ratio=0.1, tempo_stability=0.2,
            spectral_flux=0.5, vocal_presence_score=0.05, offbeat_ratio=0.1
        )

        # Instrumental
        m_inst = TrackMetadata(title="Song", artist="Artist", album="Album", genre="Rock")
        self.assertEqual(classifier._detect_vocal_language(dummy_feat_inst, m_inst), "Instrumental")

        # Arabic script
        m_ar = TrackMetadata(title="حبيبي يا ليل", artist="عمرو دياب", album="طرب", genre="عربي")
        self.assertEqual(classifier._detect_vocal_language(dummy_feat_vocal, m_ar), "Arabic")

        # Spanish
        m_es = TrackMetadata(title="Despacito Canción", artist="Luis", album="Latino", genre="Reggaeton")
        self.assertEqual(classifier._detect_vocal_language(dummy_feat_vocal, m_es), "Spanish")

        # French
        m_fr = TrackMetadata(title="La vie en rose", artist="Edith Piaf", album="Chanson", genre="French")
        self.assertEqual(classifier._detect_vocal_language(dummy_feat_vocal, m_fr), "French")

        # Japanese
        m_ja = TrackMetadata(title="夜に駆ける", artist="YOASOBI", album="Anime", genre="J-Pop")
        self.assertEqual(classifier._detect_vocal_language(dummy_feat_vocal, m_ja), "Japanese")

        # Korean
        m_ko = TrackMetadata(title="봄날 (Spring Day)", artist="BTS", album="K-Pop", genre="Korean")
        self.assertEqual(classifier._detect_vocal_language(dummy_feat_vocal, m_ko), "Korean")

        # English default vocal
        m_en = TrackMetadata(title="Love Me Like You Do", artist="Ellie Goulding", album="Halcyon", genre="Pop")
        self.assertEqual(classifier._detect_vocal_language(dummy_feat_vocal, m_en), "English")

    def test_08_modal_detection(self):
        """Verify modal detection produces valid keys and modes."""
        sr = 22050
        t = np.linspace(0, 1.0, sr, False)
        # C-F-G-C Major progression
        c = np.sin(2*np.pi*261.63*t) + np.sin(2*np.pi*329.63*t) + np.sin(2*np.pi*392.0*t)
        f = np.sin(2*np.pi*174.61*t) + np.sin(2*np.pi*220.0*t) + np.sin(2*np.pi*261.63*t)
        g = np.sin(2*np.pi*196.0*t) + np.sin(2*np.pi*246.94*t) + np.sin(2*np.pi*293.66*t)
        prog = np.concatenate([c, f, g, c])

        feats_maj = extract_features(prog, sr)
        self.assertGreater(feats_maj.mode_major_score, 0.0)
        self.assertEqual(feats_maj.detected_mode_type, "Major")
        self.assertEqual(feats_maj.detected_key, "C Major")


if __name__ == "__main__":
    unittest.main()
