"""
Unit and Integration Tests for Music Typo
Tests mid-song seek, DSP feature extraction, classifier, and playlist generation.
"""

import unittest
from pathlib import Path
import tempfile
import shutil

from core.audio_loader import load_middle_snippet, get_audio_duration
from core.metadata_reader import extract_metadata
from core.feature_extractor import extract_features
from core.classifier import MusicClassifier, CATEGORIES
from core.playlist_generator import PlaylistGenerator
from core.orchestrator import MusicTypoPipeline
from tests.generate_test_audio import generate_all_samples


class TestMusicTypo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(__file__).parent / "sample_audio"
        if not (cls.test_dir / "Synthetic_Beat_Track.wav").exists():
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

    def test_02_feature_extraction(self):
        """Verify that acoustic features are correctly computed without NaN/Inf."""
        piano_file = self.test_dir / "Synthetic_Piano_Solo.wav"
        snippet = load_middle_snippet(piano_file, snippet_duration=20.0)
        feats = extract_features(snippet.audio, snippet.sample_rate)

        self.assertFalse(any(val is None for val in feats.__dict__.values()))
        self.assertGreater(feats.harmonic_ratio, 0.0)
        self.assertGreater(feats.percussive_ratio, 0.0)
        self.assertGreater(feats.dynamic_crest, 0.0)

    def test_03_classification_types(self):
        """Verify that synthetic profiles match their expected musical styles."""
        classifier = MusicClassifier()

        samples = {
            "Synthetic_Beat_Track.wav": ["Beat"],
            "Synthetic_Piano_Solo.wav": ["Piano", "Melody", "Classical"],
            "Synthetic_Rock_Riff.wav": ["Rock", "Beat"],
            "Synthetic_Epic_Crescendo.wav": ["Epic", "Classical"],
            "Synthetic_Vocal_Melody.wav": ["Vocal", "Melody"],
        }

        for filename, expected_candidates in samples.items():
            filepath = self.test_dir / filename
            snippet = load_middle_snippet(filepath, snippet_duration=20.0)
            meta = extract_metadata(filepath)
            feats = extract_features(snippet.audio, snippet.sample_rate)
            res = classifier.classify(filepath, feats, meta, snippet.total_duration, snippet.snippet_offset)

            matched = (res.primary_type in expected_candidates) or any(
                sec in expected_candidates for sec in res.secondary_tags
            )
            self.assertTrue(
                matched,
                f"Track {filename} was classified as '{res.primary_type}' (scores: {res.category_scores}), expected one of {expected_candidates}"
            )

    def test_04_playlist_generation(self):
        """Verify that .m3u8 playlists and summary reports are created and valid."""
        out_playlist_dir = self.temp_dir / "playlists"
        pipeline = MusicTypoPipeline(snippet_duration=15.0)

        results, stats, playlists = pipeline.process_library(
            input_dir=self.test_dir,
            output_dir=out_playlist_dir,
            recursive=False
        )

        self.assertEqual(stats.processed, 5)
        self.assertEqual(stats.errors, 0)
        self.assertIn("Master", playlists)
        self.assertTrue(playlists["Master"].exists())

        # Check Master playlist content
        with open(playlists["Master"], "r", encoding="utf-8") as f:
            content = f.read()
            self.assertTrue(content.startswith("#EXTM3U"))
            self.assertIn("#EXTINF:", content)

        # Check CSV and JSON reports
        csv_file = out_playlist_dir / "classification_report.csv"
        json_file = out_playlist_dir / "summary.json"
        self.assertTrue(csv_file.exists())
        self.assertTrue(json_file.exists())


if __name__ == "__main__":
    unittest.main()
