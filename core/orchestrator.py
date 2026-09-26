"""
Orchestrator Module - Orchestrates scanning, mid-song audio analysis,
classification, playlist generation, and optional file organization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import time

from core.audio_loader import load_middle_snippet
from core.metadata_reader import extract_metadata
from core.feature_extractor import extract_features
from core.classifier import MusicClassifier, ClassificationResult
from core.playlist_generator import PlaylistGenerator
from core.file_organizer import FileOrganizer


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".wma", ".opus"
}


@dataclass
class ScanStats:
    total_found: int = 0
    processed: int = 0
    errors: int = 0
    elapsed_time: float = 0.0
    category_counts: Dict[str, int] = None


class MusicTypoPipeline:
    """
    Coordinates batch scanning and classification of audio files.
    """

    def __init__(
        self,
        snippet_duration: float = 20.0,
        secondary_threshold: float = 0.40,
        use_relative_playlists: bool = False
    ):
        self.snippet_duration = snippet_duration
        self.classifier = MusicClassifier(secondary_threshold=secondary_threshold)
        self.use_relative_playlists = use_relative_playlists

    def scan_directory(self, input_dir: Path | str, recursive: bool = True) -> List[Path]:
        """Scans input directory for all supported audio files."""
        dir_path = Path(input_dir)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        files: List[Path] = []
        pattern = "**/*" if recursive else "*"
        for p in dir_path.glob(pattern):
            if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                files.append(p)

        return sorted(files)

    def process_library(
        self,
        input_dir: Optional[Path | str] = None,
        output_dir: Path | str = "./Playlists_Output",
        files: Optional[List[Path | str]] = None,
        recursive: bool = True,
        organize_mode: Optional[str] = None,  # None, "copy", "move"
        progress_callback: Optional[Callable[[int, int, Path, Optional[ClassificationResult]], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[List[ClassificationResult], ScanStats, Dict[str, Path]]:
        """
        Executes full analysis pipeline on either a directory or an explicit list of files:
        1. Discovers audio files (or takes selected files)
        2. Samples middle of each song
        3. Classifies by musical type
        4. Generates .m3u8 playlists and reports
        5. Optionally organizes files into subfolders
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        def log(msg: str):
            if log_callback:
                log_callback(msg)

        if files:
            audio_files = [Path(f) for f in files if Path(f).is_file() and Path(f).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
            log(f"Processing {len(audio_files)} selected audio track(s)...")
        elif input_dir:
            input_path = Path(input_dir)
            log(f"Scanning '{input_path}' for music files...")
            audio_files = self.scan_directory(input_path, recursive=recursive)
        else:
            raise ValueError("Either 'input_dir' or 'files' must be provided.")

        total = len(audio_files)
        log(f"Found {total} audio track(s).")

        results: List[ClassificationResult] = []
        errors = 0
        cat_counts: Dict[str, int] = {}

        for idx, file_path in enumerate(audio_files, 1):
            log(f"[{idx}/{total}] Reading mid-song & analyzing: {file_path.name}")
            try:
                # 1. Seek to exact middle of track and extract snippet
                snippet = load_middle_snippet(file_path, snippet_duration=self.snippet_duration)

                # 2. Extract metadata
                meta = extract_metadata(file_path)

                # 3. Compute DSP features
                feats = extract_features(snippet.audio, snippet.sample_rate)

                # 4. Classify
                res = self.classifier.classify(
                    file_path=file_path,
                    features=feats,
                    metadata=meta,
                    total_duration=snippet.total_duration,
                    snippet_offset=snippet.snippet_offset
                )

                results.append(res)
                cat_counts[res.primary_type] = cat_counts.get(res.primary_type, 0) + 1

                if progress_callback:
                    progress_callback(idx, total, file_path, res)

            except Exception as e:
                errors += 1
                log(f"[Error] Failed to process {file_path.name}: {e}")
                if progress_callback:
                    progress_callback(idx, total, file_path, None)

        # 5. Generate Playlists and Reports
        log("Generating category .m3u8 playlists and reports...")
        pg = PlaylistGenerator(output_path, use_relative_paths=self.use_relative_playlists)
        playlists = pg.generate_category_playlists(results, include_secondary=True)
        csv_rep, json_rep = pg.generate_reports(results)
        log(f"Playlists saved to: {output_path}")

        # 6. Optional file organization
        if organize_mode in ("copy", "move"):
            log(f"Organizing files (mode: {organize_mode})...")
            organizer = FileOrganizer(output_path / "Organized_Music", mode=organize_mode)
            organizer.organize(results)
            log("File organization complete.")

        elapsed = time.time() - start_time
        stats = ScanStats(
            total_found=total,
            processed=len(results),
            errors=errors,
            elapsed_time=elapsed,
            category_counts=cat_counts
        )

        log(f"Finished processing {len(results)}/{total} tracks in {elapsed:.1f}s.")
        return results, stats, playlists
