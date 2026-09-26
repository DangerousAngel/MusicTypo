"""
Playlist Generator Module - Generates standard extended .m3u8 playlist files
and classification reports (CSV and JSON).
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import os

from core.classifier import ClassificationResult


class PlaylistGenerator:
    """
    Produces clean, standard UTF-8 .m3u8 playlists compatible with VLC,
    Windows Media Player, iTunes, Winamp, Foobar2000, and mobile audio players.
    """

    def __init__(self, output_dir: Path | str, use_relative_paths: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_relative_paths = use_relative_paths

    def _format_track_path(self, playlist_file_path: Path, track_file_path: Path) -> str:
        """Formats the track path as either relative or absolute."""
        if self.use_relative_paths:
            try:
                rel = os.path.relpath(track_file_path, start=playlist_file_path.parent)
                return rel.replace("\\", "/")
            except ValueError:
                # If on different Windows drives (e.g. C: and E:), relpath throws ValueError
                return str(track_file_path.resolve())
        return str(track_file_path.resolve())

    def generate_category_playlists(
        self,
        results: List[ClassificationResult],
        include_secondary: bool = False
    ) -> Dict[str, Path]:
        """
        Generates individual .m3u8 playlists for each category.
        
        Args:
            results: List of classified track results.
            include_secondary: If True, tracks with secondary tags are also
                               included in corresponding secondary playlists.
                               
        Returns:
            Dict mapping category name to generated playlist Path.
        """
        # Group tracks by category
        grouped: Dict[str, List[ClassificationResult]] = {}
        for r in results:
            # Primary category
            grouped.setdefault(r.primary_type, []).append(r)

            # Secondary categories
            if include_secondary:
                for sec in r.secondary_tags:
                    if sec != r.primary_type:
                        if r not in grouped.setdefault(sec, []):
                            grouped[sec].append(r)

        generated_playlists: Dict[str, Path] = {}

        for category, tracks in grouped.items():
            if not tracks:
                continue

            playlist_name = f"{category}.m3u8"
            playlist_path = self.output_dir / playlist_name

            lines = ["#EXTM3U", f"# Playlist: {category} ({len(tracks)} tracks)", ""]
            for t in tracks:
                duration_sec = int(round(t.total_duration)) if t.total_duration > 0 else -1
                artist_title = f"{t.artist} - {t.title}"
                # Add metadata comment
                lines.append(f"#EXTINF:{duration_sec},{artist_title}")
                lines.append(f"# [Category: {t.primary_type} | BPM: {t.bpm} | Confidence: {t.confidence:.0%}]")
                lines.append(self._format_track_path(playlist_path, t.file_path))
                lines.append("")

            with open(playlist_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            generated_playlists[category] = playlist_path

        # Generate a Master Playlist
        master_path = self.output_dir / "All_Tracks_Categorized.m3u8"
        master_lines = ["#EXTM3U", f"# Master Categorized Playlist ({len(results)} tracks)", ""]

        for category, tracks in sorted(grouped.items()):
            master_lines.append(f"# ==========================================")
            master_lines.append(f"# Category: {category} ({len(tracks)} tracks)")
            master_lines.append(f"# ==========================================")
            for t in tracks:
                duration_sec = int(round(t.total_duration)) if t.total_duration > 0 else -1
                artist_title = f"{t.artist} - {t.title}"
                master_lines.append(f"#EXTINF:{duration_sec},{artist_title}")
                master_lines.append(self._format_track_path(master_path, t.file_path))
                master_lines.append("")

        with open(master_path, "w", encoding="utf-8") as f:
            f.write("\n".join(master_lines))

        generated_playlists["Master"] = master_path
        return generated_playlists

    def generate_reports(self, results: List[ClassificationResult]) -> Tuple[Path, Path]:
        """
        Generates CSV and JSON summary reports of the classification results.
        """
        # CSV Report
        csv_path = self.output_dir / "classification_report.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Filename",
                "Title",
                "Artist",
                "Primary Type",
                "Confidence",
                "Secondary Tags",
                "BPM",
                "Total Duration (s)",
                "Snippet Offset (s)",
                "File Path"
            ])
            for r in results:
                writer.writerow([
                    r.file_path.name,
                    r.title,
                    r.artist,
                    r.primary_type,
                    f"{r.confidence:.2%}",
                    ", ".join(r.secondary_tags),
                    r.bpm,
                    round(r.total_duration, 1),
                    round(r.snippet_offset, 1),
                    str(r.file_path)
                ])

        # JSON Report
        json_path = self.output_dir / "summary.json"
        summary_data = []
        for r in results:
            summary_data.append({
                "filename": r.file_path.name,
                "path": str(r.file_path),
                "title": r.title,
                "artist": r.artist,
                "primary_type": r.primary_type,
                "confidence": r.confidence,
                "secondary_tags": r.secondary_tags,
                "category_scores": r.category_scores,
                "bpm": r.bpm,
                "duration_seconds": round(r.total_duration, 2),
                "snippet_offset_seconds": round(r.snippet_offset, 2),
            })

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        return csv_path, json_path
