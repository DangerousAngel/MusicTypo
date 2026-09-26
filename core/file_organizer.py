"""
File Organizer Module - Optionally copies, moves, or creates shortcuts for
audio files into categorized directories (e.g., Music/Rock/, Music/Piano/).
"""

import shutil
from pathlib import Path
from typing import Dict, List, Literal
import os

from core.classifier import ClassificationResult


class FileOrganizer:
    """
    Organizes physical music files into categorized directory structures.
    """

    def __init__(self, target_dir: Path | str, mode: Literal["copy", "move", "link"] = "copy"):
        self.target_dir = Path(target_dir)
        self.mode = mode

    def organize(
        self,
        results: List[ClassificationResult],
        overwrite: bool = False
    ) -> Dict[str, List[Path]]:
        """
        Organizes files into subfolders named after their primary classification type.
        
        Returns:
            Dict mapping category name to list of new target paths.
        """
        organized: Dict[str, List[Path]] = {}

        for result in results:
            category = result.primary_type
            cat_dir = self.target_dir / category
            cat_dir.mkdir(parents=True, exist_ok=True)

            src = result.file_path
            dst = cat_dir / src.name

            # Avoid collision if destination exists and overwrite is False
            if dst.exists() and not overwrite and dst.resolve() != src.resolve():
                counter = 1
                while dst.exists():
                    dst = cat_dir / f"{src.stem}_{counter}{src.suffix}"
                    counter += 1

            if src.resolve() == dst.resolve():
                # Already in place
                organized.setdefault(category, []).append(dst)
                continue

            try:
                if self.mode == "copy":
                    shutil.copy2(src, dst)
                elif self.mode == "move":
                    shutil.move(src, dst)
                elif self.mode == "link":
                    try:
                        os.symlink(src, dst)
                    except (OSError, NotImplementedError):
                        # Symlink may require admin on Windows; fallback to copy
                        shutil.copy2(src, dst)

                organized.setdefault(category, []).append(dst)
            except Exception as e:
                print(f"[Warning] Failed to {self.mode} {src.name} to {dst}: {e}")

        return organized
