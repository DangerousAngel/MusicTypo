"""
Music Typo - Command Line Interface (CLI)
Analyzes mid-song snippets, classifies music files into Rock, Beat, Piano, Melody, Epic, Vocal, etc.,
and generates .m3u8 playlists.
"""

import argparse
import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel

from core.orchestrator import MusicTypoPipeline
from core.classifier import CATEGORIES

console = Console(highlight=False)


def render_banner():
    console.print(
        Panel.fit(
            "[bold cyan]Music Typo - Intelligent Music Classifier & Playlist Generator[/bold cyan]\n"
            "[dim]Analyzes mid-song excerpts to classify into Rock, Beat, Piano, Melody, Epic, Vocal, etc.[/dim]\n"
            "[magenta]Developed by DangerousAngel[/magenta] - [link=https://github.com/DangerousAngel]https://github.com/DangerousAngel[/link]",
            border_style="cyan"
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Music Typo: Sort and organize music into categories and generate .m3u8 playlists."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Path to folder containing music files."
    )
    parser.add_argument(
        "-f", "--files",
        nargs="+",
        default=None,
        help="Specific audio files to classify (space-separated paths)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./Playlists_Output",
        help="Directory to save generated .m3u8 playlists and reports (default: ./Playlists_Output)."
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=20.0,
        help="Duration of mid-song excerpt to read in seconds (default: 20.0s)."
    )
    parser.add_argument(
        "--organize",
        choices=["copy", "move"],
        default=None,
        help="Optional: copy or move audio files into category subfolders."
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Use relative paths in .m3u8 playlists instead of absolute paths."
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not scan subdirectories recursively."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Threshold for assigning secondary category tags (default: 0.40)."
    )

    args = parser.parse_args()
    render_banner()

    if not args.input and not args.files:
        console.print("[bold red]Error:[/bold red] You must provide either --input folder or --files.")
        sys.exit(1)

    input_dir = Path(args.input).resolve() if args.input else None
    output_dir = Path(args.output).resolve()

    if input_dir and (not input_dir.exists() or not input_dir.is_dir()):
        console.print(f"[bold red]Error:[/bold red] Input directory does not exist: {input_dir}")
        sys.exit(1)

    pipeline = MusicTypoPipeline(
        snippet_duration=args.duration,
        secondary_threshold=args.threshold,
        use_relative_playlists=args.relative
    )

    if input_dir:
        console.print(f"[bold green]Input Directory:[/bold green] {input_dir}")
    if args.files:
        console.print(f"[bold green]Selected Files:[/bold green] {len(args.files)} track(s)")
    console.print(f"[bold green]Output Directory:[/bold green] {output_dir}")
    console.print(f"[bold green]Mid-song sample window:[/bold green] {args.duration}s\n")

    # Discover files first
    if args.files:
        files = [Path(f) for f in args.files if Path(f).is_file()]
    else:
        files = pipeline.scan_directory(input_dir, recursive=not args.no_recursive)

    if not files:
        console.print("[yellow]No supported audio files found.[/yellow]")
        sys.exit(0)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Classifying audio (reading mid-song)...", total=len(files))

        def on_progress(idx, total, file_path, res):
            if res:
                progress.update(task, advance=1, description=f"[cyan]Analyzed:[/cyan] {file_path.name[:35]}")
            else:
                progress.update(task, advance=1, description=f"[red]Failed:[/red] {file_path.name[:35]}")

        res_list, stats, playlists = pipeline.process_library(
            input_dir=input_dir,
            output_dir=output_dir,
            files=args.files,
            recursive=not args.no_recursive,
            organize_mode=args.organize,
            progress_callback=on_progress
        )

    # Print Results Table
    table = Table(title="Classification Results", show_lines=True)
    table.add_column("Track", style="white", no_wrap=False)
    table.add_column("Primary Type", style="bold magenta")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Secondary Tags", style="cyan")
    table.add_column("BPM", justify="right", style="yellow")
    table.add_column("Duration", justify="right", style="dim")

    for r in res_list:
        sec_str = ", ".join(r.secondary_tags) if r.secondary_tags else "-"
        table.add_row(
            r.title,
            r.primary_type,
            f"{r.confidence:.0%}",
            sec_str,
            f"{r.bpm:.0f}" if r.bpm > 0 else "-",
            f"{int(r.total_duration // 60)}:{int(r.total_duration % 60):02d}"
        )

    console.print(table)

    # Print Summary & Playlists
    summary_table = Table(title="Category Distribution & Generated Playlists", show_lines=False)
    summary_table.add_column("Category", style="bold")
    summary_table.add_column("Track Count", justify="right", style="cyan")
    summary_table.add_column("Playlist File (.m3u8)", style="blue")

    for cat in CATEGORIES:
        count = stats.category_counts.get(cat, 0)
        pl_path = playlists.get(cat)
        pl_str = pl_path.name if pl_path else "[dim]None[/dim]"
        if count > 0:
            summary_table.add_row(cat, str(count), str(pl_str))

    # Add master
    if "Master" in playlists:
        summary_table.add_row(
            "[bold white]All Categorized (Master)[/bold white]",
            str(stats.processed),
            str(playlists["Master"].name)
        )

    console.print("\n", summary_table)

    console.print(
        f"\n[bold green][OK][/bold green] Processed {stats.processed} tracks in {stats.elapsed_time:.1f} seconds."
    )
    console.print(f"Playlists & CSV/JSON reports written to: [bold]{output_dir}[/bold]\n")


if __name__ == "__main__":
    main()
