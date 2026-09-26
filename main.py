"""
Music Typo - Main Launcher
Launches the GUI application by default, or runs in CLI mode if command-line arguments are provided.
"""

import sys


def main():
    if len(sys.argv) > 1:
        # CLI Mode
        from cli import main as cli_main
        cli_main()
    else:
        # Desktop GUI Mode
        from gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
