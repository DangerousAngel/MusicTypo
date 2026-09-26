"""
Music Typo - Desktop Graphical User Interface (GUI)
Sleek Modern Dark Cyber/Glass aesthetic matching the reference UI design.
Includes Folder and Specific File selection, mid-song excerpt slider,
progress tracking, categorized .m3u8 playlist generation, and interactive track table.
"""

import os
import sys
import threading
import queue
import time
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk

# Ensure core package is found
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import MusicTypoPipeline, ScanStats, SUPPORTED_AUDIO_EXTENSIONS
from core.classifier import ClassificationResult, CATEGORIES

# Configure CustomTkinter dark appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Theme Palette Colors
COLOR_BG = "#0d1019"            # Deep midnight navy
COLOR_CARD_BG = "#141724"       # Dark slate card
COLOR_CARD_BORDER = "#33264d"   # Violet/purple glow border
COLOR_INPUT_BG = "#191d2d"      # Dark input field background
COLOR_INPUT_BORDER = "#2b3048"  # Input border
COLOR_TEXT_PRIMARY = "#ffffff"  # Crisp white
COLOR_TEXT_MUTED = "#94a3b8"    # Soft slate gray
COLOR_TEXT_LABEL = "#c4cad4"    # Subtle light gray
COLOR_ACCENT_PURPLE = "#7c3aed" # Vibrant Purple
COLOR_ACCENT_HOVER = "#6d28d9"  # Deep Purple hover
COLOR_ACCENT_CYAN = "#38bdf8"   # Bright Cyan
COLOR_TABLE_BG = "#111420"      # Dark table background
COLOR_ROW_ALT = "#151827"       # Alternating row background
COLOR_ROW_SELECT = "#581c87"    # Selected row violet glow
COLOR_BORDER_BTN = "#433560"    # Outline button border

CATEGORY_ICONS = {
    "Epic": "🎸",
    "Beat": "🥁",
    "Piano": "🎹",
    "Classical": "🎻",
    "Rock": "🎸",
    "Melody": "🎵",
    "Vocal": "🎤",
    "Ambient": "🌌"
}


class ModernMusicTypoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Music Typo - Intelligent Music Classifier")
        self.geometry("1060x780")
        self.minsize(880, 640)
        self.configure(fg_color=COLOR_BG)

        # State Variables
        self.input_path_var = tk.StringVar(value="")
        self.output_path_var = tk.StringVar(value=str(Path("./Playlists_Output").resolve()))
        self.duration_var = tk.DoubleVar(value=20.0)
        self.recursive_var = tk.BooleanVar(value=True)
        self.use_relative_var = tk.BooleanVar(value=False)
        self.organize_mode_var = tk.StringVar(value="Playlists Only (No File Moving)")

        self.selected_files = []  # Stores list of specific selected files if chosen
        self.is_running = False
        self.msg_queue = queue.Queue()

        self._configure_treeview_styles()
        self._build_ui()
        self._poll_queue()

    def _configure_treeview_styles(self):
        self.tree_style = ttk.Style()
        try:
            self.tree_style.theme_use("clam")
        except Exception:
            pass

        self.tree_style.configure(
            "Custom.Treeview",
            background=COLOR_TABLE_BG,
            foreground="#e2e8f0",
            fieldbackground=COLOR_TABLE_BG,
            rowheight=30,
            borderwidth=0,
            font=("Segoe UI", 9)
        )
        self.tree_style.configure(
            "Custom.Treeview.Heading",
            background="#181c2d",
            foreground="#94a3b8",
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padding=(8, 6)
        )
        self.tree_style.map(
            "Custom.Treeview",
            background=[("selected", COLOR_ROW_SELECT)],
            foreground=[("selected", "#ffffff")]
        )
        self.tree_style.map(
            "Custom.Treeview.Heading",
            background=[("active", "#22273e")]
        )

    def _build_ui(self):
        # Main Outer Container
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=22, pady=(16, 18))

        # 1. Header Title & Subtitle (Centered)
        header_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        header_frame.pack(fill=tk.X, pady=(0, 14))

        title_label = ctk.CTkLabel(
            header_frame,
            text="Music Typo - Intelligent Music Classifier",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color=COLOR_TEXT_PRIMARY
        )
        title_label.pack(anchor="center")

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Sorts and organizes music into categories by analyzing song excerpts.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_MUTED
        )
        subtitle_label.pack(anchor="center", pady=(2, 0))

        # DangerousAngel GitHub Route
        author_btn = ctk.CTkButton(
            header_frame,
            text="★ Developed by DangerousAngel | GitHub",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#181c2e",
            border_width=1,
            border_color="#4f3875",
            hover_color="#2a2046",
            text_color="#c4b5fd",
            height=24,
            corner_radius=12,
            command=self._open_about_github
        )
        author_btn.pack(anchor="center", pady=(5, 0))

        # 2. Main Configuration Card (Dark Glassmorphic Box with Violet Border)
        self.card = ctk.CTkFrame(
            self.main_container,
            fg_color=COLOR_CARD_BG,
            corner_radius=14,
            border_width=1.5,
            border_color=COLOR_CARD_BORDER
        )
        self.card.pack(fill=tk.X, pady=(0, 14), padx=2)

        card_inner = ctk.CTkFrame(self.card, fg_color="transparent")
        card_inner.pack(fill=tk.X, padx=18, pady=16)

        # Row 1: Music Folder / Select Files
        row1 = ctk.CTkFrame(card_inner, fg_color="transparent")
        row1.pack(fill=tk.X, pady=(0, 10))

        lbl_input = ctk.CTkLabel(
            row1,
            text="Music Folder:",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_LABEL,
            width=100,
            anchor="w"
        )
        lbl_input.pack(side=tk.LEFT)

        self.input_entry = ctk.CTkEntry(
            row1,
            textvariable=self.input_path_var,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=COLOR_INPUT_BG,
            border_color=COLOR_INPUT_BORDER,
            text_color=COLOR_TEXT_PRIMARY,
            corner_radius=8,
            height=32,
            placeholder_text="Select music folder or specific files..."
        )
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))

        btn_browse_folder = ctk.CTkButton(
            row1,
            text="Browse...",
            width=86,
            height=32,
            corner_radius=16,
            fg_color="#181c2c",
            border_width=1.2,
            border_color=COLOR_BORDER_BTN,
            hover_color="#272e48",
            text_color=COLOR_TEXT_PRIMARY,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            command=self._on_browse_folder
        )
        btn_browse_folder.pack(side=tk.LEFT, padx=(0, 6))

        btn_select_files = ctk.CTkButton(
            row1,
            text="Select Files...",
            width=96,
            height=32,
            corner_radius=16,
            fg_color="#1c2138",
            border_width=1.2,
            border_color="#4f3e72",
            hover_color="#303657",
            text_color="#c4b5fd",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            command=self._on_select_files
        )
        btn_select_files.pack(side=tk.LEFT)

        # Row 2: Output Folder
        row2 = ctk.CTkFrame(card_inner, fg_color="transparent")
        row2.pack(fill=tk.X, pady=(0, 12))

        lbl_output = ctk.CTkLabel(
            row2,
            text="Output Folder:",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLOR_TEXT_LABEL,
            width=100,
            anchor="w"
        )
        lbl_output.pack(side=tk.LEFT)

        self.output_entry = ctk.CTkEntry(
            row2,
            textvariable=self.output_path_var,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color=COLOR_INPUT_BG,
            border_color=COLOR_INPUT_BORDER,
            text_color=COLOR_TEXT_PRIMARY,
            corner_radius=8,
            height=32
        )
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))

        btn_browse_output = ctk.CTkButton(
            row2,
            text="Browse...",
            width=86,
            height=32,
            corner_radius=16,
            fg_color="#181c2c",
            border_width=1.2,
            border_color=COLOR_BORDER_BTN,
            hover_color="#272e48",
            text_color=COLOR_TEXT_PRIMARY,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            command=self._on_browse_output
        )
        btn_browse_output.pack(side=tk.LEFT)

        # Row 3: Mid-Song Read Slider + Organization Dropdown + Checkboxes
        row3 = ctk.CTkFrame(card_inner, fg_color="transparent")
        row3.pack(fill=tk.X, pady=(0, 14))

        # Left Section: Slider
        slider_box = ctk.CTkFrame(row3, fg_color="transparent")
        slider_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 20))

        self.slider_label = ctk.CTkLabel(
            slider_box,
            text=f"Mid-Song Read: {int(self.duration_var.get())}s",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_LABEL,
            anchor="w"
        )
        self.slider_label.pack(anchor="w", pady=(0, 2))

        self.slider = ctk.CTkSlider(
            slider_box,
            from_=5.0,
            to=45.0,
            variable=self.duration_var,
            command=self._on_slider_moved,
            height=14,
            progress_color="#38bdf8",
            button_color="#818cf8",
            button_hover_color="#a5b4fc",
            fg_color="#23273c"
        )
        self.slider.pack(fill=tk.X, pady=(2, 0))

        # Right Section: Organization & Checkboxes
        right_box = ctk.CTkFrame(row3, fg_color="transparent")
        right_box.pack(side=tk.RIGHT)

        # Organization dropdown
        org_box = ctk.CTkFrame(right_box, fg_color="transparent")
        org_box.pack(side=tk.LEFT, padx=(0, 16))

        lbl_org = ctk.CTkLabel(
            org_box,
            text="File Organization:",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_LABEL,
            anchor="w"
        )
        lbl_org.pack(anchor="w")

        self.org_dropdown = ctk.CTkOptionMenu(
            org_box,
            variable=self.organize_mode_var,
            values=[
                "Playlists Only (No File Moving)",
                "Copy Files to Category Folders",
                "Move Files to Category Folders"
            ],
            width=230,
            height=30,
            corner_radius=8,
            fg_color="#181c2c",
            button_color="#272e48",
            button_hover_color="#363f61",
            dropdown_fg_color="#181c2c",
            dropdown_hover_color="#2b334f",
            text_color=COLOR_TEXT_PRIMARY,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.org_dropdown.pack(pady=(2, 0))

        # Checkboxes
        cb_box = ctk.CTkFrame(right_box, fg_color="transparent")
        cb_box.pack(side=tk.LEFT, padx=(4, 0))

        self.cb_recursive = ctk.CTkCheckBox(
            cb_box,
            text="Recursive Scan",
            variable=self.recursive_var,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_LABEL,
            checkbox_width=16,
            checkbox_height=16,
            corner_radius=4,
            border_width=1.5,
            border_color="#475569",
            fg_color=COLOR_ACCENT_PURPLE,
            hover_color=COLOR_ACCENT_HOVER
        )
        self.cb_recursive.pack(anchor="w", pady=(0, 4))

        self.cb_relative = ctk.CTkCheckBox(
            cb_box,
            text="Relative Paths in Playlists",
            variable=self.use_relative_var,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_LABEL,
            checkbox_width=16,
            checkbox_height=16,
            corner_radius=4,
            border_width=1.5,
            border_color="#475569",
            fg_color=COLOR_ACCENT_PURPLE,
            hover_color=COLOR_ACCENT_HOVER
        )
        self.cb_relative.pack(anchor="w")

        # Action Buttons Row
        action_row = ctk.CTkFrame(card_inner, fg_color="transparent")
        action_row.pack(fill=tk.X, pady=(4, 2))

        self.btn_start = ctk.CTkButton(
            action_row,
            text="▶  Start Scanning & Categorizing",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=COLOR_ACCENT_PURPLE,
            hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT_PRIMARY,
            height=36,
            corner_radius=18,
            command=self._on_start_clicked
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_open_folder = ctk.CTkButton(
            action_row,
            text="📁 Open Output Folder",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="#181c2c",
            border_width=1.4,
            border_color=COLOR_ACCENT_PURPLE,
            hover_color="#282245",
            text_color="#e2e8f0",
            height=36,
            corner_radius=18,
            command=self._on_open_output_folder
        )
        self.btn_open_folder.pack(side=tk.LEFT, padx=(0, 16))

        self.status_label = ctk.CTkLabel(
            action_row,
            text="Ready to scan.",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLOR_TEXT_MUTED
        )
        self.status_label.pack(side=tk.LEFT, padx=(4, 0))

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self.card,
            height=3,
            corner_radius=0,
            fg_color="#1b1f33",
            progress_color="#38bdf8"
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # 3. Tab Bar (Classified Tracks | Activity Log)
        tabs_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        tabs_container.pack(fill=tk.X, pady=(2, 6))

        self.current_tab = "tracks"

        self.tab_tracks_btn = ctk.CTkButton(
            tabs_container,
            text="Classified Tracks",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#1b1f33",
            border_width=1,
            border_color=COLOR_ACCENT_PURPLE,
            hover_color="#262d47",
            text_color="#ffffff",
            height=28,
            corner_radius=6,
            command=self._show_tracks_tab
        )
        self.tab_tracks_btn.pack(side=tk.LEFT, padx=(0, 6))

        self.tab_log_btn = ctk.CTkButton(
            tabs_container,
            text="Activity Log",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            fg_color="transparent",
            border_width=0,
            hover_color="#1c2138",
            text_color=COLOR_TEXT_MUTED,
            height=28,
            corner_radius=6,
            command=self._show_log_tab
        )
        self.tab_log_btn.pack(side=tk.LEFT)

        # 4. Content Area (Table or Log)
        self.content_container = ctk.CTkFrame(
            self.main_container,
            fg_color=COLOR_TABLE_BG,
            corner_radius=10,
            border_width=1,
            border_color="#22283e"
        )
        self.content_container.pack(fill=tk.BOTH, expand=True)

        self._build_table_view()
        self._build_log_view()
        self._show_tracks_tab()

    def _build_table_view(self):
        self.table_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")

        columns = ("title", "category", "confidence", "secondary", "bpm", "duration", "path")
        self.tree = ttk.Treeview(
            self.table_frame,
            columns=columns,
            show="headings",
            style="Custom.Treeview",
            selectmode="browse"
        )

        self.tree.heading("title", text="Track Title", anchor="w")
        self.tree.heading("category", text="Primary Type", anchor="center")
        self.tree.heading("confidence", text="Confidence", anchor="center")
        self.tree.heading("secondary", text="Secondary Tags", anchor="w")
        self.tree.heading("bpm", text="BPM", anchor="center")
        self.tree.heading("duration", text="Length", anchor="center")
        self.tree.heading("path", text="Full Path", anchor="w")

        self.tree.column("title", width=200, minwidth=140, anchor="w")
        self.tree.column("category", width=120, minwidth=90, anchor="center")
        self.tree.column("confidence", width=95, minwidth=80, anchor="center")
        self.tree.column("secondary", width=220, minwidth=140, anchor="w")
        self.tree.column("bpm", width=65, minwidth=50, anchor="center")
        self.tree.column("duration", width=70, minwidth=60, anchor="center")
        self.tree.column("path", width=300, minwidth=180, anchor="w")

        # Scrollbar
        scroll_y = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self._on_tree_double_click)

    def _build_log_view(self):
        self.log_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")

        self.log_textbox = ctk.CTkTextbox(
            self.log_frame,
            fg_color="#0f121d",
            text_color="#cbd5e1",
            font=ctk.CTkFont(family="Consolas", size=10),
            corner_radius=8,
            wrap="word"
        )
        self.log_textbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.log_textbox.configure(state="disabled")

    def _show_tracks_tab(self):
        self.current_tab = "tracks"
        self.tab_tracks_btn.configure(
            fg_color="#1b1f33",
            border_width=1,
            text_color="#ffffff",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        )
        self.tab_log_btn.configure(
            fg_color="transparent",
            border_width=0,
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.log_frame.pack_forget()
        self.table_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    def _show_log_tab(self):
        self.current_tab = "log"
        self.tab_log_btn.configure(
            fg_color="#1b1f33",
            border_width=1,
            text_color="#ffffff",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold")
        )
        self.tab_tracks_btn.configure(
            fg_color="transparent",
            border_width=0,
            text_color=COLOR_TEXT_MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=11)
        )
        self.table_frame.pack_forget()
        self.log_frame.pack(fill=tk.BOTH, expand=True)

    def _on_slider_moved(self, val):
        self.slider_label.configure(text=f"Mid-Song Read: {int(float(val))}s")

    def _on_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Music Files")
        if folder:
            self.selected_files = []  # Clear specific files
            self.input_path_var.set(str(Path(folder).resolve()))

    def _on_select_files(self):
        """Feature: Select specific music files instead of an entire folder."""
        filetypes = [
            ("Supported Audio Files", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.aiff *.opus *.wma"),
            ("All Files", "*.*")
        ]
        files = filedialog.askopenfilenames(title="Select Specific Music Files", filetypes=filetypes)
        if files:
            self.selected_files = [Path(f).resolve() for f in files]
            if len(self.selected_files) == 1:
                self.input_path_var.set(str(self.selected_files[0]))
            else:
                names = ", ".join([f.name for f in self.selected_files[:3]])
                extra = f" (+{len(self.selected_files)-3} more)" if len(self.selected_files) > 3 else ""
                self.input_path_var.set(f"[{len(self.selected_files)} Files Selected] {names}{extra}")

    def _on_browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder for Playlists")
        if folder:
            self.output_path_var.set(str(Path(folder).resolve()))

    def _open_about_github(self):
        """Opens DangerousAngel's GitHub repository/profile."""
        webbrowser.open("https://github.com/DangerousAngel/MusicTypo")

    def _on_open_output_folder(self):
        out_dir = Path(self.output_path_var.get())
        if out_dir.exists():
            if sys.platform == "win32":
                os.startfile(str(out_dir))
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(out_dir)])
        else:
            messagebox.showinfo("Notice", f"Output folder does not exist yet: {out_dir}")

    def _on_tree_double_click(self, event):
        item_id = self.tree.focus()
        if not item_id:
            return
        values = self.tree.item(item_id, "values")
        if values and len(values) >= 7:
            filepath = values[6]
            if Path(filepath).exists() and sys.platform == "win32":
                try:
                    os.startfile(filepath)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to play file: {e}")

    def _append_log(self, text: str):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", text + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def _on_start_clicked(self):
        input_str = self.input_path_var.get().strip()
        out_str = self.output_path_var.get().strip()

        # Check if files or folder are specified
        is_files_mode = bool(self.selected_files)
        is_folder_mode = bool(input_str and Path(input_str).is_dir())

        if not is_files_mode and not is_folder_mode:
            messagebox.showwarning(
                "Input Required",
                "Please select a music folder using 'Browse...' or specific files using 'Select Files...'."
            )
            return

        if not out_str:
            messagebox.showwarning("Output Required", "Please select an output folder for playlists.")
            return

        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.status_label.configure(text="Initializing scan...")
        self.progress_bar.set(0)

        # Clear existing table items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Determine file organization mode
        org_choice = self.organize_mode_var.get()
        org_mode = None
        if "Copy" in org_choice:
            org_mode = "copy"
        elif "Move" in org_choice:
            org_mode = "move"

        target_files = list(self.selected_files) if is_files_mode else None
        target_dir = Path(input_str) if is_folder_mode else None

        # Launch background thread
        threading.Thread(
            target=self._worker_thread,
            args=(
                target_dir,
                Path(out_str),
                target_files,
                self.duration_var.get(),
                self.recursive_var.get(),
                self.use_relative_var.get(),
                org_mode
            ),
            daemon=True
        ).start()

    def _worker_thread(self, in_dir, out_dir, files, duration, recursive, relative, org_mode):
        try:
            pipeline = MusicTypoPipeline(
                snippet_duration=duration,
                secondary_threshold=0.40,
                use_relative_playlists=relative
            )

            def progress_cb(idx, total, fpath, res):
                self.msg_queue.put(("progress", (idx, total, fpath, res)))

            def log_cb(msg):
                self.msg_queue.put(("log", msg))

            res_list, stats, playlists = pipeline.process_library(
                input_dir=in_dir,
                output_dir=out_dir,
                files=files,
                recursive=recursive,
                organize_mode=org_mode,
                progress_callback=progress_cb,
                log_callback=log_cb
            )

            self.msg_queue.put(("done", (stats, playlists)))

        except Exception as e:
            self.msg_queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()

                if msg_type == "log":
                    self._append_log(data)

                elif msg_type == "progress":
                    idx, total, fpath, res = data
                    frac = (idx / total) if total > 0 else 0
                    self.progress_bar.set(frac)
                    self.status_label.configure(
                        text=f"Reading mid-song [{idx}/{total}]: {fpath.name[:32]}"
                    )

                    if res:
                        # Icon formatting matching screenshot
                        cat_icon = CATEGORY_ICONS.get(res.primary_type, "🎵")
                        prim_str = f"{cat_icon} {res.primary_type}"

                        # Secondary tags formatting with icons
                        sec_formatted = []
                        for sec in res.secondary_tags:
                            sec_icon = CATEGORY_ICONS.get(sec, "")
                            sec_formatted.append(f"{sec_icon} {sec}".strip())
                        sec_str = ", ".join(sec_formatted) if sec_formatted else "-"

                        # Duration formatting
                        dur_str = f"{int(res.total_duration // 60)}:{int(res.total_duration % 60):02d}"

                        # Add row to table
                        self.tree.insert(
                            "",
                            tk.END,
                            values=(
                                res.title,
                                prim_str,
                                f"{res.confidence:.0%}",
                                sec_str,
                                f"{int(round(res.bpm))}" if res.bpm > 0 else "-",
                                dur_str,
                                str(res.file_path)
                            )
                        )

                elif msg_type == "done":
                    stats, playlists = data
                    self.is_running = False
                    self.btn_start.configure(state="normal")
                    self.progress_bar.set(1.0)
                    self.status_label.configure(
                        text=f"Completed {stats.processed} tracks in {stats.elapsed_time:.1f}s."
                    )

                elif msg_type == "error":
                    self.is_running = False
                    self.btn_start.configure(state="normal")
                    self.status_label.configure(text="An error occurred.")
                    messagebox.showerror("Scan Error", f"Processing halted with error:\n{data}")

        except queue.Empty:
            pass

        self.after(80, self._poll_queue)


def main():
    app = ModernMusicTypoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
