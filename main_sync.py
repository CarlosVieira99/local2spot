"""Sequential local MP3 → Spotify Liked Songs ingestor.

This script scans a local folder for `.mp3` files, recognizes each track with
Shazam (async API), searches it on Spotify, and adds it to your Liked Songs.
Processing is sequential (one-by-one); `asyncio` is used only to call Shazam's
asynchronous client.

Environment variables (can be overridden via CLI flags):
    - MUSIC_FOLDER: Root folder containing `.mp3` files (recursively scanned)
    - SPOTIFY_CLIENT_ID: Spotify app Client ID
    - SPOTIFY_CLIENT_SECRET: Spotify app Client Secret
    - SPOTIFY_REDIRECT_URI: Redirect URI configured in your Spotify app
    - LOG_FILE: Optional path for the failures log (default: failed_tracks.log)
    - SPOTIFY_SEARCH_LIMIT: Optional search result limit (default: 3)

Spotify OAuth scopes required:
    - user-library-read
    - user-library-modify

Quick usage:
    python main_sync.py --music-folder C:\\Music --client-id <id> --client-secret <secret> --redirect-uri http://localhost:8080/callback
"""

# Import Basic Dependencies
import os
import sys
import time
import dotenv
import asyncio
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

# Import Music Recognition Dependencies
from shazamio import Shazam

# Import Spotify Dependencies
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Import UI Dependencies
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich import box

# Load Environment Variables
dotenv.load_dotenv(override=True)

# Constants (can be overridden by env vars above)
LOG_FILE: str = os.getenv("LOG_FILE", "failed_tracks.log")
SPOTIFY_SEARCH_LIMIT: int = int(os.getenv("SPOTIFY_SEARCH_LIMIT", "3"))
SPOTIFY_REQUEST_TIMEOUT: int = int(os.getenv("SPOTIFY_REQUEST_TIMEOUT", "30"))
SPOTIFY_RETRIES: int = int(os.getenv("SPOTIFY_RETRIES", "3"))
SPOTIFY_STATUS_RETRIES: int = int(os.getenv("SPOTIFY_STATUS_RETRIES", "3"))
SPOTIFY_BACKOFF_FACTOR: float = float(os.getenv("SPOTIFY_BACKOFF_FACTOR", "0.4"))
SPOTIFY_MAX_CALL_RETRIES: int = int(os.getenv("SPOTIFY_MAX_CALL_RETRIES", "2"))
SPOTIFY_CALL_BACKOFF_BASE: float = float(os.getenv("SPOTIFY_CALL_BACKOFF_BASE", "0.6"))

SHAZAM_TIMEOUT_SECONDS: int = int(os.getenv("SHAZAM_TIMEOUT_SECONDS", "20"))
SHAZAM_MAX_RETRIES: int = int(os.getenv("SHAZAM_MAX_RETRIES", "3"))
SHAZAM_BACKOFF_BASE: float = float(os.getenv("SHAZAM_BACKOFF_BASE", "0.8"))

# Markdown results output (can be overridden via CLI)
RESULTS_MD_FILE: str = os.getenv("RESULTS_MD_FILE", "results.md")

def _auth_spotify(client_id: str, client_secret: str, redirect_uri: str, console: Console) -> Optional[spotipy.Spotify]:
    """Authenticate with Spotify and return a client or None on failure."""
    try:
        sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-library-modify user-library-read",
            ),
            requests_timeout=SPOTIFY_REQUEST_TIMEOUT,
            retries=SPOTIFY_RETRIES,
            status_retries=SPOTIFY_STATUS_RETRIES,
            backoff_factor=SPOTIFY_BACKOFF_FACTOR,
        )
        user_info = sp.current_user()
        console.print(f"[green]✓ Authenticated as: {user_info.get('display_name', 'Unknown')}[/green]")
        return sp
    except Exception as e:
        console.print(f"[red]✗ Spotify authentication failed: {e}[/red]")
        console.print("[yellow]Make sure your Spotify app credentials are correct and you've completed the OAuth flow.[/yellow]")
        return None


def _discover_mp3s(music_folder: str, console: Console) -> List[Path]:
    """Return a sorted list of .mp3 Paths under the given folder, or an empty list with a message."""
    music_path = Path(music_folder)
    if not music_path.exists():
        console.print(f"[red]Music folder not found:[/red] {music_path}")
        return []
    mp3_files = sorted([p for p in music_path.rglob("*.mp3")])
    if not mp3_files:
        console.print(f"[yellow]No .mp3 files found under[/yellow] {music_path}")
    return mp3_files


async def _recognize_track(shazam: Shazam, mp3_file: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Recognize a track with Shazam using timeout and retries.

    Returns (title, artist, error). On success, error is None; on failure,
    title and artist are None and error contains a brief reason.
    """
    last_err: Optional[str] = None
    for attempt in range(1, SHAZAM_MAX_RETRIES + 1):
        try:
            shazam_result = await asyncio.wait_for(shazam.recognize(str(mp3_file)), timeout=SHAZAM_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            last_err = "timeout"
            shazam_result = None
        except Exception as e:
            last_err = f"exception: {e}"
            shazam_result = None

        if shazam_result and "track" in shazam_result:
            title = shazam_result["track"].get("title")
            artist = shazam_result["track"].get("subtitle")
            return title, artist, None

        if attempt < SHAZAM_MAX_RETRIES:
            backoff = SHAZAM_BACKOFF_BASE * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)

    return None, None, (last_err or "no_track_in_result")


def _normalize_query_value(value: Optional[str]) -> str:
    """Normalize query value by collapsing whitespace/newlines and escaping quotes."""
    if not value:
        return ""
    s = " ".join(str(value).split())  # collapse all whitespace/newlines
    return s.replace('"', '\\"')


def _search_spotify(sp: spotipy.Spotify, query_title: str, query_artist: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
    """Search Spotify with retries.

    Returns (track_id, track_name, track_artist, error_message, query_used).
    On failure, track_* are None and error_message is set. query_used is always returned.
    """
    last_err: Optional[str] = None
    norm_title = _normalize_query_value(query_title)
    norm_artist = _normalize_query_value(query_artist)
    if norm_title and norm_artist:
        query = f'track:"{norm_title}" artist:"{norm_artist}"'
    elif norm_title:
        query = f'track:"{norm_title}"'
    elif norm_artist:
        query = f'artist:"{norm_artist}"'
    else:
        query = ""
    for attempt in range(1, SPOTIFY_MAX_CALL_RETRIES + 1):
        try:
            sp_result = sp.search(q=query, limit=SPOTIFY_SEARCH_LIMIT)
            items = sp_result.get("tracks", {}).get("items", []) if sp_result else []
            if items:
                first = items[0]
                track_id = first.get("id")
                track_name = first.get("name")
                track_artists = ", ".join(a.get("name", "") for a in first.get("artists", []) if isinstance(a, dict)) or None
                return track_id, track_name, track_artists, None, query
            last_err = "no_match"
        except spotipy.exceptions.SpotifyException as e:
            if getattr(e, "http_status", None) == 429:
                retry_after = 1
                try:
                    headers = getattr(e, "headers", None)
                    if headers and isinstance(headers, dict):
                        ra = headers.get("Retry-After") or headers.get("retry-after")
                        if ra is not None:
                            retry_after = int(str(ra))
                except Exception:
                    retry_after = 1
                last_err = f"rate_limited_{retry_after}s"
                time.sleep(retry_after)
            else:
                last_err = f"spotify_exception: {e}"
        except Exception as e:
            last_err = f"exception: {e}"

        if attempt < SPOTIFY_MAX_CALL_RETRIES:
            backoff = SPOTIFY_CALL_BACKOFF_BASE * (2 ** (attempt - 1))
            time.sleep(backoff)

    return None, None, None, last_err, query


def _update_liked(sp: spotipy.Spotify, track_id: str) -> Tuple[str, Optional[str]]:
    """Check/add track to Liked Songs with retries.

    Returns (status, error) where status is 'already' or 'added'. On repeated failure,
    returns ("error", message).
    """
    last_err: Optional[str] = None
    for attempt in range(1, SPOTIFY_MAX_CALL_RETRIES + 1):
        try:
            contains = sp.current_user_saved_tracks_contains(tracks=[track_id])[0]
            if contains:
                return "already", None
            # sp.current_user_saved_tracks_add(tracks=[track_id])
            return "added", None
        except spotipy.exceptions.SpotifyException as e:
            if getattr(e, "http_status", None) == 429:
                retry_after = 1
                try:
                    headers = getattr(e, "headers", None)
                    if headers and isinstance(headers, dict):
                        ra = headers.get("Retry-After") or headers.get("retry-after")
                        if ra is not None:
                            retry_after = int(str(ra))
                except Exception:
                    retry_after = 1
                last_err = f"rate_limited_{retry_after}s"
                time.sleep(retry_after)
            else:
                last_err = f"spotify_exception: {e}"
            if attempt < SPOTIFY_MAX_CALL_RETRIES:
                backoff = SPOTIFY_CALL_BACKOFF_BASE * (2 ** (attempt - 1))
                time.sleep(backoff)
        except Exception as e:
            last_err = f"exception: {e}"
            if attempt < SPOTIFY_MAX_CALL_RETRIES:
                backoff = SPOTIFY_CALL_BACKOFF_BASE * (2 ** (attempt - 1))
                time.sleep(backoff)

    return "error", last_err


FailedRecord = Tuple[str, str, Optional[str]]  # (path, stage, message)


def _write_fail_log(log_file: str, failed_tracks: List[FailedRecord]) -> None:
    """Write failures to log file as tab-separated lines: path, stage, message."""
    if not failed_tracks:
        return
    with open(log_file, "w", encoding="utf-8") as f:
        for path, stage, message in failed_tracks:
            msg = message or ""
            f.write(f"{path}\t{stage}\t{msg}\n")


def _md_escape(cell: Optional[str]) -> str:
    """Escape Markdown table cell content (pipes and newlines)."""
    if not cell:
        return ""
    return str(cell).replace("|", "\\|").replace("\n", " ")


def _write_results_md(md_file: str, rows: List[Tuple[str, str, str, str]]) -> None:
    """Write results to a Markdown table.

    Columns: Filename | Search Query | Spotify Track | Artist
    """
    if not rows:
        return
    header = "| Filename | Search Query | Spotify Track | Artist |\n|---|---|---|---|\n"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(header)
        for filename, query, track_name, track_artist in rows:
            f.write(
                f"| {_md_escape(filename)} | {_md_escape(query)} | {_md_escape(track_name)} | {_md_escape(track_artist)} |\n"
            )


async def main() -> int:
    # Set up Input Arguments
    parser = argparse.ArgumentParser(
        description=(
            "Scan a folder of .mp3 files, recognize tracks with Shazam, and add matches to Spotify Liked Songs. "
            "CLI flags override environment variables."
        )
    )
    parser.add_argument("--music-folder", type=str, default=os.getenv("MUSIC_FOLDER"), help="Path to the music folder")
    parser.add_argument("--client-id", type=str, default=os.getenv("SPOTIFY_CLIENT_ID"), help="Spotify Client ID")
    parser.add_argument("--client-secret", type=str, default=os.getenv("SPOTIFY_CLIENT_SECRET"), help="Spotify Client Secret")
    parser.add_argument("--redirect-uri", type=str, default=os.getenv("SPOTIFY_REDIRECT_URI"), help="Spotify Redirect URI")
    parser.add_argument("--results-md", type=str, default=RESULTS_MD_FILE, help="Path to write Markdown results table")
    args = parser.parse_args()

    # Validate the Arguments
    if not args.music_folder:
        parser.error("Music folder path is required.")
    if not args.client_id:
        parser.error("Spotify Client ID is required.")
    if not args.client_secret:
        parser.error("Spotify Client Secret is required.")
    if not args.redirect_uri:
        parser.error("Spotify Redirect URI is required.")

    # Global Variables
    console = Console()
    shazam = Shazam()

    # Spotify Authentication
    sp = _auth_spotify(args.client_id, args.client_secret, args.redirect_uri, console)
    if sp is None:
        return 1
    
    # Stats
    num_tracks = 0
    num_added = 0
    num_failed = 0
    num_already_added = 0
    failed_tracks: List[FailedRecord] = []
    results_rows: List[Tuple[str, str, str, str]] = []  # Filename, Search Query, Spotify Track, Artist

    # Get the File List
    mp3_files = _discover_mp3s(args.music_folder, console)
    if not mp3_files:
        return 0

    num_tracks = len(mp3_files)

    # Start timing for ETA estimation
    start_time = time.perf_counter()

    # Helper to render a live status panel
    def render_status_panel(current_index: int, total: int, file_name: str,
                            status: str = "recognizing with Shazam…", recognized: Optional[str] = None) -> Panel:
        """Render a Rich Panel with progress, current file, ETA, and stats.

        Args:
            current_index: 1-based index of the current item.
            total: total number of items.
            file_name: name of the file being processed.
            status: short status text (e.g., recognition/search/added).
            recognized: recognized track display name, if available.

        Returns:
            Panel: A Rich Panel object to be displayed live.
        """
        # Truncate and pad file name for neat display
        file_name = file_name[:30] + "   "

        # Left side: info table
        info = Table.grid(padding=(0, 1), expand=True)
        info.add_column(justify="left")
        info.add_column(justify="left")
        info.add_row("Progress", f"{current_index}/{total}")
        info.add_row("File", file_name)
        info.add_row("Status", status or "-")

        # ETA estimation (simple average of elapsed per processed item)
        def _fmt_eta(seconds: int) -> str:
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        elapsed = max(0.0, time.perf_counter() - start_time)
        processed = max(1, current_index - 1)
        avg_per_item = elapsed / processed
        remaining_items = max(0, total - current_index + 1)
        eta_seconds = int(avg_per_item * remaining_items)
        info.add_row("ETA", _fmt_eta(eta_seconds))

        # Right side: stats table
        stats = Table.grid(padding=(0, 1))
        stats.add_column(justify="left")
        stats.add_row(f"Added: [green]{num_added}[/green]")
        stats.add_row(f"Already liked: [cyan]{num_already_added}[/cyan]")
        stats.add_row(f"Failed: [red]{num_failed}[/red]")

        # Wrap both into one panel
        wrapper = Table.grid(expand=True)
        wrapper.add_column()
        wrapper.add_column(ratio=1)
        wrapper.add_row(info, stats)
        return Panel(wrapper, title="Spotify Ingestion", border_style="blue")

    # Process Each File with a live Rich panel
    with Live(Panel("Preparing…", title="Spotify Ingestion", border_style="blue"), console=console, refresh_per_second=6) as live:
        for idx, mp3_file in enumerate(mp3_files, start=1):
            current_file_name = mp3_file.name
            recognized_title: Optional[str] = None
            recognized_artist: Optional[str] = None

            # Update: starting recognition
            live.update(render_status_panel(idx, num_tracks, current_file_name, status="recognizing with Shazam…"))

            # Recognize the Track using Shazam
            recognized_title, recognized_artist, recog_err = await _recognize_track(shazam, mp3_file)
            if not recognized_title:
                num_failed += 1
                failed_tracks.append((str(mp3_file), "recognition_failed", recog_err))
                # Include row for recognition failure
                results_rows.append((current_file_name, "", "-", "-"))
                live.update(render_status_panel(idx, num_tracks, current_file_name, status="failed to recognize with Shazam", recognized=recognized_title))
                continue

            # Update: searching on Spotify
            live.update(render_status_panel(idx, num_tracks, current_file_name, status="searching on Spotify…", recognized=recognized_title))

            # Search for the Track on Spotify
            track_id, track_name, track_artist, search_err, query_used = _search_spotify(sp, recognized_title, recognized_artist)
            if not track_id:
                num_failed += 1
                failed_tracks.append((str(mp3_file), "spotify_no_match", search_err))
                # Collect row even for failures
                results_rows.append((current_file_name, query_used or "", "-", "-"))
                live.update(render_status_panel(idx, num_tracks, current_file_name, status="no Spotify match found", recognized=recognized_title))
                continue

            status, like_err = _update_liked(sp, track_id)
            if status == "already":
                num_already_added += 1
                results_rows.append((current_file_name, query_used or "", track_name or "", track_artist or ""))
                live.update(render_status_panel(idx, num_tracks, current_file_name, status="already in Liked Songs", recognized=recognized_title))
            elif status == "added":
                num_added += 1
                results_rows.append((current_file_name, query_used or "", track_name or "", track_artist or ""))
                live.update(render_status_panel(idx, num_tracks, current_file_name, status="added to Liked Songs", recognized=recognized_title))
            else:
                num_failed += 1
                failed_tracks.append((str(mp3_file), "spotify_like_error", like_err))
                results_rows.append((current_file_name, query_used or "", track_name or "", track_artist or ""))
                live.update(render_status_panel(idx, num_tracks, current_file_name, status="failed to like on Spotify", recognized=recognized_title))

            # Limit the Spotify API Call Rate without blocking the event loop
            await asyncio.sleep(0.01)

    # Save the Log with all the Failed Tracks
    _write_fail_log(LOG_FILE, failed_tracks)

    # Write Markdown results table
    try:
        _write_results_md(args.results_md, results_rows)
        results_md_note = args.results_md
    except Exception:
        results_md_note = None

    # Add some Statistics of the Process (styled)
    def _fmt_duration(seconds: int) -> str:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

    elapsed_total = int(max(0.0, time.perf_counter() - start_time))
    border = "green" if num_failed == 0 else ("yellow" if num_failed <= 3 else "red")

    summary = Table.grid(expand=True)
    summary.add_column(justify="left", style="bold")
    summary.add_column(justify="right")
    summary.add_row("🎵 Found", f"[bold]{num_tracks}[/bold] tracks")
    summary.add_row("💚 Already liked", f"[cyan]{num_already_added}[/cyan]")
    summary.add_row("➕ Added", f"[green]{num_added}[/green]")
    if num_failed > 0:
        summary.add_row("❌ Failed", f"[red]{num_failed}[/red]")
        summary.add_row("🗒 Log", f"[magenta]{LOG_FILE}[/magenta]")
    if results_md_note:
        summary.add_row("📄 Results", f"[magenta]{results_md_note}[/magenta]")

    Console().print(
        Panel(
            summary,
            title="[bold]Spotify Ingestion Summary[/bold]",
            subtitle=f"Duration: {_fmt_duration(elapsed_total)}",
            subtitle_align="left",
            border_style=border,
            box=box.ROUNDED,
        )
    )

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))