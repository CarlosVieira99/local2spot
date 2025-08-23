"""Async local MP3 → Spotify Liked Songs ingestor with bounded concurrency.

This script scans a local folder for `.mp3` files, recognizes each track with
Shazam (async API), searches it on Spotify, and adds it to your Liked Songs.
It applies async concurrency patterns while preserving the original logic,
retries, logging, and output format.

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
    python main_async.py --music-folder C:\\Music --client-id <id> --client-secret <secret> --redirect-uri http://localhost:8080/callback [--concurrency 4]
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
from concurrent.futures import ThreadPoolExecutor

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

# Markdown Results Output
RESULTS_MD_FILE: str = os.getenv("RESULTS_MD_FILE", "results.md")

# Load Environment Variables
dotenv.load_dotenv(override=True)

# Constants (can be overridden by env vars above)
LOG_FILE: str = os.getenv("LOG_FILE", "failed_tracks.log")
SPOTIFY_SEARCH_LIMIT: int = int(os.getenv("SPOTIFY_SEARCH_LIMIT", "3"))
SPOTIFY_REQUEST_TIMEOUT: int = int(os.getenv("SPOTIFY_REQUEST_TIMEOUT", "10"))
SPOTIFY_RETRIES: int = int(os.getenv("SPOTIFY_RETRIES", "3"))
SPOTIFY_STATUS_RETRIES: int = int(os.getenv("SPOTIFY_STATUS_RETRIES", "3"))
SPOTIFY_BACKOFF_FACTOR: float = float(os.getenv("SPOTIFY_BACKOFF_FACTOR", "0.4"))
SPOTIFY_MAX_CALL_RETRIES: int = int(os.getenv("SPOTIFY_MAX_CALL_RETRIES", "2"))
SPOTIFY_CALL_BACKOFF_BASE: float = float(os.getenv("SPOTIFY_CALL_BACKOFF_BASE", "0.6"))

SHAZAM_TIMEOUT_SECONDS: int = int(os.getenv("SHAZAM_TIMEOUT_SECONDS", "20"))
SHAZAM_MAX_RETRIES: int = int(os.getenv("SHAZAM_MAX_RETRIES", "3"))
SHAZAM_BACKOFF_BASE: float = float(os.getenv("SHAZAM_BACKOFF_BASE", "0.8"))


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


def _markdown_cell(value: Optional[str]) -> str:
    """Sanitize a value for inclusion in a Markdown table cell."""
    if value is None:
        return ""
    s = str(value).replace("\n", " ").strip()
    # Escape pipe which breaks table formatting
    return s.replace("|", "\\|")


async def _search_spotify_async(sp: spotipy.Spotify, query_title: str, query_artist: str, *, call_spotify, api_backoff_base: float) -> Tuple[Optional[str], Optional[str]]:
    """Search Spotify with retries using async executor wrapper. Returns (track_id, error)."""
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
            sp_result = await call_spotify(sp.search, q=query, limit=SPOTIFY_SEARCH_LIMIT, timeout=20)
            items = sp_result.get("tracks", {}).get("items", []) if sp_result else []
            if items:
                return items[0]["id"], None
            last_err = "no_match"
        except spotipy.exceptions.SpotifyException as e:
            # Explicitly handle rate limiting with Retry-After header when available
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
                await asyncio.sleep(retry_after)
            else:
                last_err = f"spotify_exception: {e}"
        except Exception as e:
            last_err = f"exception: {e}"

        if attempt < SPOTIFY_MAX_CALL_RETRIES:
            backoff = api_backoff_base * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)

    return None, last_err


async def _update_liked_async(sp: spotipy.Spotify, track_id: str, *, call_spotify, api_backoff_base: float) -> Tuple[str, Optional[str]]:
    """Check/add track to Liked Songs with retries using async executor wrapper.

    Returns (status, error) where status is 'already' or 'added'. On repeated failure,
    returns ("error", message).

    Note: To preserve original behavior, we DO NOT actually add the track.
    """
    last_err: Optional[str] = None
    for attempt in range(1, SPOTIFY_MAX_CALL_RETRIES + 1):
        try:
            contains_list = await call_spotify(sp.current_user_saved_tracks_contains, tracks=[track_id], timeout=20)
            contains = bool(contains_list[0]) if contains_list else False
            if contains:
                return "already", None
            await call_spotify(sp.current_user_saved_tracks_add, tracks=[track_id], timeout=20)
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
                await asyncio.sleep(retry_after)
            else:
                last_err = f"spotify_exception: {e}"
        except Exception as e:
            last_err = f"exception: {e}"
            if attempt < SPOTIFY_MAX_CALL_RETRIES:
                backoff = api_backoff_base * (2 ** (attempt - 1))
                await asyncio.sleep(backoff)

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
    parser.add_argument(
        "--concurrency",
        type=int,
        default= max(2, min(8, (os.cpu_count() or 4))),
        help="Max number of files to process concurrently",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=os.getenv("RESULTS_MD_FILE", RESULTS_MD_FILE),
        help="Path to write Markdown summary table",
    )
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

    # Concurrency Level
    console.print(f"[blue]Using concurrency level: {args.concurrency}[/blue]")

    # Spotify Authentication
    sp = _auth_spotify(args.client_id, args.client_secret, args.redirect_uri, console)
    if sp is None:
        return 1

    # Stats shared across tasks
    num_tracks = 0
    num_added = 0
    num_failed = 0
    num_already_added = 0
    processed_count = 0
    failed_tracks: List[FailedRecord] = []
    # Collected results for Markdown export: (Filename, Search Query, Spotify Track, Artist)
    md_rows: List[Tuple[str, str, str, str]] = []

    # Get the File List
    mp3_files = _discover_mp3s(args.music_folder, console)
    if not mp3_files:
        return 0

    num_tracks = len(mp3_files)

    # Start timing for ETA estimation
    start_time = time.perf_counter()

    # Concurrency controls and Spotify executor (single-thread due to requests.Session thread-safety)
    max_concurrency = max(1, int(args.concurrency))
    api_pause = 0.01  # light pacing between Spotify calls
    lock = asyncio.Lock()  # to guard stats and UI updates
    spotify_executor = ThreadPoolExecutor(max_workers=1)
    per_file_timeout = int(os.getenv("PER_FILE_TIMEOUT_SECONDS", "120"))

    # Async wrapper for Spotify calls using a dedicated executor and small pacing between calls
    async def call_spotify(fn, *f_args, timeout: Optional[float] = None, **f_kwargs):
        loop = asyncio.get_running_loop()
        def _call():
            return fn(*f_args, **f_kwargs)
        try:
            res = await asyncio.wait_for(loop.run_in_executor(spotify_executor, _call), timeout=timeout)
        finally:
            await asyncio.sleep(api_pause)
        return res

    # Helper to render a live status panel using shared counters
    def render_status_panel(current_file: str = "", status: str = "") -> Panel:
        nonlocal processed_count, num_added, num_already_added, num_failed
        info = Table.grid(padding=(0, 1), expand=True)
        info.add_column(justify="left")
        info.add_column(justify="left")
        info.add_row("Progress", f"{processed_count}/{num_tracks}")
        info.add_row("File", (current_file[:30] + "   ") if current_file else "-")
        info.add_row("Status", status or "-")

        # ETA estimation (simple average of elapsed per processed item)
        eta_str = "--:--"
        if processed_count > 0:
            elapsed = max(0.0, time.perf_counter() - start_time)
            avg_per_item = elapsed / processed_count
            remaining_items = max(0, num_tracks - processed_count)
            eta_seconds = int(avg_per_item * remaining_items)
            # format mm:ss or h:mm:ss
            h, rem = divmod(eta_seconds, 3600)
            m, s = divmod(rem, 60)
            eta_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        info.add_row("ETA", eta_str)

        stats = Table.grid(padding=(0, 1))
        stats.add_column(justify="left")
        stats.add_row(f"Added: [green]{num_added}[/green]")
        stats.add_row(f"Already liked: [cyan]{num_already_added}[/cyan]")
        stats.add_row(f"Failed: [red]{num_failed}[/red]")

        wrapper = Table.grid(expand=True)
        wrapper.add_column()
        wrapper.add_column(ratio=1)
        wrapper.add_row(info, stats)
        return Panel(wrapper, title="Spotify Ingestion", border_style="blue")

    async def update_status(live: Live, file_name: str, status: str):
        live.update(render_status_panel(file_name, status))

    async def process_file(file_path: Path, live: Live, shazam: Shazam):
        nonlocal num_added, num_failed, num_already_added, processed_count
        file_name = file_path.name

        await update_status(live, file_name, "recognizing with Shazam…")

        # Reuse a dedicated Shazam instance per worker to avoid shared state issues
        recognized_title: Optional[str] = None
        recognized_artist: Optional[str] = None
        try:
            recognized_title, recognized_artist, recog_err = await _recognize_track(shazam, file_path)
        except Exception as e:
            recognized_title, recognized_artist, recog_err = None, None, f"exception: {e}"

        if not recognized_title:
            async with lock:
                num_failed += 1
                failed_tracks.append((str(file_path), "recognition_failed", recog_err))
                processed_count += 1
                # Append empty query/track info for this file
                md_rows.append((file_name, "", "", ""))
            await update_status(live, file_name, "failed to recognize with Shazam")
            return

        await update_status(live, file_name, "searching on Spotify…")

        # Build the exact search query string (same logic as in _search_spotify_async)
        norm_title = _normalize_query_value(recognized_title)
        norm_artist = _normalize_query_value(recognized_artist)
        if norm_title and norm_artist:
            search_query = f'track:"{norm_title}" artist:"{norm_artist}"'
        elif norm_title:
            search_query = f'track:"{norm_title}"'
        elif norm_artist:
            search_query = f'artist:"{norm_artist}"'
        else:
            search_query = ""

        # Search Spotify using the original query logic but via async wrapper
        track_id, search_err = await _search_spotify_async(
            sp,
            recognized_title,
            recognized_artist,
            call_spotify=call_spotify,
            api_backoff_base=SPOTIFY_CALL_BACKOFF_BASE,
        )
        if not track_id:
            async with lock:
                num_failed += 1
                failed_tracks.append((str(file_path), "spotify_no_match", search_err))
                processed_count += 1
                # Append row with query but no match
                md_rows.append((file_name, search_query, "", ""))
            await update_status(live, file_name, "no Spotify match found")
            return

        # Get track details for Markdown table (name and primary artist)
        track_name: str = ""
        artist_name: str = ""
        try:
            track_obj = await call_spotify(sp.track, track_id, timeout=20)
            if isinstance(track_obj, dict):
                track_name = track_obj.get("name") or ""
                artists = track_obj.get("artists") or []
                if artists and isinstance(artists, list):
                    artist_name = artists[0].get("name", "")
        except Exception:
            # Non-fatal; leave names empty if fetch fails
            track_name = track_name or ""
            artist_name = artist_name or ""

        # Check/add liked status using async wrapper (no actual add to preserve behavior)
        status, like_err = await _update_liked_async(
            sp, track_id, call_spotify=call_spotify, api_backoff_base=SPOTIFY_CALL_BACKOFF_BASE
        )
        if status == "already":
            async with lock:
                num_already_added += 1
                processed_count += 1
                md_rows.append((file_name, search_query, track_name, artist_name))
            await update_status(live, file_name, "already in Liked Songs")
        elif status == "added":
            async with lock:
                num_added += 1
                processed_count += 1
                md_rows.append((file_name, search_query, track_name, artist_name))
            await update_status(live, file_name, "added to Liked Songs")
        else:
            async with lock:
                num_failed += 1
                failed_tracks.append((str(file_path), "spotify_like_error", like_err))
                processed_count += 1
                md_rows.append((file_name, search_query, track_name, artist_name))
            await update_status(live, file_name, "failed to like on Spotify")

        await asyncio.sleep(0)

    async def runner(files: List[Path]):
        queue: asyncio.Queue[Path] = asyncio.Queue()
        for f in files:
            queue.put_nowait(f)

        async def worker(live: Live, wid: int):
            nonlocal num_failed, processed_count
            shazam = Shazam()
            while True:
                try:
                    f = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if queue.empty():
                        break
                    await asyncio.sleep(0)
                    continue
                try:
                    try:
                        await asyncio.wait_for(process_file(f, live, shazam), timeout=per_file_timeout)
                    except asyncio.TimeoutError:
                        async with lock:
                            num_failed += 1
                            failed_tracks.append((str(f), "file_timeout", f"exceeded {per_file_timeout}s"))
                            processed_count += 1
                            # Append empty row on timeout
                            md_rows.append((f.name, "", "", ""))
                        await update_status(live, f.name, "timed out; skipping")
                finally:
                    queue.task_done()

        with Live(Panel("Preparing…", title="Spotify Ingestion", border_style="blue"), console=console, refresh_per_second=8) as live:
            workers = [asyncio.create_task(worker(live, i)) for i in range(max_concurrency)]
            await queue.join()
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    try:
        await runner(mp3_files)
    finally:
        # Ensure the Spotify executor is shutdown cleanly
        spotify_executor.shutdown(wait=True, cancel_futures=False)

    # Save the Log with all the Failed Tracks
    _write_fail_log(LOG_FILE, failed_tracks)

    # Write Markdown results table
    try:
        with open(args.markdown_out, "w", encoding="utf-8") as f:
            f.write("| Filename | Search Query | Spotify Track | Artist |\n")
            f.write("|---|---|---|---|\n")
            for fn, q, tn, ar in md_rows:
                f.write(
                    f"| {_markdown_cell(fn)} | {_markdown_cell(q)} | {_markdown_cell(tn)} | {_markdown_cell(ar)} |\n"
                )
    except Exception as e:
        console.print(f"[yellow]Warning: failed to write markdown output: {e}[/yellow]")

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
    summary.add_row("📄 Markdown", f"[magenta]{args.markdown_out}[/magenta]")

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

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))