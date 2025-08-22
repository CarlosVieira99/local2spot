# local2spot

Recognize local `.mp3` files with Shazam and add matches to your Spotify Liked Songs. Includes both a simple sequential script and a fast async version with bounded concurrency, a live TUI, retries/backoff, and failure logging.

Note: By default, the scripts do not perform the final “add to Liked Songs” call; they only check if a track is already liked and report “added” as a dry run. See the note in Usage to enable actual adding.

## Features and File Overview

- Shazam recognition: identifies local `.mp3` tracks using the `shazamio` async API
	- `main_sync.py` (sequential; async only to call Shazam)
	- `main_async.py` (concurrent; bounded workers for faster throughput)
- Spotify integration: searches Spotify and checks if tracks are already in Liked Songs
- Dry-run by default: reports “added” without modifying your library; toggle a single line to enable real adds
- Live terminal UI: progress, ETA, and stats rendered with `rich`
- Resilient: timeouts, retries, and exponential backoff for Shazam and Spotify calls
- Configurable: `.env` support via `python-dotenv`; CLI flags override env vars
- Failure log: saves unrecognized or unmatched files to `failed_tracks.log`

Other files:

- `.env.template` – copy to `.env` and fill in credentials and options
- `music/` – put your `.mp3` files here (or point `MUSIC_FOLDER` elsewhere)
- `requirements.txt`, `uv.lock`, `pyproject.toml` – dependency and project metadata

## Getting Started

### 1. Clone the Repository

```sh
git clone <this-repo-url>
cd <this-repo>
```

### 2. Setup Python Environment and Install Dependencies

Requires UV to be installed (Ruff is optional for linting):

```sh
uv sync
```

Alternatively (without UV):

```sh
python -m venv .venv
. .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.template` to `.env` and fill values according to your setup. You’ll need a Spotify Developer app with the proper redirect URI.

Required:

- `SPOTIFY_CLIENT_ID` – Your Spotify application Client ID
- `SPOTIFY_CLIENT_SECRET` – Your Spotify application Client Secret
- `SPOTIFY_REDIRECT_URI` – Redirect URI configured in your Spotify app (e.g., `http://127.0.0.1:8888/callback`)
- `MUSIC_FOLDER` – Path to the root folder containing `.mp3` files (recursively scanned)

Optional (tuning):

- `SPOTIFY_SEARCH_LIMIT` – Search result limit (default: 3)
- `SPOTIFY_REQUEST_TIMEOUT`, `SPOTIFY_RETRIES`, `SPOTIFY_STATUS_RETRIES`, `SPOTIFY_BACKOFF_FACTOR`
- `SPOTIFY_MAX_CALL_RETRIES`, `SPOTIFY_CALL_BACKOFF_BASE`
- `SHAZAM_TIMEOUT_SECONDS`, `SHAZAM_MAX_RETRIES`, `SHAZAM_BACKOFF_BASE`
- `LOG_FILE` – Where to write the failures log (default: `failed_tracks.log`)

Scopes: The app needs `user-library-read` and `user-library-modify` configured. During the first run you’ll complete the OAuth flow in the browser.

### 4. Usage

Run sequential (simpler; good baseline):

```sh
uv run main_sync.py --music-folder "<path-to-music>" --client-id <id> --client-secret <secret> --redirect-uri "http://127.0.0.1:8888/callback"
```

Run concurrent (faster on big folders):

```sh
uv run main_async.py --music-folder "<path-to-music>" --client-id <id> --client-secret <secret> --redirect-uri "http://127.0.0.1:8888/callback" --concurrency 4
```

Notes:

- CLI flags override values from `.env`.
- OAuth opens a browser; return to the terminal after authorizing.
- Dry run by default: to actually add tracks to Liked Songs, open the files and uncomment the lines with `current_user_saved_tracks_add`.
	- Search in `main_sync.py` and `main_async.py` for `current_user_saved_tracks_add` and remove the comment to enable real adds.

## Why `main_sync.py` uses asyncio

`main_sync.py` processes files one-by-one (sequentially). It still defines `async def main()` and runs with `asyncio.run(...)` because the Shazam client (`shazamio`) exposes an asynchronous API. In short:

- The pipeline is sequential; files are handled in series.
- `async/await` is used solely to call Shazam’s async recognition method.
- There’s no concurrency or task scheduling; each file is awaited before moving to the next.

This keeps behavior simple and predictable while satisfying the async requirement of the Shazam library. If you want true parallelism, use `main_async.py`.

## Contact

For inquiries or further information, please reach out:

- Carlos Vieira — [car.vieira@outlook.pt](mailto:car.vieira@outlook.pt)

