# Import Dependencies
import asyncio
from shazamio import Shazam
import argparse
import dotenv
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time

# Load Environment Variables
dotenv.load_dotenv(override=True)

async def main():
    # Set up Input Arguments
    parser = argparse.ArgumentParser(description="Add tracks from a Markdown results table to Spotify Liked Songs")
    parser.add_argument("--music-folder", type=str, default=os.getenv("MUSIC_FOLDER"), help="Path to the music folder")
    parser.add_argument("--client-id", type=str, default=os.getenv("SPOTIFY_CLIENT_ID"), help="Spotify Client ID")
    parser.add_argument("--client-secret", type=str, default=os.getenv("SPOTIFY_CLIENT_SECRET"), help="Spotify Client Secret")
    parser.add_argument("--redirect-uri", type=str, default=os.getenv("SPOTIFY_REDIRECT_URI"), help="Spotify Redirect URI")
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
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=args.client_id,
                                                   client_secret=args.client_secret,
                                                   redirect_uri=args.redirect_uri,
                                                   scope="user-library-modify user-library-read"))
    LOG_FILE = "failed_tracks.log"
    
    # Stats
    num_tracks = 0
    num_added = 0
    num_failed = 0
    num_already_added = 0
    failed_tracks = []

    # Get the File List
    music_path = Path(args.music_folder)
    if not music_path.exists():
        console.print(f"[red]Music folder not found:[/red] {music_path}")
        return 0
    else:
        mp3_files = sorted([p for p in music_path.rglob("*.mp3")])
        if mp3_files:
            console.print(f"[green]Found {len(mp3_files)} .mp3 files in the selected directory[/green]")
        else:
            console.print(f"[yellow]No .mp3 files found under[/yellow] {music_path}")
            return 0

    # [DEVELOPMENT | REMOVE FOR PROD]
    mp3_files = mp3_files[:10]

    num_tracks = len(mp3_files)

    # Process Each File
    for mp3_file in mp3_files:

        # Recognize the Track using Shazam
        print(f"[yellow]Recognizing:[/yellow] {mp3_file.name}")
        shazam_result = await shazam.recognize(str(mp3_file))
        if shazam_result and "track" in shazam_result:
            music_name = shazam_result['track']['share']['text']
        else:
            num_failed += 1
            failed_tracks.append(str(mp3_file))
            continue

        # Search for the Track on Spotify
        print(f"[yellow]Searching on Spotify:[/yellow] {music_name}")
        sp_result = sp.search(q=music_name, limit=3)
        contains = sp.current_user_saved_tracks_contains(tracks=[sp_result['tracks']['items'][0]['id']])[0]
        if contains:
            num_already_added += 1
        else:
            sp.current_user_saved_tracks_add(tracks=[sp_result['tracks']['items'][0]['id']])
            num_added += 1

        # Limit the Spotify API Call Rate
        print(f"[yellow]Limiting API call rate:[/yellow] Sleeping for 0.01 seconds")
        time.sleep(0.01)

    # Save the Log with all the Failed Tracks
    with open(LOG_FILE, "w") as log_file:
        for track in failed_tracks:
            log_file.write(f"{track}\n")

    # Add some Statistics of the Process
    summary = Table.grid(expand=True)
    summary.add_column(justify="left")
    summary.add_column(justify="right")
    summary.add_row(f"Found {str(num_tracks)} tracks.")
    summary.add_row(f"You had already liked {str(num_already_added)} of them.")
    summary.add_row(f"So {num_added} tracks were added.")
    if num_failed > 0:
        summary.add_row(f"Unfortunately failed to resolve {str(num_failed)} tracks.")
        summary.add_row(f"You can find the failed tracks in the log at {LOG_FILE}")
    Console().print(Panel(summary, title="Spotify Ingestion Summary", border_style="green"))

if __name__ == "__main__":
  asyncio.run(main())