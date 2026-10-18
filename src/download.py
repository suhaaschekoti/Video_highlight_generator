"""Fetch a video from a URL (YouTube etc.) with yt-dlp into data/input/. Free, no key."""

import subprocess
from pathlib import Path


def download(url: str, out_dir: str = "data/input", max_height: int = 720) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tmpl = str(Path(out_dir) / "%(id)s.%(ext)s")
    fmt = f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best"
    r = subprocess.run(["yt-dlp", "-f", fmt, "--merge-output-format", "mp4", "--no-playlist",
                        "-o", tmpl, "--print", "after_move:filepath", url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Download failed: {r.stderr.strip()[-300:]}")
    path = r.stdout.strip().splitlines()[-1]
    return path
