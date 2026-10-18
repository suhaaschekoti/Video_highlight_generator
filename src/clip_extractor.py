"""Stage 4: FFmpeg cutting + concat, captions, chapters, manifests, label export."""

import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from src.fusion import Clip


def _cut(video_path: str, clip: Clip, out_path: str, has_video: bool) -> None:
    cmd = ["ffmpeg", "-y", "-ss", f"{clip.start:.3f}", "-i", video_path, "-t", f"{clip.end - clip.start:.3f}"]
    if has_video:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-r", "30", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    else:
        cmd += ["-vn"]
    cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", "-avoid_negative_ts", "make_zero", out_path]
    subprocess.run(cmd, check=True, capture_output=True)


def _srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def write_srt(clips: list[Clip], path: str) -> None:
    t = 0.0
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(clips, 1):
            d = c.end - c.start
            f.write(f"{i}\n{_srt_time(t)} --> {_srt_time(t + d)}\n{c.text.strip()}\n\n")
            t += d


def write_chapters(chapters: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chapters:
            m, s = divmod(int(c["start"]), 60)
            h, m = divmod(m, 60)
            stamp = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
            f.write(f"{stamp} {c['title']}\n")


def build_reel(video_path: str, clips: list[Clip], out_path: str, has_video: bool = True,
               burn_subtitles: bool = False, srt_path: str | None = None) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, c in enumerate(clips):
            p = str(Path(tmp) / f"clip_{i:03d}.mp4")
            _cut(video_path, c, p, has_video)
            parts.append(p)
        lst = str(Path(tmp) / "list.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{p}'\n" for p in parts)
        concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy"]
        if burn_subtitles and srt_path and has_video:
            tmp_out = str(Path(tmp) / "concat.mp4")
            subprocess.run(concat + [tmp_out], check=True, capture_output=True)
            srt_esc = str(Path(srt_path).resolve()).replace("\\", "/").replace(":", "\\:")
            subprocess.run(["ffmpeg", "-y", "-i", tmp_out, "-vf", f"subtitles='{srt_esc}'", "-c:a", "copy",
                            out_path], check=True, capture_output=True)
        else:
            subprocess.run(concat + [out_path], check=True, capture_output=True)
    return out_path


def write_manifests(clips: list[Clip], scored: list[dict], meta: dict, chapters: list[dict],
                    json_path: str, txt_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "clips": [asdict(c) for c in clips], "chapters": chapters,
                   "segments": scored}, f, indent=1, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{meta.get('title', 'Highlights')}\n{'=' * 60}\n{meta.get('summary', '')}\n\n")
        w = meta.get("weights", {})
        f.write(f"Source: {meta.get('source')} | domain: {meta.get('domain')} | weights "
                + ", ".join(f"{k}={v:.2f}" for k, v in w.items()) + "\n")
        f.write(f"Reel {meta.get('reel_duration_s', 0):.1f}s from {meta.get('source_duration_s', 0):.1f}s "
                f"({meta.get('compression_ratio', 0):.1f}x)\n\n")
        if chapters:
            f.write("Chapters\n")
            for c in chapters:
                f.write(f"  {int(c['start']) // 60}:{int(c['start']) % 60:02d}  {c['title']}\n")
            f.write("\n")
        for i, c in enumerate(clips, 1):
            f.write(f"Clip {i}: [{c.start:8.1f}s - {c.end:8.1f}s]  score={c.score:.2f}\n"
                    f"   why: {c.reason}\n   \"{c.text.strip()}\"\n\n")


def export_truth(clips: list[Clip], path: str) -> None:
    """Save the (possibly hand-edited) clip set as a ground-truth label file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"highlights": [{"start": c.start, "end": c.end} for c in clips]}, f, indent=1)
