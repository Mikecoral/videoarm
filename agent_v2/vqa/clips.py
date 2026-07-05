from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def ffmpeg_executable() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _safe_name(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text.strip("_") or "item"


def clip_filename(video_id: str, start: float, end: float) -> str:
    start_text = f"{start:.2f}".replace(".", "p")
    end_text = f"{end:.2f}".replace(".", "p")
    return f"{_safe_name(video_id)}_{start_text}_{end_text}.mp4"


def make_clip(
    *,
    dataset_root: str | Path,
    video_path: str,
    video_id: str,
    start: float,
    end: float,
    clips_dir: str | Path,
) -> str | None:
    exe = ffmpeg_executable()
    if not exe:
        return None

    root = Path(dataset_root)
    source = Path(video_path)
    if not source.is_absolute():
        source = root / source
    if not source.exists():
        return None

    out_dir = Path(clips_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / clip_filename(video_id, start, end)
    if output.exists() and output.stat().st_size > 0:
        return str(output)

    duration = max(0.1, float(end) - float(start))
    cmd = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(start):.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return str(output)
