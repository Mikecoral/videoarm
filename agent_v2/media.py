"""Frame sampling and encoding.

Frames are always extracted fresh from the source video at a fixed fps with
OpenCV, down-scaled to keep base64 payloads small, and cached to disk so
re-runs are cheap.  Pre-extracted release frames are never reused.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
from PIL import Image

from . import config


@dataclass
class Frame:
    timestamp: float          # seconds into the video (rounded to int second)
    path: Path                # cached jpg on disk


@dataclass
class VideoInfo:
    video_id: str
    path: Path
    duration: float
    fps: float
    width: int
    height: int
    frames: List[Frame]


def probe(video_path: Path) -> tuple[float, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = n / fps if fps else 0.0
    return duration, fps, w, h


def _save_resized(img: Image.Image, out_path: Path) -> None:
    img = img.convert("RGB")
    img.thumbnail((config.FRAME_MAX_SIDE, config.FRAME_MAX_SIDE))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=config.FRAME_JPEG_QUALITY)


def _extract_with_opencv(video_path: Path, cache_dir: Path, fps_target: float) -> List[Frame]:
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / fps_target)))
    frames: List[Frame] = []
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            break
        ts = int(round(idx / src_fps))
        out_path = cache_dir / f"{ts:06d}.jpg"
        if not out_path.exists():
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _save_resized(Image.fromarray(rgb), out_path)
        frames.append(Frame(timestamp=float(ts), path=out_path))
        idx += step
    cap.release()
    return frames



def load_video(video_id: str, video_path: Path, *, cache_root: Path) -> VideoInfo:
    """Probe video metadata and extract frames at FRAME_FPS via OpenCV.

    Frames are always extracted fresh from the source video — pre-extracted
    release frames are never reused, so every run observes the same stream.
    """
    duration, fps, w, h = probe(video_path)
    cache_dir = cache_root / video_id
    frames = _extract_with_opencv(video_path, cache_dir, config.FRAME_FPS)
    return VideoInfo(video_id=video_id, path=video_path, duration=duration,
                     fps=fps, width=w, height=h, frames=frames)


def to_data_url(frame: Frame) -> str:
    raw = frame.path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:image/jpeg;base64,{b64}"


def frames_in_window(frames: List[Frame], start: float, end: float, k: int) -> List[Frame]:
    """Pick up to ``k`` frames spread across [start, end]."""
    inside = [f for f in frames if start <= f.timestamp <= end]
    if not inside:
        # nearest single frame to the window centre
        if not frames:
            return []
        centre = (start + end) / 2
        return [min(frames, key=lambda f: abs(f.timestamp - centre))]
    if len(inside) <= k:
        return inside
    picks = []
    for i in range(k):
        pos = i / (k - 1) if k > 1 else 0.5
        target = start + pos * (end - start)
        picks.append(min(inside, key=lambda f: abs(f.timestamp - target)))
    # de-duplicate while preserving order
    seen, out = set(), []
    for f in picks:
        if f.timestamp not in seen:
            seen.add(f.timestamp)
            out.append(f)
    return out
