"""Structured run-trace recorder -> run_log.json."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, List


class RunLog:
    def __init__(self) -> None:
        self.videos: List[Dict[str, Any]] = []
        self._current: Dict[str, Any] | None = None

    def start_video(self, video_id: str, video_info: Dict[str, Any]) -> None:
        self._current = {"video_id": video_id, "video_info": video_info, "steps": []}
        self.videos.append(self._current)

    def add(self, step: str, *, inputs: Any = None, output: Any = None,
            elapsed: float | None = None, **extra: Any) -> None:
        entry: Dict[str, Any] = {"step": step}
        if inputs is not None:
            entry["input"] = inputs
        if output is not None:
            entry["output"] = output
        if elapsed is not None:
            entry["elapsed"] = round(elapsed, 3)
        entry.update(extra)
        assert self._current is not None, "start_video() must be called first"
        self._current["steps"].append(entry)

    @contextmanager
    def timed(self, step: str, *, inputs: Any = None, **extra: Any):
        t0 = time.time()
        holder: Dict[str, Any] = {}
        try:
            yield holder
        finally:
            self.add(step, inputs=inputs, output=holder.get("output"),
                     elapsed=time.time() - t0, **extra)

    def as_list(self) -> List[Dict[str, Any]]:
        return self.videos
