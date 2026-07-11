from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np

from preprocessing.types import PreprocessingError, VideoMetadata


def ensure_ffmpeg() -> tuple[str, str]:
    """Return (ffmpeg, ffprobe) paths or raise if either is missing."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise PreprocessingError("ffmpeg/ffprobe not found on PATH")
    return ffmpeg, ffprobe


def _parse_fps(value: str | None) -> float:
    if not value or value in {"0/0", "0"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _scaled_dims(width: int, height: int, max_dim: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise PreprocessingError("Video has invalid dimensions")
    longest = max(width, height)
    if longest <= max_dim:
        return width, height
    scale = max_dim / float(longest)
    return max(1, int(width * scale)), max(1, int(height * scale))


def probe_metadata(video_path: Path | str) -> VideoMetadata:
    """Read video metadata via ffprobe."""
    _, ffprobe = ensure_ffmpeg()
    video_path = Path(video_path)

    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "unknown ffprobe error"
        raise PreprocessingError(f"ffprobe failed for {video_path}: {stderr}")

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as error:
        raise PreprocessingError(f"ffprobe returned invalid JSON for {video_path}") from error

    streams = payload.get("streams") or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise PreprocessingError(f"No video stream found in {video_path}")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))

    frame_count = 0
    nb_frames = video_stream.get("nb_frames")
    if nb_frames is not None:
        try:
            frame_count = int(nb_frames)
        except (TypeError, ValueError):
            frame_count = 0

    duration = 0.0
    format_info = payload.get("format") or {}
    if format_info.get("duration") is not None:
        try:
            duration = float(format_info["duration"])
        except (TypeError, ValueError):
            duration = 0.0
    elif video_stream.get("duration") is not None:
        try:
            duration = float(video_stream["duration"])
        except (TypeError, ValueError):
            duration = 0.0

    if frame_count <= 0 and duration > 0 and fps > 0:
        frame_count = int(round(duration * fps))

    if duration <= 0 and frame_count > 0 and fps > 0:
        duration = frame_count / fps

    return VideoMetadata(
        path=str(video_path),
        duration_sec=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def iter_frames(
    video_path: Path | str,
    *,
    max_dim: int,
    metadata: VideoMetadata | None = None,
) -> Iterator[np.ndarray]:
    """Decode and downscale frames via ffmpeg, yielding BGR numpy arrays."""
    ffmpeg, _ = ensure_ffmpeg()
    video_path = Path(video_path)
    meta = metadata or probe_metadata(video_path)
    target_w, target_h = _scaled_dims(meta.width, meta.height, max_dim)
    frame_bytes = target_w * target_h * 3

    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={target_w}:{target_h}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.stdout is None:
        proc.kill()
        proc.wait()
        raise PreprocessingError(f"ffmpeg failed to open stdout pipe for {video_path}")

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(target_h, target_w, 3).copy()
    finally:
        proc.stdout.close()
        return_code = proc.wait()
        if return_code != 0:
            raise PreprocessingError(f"ffmpeg decode failed for {video_path} (exit {return_code})")
