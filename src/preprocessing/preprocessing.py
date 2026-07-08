"""
Video preprocessing pipeline.

Turns an already-downloaded .mp4 (from /video) into a small, bounded set
of resized frames that a VLM can consume cheaply and quickly, instead of
decoding the full clip downstream.

Downloading is NOT this module's responsibility - another component places
files in /video before this runs. Other modules should not import this
file directly - go through `api.py`.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import structlog

log = structlog.get_logger(__name__)

DEFAULT_MAX_FRAMES = 240
DEFAULT_MAX_DIM = 512
DEFAULT_TARGET_FPS = 1.0
DEFAULT_MIN_FRAMES = 8
DEFAULT_VIDEO_DIR = Path("videos")
DEFAULT_OUTPUT_DIR = Path("preprocessed_input")


class PreprocessingError(Exception):
    """Raised for any unrecoverable failure in the preprocessing stage."""


@dataclass
class Frame:
    index: int            # position within the sampled sequence
    timestamp: float      # seconds into the source video
    image: np.ndarray     # BGR ndarray (opencv convention)


@dataclass
class VideoMetadata:
    path: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass
class PreprocessResult:
    task_id: str
    source: str
    metadata: VideoMetadata
    frames: List[Frame]
    saved_paths: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Locating the already-downloaded video file
# --------------------------------------------------------------------------

def _candidate_filenames(video_ref: str, task_id: Optional[str]) -> List[str]:
    """Build a list of filenames to look for in video_dir, in priority
    order: exact ref as given, basename of a URL-shaped ref, task_id with
    common extensions."""
    candidates = [video_ref]

    parsed = urlparse(video_ref)
    if parsed.scheme in ("http", "https") and parsed.path:
        basename = Path(parsed.path).name
        if basename:
            candidates.append(basename)

    if task_id:
        for ext in (".mp4", ".mov", ".mkv", ".webm"):
            candidates.append(f"{task_id}{ext}")

    # de-dupe while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def resolve_video(video_ref: str, video_dir: Path = DEFAULT_VIDEO_DIR, task_id: Optional[str] = None) -> Path:
    """Locate an already-downloaded video file for this task.

    `video_ref` is normally the task's `video_url` from tasks.json (or,
    for local testing, a bare filename / full path). This function never
    downloads anything - it only looks on disk, either at `video_ref`
    directly (if it's already a valid path) or inside `video_dir`, trying
    the original URL's basename and, as a last resort, `{task_id}.<ext>`.
    """
    direct = Path(video_ref)
    if direct.is_file():
        return direct

    video_dir = Path(video_dir)
    for name in _candidate_filenames(video_ref, task_id):
        candidate = video_dir / name
        if candidate.is_file():
            return candidate

    available = sorted(p.name for p in video_dir.glob("*")) if video_dir.is_dir() else []
    raise PreprocessingError(
        f"Could not find a local video for ref={video_ref!r} task_id={task_id!r} "
        f"in {video_dir}. Files present: {available}"
    )


# --------------------------------------------------------------------------
# Frame budget
# --------------------------------------------------------------------------

def compute_frame_budget(
    duration_sec: float,
    max_frames: int = DEFAULT_MAX_FRAMES,
    target_fps: float = DEFAULT_TARGET_FPS,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> int:
    """Derive a frame count from clip length (~target_fps), capped at max_frames."""
    if duration_sec <= 0:
        return min_frames
    return min(max_frames, max(min_frames, int(duration_sec * target_fps)))


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

def read_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreprocessingError(f"Could not open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        if duration <= 0:
            duration = _probe_duration_ffprobe(video_path)
        return VideoMetadata(
            path=str(video_path), duration_sec=duration, fps=fps,
            frame_count=frame_count, width=width, height=height,
        )
    finally:
        cap.release()


def _probe_duration_ffprobe(video_path: Path) -> float:
    """Fallback for containers where opencv reports a bogus frame count
    (common with VFR mp4s). Requires ffmpeg/ffprobe on PATH."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001
        log.warning("ffprobe_duration_fallback_failed", path=str(video_path))
        return 0.0


# --------------------------------------------------------------------------
# Resizing
# --------------------------------------------------------------------------

def _resize_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _resized_dims(width: int, height: int, max_dim: int) -> Tuple[int, int]:
    longest = max(width, height)
    if longest <= max_dim:
        return width, height
    scale = max_dim / float(longest)
    return max(1, int(width * scale)), max(1, int(height * scale))


def _effective_fps(metadata: VideoMetadata) -> float:
    if metadata.fps > 0:
        return metadata.fps
    if metadata.duration_sec > 0 and metadata.frame_count > 0:
        return metadata.frame_count / metadata.duration_sec
    return 0.0


def _iter_frames_ffmpeg(video_path: Path, metadata: VideoMetadata, max_dim: int) -> Iterator[Tuple[float, np.ndarray]]:
    out_w, out_h = _resized_dims(metadata.width, metadata.height, max_dim)
    frame_bytes = out_w * out_h * 3
    fps = _effective_fps(metadata)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"scale={out_w}:{out_h}",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-vsync",
        "0",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdout is None:
        proc.kill()
        raise PreprocessingError("ffmpeg stdout pipe unavailable")

    frame_idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            img = np.frombuffer(raw, dtype=np.uint8).reshape((out_h, out_w, 3))
            ts = (frame_idx / fps) if fps > 0 else 0.0
            yield ts, img
            frame_idx += 1
    finally:
        if proc.stdout:
            proc.stdout.close()

    stderr_txt = ""
    if proc.stderr:
        stderr_txt = proc.stderr.read().decode("utf-8", errors="replace").strip()
        proc.stderr.close()
    exit_code = proc.wait()
    if exit_code != 0:
        raise PreprocessingError(f"ffmpeg decode failed with code={exit_code}: {stderr_txt}")


def _iter_frames_cv2(video_path: Path, metadata: VideoMetadata, max_dim: int) -> Iterator[Tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PreprocessingError(f"Could not open video for sampling: {video_path}")
    fps = _effective_fps(metadata)
    idx = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok or img is None:
                break
            ts = (idx / fps) if fps > 0 else 0.0
            yield ts, _resize_max_dim(img, max_dim)
            idx += 1
    finally:
        cap.release()


def _iter_frames_sequential(video_path: Path, metadata: VideoMetadata, max_dim: int) -> Iterator[Tuple[float, np.ndarray]]:
    try:
        yield from _iter_frames_ffmpeg(video_path, metadata, max_dim)
        return
    except Exception as exc:  # noqa: BLE001
        log.info("ffmpeg_decode_fallback_to_cv2", reason=str(exc), path=str(video_path))
    yield from _iter_frames_cv2(video_path, metadata, max_dim)


# --------------------------------------------------------------------------
# Frame sampling
# --------------------------------------------------------------------------

def _uniform_timestamps(duration: float, n: int) -> List[float]:
    """n timestamps spread across the clip. Trims a small margin off each
    end since first/last frames are often black or mid-transition."""
    if n <= 1 or duration <= 0:
        return [duration / 2.0] if duration > 0 else [0.0]
    margin = duration * 0.03
    start, end = margin, duration - margin
    if end <= start:
        start, end = 0.0, duration
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def sample_frames_uniform(
    video_path: Path, metadata: VideoMetadata, max_frames: int, max_dim: int
) -> List[Frame]:
    """Evenly spaced frames from a single sequential decode pass."""
    timestamps = _uniform_timestamps(metadata.duration_sec, max_frames)
    if not timestamps:
        raise PreprocessingError("No target timestamps generated")

    frames: List[Frame] = []
    target_idx = 0
    prev: Optional[Tuple[float, np.ndarray]] = None

    for ts, img in _iter_frames_sequential(video_path, metadata, max_dim):
        while target_idx < len(timestamps) and ts >= timestamps[target_idx]:
            target_ts = timestamps[target_idx]
            chosen_ts, chosen_img = ts, img
            if prev is not None and abs(prev[0] - target_ts) <= abs(ts - target_ts):
                chosen_ts, chosen_img = prev
            frames.append(Frame(index=len(frames), timestamp=chosen_ts, image=chosen_img))
            target_idx += 1
        prev = (ts, img)
        if target_idx >= len(timestamps):
            break

    # If timestamps remain, duplicate the last decoded frame to keep cardinality stable.
    if prev is not None:
        while target_idx < len(timestamps):
            frames.append(Frame(index=len(frames), timestamp=prev[0], image=prev[1]))
            target_idx += 1

    if not frames:
        raise PreprocessingError(f"No frames could be extracted from {video_path}")
    return frames


def sample_frames_adaptive(
    video_path: Path, metadata: VideoMetadata, max_frames: int, max_dim: int
) -> List[Frame]:
    """Over-samples uniformly, scores frame-to-frame visual change, then
    keeps the highest-motion frames (plus first/last) so multi-shot or
    fast-cut clips aren't under-represented. Falls back to uniform sampling
    on any failure so this is always safe to call."""
    try:
        candidate_n = min(max_frames * 2, 120)
        candidates = sample_frames_uniform(video_path, metadata, candidate_n, max_dim)
        if len(candidates) <= max_frames:
            return candidates

        diffs: List[float] = [0.0]
        prev_gray = None
        for frame in candidates:
            gray = cv2.cvtColor(cv2.resize(frame.image, (64, 36)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diffs.append(float(np.mean(cv2.absdiff(gray, prev_gray))))
            prev_gray = gray

        first_idx = 0
        last_idx = len(candidates) - 1
        scored = list(zip(range(len(candidates)), diffs))
        pool = [v for v in scored if v[0] not in (first_idx, last_idx)]
        pool.sort(key=lambda v: v[1], reverse=True)

        chosen = {first_idx, last_idx}
        for i, _score in pool:
            if len(chosen) >= max_frames:
                break
            chosen.add(i)

        selected = [candidates[i] for i in sorted(chosen)]
        return [
            Frame(index=out_idx, timestamp=f.timestamp, image=f.image)
            for out_idx, f in enumerate(selected)
        ]
    except Exception as exc:  # noqa: BLE001
        log.info("adaptive_fallback_to_uniform", reason=str(exc))
        return sample_frames_uniform(video_path, metadata, max_frames, max_dim)


# --------------------------------------------------------------------------
# Saving (results land in /preprocessed_input)
# --------------------------------------------------------------------------

def save_frames(frames: List[Frame], out_dir: Path, task_id: str, quality: int = 85) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in frames:
        fname = f"{task_id}_frame{f.index:02d}_t{f.timestamp:.2f}.jpg"
        fpath = out_dir / fname
        ok = cv2.imwrite(str(fpath), f.image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise PreprocessingError(f"Failed to write frame: {fpath}")
        paths.append(str(fpath))
    return paths


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def preprocess_video(
    task_id: str,
    video_ref: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_dim: int = DEFAULT_MAX_DIM,
    strategy: str = "adaptive",
    video_dir: Path = DEFAULT_VIDEO_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    save_to_disk: bool = True,
) -> PreprocessResult:
    """Full pipeline: locate local file in video_dir -> metadata -> sample -> (save).

    `video_ref` is typically the task's `video_url` field - it's used only
    to figure out which file in `video_dir` belongs to this task, nothing
    is fetched over the network here.
    """
    local_path = resolve_video(video_ref, video_dir, task_id=task_id)
    metadata = read_metadata(local_path)
    frame_budget = compute_frame_budget(metadata.duration_sec, max_frames=max_frames)

    log.info(
        "preprocessing_started",
        task_id=task_id,
        path=str(local_path),
        duration_sec=metadata.duration_sec,
        resolution=f"{metadata.width}x{metadata.height}",
        frame_budget=frame_budget,
        strategy=strategy,
    )

    if strategy == "uniform":
        frames = sample_frames_uniform(local_path, metadata, frame_budget, max_dim)
    else:
        frames = sample_frames_adaptive(local_path, metadata, frame_budget, max_dim)

    saved_paths: List[str] = []
    if save_to_disk:
        saved_paths = save_frames(frames, Path(output_dir) / task_id, task_id)

    log.info(
        "preprocessing_complete",
        task_id=task_id,
        duration_sec=metadata.duration_sec,
        num_frames=len(frames),
        frame_budget=frame_budget,
        strategy=strategy,
    )
    return PreprocessResult(
        task_id=task_id, source=str(local_path), metadata=metadata, frames=frames, saved_paths=saved_paths,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python -m preprocessing.preprocessing <task_id> <filename in /video>")
        sys.exit(1)
    tid, ref = sys.argv[1], sys.argv[2]
    try:
        res = preprocess_video(tid, ref)
        print(f"{tid}: {len(res.frames)} frames, duration={res.metadata.duration_sec:.1f}s, "
              f"size={res.metadata.width}x{res.metadata.height}")
    except PreprocessingError as e:
        print(f"{tid}: FAILED - {e}")
        sys.exit(1)
