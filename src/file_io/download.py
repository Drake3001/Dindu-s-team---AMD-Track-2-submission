from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import structlog

log = structlog.get_logger(__name__)

CHUNK_SIZE = 1024 * 1024
DEFAULT_VIDEO_EXT = ".mp4"


def _extension_from_url(url: str) -> str:
    """Derive a file extension from a video URL, defaulting to .mp4."""
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix else DEFAULT_VIDEO_EXT


def expected_video_path(
    task_id: str,
    url: str,
    videos_dir: str | Path = "videos",
) -> Path:
    """Return the expected local path for a task's downloaded video."""
    return Path(videos_dir) / f"{task_id}{_extension_from_url(url)}"


def download_for_task(
    task_id: str,
    url: str,
    videos_dir: str | Path = "videos",
) -> Path:
    """Download a task video and save it as videos/{task_id}{ext}."""
    filename = f"{task_id}{_extension_from_url(url)}"
    return download_video(url, videos_dir=videos_dir, filename=filename)


def download_video(
    url: str,
    videos_dir: str | Path = "videos",
    filename: str | None = None,
) -> Path:
    """Download a video from a URL into the videos directory."""
    directory = Path(videos_dir)
    directory.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = Path(urlparse(url).path).name or "video.mp4"

    output_path = directory / filename
    log.info("downloading_video", url=url, path=str(output_path))

    with urlopen(url) as response, output_path.open("wb") as f:
        while chunk := response.read(CHUNK_SIZE):
            f.write(chunk)

    log.info("video_downloaded", path=str(output_path), size_bytes=output_path.stat().st_size)
    return output_path
