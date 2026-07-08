from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import structlog

log = structlog.get_logger(__name__)

CHUNK_SIZE = 1024 * 1024


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
