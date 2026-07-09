import json
from datetime import datetime, timezone
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

DEFAULT_OUTPUT_DIR = "output"


def resolve_output_dir(output_dir: str | Path | None = None) -> Path:
    """Resolve the output directory, defaulting to `output/`."""
    return Path(output_dir or DEFAULT_OUTPUT_DIR)


def write_output(
    data: list[dict],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    filename_prefix: str = "results",
) -> Path:
    """Write a list of dictionaries to a single timestamped JSON file.

    Creates `output_dir` if it does not exist. The entire list is written
    into one file named `{filename_prefix}_YYYYMMDD_HHMMSS.json`.
    """
    directory = resolve_output_dir(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = directory / f"{filename_prefix}_{timestamp}.json"

    log.info("writing_output", path=str(output_path), count=len(data))

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    log.info("output_written", path=str(output_path))
    return output_path
