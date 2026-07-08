import json
from datetime import datetime, timezone
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def write_output(data: list[dict], output_dir: str | Path = "output") -> Path:
    """Write a list of dictionaries to a timestamped JSON file."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = directory / f"results_{timestamp}.json"

    log.info("writing_output", path=str(output_path), count=len(data))

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    log.info("output_written", path=str(output_path))
    return output_path
