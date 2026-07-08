import json
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def read_input(file_path: str | Path) -> list[dict]:
    """Read a JSON file and return its contents as a list of dictionaries."""
    path = Path(file_path)
    log.info("reading_input", path=str(path))

    if not path.exists():
        log.error("input_file_not_found", path=str(path))
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        log.error("invalid_input_format", expected="list", got=type(data).__name__)
        raise ValueError("Input JSON must be a list of dictionaries")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            log.error("invalid_item_format", index=index, expected="dict", got=type(item).__name__)
            raise ValueError(f"Item at index {index} must be a dictionary")

    log.info("input_loaded", count=len(data))
    return data
