import logging
from pathlib import Path

import structlog

from file_io.download import download_video
from file_io.drive_upload import upload_outputs
from file_io.input import read_input
from file_io.output import resolve_output_dir, write_output


def configure_logging() -> None:
    """Configure structlog for structured console output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def load_input(file_path: str | Path) -> list[dict]:
    """Load tasks from a JSON input file."""
    return read_input(file_path)


def save_output(
    data: list[dict],
    output_dir: str | Path = "output",
    filename_prefix: str = "results",
) -> Path:
    """Save processed data to a timestamped JSON output file in the given directory."""
    return write_output(data, output_dir, filename_prefix=filename_prefix)


def upload_to_drive(
    output_dir: str | Path = "output",
    subfolder_name: str | None = None,
) -> list[str]:
    """Upload JSON output files to Google Drive and remove uploaded local files."""
    return upload_outputs(output_dir, subfolder_name=subfolder_name)
