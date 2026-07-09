"""CLI for validating Drive credentials and uploading output files."""

from __future__ import annotations

import fire
import structlog
from googleapiclient.errors import HttpError

from file_io.api import configure_logging
from file_io.drive_upload import login, upload_outputs, validate_drive_config, verify_drive_target

log = structlog.get_logger(__name__)


def check() -> int:
    """Validate Google Drive OAuth config and target folder access."""
    configure_logging()

    try:
        target = verify_drive_target()
    except HttpError as error:
        print(f"Could not verify target folder: {error}")
        return 1

    if not target["ok"]:
        print("Drive configuration is incomplete:")
        for issue in target["issues"]:
            print(f"  - {issue}")
        return 1

    if target.get("target_issue"):
        print(f"Target folder issue: {target['target_issue']}")
        return 1

    print("Drive configuration looks valid.")
    print(f"  folder_id: {target['folder_id']}")
    print(f"  oauth_client: {target['oauth_client_file']}")
    print(f"  oauth_token: {target['oauth_token_file']}")
    print(f"  target folder: {target['folder_name']}")
    return 0


def auth() -> int:
    """Authorize Google Drive access via OAuth (opens browser once)."""
    configure_logging()
    try:
        login()
    except (FileNotFoundError, ValueError) as error:
        print(str(error))
        return 1

    print("OAuth login successful. Token saved for future uploads.")
    return 0


def upload(output_dir: str = "output", subfolder_name: str | None = None) -> int:
    """Upload JSON files from a local directory to Google Drive."""
    configure_logging()
    result = validate_drive_config()
    if not result["ok"]:
        print("Drive configuration is incomplete:")
        for issue in result["issues"]:
            print(f"  - {issue}")
        return 1

    try:
        uploaded_ids = upload_outputs(output_dir, subfolder_name=subfolder_name)
    except (ValueError, FileNotFoundError) as error:
        print(str(error))
        return 1
    except HttpError as error:
        print(f"Google Drive upload failed: {error}")
        return 1

    if not uploaded_ids:
        print(f"No JSON files found in {output_dir}")
        return 0

    print(f"Uploaded {len(uploaded_ids)} file(s) to Google Drive.")
    return 0


def cli() -> None:
    fire.Fire({"check": check, "login": auth, "upload": upload})


if __name__ == "__main__":
    cli()
