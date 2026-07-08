import os
import re
from pathlib import Path

import structlog
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

log = structlog.get_logger(__name__)

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
JSON_MIME_TYPE = "application/json"

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv()
        _ENV_LOADED = True


def _parse_folder_id(value: str) -> str:
    """Extract a Google Drive folder ID from a URL or return the raw ID."""
    value = value.strip()
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", value)
    if match:
        return match.group(1)
    return value


def _get_config() -> tuple[str, Path]:
    """Load Drive folder ID and service account file path from environment."""
    _ensure_env_loaded()

    folder_value = os.getenv("GOOGLE_DRIVE") or os.getenv("GDRIVE_FOLDER_ID")
    if not folder_value:
        log.error("drive_config_missing", variable="GOOGLE_DRIVE or GDRIVE_FOLDER_ID")
        raise ValueError("Set GOOGLE_DRIVE or GDRIVE_FOLDER_ID in .env")

    service_account_file = os.getenv("GDRIVE_SERVICE_ACCOUNT_FILE")
    if not service_account_file:
        log.error("drive_config_missing", variable="GDRIVE_SERVICE_ACCOUNT_FILE")
        raise ValueError("Set GDRIVE_SERVICE_ACCOUNT_FILE in .env")

    service_account_path = Path(service_account_file)
    if not service_account_path.exists():
        log.error("service_account_not_found", path=str(service_account_path))
        raise FileNotFoundError(f"Service account file not found: {service_account_path}")

    folder_id = _parse_folder_id(folder_value)
    log.info("drive_config_loaded", folder_id=folder_id, service_account=str(service_account_path))
    return folder_id, service_account_path


def get_drive_service(service_account_file: str | Path):
    """Create an authenticated Google Drive API service client."""
    credentials = Credentials.from_service_account_file(
        str(service_account_file),
        scopes=[DRIVE_SCOPE],
    )
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def ensure_subfolder(service, parent_folder_id: str, subfolder_name: str) -> str:
    """Return an existing subfolder ID or create one under the parent folder."""
    query = (
        f"name = '{subfolder_name}' and "
        f"'{parent_folder_id}' in parents and "
        f"mimeType = '{FOLDER_MIME_TYPE}' and trashed = false"
    )
    response = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)", pageSize=1)
        .execute()
    )
    files = response.get("files", [])
    if files:
        folder_id = files[0]["id"]
        log.info("drive_subfolder_exists", name=subfolder_name, folder_id=folder_id)
        return folder_id

    metadata = {
        "name": subfolder_name,
        "mimeType": FOLDER_MIME_TYPE,
        "parents": [parent_folder_id],
    }
    created = service.files().create(body=metadata, fields="id").execute()
    folder_id = created["id"]
    log.info("drive_subfolder_created", name=subfolder_name, folder_id=folder_id)
    return folder_id


def upload_file_to_drive(service, local_path: Path, folder_id: str) -> str:
    """Upload a local file to the given Drive folder and return the file ID."""
    metadata = {"name": local_path.name, "parents": [folder_id]}
    media = MediaFileUpload(str(local_path), mimetype=JSON_MIME_TYPE, resumable=True)

    log.info("drive_upload_started", path=str(local_path), folder_id=folder_id)
    created = (
        service.files()
        .create(body=metadata, media_body=media, fields="id")
        .execute()
    )
    file_id = created["id"]
    log.info("drive_upload_completed", path=str(local_path), file_id=file_id)
    return file_id


def upload_outputs(
    output_dir: str | Path = "output",
    subfolder_name: str | None = None,
) -> list[str]:
    """Upload JSON files from a local directory to Google Drive.

    Reads configuration from `.env`:
    - `GOOGLE_DRIVE` or `GDRIVE_FOLDER_ID`: target folder URL or ID
    - `GDRIVE_SERVICE_ACCOUNT_FILE`: path to service account JSON key

    After each successful upload, the local file is deleted to avoid
    duplicate uploads on subsequent runs.
    """
    directory = Path(output_dir)
    if not directory.exists():
        log.error("output_dir_not_found", path=str(directory))
        raise FileNotFoundError(f"Output directory not found: {directory}")

    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        log.info("no_output_files_to_upload", path=str(directory))
        return []

    parent_folder_id, service_account_file = _get_config()
    service = get_drive_service(service_account_file)

    target_folder_id = parent_folder_id
    if subfolder_name:
        target_folder_id = ensure_subfolder(service, parent_folder_id, subfolder_name)

    uploaded_ids: list[str] = []
    for json_file in json_files:
        file_id = upload_file_to_drive(service, json_file, target_folder_id)
        uploaded_ids.append(file_id)
        json_file.unlink()
        log.info("local_output_removed", path=str(json_file))

    log.info("drive_upload_batch_completed", count=len(uploaded_ids), folder_id=target_folder_id)
    return uploaded_ids
