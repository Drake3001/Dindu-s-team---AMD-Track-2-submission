import json
import os
import re
from pathlib import Path

import structlog
from dotenv import load_dotenv, find_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = structlog.get_logger(__name__)

# drive.file only allows access to files the app created/opened interactively.
# Full drive scope is required to upload into an existing folder configured in .env.
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
JSON_MIME_TYPE = "application/json"

DEFAULT_OAUTH_CLIENT_FILE = Path("credentials/oauth-client.json")
DEFAULT_OAUTH_TOKEN_FILE = Path("credentials/token.json")

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(find_dotenv(filename="credentials/.env"))
        _ENV_LOADED = True


def _parse_folder_id(value: str) -> str:
    """Extract a Google Drive folder ID from a URL or return the raw ID."""
    value = value.strip()
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", value)
    if match:
        return match.group(1)
    return value


def _oauth_paths() -> tuple[Path, Path]:
    _ensure_env_loaded()
    client_file = Path(
        os.getenv("GDRIVE_OAUTH_CLIENT_FILE", str(DEFAULT_OAUTH_CLIENT_FILE))
    )
    token_file = Path(
        os.getenv("GDRIVE_OAUTH_TOKEN_FILE", str(DEFAULT_OAUTH_TOKEN_FILE))
    )
    return client_file, token_file


def validate_drive_config() -> dict:
    """Check that Drive upload configuration files are present."""
    _ensure_env_loaded()

    issues: list[str] = []
    folder_value = os.getenv("GOOGLE_DRIVE") or os.getenv("GDRIVE_FOLDER_ID")
    client_file, token_file = _oauth_paths()

    if not folder_value:
        issues.append("Set GOOGLE_DRIVE or GDRIVE_FOLDER_ID in credentials/.env")

    if not client_file.exists():
        issues.append(
            f"OAuth client file not found: {client_file}. "
            "Download a Desktop OAuth client JSON from Google Cloud Console."
        )
    else:
        try:
            data = json.loads(client_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            issues.append(f"OAuth client file is not valid JSON: {client_file}")
        else:
            client_config = data.get("installed") or data.get("web")
            if not client_config:
                issues.append(
                    f"OAuth client JSON must contain an 'installed' or 'web' section: {client_file}"
                )
            elif "client_id" not in client_config or "client_secret" not in client_config:
                issues.append(f"OAuth client JSON is missing client_id or client_secret: {client_file}")

    if not token_file.exists():
        issues.append(
            f"OAuth token not found: {token_file}. Run: uv run upload-drive login"
        )

    folder_id = _parse_folder_id(folder_value) if folder_value else None
    ok = len(issues) == 0
    if ok:
        log.info(
            "drive_config_valid",
            folder_id=folder_id,
            oauth_client=str(client_file),
            oauth_token=str(token_file),
        )
    else:
        for issue in issues:
            log.error("drive_config_invalid", issue=issue)

    return {
        "ok": ok,
        "issues": issues,
        "folder_id": folder_id,
        "oauth_client_file": str(client_file),
        "oauth_token_file": str(token_file),
    }


def login() -> Credentials:
    """Run the OAuth installed-app flow and save the token for future uploads."""
    _ensure_env_loaded()
    client_file, token_file = _oauth_paths()

    if not client_file.exists():
        raise FileNotFoundError(
            f"OAuth client file not found: {client_file}. "
            "Create a Desktop OAuth client in Google Cloud Console and download the JSON."
        )

    log.info("oauth_login_started", client_file=str(client_file))
    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), DRIVE_SCOPES)
    credentials = flow.run_local_server(port=0)

    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(credentials.to_json(), encoding="utf-8")
    log.info("oauth_login_completed", token_file=str(token_file))
    return credentials


def get_drive_credentials() -> Credentials:
    """Load cached OAuth credentials, refreshing the access token when needed."""
    client_file, token_file = _oauth_paths()

    if not client_file.exists():
        raise FileNotFoundError(f"OAuth client file not found: {client_file}")

    if not token_file.exists():
        raise ValueError(
            f"No OAuth token found at {token_file}. Run: uv run upload-drive login"
        )

    credentials = Credentials.from_authorized_user_file(str(token_file), DRIVE_SCOPES)
    if credentials.expired and credentials.refresh_token:
        log.info("oauth_token_refreshing")
        credentials.refresh(Request())
        token_file.write_text(credentials.to_json(), encoding="utf-8")
        log.info("oauth_token_refreshed", token_file=str(token_file))
    elif not credentials.valid:
        raise ValueError(
            f"OAuth token at {token_file} is invalid or expired. Run: uv run upload-drive login"
        )

    return credentials


def verify_drive_target() -> dict:
    """Validate config, OAuth token, and access to the target Drive folder."""
    result = validate_drive_config()
    if not result["ok"]:
        return {**result, "folder_name": None, "target_issue": None, "authenticated": False}

    try:
        folder_id = _get_folder_id()
        service = get_drive_service()
        folder = verify_folder_access(service, folder_id)
    except (ValueError, FileNotFoundError) as error:
        return {
            **result,
            "folder_name": None,
            "target_issue": str(error),
            "authenticated": False,
        }

    return {
        **result,
        "folder_name": folder.get("name"),
        "target_issue": None,
        "authenticated": True,
    }


def _get_folder_id() -> str:
    """Load the target Drive folder ID from environment."""
    _ensure_env_loaded()

    folder_value = os.getenv("GOOGLE_DRIVE") or os.getenv("GDRIVE_FOLDER_ID")
    if not folder_value:
        log.error("drive_config_missing", variable="GOOGLE_DRIVE or GDRIVE_FOLDER_ID")
        raise ValueError("Set GOOGLE_DRIVE or GDRIVE_FOLDER_ID in credentials/.env")

    folder_id = _parse_folder_id(folder_value)
    log.info("drive_config_loaded", folder_id=folder_id)
    return folder_id


def _drive_kwargs() -> dict:
    return {"supportsAllDrives": True}


def _list_kwargs() -> dict:
    return {"supportsAllDrives": True, "includeItemsFromAllDrives": True}


def _escape_query_value(value: str) -> str:
    return value.replace("'", "\\'")


def verify_folder_access(service, folder_id: str) -> dict:
    """Confirm the authenticated user can access the target folder."""
    try:
        folder = (
            service.files()
            .get(fileId=folder_id, fields="id,name", **_drive_kwargs())
            .execute()
        )
    except HttpError as error:
        if error.resp.status == 404:
            raise ValueError(
                f"Folder not found or not accessible: {folder_id}. "
                "Check GOOGLE_DRIVE in credentials/.env points to a folder you own "
                "or can edit, then re-run: uv run upload-drive login"
            ) from error
        raise

    log.info(
        "drive_target_verified",
        folder_id=folder_id,
        folder_name=folder.get("name"),
    )
    return folder


def get_drive_service():
    """Create an authenticated Google Drive API service client using OAuth."""
    credentials = get_drive_credentials()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def ensure_subfolder(service, parent_folder_id: str, subfolder_name: str) -> str:
    """Return an existing subfolder ID or create one under the parent folder."""
    safe_name = _escape_query_value(subfolder_name)
    query = (
        f"name = '{safe_name}' and "
        f"'{parent_folder_id}' in parents and "
        f"mimeType = '{FOLDER_MIME_TYPE}' and trashed = false"
    )
    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id, name)",
            pageSize=1,
            **_list_kwargs(),
        )
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
    created = (
        service.files()
        .create(body=metadata, fields="id", **_drive_kwargs())
        .execute()
    )
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
        .create(body=metadata, media_body=media, fields="id", **_drive_kwargs())
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

    Reads configuration from `credentials/.env`:
    - `GOOGLE_DRIVE` or `GDRIVE_FOLDER_ID`: target folder URL or ID
    - `GDRIVE_OAUTH_CLIENT_FILE`: OAuth desktop client JSON (optional)
    - `GDRIVE_OAUTH_TOKEN_FILE`: saved OAuth token (optional)

    Run `uv run upload-drive login` once to authorize access.

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

    parent_folder_id = _get_folder_id()
    service = get_drive_service()
    verify_folder_access(service, parent_folder_id)

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
