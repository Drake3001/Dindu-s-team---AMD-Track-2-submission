# AMD

## Setup

```bash
uv sync
```

## Output

Write a list of dicts to a timestamped JSON file in any directory:

```python
from file_io.api import configure_logging, save_output

configure_logging()
output_path = save_output(tasks, output_dir="output")
```

## Google Drive Upload

Upload JSON files from a local output directory to Google Drive. After each successful upload, the local file is deleted to avoid duplicate uploads.

### 1. Create a service account

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project and enable the **Google Drive API**.
3. Create a **Service Account** and download its JSON key.
4. Save the key locally, e.g. `credentials/service-account.json`.

### 2. Share the target Drive folder

1. Open your Google Drive folder.
2. Share it with the service account email (from the JSON file, e.g. `...@....iam.gserviceaccount.com`) as **Editor**.

### 3. Configure `.env`

Copy `.env.example` to `.env` and set:

```env
GOOGLE_DRIVE=https://drive.google.com/drive/folders/YOUR_FOLDER_ID
GDRIVE_SERVICE_ACCOUNT_FILE=credentials/service-account.json
```

`GOOGLE_DRIVE` accepts either a shared folder URL or a raw folder ID. You can also use `GDRIVE_FOLDER_ID` instead.

### 4. Upload outputs

```python
from file_io.api import configure_logging, save_output, upload_to_drive

configure_logging()

output_path = save_output(tasks, output_dir="output")
uploaded_ids = upload_to_drive("output", subfolder_name="run_2026_07_08")
```

If `subfolder_name` is provided, the uploader creates that subfolder under the configured Drive folder when it does not already exist.
