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

## Model Client

Call OpenAI-compatible chat completion APIs through OpenRouter or Fireworks using the official `openai` Python SDK:

```env
MODEL_PROVIDER=openrouter
OPENROUTER_API_KEY=YOUR_OPENROUTER_API_KEY
OPENROUTER_MODEL=YOUR_OPENROUTER_MODEL

FIREWORKS_API_KEY=YOUR_FIREWORKS_API_KEY
FIREWORKS_MODEL=YOUR_FIREWORKS_MODEL
MODEL_TIMEOUT_SECONDS=60
MODEL_TEMPERATURE=0.7
MODEL_MAX_TOKENS=512
```

```python
from model_client import generate_text

text = generate_text(
    "You write concise captions for videos.",
    "Write one formal caption for a product demo clip.",
)
```

For VLM models, pass base64 images as raw base64 strings or full data URLs:

```python
from model_client import generate_from_images_base64

text = generate_from_images_base64(
    ["BASE64_FRAME_1", "data:image/jpeg;base64,BASE64_FRAME_2"],
    "You analyze video frames.",
    "Describe what happens across these frames.",
)
```

For one image containing a grid of video frames:

```python
from model_client import generate_from_frame_grid_base64

text = generate_from_frame_grid_base64(
    "BASE64_FRAME_GRID",
    "You analyze video frame grids.",
    "Describe the video sequence.",
)
```

## Google Drive Upload (OAuth)

Upload JSON files from a local output directory to your personal Google Drive folder. After each successful upload, the local file is deleted to avoid duplicate uploads.

### 1. Enable Google Drive API

1. Open [Google Cloud Console](https://console.cloud.google.com/).
2. Create or select a project and enable the **Google Drive API**.
3. Configure the **OAuth consent screen** (External is fine for personal use; add your email as a test user if the app is in Testing mode).
4. Create an **OAuth client ID** of type **Desktop app**.
5. Download the JSON and save it as `credentials/oauth-client.json`.

### 2. Configure `credentials/.env`

Copy `credentials/.env.example` to `credentials/.env` and set:

```env
GOOGLE_DRIVE=https://drive.google.com/drive/folders/YOUR_FOLDER_ID
GDRIVE_OAUTH_CLIENT_FILE=credentials/oauth-client.json
GDRIVE_OAUTH_TOKEN_FILE=credentials/token.json
```

`GOOGLE_DRIVE` accepts either a shared folder URL or a raw folder ID. You can also use `GDRIVE_FOLDER_ID` instead.

### 3. Authorize once

```bash
uv run upload-drive login
```

This opens a browser so you can sign in with your Google account and grant Drive access. The token is saved locally to `credentials/token.json`.

### 4. Validate

```bash
uv run upload-drive check
```

### 5. Upload outputs

```bash
uv run upload-drive upload --output_dir output
uv run upload-drive upload --output_dir output --subfolder_name run_2026_07_09
```

Or from Python:

```python
from file_io.api import configure_logging, save_output, upload_to_drive

configure_logging()

output_path = save_output(tasks, output_dir="output")
uploaded_ids = upload_to_drive("output", subfolder_name="run_2026_07_08")
```

If `subfolder_name` is provided, the uploader creates that subfolder under the configured Drive folder when it does not already exist.
