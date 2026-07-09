# Scripts

All commands are run from the project root (`AMD/`).

Install dependencies and register scripts first:

```bash
uv sync
```

---

## `bench`

**Purpose:** Benchmark the preprocessing pipeline for all tasks in `tasks.json`. Optionally downloads videos (renamed to `{task_id}.mp4`), preprocesses each video sequentially, and writes a JSON timing report to `output/processing/`.

**Defined in:** `pyproject.toml` → `[project.scripts]`

**Usage:**

```bash
uv run bench
uv run bench --tasks input/tasks.json
uv run bench --skip_download=True
uv run bench --fps 2 --max_dim 384 --prune_threshold 8 --grid_cols 3 --grid_rows 3
uv run bench --runs 3
uv run bench --help
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tasks` | str | `input/tasks.json` | Path to tasks JSON file (always read) |
| `--videos_dir` | str | `videos` | Directory for downloaded videos |
| `--output_dir` | str | `output` | Base output directory; reports go to `output/processing/` |
| `--fps` | float | `1.0` | Target frames per second to sample |
| `--max_dim` | int | `512` | Max longest edge when resizing frames |
| `--prune_threshold` | float | `5.0` | Min mean abs-diff to keep a frame (drops near-duplicates) |
| `--grid_cols` | int | `4` | Columns in each base64 grid montage |
| `--grid_rows` | int | `4` | Rows in each base64 grid montage |
| `--max_frames` | int | `240` | Safety cap on sampled frames |
| `--skip_download` | bool | `False` | Skip download; assume `videos/{task_id}.ext` exists |
| `--runs` | int | `1` | Repeat full benchmark N times |

**Output:** `output/processing/bench_YYYYMMDD_HHMMSS.json` with run parameters, per-video metadata, frame/grid counts, and per-phase timings.

---

## `main`

**Purpose:** Demo script for the `file_io` module. Reads `input/tasks.json` and writes a timestamped JSON file to `output/`.

**Defined in:** `src/main.py` (not registered in `pyproject.toml`)

**Usage:**

```bash
uv run python src/main.py
```

**Arguments:** None.

---

## `preprocessing.preprocessing`

**Purpose:** Run the full preprocessing pipeline once for a single task — sample+downscale, prune, and build in-memory base64 grids.

**Defined in:** `src/preprocessing/preprocessing.py` (`__main__`)

**Usage:**

```bash
uv run python -m preprocessing.preprocessing v1 videos/v1.mp4
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `task_id` | yes | Task identifier |
| `video_path` | yes | Path to the local video file |

Processed output is held in memory (base64 grids). Use `preprocessing.api.preprocess()` from Python code for the public interface.

---

## Adding a new script

1. Add a `cli()` entry point (or similar) in your module.
2. Register it in `pyproject.toml`:

```toml
[project.scripts]
my-script = "mypackage.mymodule:cli"
```

3. Run `uv sync` to install the new script.
4. Document it in this file.
