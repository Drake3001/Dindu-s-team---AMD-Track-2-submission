# Scripts

All commands are run from the project root (`AMD/`).

Install dependencies and register scripts first:

```bash
uv sync
```

---

## `bench`

**Purpose:** Benchmark preprocessing timing for a single already-downloaded video. Reports per-phase timings (metadata read, frame sampling, disk save) and optional multi-run averages.

**Defined in:** `pyproject.toml` → `[project.scripts]`

**Usage:**

```bash
uv run bench --task_id v1 --video clip1.mp4
uv run bench v1 clip1.mp4
uv run bench --task_id v1 --video clip1.mp4 --runs 5 --save=False
uv run bench --task_id v1 --video clip1.mp4 --strategy uniform --max_frames 8
uv run bench --help
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `task_id` | str | *(required)* | Task identifier (positional or `--task_id`) |
| `video` | str | *(required)* | Filename inside `videos/`, or a full path |
| `--video_dir` | str | `videos` | Directory containing downloaded videos |
| `--output_dir` | str | `preprocessed_input` | Where sampled frames are saved |
| `--max_frames` | int | `240` | Upper bound on frames to sample |
| `--max_dim` | int | `512` | Max width/height when resizing frames |
| `--strategy` | str | `adaptive` | `uniform` or `adaptive` sampling |
| `--runs` | int | `3` | Repeat benchmark N times and log averages |
| `--save` | bool | `True` | Write frames to disk; use `--save=False` to skip |

**Short flags:** `-v` (`video_dir`), `-o` (`output_dir`), `-r` (`runs`)

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

**Purpose:** Run the full preprocessing pipeline once for a single task — locate video, read metadata, sample frames, and save to disk.

**Defined in:** `src/preprocessing/preprocessing.py` (`__main__`)

**Usage:**

```bash
uv run python -m preprocessing.preprocessing v1 clip1.mp4
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `task_id` | yes | Task identifier |
| `video` | yes | Filename inside `videos/` (or path resolvable by the pipeline) |

Uses default directories: `videos/` (input), `preprocessed_input/` (output).

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
