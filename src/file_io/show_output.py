"""Print model responses for a task from a bench2 report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

DEFAULT_VLM_OUTPUT_DIR = Path("output") / "vlm_output"


def resolve_report_path(file: str) -> Path:
    """Resolve a report path, falling back to output/vlm_output/<file>."""
    path = Path(file)
    if path.is_file():
        return path

    fallback = DEFAULT_VLM_OUTPUT_DIR / file
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(f"Report file not found: {file} (also tried {fallback})")


def _collect_task_entries(report: dict, task_id: str) -> list[tuple[int | None, dict]]:
    """Return (run_index, task) pairs matching task_id. run_index is None for top-level tasks."""
    matches: list[tuple[int | None, dict]] = []

    for task in report.get("tasks", []):
        if task.get("task_id") == task_id:
            matches.append((None, task))

    for run in report.get("runs", []):
        run_index = run.get("run")
        for task in run.get("tasks", []):
            if task.get("task_id") == task_id:
                matches.append((run_index, task))

    return matches


def _available_task_ids(report: dict) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    for task in report.get("tasks", []):
        tid = task.get("task_id")
        if isinstance(tid, str) and tid not in seen:
            ids.append(tid)
            seen.add(tid)

    for run in report.get("runs", []):
        for task in run.get("tasks", []):
            tid = task.get("task_id")
            if isinstance(tid, str) and tid not in seen:
                ids.append(tid)
                seen.add(tid)

    return ids


def _format_output_body(output: dict) -> str:
    response = output.get("response")
    if isinstance(response, str):
        if output.get("valid_json"):
            try:
                parsed = json.loads(response)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        return response

    preview = output.get("response_preview", "")
    return (
        f"{preview}\n\n"
        "(Full response not stored. Re-run bench2 with --include_responses=True.)"
    )


def print_task_outputs(report: dict, task_id: str) -> None:
    """Print formatted model responses for a task."""
    entries = _collect_task_entries(report, task_id)
    if not entries:
        available = ", ".join(_available_task_ids(report)) or "(none)"
        raise ValueError(f"Task '{task_id}' not found. Available task ids: {available}")

    for run_index, task in entries:
        outputs = task.get("outputs") or []
        if not outputs:
            header = f"--- {task_id}"
            if run_index is not None:
                header += f" (run {run_index})"
            print(f"{header} ---")
            print("(No model outputs recorded for this task.)")
            print()
            continue

        for output in outputs:
            prompt = output.get("prompt", "unknown")
            valid_json = output.get("valid_json", False)
            header = f"--- {task_id} / {prompt} (valid_json={valid_json})"
            if run_index is not None:
                header += f" run {run_index}"
            print(header)
            print(_format_output_body(output))
            print()


def main(file: str, task_id: str) -> int:
    """Print model response(s) for a task from a bench report file."""
    try:
        report_path = resolve_report_path(file)
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        print_task_outputs(report, task_id)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as error:
        print(str(error))
        return 1
    return 0


def cli() -> None:
    sys.exit(fire.Fire(main) or 0)
