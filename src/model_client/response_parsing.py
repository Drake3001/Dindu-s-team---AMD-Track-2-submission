from __future__ import annotations

import json
import re
from typing import Any

from model_client.types import ModelResponseError

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_from_model_response(text: str) -> Any:
    """Parse JSON from a model response, tolerating common prose/code fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = text.strip()
    fenced = _FENCED_JSON_RE.search(cleaned)
    if fenced:
        return _loads_or_raise(fenced.group(1).strip())

    for candidate in _json_candidates(cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ModelResponseError("Model response does not contain a valid JSON object or array")


def format_json_for_prompt(value: Any) -> str:
    """Format parsed JSON data for insertion into a text prompt."""
    return json.dumps(value, indent=2, ensure_ascii=False)


def _extract_outer_json(text: str, start_char: str, end_char: str) -> str | None:
    start = text.find(start_char)
    end = text.rfind(end_char)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _json_candidates(text: str) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        candidate = _extract_outer_json(text, start_char, end_char)
        if start != -1 and candidate is not None:
            candidates.append((start, candidate))
    return [candidate for _, candidate in sorted(candidates, key=lambda item: item[0])]


def _loads_or_raise(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ModelResponseError("Extracted JSON from model response is invalid") from error
