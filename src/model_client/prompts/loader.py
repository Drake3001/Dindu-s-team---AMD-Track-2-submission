from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "video_analysis"

_SECTION_RE = re.compile(r"^#\s*(System|User)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Prompt:
    name: str
    system: str
    user: str


def _parse(text: str) -> tuple[str, str]:
    matches = list(_SECTION_RE.finditer(text))
    if len(matches) < 2:
        raise ValueError("Prompt file must contain '# System' and '# User' sections")

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        name = match.group(1).lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    system = sections.get("system", "")
    user = sections.get("user", "")
    if not system or not user:
        raise ValueError("Prompt file must contain non-empty '# System' and '# User' sections")
    return system, user


def list_prompt_names() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))


def load_prompt(name: str) -> Prompt:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        available = ", ".join(list_prompt_names()) or "(none)"
        raise FileNotFoundError(f"Unknown prompt '{name}'. Available: {available}")
    system, user = _parse(path.read_text(encoding="utf-8"))
    return Prompt(name=name, system=system, user=user)


def load_prompts(names: list[str] | None = None) -> list[Prompt]:
    return [load_prompt(n) for n in (names or list_prompt_names())]
