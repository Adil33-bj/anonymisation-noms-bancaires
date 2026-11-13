"""Helpers to keep processed data versions explicit and documented."""

from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path


def append_data_changelog(changelog_path: Path, title: str, bullets: list[str]) -> None:
    """Append a markdown changelog entry with UTC timestamp."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [f"## {title}", f"- Date: {timestamp}"]
    lines.extend(f"- {item}" for item in bullets)
    lines.append("")

    if changelog_path.exists():
        previous = changelog_path.read_text(encoding="utf-8")
    else:
        previous = "# Data Processed Changelog\n\n"
    changelog_path.write_text(previous + "\n".join(lines), encoding="utf-8")
