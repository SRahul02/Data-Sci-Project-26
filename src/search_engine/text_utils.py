from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split free text into alpha tokens."""
    return TOKEN_PATTERN.findall(text.lower())
