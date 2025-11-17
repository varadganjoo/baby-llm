"""Simple tokenizer helpers for the baby language model."""

from __future__ import annotations

import re
from typing import Iterable, List

TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;:\"()\-]+")


def tokenize(text: str) -> List[str]:
    """Lower-case text and return a list of tokens."""
    tokens = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        if token.isalpha() or token.isalnum() or token == "'":
            tokens.append(token.lower())
        else:
            tokens.append(token)
    return tokens


def iter_token_slices(chunks: Iterable[str]) -> Iterable[str]:
    """Tokenize multiple pieces of text lazily."""
    for chunk in chunks:
        for token in tokenize(chunk):
            yield token
