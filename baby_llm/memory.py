"""Replay buffer for incremental training."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List


class TrainingMemory:
    """Stores recent token sequences for replay."""

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._buffer: Deque[List[str]] = deque(maxlen=capacity)

    def add(self, tokens: Iterable[str]) -> None:
        tokens_list = [token for token in tokens if token]
        if tokens_list:
            self._buffer.append(tokens_list)

    def sequences(self) -> List[List[str]]:
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)
