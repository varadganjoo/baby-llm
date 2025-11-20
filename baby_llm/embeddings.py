"""Simple distributed word representations for the baby model."""

from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Vector = List[float]


class BabyEmbeddings:
    """Tracks lightweight word representations via co-occurrence updates."""

    def __init__(
        self,
        *,
        dim: int = 64,
        base_learning_rate: float = 0.08,  # Increased from 0.05
        decay: float = 0.99,
        window: int = 4,  # Increased from 2
        negative_samples: int = 5,  # Increased from 2
        rng_seed: Optional[int] = None,
    ) -> None:
        self.dim = dim
        self.base_learning_rate = base_learning_rate
        self.decay = decay
        self.window = window
        self.negative_samples = negative_samples
        self.vectors: Dict[str, Vector] = {}
        self.counts: Counter[str] = Counter()
        self._rng = random.Random(rng_seed)

    # ------------------------------------------------------------------ Updates
    def observe_sequence(self, tokens: Iterable[str]) -> None:
        words = [token for token in tokens if token]
        if not words:
            return
        for index, token in enumerate(words):
            self.counts[token] += 1
            self._ensure_vector(token)
            for offset in range(1, self.window + 1):
                left_index = index - offset
                if left_index >= 0:
                    self._apply_context(token, words[left_index], offset)
                right_index = index + offset
                if right_index < len(words):
                    self._apply_context(token, words[right_index], offset)

    def _apply_context(self, token: str, neighbor: str, distance: int) -> None:
        if not neighbor:
            return
        primary_vec = self._ensure_vector(token)
        neighbor_vec = self._ensure_vector(neighbor)
        weight = self._context_weight(distance)
        primary_lr = self._learning_rate_for(token)
        neighbor_lr = self._learning_rate_for(neighbor)
        self._blend_vectors(primary_vec, neighbor_vec, primary_lr * weight)
        self._blend_vectors(neighbor_vec, primary_vec, neighbor_lr * weight)
        if self.negative_samples > 0:
            self._apply_negative_samples(
                token, primary_vec, weight, exclude={token, neighbor}
            )

    def _blend_vectors(self, target: Vector, source: Vector, rate: float) -> None:
        if rate <= 0:
            return
        for idx in range(self.dim):
            target[idx] = target[idx] * self.decay + rate * source[idx]
        self._normalize(target)

    def _apply_negative_samples(
        self, token: str, token_vec: Vector, weight: float, *, exclude: Sequence[str]
    ) -> None:
        negatives = self._sample_negative_tokens(exclude, self.negative_samples)
        if not negatives:
            return
        token_snapshot = token_vec[:]
        token_lr = self._learning_rate_for(token) * weight
        for negative in negatives:
            neg_vec = self._ensure_vector(negative)
            neg_lr = self._learning_rate_for(negative) * weight
            for idx in range(self.dim):
                token_vec[idx] -= token_lr * neg_vec[idx]
            self._normalize(token_vec)
            for idx in range(self.dim):
                neg_vec[idx] -= neg_lr * token_snapshot[idx]
            self._normalize(neg_vec)

    def _sample_negative_tokens(
        self, exclude: Sequence[str], count: int
    ) -> List[str]:
        if count <= 0 or not self.counts:
            return []
        exclude_set = set(exclude)
        candidates = [
            (word, freq) for word, freq in self.counts.items() if word not in exclude_set
        ]
        if not candidates:
            return []
        total = float(sum(freq for _, freq in candidates))
        picks: List[str] = []
        for _ in range(count):
            threshold = self._rng.uniform(0, total)
            cumulative = 0.0
            for word, freq in candidates:
                cumulative += freq
                if cumulative >= threshold:
                    picks.append(word)
                    break
        return picks

    def _learning_rate_for(self, token: str) -> float:
        count = self.counts.get(token, 1)
        # Slower decay than sqrt (power 0.4) to keep learning active longer
        return self.base_learning_rate / (count ** 0.4)

    def _context_weight(self, distance: int) -> float:
        # Smoother distance penalty
        return 1.0 / (distance ** 0.5)

    def _ensure_vector(self, token: str) -> Vector:
        if token not in self.vectors:
            self.vectors[token] = [
                self._rng.uniform(-0.5, 0.5) / max(1, self.dim) for _ in range(self.dim)
            ]
            self._normalize(self.vectors[token])
        return self.vectors[token]

    # ------------------------------------------------------------------ Queries
    def embedding_for_tokens(self, tokens: Iterable[str]) -> Optional[Vector]:
        vectors = [self.vectors[token] for token in tokens if token in self.vectors]
        if not vectors:
            return None
        averaged = [0.0 for _ in range(self.dim)]
        for vector in vectors:
            for idx, value in enumerate(vector):
                averaged[idx] += value
        count = float(len(vectors))
        averaged = [value / count for value in averaged]
        self._normalize(averaged)
        return averaged

    def similarity(self, token: str, vector: Vector) -> float:
        token_vec = self.vectors.get(token)
        if token_vec is None:
            return 0.0
        return sum(a * b for a, b in zip(token_vec, vector))

    def most_similar(
        self, token: str, *, top_n: int = 5, min_count: int = 2
    ) -> List[Tuple[str, float]]:
        base = self.vectors.get(token)
        if base is None or self.counts[token] < min_count:
            return []
        scores: List[Tuple[float, str]] = []
        for other, vector in self.vectors.items():
            if other == token or self.counts[other] < min_count:
                continue
            similarity = sum(a * b for a, b in zip(base, vector))
            scores.append((similarity, other))
        scores.sort(reverse=True)
        return [(other, score) for score, other in scores[:top_n] if score > 0]

    # ------------------------------------------------------------------ Helpers
    def _normalize(self, vector: Vector) -> None:
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return
        inv = 1.0 / magnitude
        for idx in range(len(vector)):
            vector[idx] *= inv

    def __len__(self) -> int:
        return len(self.vectors)

    # ------------------------------------------------------------------ Persistence
    def to_dict(self) -> Dict[str, object]:
        return {
            "dim": self.dim,
            "base_learning_rate": self.base_learning_rate,
            "decay": self.decay,
            "window": self.window,
            "negative_samples": self.negative_samples,
            "vectors": {token: vector[:] for token, vector in self.vectors.items()},
            "counts": dict(self.counts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "BabyEmbeddings":
        emb = cls(
            dim=int(data.get("dim", 64)),
            base_learning_rate=float(data.get("base_learning_rate", 0.05)),
            decay=float(data.get("decay", 0.99)),
            window=int(data.get("window", 2)),
            negative_samples=int(data.get("negative_samples", 2)),
        )
        emb.vectors = {
            token: [float(value) for value in vector]
            for token, vector in data.get("vectors", {}).items()
        }
        emb.counts.update(
            {token: int(count) for token, count in data.get("counts", {}).items()}
        )
        return emb

    # ------------------------------------------------------------------ Pre-trained import
    def load_glove_format(
        self, path: Path | str, *, limit: Optional[int] = None
    ) -> int:
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"No embedding file at {file_path}")
        loaded = 0
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and loaded >= limit:
                    break
                parts = line.strip().split()
                if len(parts) <= 2:
                    continue
                word = parts[0]
                vector = [float(value) for value in parts[1:]]
                if not vector:
                    continue
                if not self.vectors:
                    self.dim = len(vector)
                if len(vector) != self.dim:
                    raise ValueError(
                        f"Embedding dimension mismatch (expected {self.dim}, found {len(vector)})"
                    )
                self.vectors[word] = vector
                self._normalize(self.vectors[word])
                self.counts[word] = max(self.counts.get(word, 0), 1)
                loaded += 1
        if loaded == 0:
            raise ValueError(f"No embeddings found in {file_path}")
        return loaded
