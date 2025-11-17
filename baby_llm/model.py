"""Incremental language model that slowly improves as it is exposed to text."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .embeddings import BabyEmbeddings, Vector
from .memory import TrainingMemory
from .rnn_head import BabyRNNHead
from .tokenizer import tokenize


@dataclass
class ModelStats:
    """Simple container for user-facing statistics."""

    tokens_seen: int
    utterances_seen: int
    vocabulary_size: int
    bigram_transitions: int
    trigram_transitions: int
    embedding_terms: int
    head_device: str


class BabyLanguageModel:
    """A tiny n-gram language learner intended to feel like a growing baby."""

    def __init__(
        self,
        *,
        rng_seed: Optional[int] = None,
        label: str = "Baby",
        embedding_dim: int = 64,
        device: Optional[str] = None,
    ) -> None:
        self.unigram: Counter[str] = Counter()
        self.bigram: Dict[str, Counter[str]] = defaultdict(Counter)
        self.trigram: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.turns_seen = 0
        self.total_tokens = 0
        self.label = label
        self._rng = random.Random(rng_seed)
        self.embeddings = BabyEmbeddings(dim=embedding_dim, rng_seed=rng_seed)
        self._topic_bonus = 0.75
        self._head_bonus = 0.5
        self._device_preference = device
        self.rnn_head = BabyRNNHead(embedding_dim=embedding_dim, device=device)
        self.memory = TrainingMemory()
        self.vocab_index: Dict[str, int] = {}
        self._log_path = Path("training_log.jsonl")

    # ------------------------------------------------------------------ Learning
    def train_on_text(self, text: str) -> int:
        """Update the model with raw text and return the number of tokens consumed."""
        tokens = tokenize(text)
        if tokens:
            self._update_vocab(tokens)
            self.memory.add(tokens)
            self._append_log(text)
            self.embeddings.observe_sequence(tokens)
            self._train_rnn_head(tokens)
        processed = self._train_on_tokens(tokens)
        if processed:
            self.turns_seen += 1
        return processed

    def _train_on_tokens(self, tokens: Iterable[str]) -> int:
        processed = 0
        previous_token: Optional[str] = None
        previous_bigram: Optional[Tuple[str, str]] = None
        for token in tokens:
            if not token:
                continue
            processed += 1
            self.total_tokens += 1
            self.unigram[token] += 1
            if previous_token is not None:
                self.bigram[previous_token][token] += 1
                if previous_bigram is not None:
                    self.trigram[previous_bigram][token] += 1
            if previous_token is not None:
                previous_bigram = (previous_token, token)
            previous_token = token
        return processed

    def _train_rnn_head(self, tokens: Iterable[str]) -> None:
        vectors = self._vectors_for_tokens(tokens)
        if not vectors:
            return
        self.rnn_head.train_from_vectors(vectors)
        token_indices = self._token_indices(tokens)
        embedding_matrix = self._embedding_matrix()
        if token_indices and embedding_matrix:
            self.rnn_head.train_token_sequence(
                vectors, token_indices, embedding_matrix
            )

    def _vectors_for_tokens(self, tokens: Iterable[str]) -> List[Vector]:
        vectors: List[Vector] = []
        for token in tokens:
            vector = self.embeddings.vectors.get(token)
            if vector is not None:
                vectors.append(vector)
        return vectors

    def _update_vocab(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            if token not in self.vocab_index:
                self.vocab_index[token] = len(self.vocab_index)

    def _token_indices(self, tokens: Iterable[str]) -> List[int]:
        indices: List[int] = []
        for token in tokens:
            idx = self.vocab_index.get(token)
            if idx is not None:
                indices.append(idx)
        return indices[1:]

    def _embedding_matrix(self) -> List[List[float]]:
        if not self.vocab_index:
            return []
        vocab_tokens = sorted(self.vocab_index.items(), key=lambda item: item[1])
        matrix: List[List[float]] = []
        size = len(vocab_tokens)
        matrix: List[List[float]] = [[0.0 for _ in range(self.embeddings.dim)] for _ in range(size)]
        for token, idx in vocab_tokens:
            vector = self.embeddings.vectors.get(token)
            if vector is not None:
                matrix[idx] = list(vector)
        return matrix

    def _append_log(self, text: str) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"text": text}) + "\n")
        except Exception:
            pass

    def _predict_head_vector(self, tokens: List[str]) -> Optional[Vector]:
        vectors = self._vectors_for_tokens(tokens)
        if not vectors:
            return None
        return self.rnn_head.predict_next(vectors)

    # ------------------------------------------------------------------ Generation
    def reply(self, prompt: str, *, max_words: int = 40) -> str:
        """Generate a response after training on a user prompt."""
        sequence = self._generate_sequence(prompt, max_words=max_words)
        if not sequence:
            return self._fallback_response()

        sentence = self._format_sentence(sequence)
        if self.turns_seen < 5:
            return f"I'm still tiny but I'm trying: {sentence}"
        if self.turns_seen < 20:
            return f"I think about this and say: {sentence}"
        if self._rng.random() < 0.3:
            return f"{sentence} (learning...)"
        return sentence

    def _generate_sequence(self, prompt: str, *, max_words: int) -> List[str]:
        if not self.unigram:
            return []

        prompt_tokens = tokenize(prompt)
        topic_vector = self.embeddings.embedding_for_tokens(prompt_tokens)
        head_vector = self._predict_head_vector(prompt_tokens)
        start = self._select_start_token(prompt_tokens)
        if not start:
            start = self._choose_seed_word(
                topic_vector=topic_vector, head_vector=head_vector
            )
        if not start:
            return []

        generated: List[str] = [start]
        for _ in range(max_words - 1):
            history_head = self._predict_head_vector(generated)
            nxt = self._choose_next_word(
                generated,
                topic_vector=topic_vector,
                head_vector=history_head,
            )
            if not nxt:
                break
            generated.append(nxt)
        return generated

    def _choose_seed_word(
        self, *, topic_vector: Optional[Vector], head_vector: Optional[Vector]
    ) -> Optional[str]:
        return self._weighted_choice(
            self.unigram, topic_vector=topic_vector, head_vector=head_vector
        )

    def _choose_next_word(
        self,
        history: List[str],
        *,
        topic_vector: Optional[Vector],
        head_vector: Optional[Vector],
    ) -> Optional[str]:
        if len(history) >= 2:
            trigram_key = (history[-2], history[-1])
            tri_candidates = self.trigram.get(trigram_key)
            if tri_candidates:
                choice = self._weighted_choice(
                    tri_candidates,
                    topic_vector=topic_vector,
                    head_vector=head_vector,
                )
                if choice:
                    return choice

        previous = history[-1]
        candidates = self.bigram.get(previous)
        if candidates:
            choice = self._weighted_choice(
                candidates, topic_vector=topic_vector, head_vector=head_vector
            )
            if choice:
                return choice
        # Fallback to unigram sampling to avoid dead-ends.
        return self._weighted_choice(
            self.unigram, topic_vector=topic_vector, head_vector=head_vector
        )

    def _weighted_choice(
        self,
        counter: Counter[str],
        *,
        topic_vector: Optional[Vector],
        head_vector: Optional[Vector],
    ) -> Optional[str]:
        if not counter:
            return None
        logits: List[Tuple[str, float]] = []
        for word, count in counter.items():
            if count <= 0:
                continue
            logit = math.log(float(count))
            if topic_vector is not None:
                logit += self._topic_bonus * self.embeddings.similarity(
                    word, topic_vector
                )
            if head_vector is not None:
                logit += self._head_bonus * self.embeddings.similarity(
                    word, head_vector
                )
            logits.append((word, logit))
        if not logits:
            return None
        max_logit = max(logit for _, logit in logits)
        exp_values = [
            (word, math.exp(logit - max_logit)) for word, logit in logits
        ]
        total = sum(value for _, value in exp_values)
        if total <= 0:
            return None
        pick = self._rng.uniform(0, total)
        cumulative = 0.0
        for word, value in exp_values:
            cumulative += value
            if cumulative >= pick:
                return word
        return exp_values[-1][0]

    # ------------------------------------------------------------------ Formatting
    def _format_sentence(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        no_space_before = {".", ",", "!", "?", ";", ":", ")", "]"}
        no_space_after = {"(", "["}
        parts: List[str] = []
        previous = ""
        for token in tokens:
            if not parts:
                parts.append(token)
            elif token in no_space_before or previous in no_space_after:
                parts.append(token)
            else:
                parts.append(f" {token}")
            previous = token

        sentence = "".join(parts).strip()
        # Capitalize first alphabetical character.
        for idx, char in enumerate(sentence):
            if char.isalpha():
                sentence = sentence[:idx] + char.upper() + sentence[idx + 1 :]
                break
        else:
            sentence = sentence.capitalize()
        if sentence.endswith(tuple(".!?")):
            return sentence
        return f"{sentence}."

    def _is_word_token(self, token: str) -> bool:
        return any(char.isalnum() for char in token)

    def _select_start_token(self, tokens: List[str]) -> Optional[str]:
        for token in reversed(tokens):
            if self._is_word_token(token):
                return token
        return None

    def _fallback_response(self) -> str:
        if self.turns_seen == 0:
            return "I don't know many words yet. Please talk to me or feed me documents."
        if self.turns_seen < 5:
            return "I'm still figuring out sounds. Can you share more words?"
        return "I'm thinking quietly; maybe give me more text to learn."

    # ------------------------------------------------------------------ Persistence
    def save(self, path: Path | str) -> None:
        """Serialize the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "unigram": dict(self.unigram),
            "bigram": {key: dict(counter) for key, counter in self.bigram.items()},
            "trigram": {
                f"{key[0]}\u0000{key[1]}": dict(counter)
                for key, counter in self.trigram.items()
            },
            "turns_seen": self.turns_seen,
            "total_tokens": self.total_tokens,
            "label": self.label,
            "embeddings": self.embeddings.to_dict(),
            "rnn_head": self.rnn_head.to_dict(),
            "vocab_index": self.vocab_index,
            "log_path": str(self._log_path),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, path: Path | str, *, rng_seed: Optional[int] = None, device: Optional[str] = None
    ) -> "BabyLanguageModel":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        embeddings_data = data.get("embeddings")
        embedding_dim = (
            int(embeddings_data.get("dim"))
            if isinstance(embeddings_data, dict) and "dim" in embeddings_data
            else 64
        )
        model = cls(
            rng_seed=rng_seed,
            label=data.get("label", "Baby"),
            embedding_dim=embedding_dim,
            device=device,
        )
        model.unigram.update(data.get("unigram", {}))
        for key, counter in data.get("bigram", {}).items():
            model.bigram[key].update(counter)
        for key, counter in data.get("trigram", {}).items():
            if "\u0000" in key:
                a, b = key.split("\u0000", 1)
                model.trigram[(a, b)].update(counter)
        model.turns_seen = data.get("turns_seen", 0)
        model.total_tokens = data.get("total_tokens", sum(model.unigram.values()))
        if isinstance(embeddings_data, dict):
            model.embeddings = BabyEmbeddings.from_dict(embeddings_data)
        head_data = data.get("rnn_head")
        if isinstance(head_data, dict):
            model.rnn_head = BabyRNNHead.from_dict(head_data, device=device)
        else:
            model.rnn_head = BabyRNNHead(
                embedding_dim=model.embeddings.dim, device=device
            )
        vocab_index = data.get("vocab_index")
        if isinstance(vocab_index, dict):
            model.vocab_index = {
                str(token): int(idx) for token, idx in vocab_index.items()
            }
        log_path = data.get("log_path")
        if isinstance(log_path, str):
            model._log_path = Path(log_path)
        return model

    # ------------------------------------------------------------------ Reporting
    def stats(self) -> ModelStats:
        bigram_transitions = sum(sum(counter.values()) for counter in self.bigram.values())
        trigram_transitions = sum(
            sum(counter.values()) for counter in self.trigram.values()
        )
        return ModelStats(
            tokens_seen=self.total_tokens,
            utterances_seen=self.turns_seen,
            vocabulary_size=len(self.unigram),
            bigram_transitions=bigram_transitions,
            trigram_transitions=trigram_transitions,
            embedding_terms=len(self.embeddings),
            head_device=str(self.rnn_head.device),
        )

    @property
    def device_preference(self) -> Optional[str]:
        return self._device_preference

    def replay_memory(self, rounds: int = 1) -> int:
        if len(self.memory) == 0:
            return 0
        sequences = self.memory.sequences()
        total = 0
        for _ in range(max(1, rounds)):
            for sequence in sequences:
                self._train_rnn_head(sequence)
                total += 1
        return total

    def apply_feedback(self, text: str, positive: bool) -> None:
        tokens = tokenize(text)
        if not tokens:
            return
        delta = 1 if positive else -1
        for token in tokens:
            self.unigram[token] = max(0, self.unigram[token] + delta)
        if positive:
            self.embeddings.observe_sequence(tokens)
            self.memory.add(tokens)
            self._train_rnn_head(tokens)

    def export_head(self, path: Path | str) -> None:
        data = self.rnn_head.to_dict()
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def import_head(self, path: Path | str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.rnn_head = BabyRNNHead.from_dict(data, device=self.device_preference)

    def retrain_from_log(self, *, reset_head: bool = False) -> int:
        if reset_head:
            self.rnn_head = BabyRNNHead(
                embedding_dim=self.embeddings.dim,
                device=self.device_preference,
            )
        if not self._log_path.exists():
            return 0
        count = 0
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = payload.get("text")
            if not isinstance(text, str):
                continue
            tokens = tokenize(text)
            if tokens:
                self._train_rnn_head(tokens)
                count += 1
        return count

    def similar_words(
        self, token: str, *, top_n: int = 5, min_count: int = 2
    ) -> List[tuple[str, float]]:
        return self.embeddings.most_similar(token, top_n=top_n, min_count=min_count)

    def import_embeddings(
        self, path: Path | str, *, limit: Optional[int] = None
    ) -> int:
        return self.embeddings.load_glove_format(path, limit=limit)
