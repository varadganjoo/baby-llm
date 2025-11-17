"""Tiny PyTorch RNN head that predicts next-word embeddings."""

from __future__ import annotations

import base64
import io
import warnings
from typing import List, Optional, Sequence

import torch
from torch import nn

Vector = List[float]


def _resolve_device(preference: Optional[str] = None) -> torch.device:
    requested = preference or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA requested but PyTorch is CPU-only. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        requested = "cpu"
    return torch.device(requested)


class _TinyRNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.proj = nn.Linear(hidden_size, embedding_dim)

    def forward(self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        rnn_out, hidden = self.rnn(inputs, hidden)
        attn_out, _ = self.attn(rnn_out, rnn_out, rnn_out)
        prediction = self.proj(attn_out)
        return prediction, attn_out, hidden


class BabyRNNHead:
    """Learns to predict next embeddings using PyTorch."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_size: int = 64,
        attention_heads: int = 2,
        learning_rate: float = 0.01,
        max_sequence: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_sequence = max_sequence
        self.device = _resolve_device(device)

        self.attention_heads = attention_heads
        self.model = _TinyRNN(embedding_dim, hidden_size, attention_heads).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------ Training
    def train_from_vectors(self, vectors: Sequence[Vector]) -> Optional[float]:
        vectors = self._trim_sequence(vectors)
        if len(vectors) < 2:
            return None
        inputs = torch.tensor(
            [vectors[:-1]], dtype=torch.float32, device=self.device
        )
        targets = torch.tensor(
            [vectors[1:]], dtype=torch.float32, device=self.device
        )
        self.model.train()
        self.optimizer.zero_grad()
        predictions, _, _ = self.model(inputs)
        loss = nn.functional.mse_loss(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------ Inference
    def predict_next(self, vectors: Sequence[Vector]) -> Optional[Vector]:
        vectors = self._trim_sequence(vectors)
        if not vectors:
            return None
        self.model.eval()
        with torch.no_grad():
            hidden = None
            hidden = None
            if len(vectors) > 1:
                context = torch.tensor(
                    [vectors[:-1]], dtype=torch.float32, device=self.device
                )
                _, _, hidden = self.model(context)
            last_input = torch.tensor(
                [[vectors[-1]]], dtype=torch.float32, device=self.device
            )
            prediction, _, _ = self.model(last_input, hidden)
        return prediction[0, -1].cpu().tolist()

    def train_token_sequence(
        self,
        vectors: Sequence[Vector],
        token_indices: Sequence[int],
        embedding_matrix: Sequence[Sequence[float]],
    ) -> Optional[float]:
        vectors = self._trim_sequence(vectors)
        if len(vectors) < 2 or len(token_indices) < 1 or not embedding_matrix:
            return None
        seq_len = min(len(vectors) - 1, len(token_indices))
        inputs = torch.tensor(
            [vectors[: seq_len]], dtype=torch.float32, device=self.device
        )
        targets = torch.tensor(
            [token_indices[:seq_len]], dtype=torch.long, device=self.device
        )
        matrix = torch.tensor(
            embedding_matrix, dtype=torch.float32, device=self.device
        )
        self.model.train()
        self.optimizer.zero_grad()
        predictions, _, _ = self.model(inputs)
        logits = torch.matmul(predictions, matrix.t())
        loss = nn.functional.cross_entropy(
            logits.view(-1, matrix.shape[0]), targets.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    # ------------------------------------------------------------------ Persistence
    def to_dict(self) -> dict:
        buffer = io.BytesIO()
        config = {
            "embedding_dim": self.embedding_dim,
            "hidden_size": self.hidden_size,
                "attention_heads": self.attention_heads,
            "learning_rate": self.learning_rate,
            "max_sequence": self.max_sequence,
        }
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": config,
            },
            buffer,
        )
        return {
            "device": str(self.device),
            "config": config,
            "state": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: dict, device: Optional[str] = None) -> "BabyRNNHead":
        config = data.get("config") or {}
        head = cls(
            embedding_dim=int(config.get("embedding_dim", 64)),
            hidden_size=int(config.get("hidden_size", 64)),
            attention_heads=int(config.get("attention_heads", 2)),
            learning_rate=float(config.get("learning_rate", 0.01)),
            max_sequence=int(config.get("max_sequence", 128)),
            device=device or data.get("device"),
        )
        encoded = data.get("state")
        if encoded:
            buffer = io.BytesIO(base64.b64decode(encoded))
            saved = torch.load(buffer, map_location=head.device)
            try:
                head.model.load_state_dict(saved["model"])
            except RuntimeError:
                head = cls(
                    embedding_dim=head.embedding_dim,
                    hidden_size=head.hidden_size,
                    attention_heads=head.attention_heads,
                    learning_rate=head.learning_rate,
                    max_sequence=head.max_sequence,
                    device=str(head.device),
                )
            else:
                try:
                    head.optimizer.load_state_dict(saved["optimizer"])
                except Exception:
                    pass
        return head

    # ------------------------------------------------------------------ Helpers
    def _trim_sequence(self, vectors: Sequence[Vector]) -> List[Vector]:
        if not vectors:
            return []
        if len(vectors) > self.max_sequence:
            vectors = vectors[-self.max_sequence :]
        return [list(vector) for vector in vectors]
