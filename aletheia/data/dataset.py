"""
Unified Dataset and DataLoader (§2.1, §2.2)

Combined dataset class that loads pre-processed JSONL data from both
the classical corpus and modern primary sources. Applies quarantine
checks and produces tokenized sequences for training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from aletheia.data.quarantine import quarantine_check


class AletheiaDataset(Dataset):
    """Map-style dataset loaded from JSONL files.

    Each line in the JSONL file is: {"text": ..., "lang": ..., "date": ..., "source": ...}
    Quarantine checks are applied lazily on access.
    """

    def __init__(
        self,
        data_paths: list[str | Path],
        max_seq_len: int = 2048,
        corpus_type: str = "classical",
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.corpus_type = corpus_type
        self.documents: list[dict] = []

        for path in data_paths:
            path = Path(path)
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    passed, _ = quarantine_check(doc, corpus_type)
                    if passed:
                        self.documents.append(doc)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> dict:
        doc = self.documents[idx]
        return {
            "text": doc["text"][:self.max_seq_len * 10],  # rough char limit
            "lang": doc.get("lang", "unknown"),
            "source": doc.get("source", ""),
        }


class StreamingAletheiaDataset(IterableDataset):
    """Streaming dataset for large JSONL files that don't fit in memory."""

    def __init__(
        self,
        data_paths: list[str | Path],
        max_seq_len: int = 2048,
        corpus_type: str = "classical",
    ) -> None:
        super().__init__()
        self.data_paths = [Path(p) for p in data_paths]
        self.max_seq_len = max_seq_len
        self.corpus_type = corpus_type

    def __iter__(self) -> Iterator[dict]:
        for path in self.data_paths:
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    passed, _ = quarantine_check(doc, self.corpus_type)
                    if passed:
                        yield {
                            "text": doc["text"][:self.max_seq_len * 10],
                            "lang": doc.get("lang", "unknown"),
                            "source": doc.get("source", ""),
                        }


def create_dataloader(
    data_paths: list[str | Path],
    batch_size: int = 4,
    max_seq_len: int = 2048,
    corpus_type: str = "classical",
    streaming: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader with quarantine-validated data.

    Args:
        data_paths:  List of JSONL file paths.
        batch_size:  Batch size.
        max_seq_len: Maximum sequence length.
        corpus_type: "classical" or "modern".
        streaming:   Use streaming (iterable) dataset for large files.
        num_workers: Number of dataloader workers.
    """
    if streaming:
        dataset = StreamingAletheiaDataset(data_paths, max_seq_len, corpus_type)
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    else:
        dataset = AletheiaDataset(data_paths, max_seq_len, corpus_type)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )
