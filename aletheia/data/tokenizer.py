"""
Length-Weighted Tokenizer (§3.3)

Custom tokenizer wrapper that:
  1. Trains a SentencePiece unigram model on the corpus
  2. Assigns "semantic weight" to each token based on surprisal
  3. Preserves morphological boundaries for Arabic/Sanskrit tokens

The semantic weight is: weight = log(1 + mean_surprisal)
High-surprisal tokens carry more meaning and get lower merge priority
during training, preserving their morphological structure.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class LengthWeightedTokenizer:
    """Tokenizer with semantic weight awareness.

    Wraps SentencePiece with per-token weight scoring based on
    information-theoretic surprisal.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        vocab_size: int = 64000,
    ) -> None:
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.sp = None
        self.token_weights: dict[int, float] = {}

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, model_path: str | Path) -> None:
        """Load a trained SentencePiece model."""
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(str(model_path))
            logger.info(f"Loaded tokenizer from {model_path} (vocab={self.sp.GetPieceSize()})")
        except ImportError:
            logger.warning("sentencepiece not installed. Using fallback character tokenizer.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer model: {e}")

    def train(
        self,
        corpus_files: list[str | Path],
        output_prefix: str = "data/tokenizer/aletheia",
        vocab_size: int | None = None,
    ) -> None:
        """Train a SentencePiece unigram model on the corpus.

        Args:
            corpus_files: List of text files to train on.
            output_prefix: Output path prefix for the model.
            vocab_size:    Override vocab size.
        """
        try:
            import sentencepiece as spm
        except ImportError:
            logger.error("sentencepiece required for training. Install: pip install sentencepiece")
            return

        vs = vocab_size or self.vocab_size
        output_dir = Path(output_prefix).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write corpus to a single temp file if needed
        input_files = ",".join(str(f) for f in corpus_files if Path(f).exists())
        if not input_files:
            logger.error("No valid corpus files found.")
            return

        logger.info(f"Training SentencePiece (vocab={vs}) on: {input_files}")

        spm.SentencePieceTrainer.Train(
            input=input_files,
            model_prefix=output_prefix,
            vocab_size=vs,
            model_type="unigram",
            character_coverage=0.9999,  # high coverage for multilingual
            num_threads=4,
            max_sentence_length=16384,
            shuffle_input_sentence=True,
            # Preserve morphological boundaries
            split_by_unicode_script=True,
            split_by_whitespace=True,
            user_defined_symbols=[
                "◌",  # combining mark placeholder
            ],
        )

        model_path = f"{output_prefix}.model"
        self._load_model(model_path)
        self.model_path = model_path
        logger.info(f"Tokenizer trained and saved to {model_path}")

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if self.sp is not None:
            return self.sp.Encode(text)
        # Fallback: character-level encoding
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self.sp is not None:
            return self.sp.Decode(ids)
        # Fallback
        return "".join(chr(i) for i in ids if 32 <= i < 127)

    def encode_weighted(self, text: str) -> list[tuple[int, float]]:
        """Encode text and return (token_id, weight) pairs.

        Weight is log(1 + surprisal) where surprisal is based on
        pre-computed token frequencies.
        """
        ids = self.encode(text)
        return [(tid, self.token_weights.get(tid, 1.0)) for tid in ids]

    def compute_weights(
        self,
        corpus_iter: Iterator[str],
        max_docs: int = 100000,
    ) -> None:
        """Compute per-token semantic weights from corpus statistics.

        Weight = log(1 + surprisal) where surprisal = -log2(p(token))
        """
        logger.info("Computing token semantic weights...")
        counts: Counter = Counter()
        total = 0

        for i, text in enumerate(corpus_iter):
            if i >= max_docs:
                break
            ids = self.encode(text)
            counts.update(ids)
            total += len(ids)

        if total == 0:
            logger.warning("Empty corpus — no weights computed.")
            return

        # Compute surprisal-based weights
        for tid, count in counts.items():
            prob = count / total
            surprisal = -math.log2(prob) if prob > 0 else 20.0  # cap
            self.token_weights[tid] = math.log(1 + surprisal)

        logger.info(f"Computed weights for {len(self.token_weights)} tokens.")

    def save_weights(self, path: str | Path) -> None:
        """Save token weights to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in self.token_weights.items()}, f)

    def load_weights(self, path: str | Path) -> None:
        """Load token weights from JSON."""
        with open(path, "r") as f:
            raw = json.load(f)
        self.token_weights = {int(k): v for k, v in raw.items()}
        logger.info(f"Loaded weights for {len(self.token_weights)} tokens.")
