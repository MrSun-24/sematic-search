import torch
import logging
from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None,
        batch_size: int = 32,
        normalize: bool = True
    ):
        """
        model_name: tên model SBERT
        device: "cpu" | "cuda" | None (auto detect)
        batch_size: batch encode size
        normalize: chuẩn hoá vector (rất quan trọng cho cosine similarity)
        """

        # Auto detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading model on {self.device}...")

        self.model = SentenceTransformer(model_name, device=self.device)

        self.batch_size = batch_size
        self.normalize = normalize

        logger.info(f"Model loaded: {model_name}")

    # -----------------------
    # Encode 1 text
    # -----------------------
    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    # -----------------------
    # Encode batch
    # -----------------------
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )

        return embeddings

    # -----------------------
    # Encode chunks (from chunker)
    # -----------------------
    def encode_chunks(
        self,
        chunks: List[dict],
        text_key: str = "text"
    ) -> List[dict]:
        """
        Input:
            [
                {"chunk_id": ..., "text": "..."},
                ...
            ]

        Output:
            [
                {"chunk_id": ..., "embedding": [...]},
                ...
            ]
        """

        texts = [c[text_key] for c in chunks]

        embeddings = self.encode_batch(texts)

        results = []
        for chunk, emb in zip(chunks, embeddings):
            results.append({
                "chunk_id": chunk["chunk_id"],
                "embedding": emb
            })

        return results

    # -----------------------
    # Streaming encode (advanced)
    # -----------------------
    def encode_stream(
        self,
        chunk_generator,
        text_key: str = "text"
    ):
        """
        Nhận generator từ chunker → encode theo batch → yield ra
        """

        batch = []

        for chunk in chunk_generator:
            batch.append(chunk)

            if len(batch) >= self.batch_size:
                yield from self._process_batch(batch, text_key)
                batch = []

        # xử lý batch cuối
        if batch:
            yield from self._process_batch(batch, text_key)

    def _process_batch(self, batch, text_key):
        texts = [c[text_key] for c in batch]
        embeddings = self.encode_batch(texts)

        for chunk, emb in zip(batch, embeddings):
            yield {
                "chunk_id": chunk["chunk_id"],
                "embedding": emb
            }