import torch
import logging
import numpy as np
from typing import List, Optional, Generator, Any
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Pure embedding component.
    Chỉ nhận text(s) và trả về vector(s).
    Không biết gì về chunking hay metadata.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",   # Model mạnh và phổ biến 2025-2026 tren huggingface, hỗ trợ đa ngôn ngữ, kích thước embedding 384, hiệu suất tốt cho nhiều tác vụ
        device: Optional[str] = None, # kiem tra device tu user, neu khong co thi tu dong phat hien (cuda > cpu)
        batch_size: int = 32, # batch size cho encode_batch, can co de tang toc do khi encode nhieu van ban 
        normalize: bool = True, # co the chuan hoa vector (L2 normalization) de tang toc do khi tinh cosine similarity sau nay
        max_seq_length: int = 512, # giới hạn độ dài sequence để tránh lỗi khi encode, mặc dù model có thể hỗ trợ hơn nhưng thường không cần thiết và có thể gây lỗi nếu quá dài
        cache_folder: Optional[str] = None, # tuỳ chọn thư mục cache để lưu model đã tải về, giúp tiết kiệm băng thông và thời gian khi sử dụng lại sau này
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_seq_length = max_seq_length

        # Device detection
        if device is None:
            # Tự động phát hiện device: ưu tiên CUDA(GPU) nếu có, ngược lại dùng CPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            logger.info(f"Using specified device: {device}")
            self.device = device

        logger.info(f"Loading embedding model: {model_name} on {self.device}")

        # Tải model với cấu hình đã chọn
        self.model = SentenceTransformer(
            model_name=model_name,
            device=self.device,
            cache_folder=cache_folder,
        )

        # Buộc model không được truncate ngầm. Nếu không có dòng này, SentenceTransformer có thể tự cắt mà không báo lỗi → mất thông tin.
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = max_seq_length

        logger.info(f"Model loaded successfully. Max seq length: {self.model.max_seq_length}")

    def encode(self, text: str) -> np.ndarray:
        """Encode một văn bản duy nhất"""
        # Trả về vector zero nếu text rỗng hoặc chỉ chứa whitespace, tránh lỗi khi encode và đảm bảo consistent output
        if not text or not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension(), dtype=np.float32)
        
        # Sử dụng encode_batch để tận dụng batch processing ngay cả khi chỉ có 1 văn bản, giúp tận dụng tối đa hiệu suất của model
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode nhiều văn bản cùng lúc - hiệu suất cao"""
        if not texts:
            return np.array([], dtype=np.float32)

        # Truncate an toàn trước khi encode (tránh lỗi vượt max length)
        safe_texts = [text[:self.max_seq_length * 4] for text in texts]  # ~4 chars ≈ 1 token

        embeddings = self.model.encode(
            safe_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        return embeddings

    def encode_stream(
        self,
        text_generator: Generator[str, None, None],
    ) -> Generator[np.ndarray, None, None]:
        """
        Nhận generator của text → trả về generator của embedding
        Tiết kiệm memory khi xử lý tài liệu rất lớn.
        """
        batch = []
        for text in text_generator:
            batch.append(text)

            if len(batch) >= self.batch_size:
                embeddings = self.encode_batch(batch)
                yield from embeddings
                batch = []

        # Batch cuối cùng
        if batch:
            embeddings = self.encode_batch(batch)
            yield from embeddings

    # ====================== Utility methods ======================

    def get_embedding_dimension(self) -> int:
        """Trả về kích thước vector embedding"""
        return self.model.get_sentence_embedding_dimension()

    def check_compatibility(self, chunker_max_tokens: int) -> bool:
        """Kiểm tra xem chunk_size từ chunker có an toàn không"""
        safe_limit = self.max_seq_length - 20  # để dư chỗ cho special tokens
        if chunker_max_tokens > safe_limit:
            logger.warning(
                f"Chunker chunk_size ({chunker_max_tokens}) > safe limit for embedding model ({safe_limit}). "
                f"Recommend reducing chunk_size to <= {safe_limit}"
            )
            return False
        logger.info(f"Chunk size compatible: {chunker_max_tokens} <= {safe_limit}")
        return True