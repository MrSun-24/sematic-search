import tiktoken
import spacy
import logging
import re
import hashlib
from typing import List, Dict, Generator, Optional, Union



logger = logging.getLogger(__name__) #ghi log trong qua trinh chay(debug, warning, error...)

class HybridChunker:
    def __init__(
    self, 
    chunk_size: int = 280,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",
    language_model: str = "en_core_web_sm"
    ):
        self.chunk_size = chunk_size         
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding(encoding_name) #token hoa(gan ma dinh danh(id) cho tung tu tuy tinh huong) => Tokenization
        #Văn bản → Tokenization (Tách mảnh) → Encoding (Gán mã số).
        #Khi AI trả lời: Mã số → Decoding (Giải mã) → Chữ viết.

        #Neu co spacy thi tach cau chinh xac hon, neu khong thi van co regex de dam bao ct hoat dong
        try:
            self.nlp = spacy.load(language_model)
            self.use_spacy = True
            logger.info(f"spaCy model '{language_model}' loaded successfully.")
        except Exception as e:
            logger.warning(f"spaCy load failed: {e}. Using regex fallback.")
            self.use_spacy = False
            self.nlp = None

    # -----------------------
    # Sentence Splitter (tach cau)
    # . Nếu có spaCy thì dùng doc.sents để tách câu
    # . Nếu spaCy lỗi hoặc không có, fallback sang regex (?<=[.!?])\s+
    # . Có logging để biết khi nào fallback(truong hop du phong)
    # -----------------------

    def split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        
        #Hello world!   This is a test.  
        #Another sentence here?
        if self.use_spacy:
            try:
                doc = self.nlp(text) #doc.sents = <_cython_3_2_1.generator object at 0x000001D8375BC5F0> la mot generator
                return [s.text.strip() for s in doc.sents if s.text.strip()] 
            except Exception as e:
                logger.warning(f"spacy sentence split failed: {e}")
        
        logger.info("Fallback to regex sentence splitting")
        return re.split(r'(?<=[.!?])\s+', text)
    
    # -----------------------
    # Hash for chunk_id 
    # Tao mot ma bam duy nhat tu danh sach token cua chunk
    # -----------------------

    def _hash_tokens(self, tokens: List[int], doc_id: str) ->str:
        raw = f"{doc_id}_{tokens}"
        return hashlib.md5(raw.encode()).hexdigest() # "9f8c3a5d2b5a6e7c1d8f4a9b3c2d1e0f"

    # -----------------------
    # Overlap builder (Xay dung chong cheo)
    # -----------------------

    def _get_overlap(self, sent_tokens_list: List[List[int]]) -> List[int]:
        overlap = [] 
        total = 0

        for tokens in reversed(sent_tokens_list):
            if total + len(tokens) <= self.overlap_tokens:
                overlap = tokens + overlap
                total += len(tokens)
            else:
                break
        return overlap

    # -----------------------
    # Long sentence fallback
    # Neu mot cau(sau khi duoc ma hoa thanh token) dai hon chunk_size cho phep -> khong the gop nguyen cau vao mot chunk
    # Muc dich chia nho cau do thanh nhieu chunk nho hon, dao bao moi chunk khong vuot qua chunk_size
    # Van giu co che overlap
    # -----------------------
    
    def _split_long_tokens(
        self,
        tokens: List[int],
        doc_id: str,
        source: Optional[str],
        return_format: str,
        return_text: bool,
        original_text: Optional[str] = None
    ) -> Generator[Union[Dict, str, List[int]], None, None]:
        
        start = 0
    
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens)) 
            chunk_tokens = tokens[start:end] 

            chunk_id = self._hash_tokens(chunk_tokens, doc_id)

            chunk_data = {
                "chunk_id": chunk_id, 
                "tokens": chunk_tokens, 
                "token_count": len(chunk_tokens), #8
                "source": source
            }

            # fallback: decode token
            if return_text:
                chunk_data["text"] = self.encoding.decode(chunk_tokens)

            yield self._format_output(chunk_data, return_format)

            start += self.chunk_size - self.overlap_tokens

    # -----------------------
    # Output formatter
    # Muon output trả về text, tokens, hay cả dict thì dung cai nay
    # -----------------------

    def _format_output(
        self,
        chunk_data: Dict,
        return_format: str
    ) -> Union[Dict, str, List[int]]:

        if return_format == "text":
            return chunk_data.get("text", "")

        elif return_format == "tokens":
            return chunk_data["tokens"]

        return chunk_data  # default dict

    # -----------------------
    # Main chunking (streaming)
    # -----------------------

    def chunk(
        self, 
        text: str, 
        doc_id: str = "doc",
        source: Optional[str] = None,
        return_format: str = "dict", #dict | text | tokens
        return_text: bool = True # lazy decode control
    ) -> Generator[Union[Dict, str, List[int]], None, None]:
        
        if not text or not text.strip():
            return
        
        sentences = self.split_sentences(text)
        if not sentences:
            return 
        
        #encode 1 lan duy nhat
        sentence_token_list = [self.encoding.encode(s) for s in sentences]

        current_tokens: List[int] = [] #tenbien: kieu du lieu = gia tri | token của chunk hiện tại. [1, 2, 3, 4]
        current_sent_tokens: List[List[int]] = [] #danh sách các câu (dưới dạng token) trong chunk. [[1, 2, 3, 4]]
        current_sent_texts: List[str] = []
        current_length = 0 #tổng số token hiện tại.

        for idx, sent_tokens in enumerate(sentence_token_list):
            sent_len = len(sent_tokens)

            # Neu cau qua dai, chia nho cau do thanh cac chunk nho hon
            if sent_len > self.chunk_size:
                logger.warning(f"Long sentence detected (length={sent_len}). Splitting...")
                yield from self._split_long_tokens(
                    sent_tokens,
                    doc_id,
                    source,
                    return_format,
                    return_text
                )
                continue
            
            # Thêm câu vào chunk hiện tại
            if current_length + sent_len <= self.chunk_size:
                current_tokens.extend(sent_tokens)
                current_sent_tokens.append(sent_tokens)
                current_sent_texts.append(sentences[idx])
                current_length += sent_len
            else:
               # Tạo chunk mới
                if current_tokens:
                    yield self._create_chunk(
                        current_tokens, current_sent_texts, doc_id, source, return_format, return_text
                    )

                # Tính overlap
                overlap = self._get_overlap(current_sent_tokens)

                # Reset cho chunk mới
                current_tokens = overlap + sent_tokens
                current_sent_tokens = [sent_tokens]
                current_sent_texts = [sentences[idx]]
                current_length = len(current_tokens)

        # chunk cuối
        if current_tokens:
            yield self._create_chunk(
                current_tokens, current_sent_texts, doc_id, source, return_format, return_text
            )

# Hàm helper mới (code sạch hơn)
    def _create_chunk(
        self,
        tokens: List[int],
        sent_texts: List[str],
        doc_id: str,
        source: Optional[str],
        return_format: str,
        return_text: bool
    ):
        chunk_id = self._hash_tokens(tokens, doc_id)

        chunk_data = {
            "chunk_id": chunk_id,
            "tokens": tokens,
            "token_count": len(tokens),
            "source": source,
        }

        if return_text:
            chunk_data["text"] = " ".join(sent_texts).strip()

        return self._format_output(chunk_data, return_format)
    
# Text gốc → split_sentences → danh sách câu.

# Câu → encoding.encode → token.

# Token → gom chunk theo chunk_size.

# Nếu câu quá dài → _split_long_tokens.

# Nếu chunk đầy → tạo chunk, thêm overlap bằng _get_overlap.

# Chunk được gắn chunk_id bằng _hash_tokens.

# Output chunk qua _format_output.