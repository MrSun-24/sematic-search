import pytest
import os
from src.core.chunker import HybridChunker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_text(path):
    file = os.path.join(BASE_DIR, path)

    with open(file, encoding="utf-8") as f:
        doc = f.read()

    return doc

def test_basic_chunking():
    doc = "file1.txt"
    data = load_text(doc)
    chunker = HybridChunker(chunk_size=8, overlap_tokens=2)
    print(chunker.split_sentences(data))
    for c in chunker.chunk(data, return_format="dict", return_text=True):
        print(c)
    for idx, c in enumerate(chunker.chunk(data, return_format="dict", return_text=True)):
        print(f"Chunk {idx}: text={c['text']}, tokens={c['tokens']}")

    # chunks = list(chunker.chunk(data, return_format="dict", return_text=True))
    # assert isinstance(chunks, list)     
    # assert len(chunks) > 0                                                                                  
    # assert all(isinstance(c, dict) for c in chunks)
    # assert chunks[1]['tokens'][:2] == chunks[0]['tokens'][-2:]  # Overlap check
    # print(chunks)
