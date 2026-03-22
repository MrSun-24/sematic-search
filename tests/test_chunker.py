import pytest
import os
from src.core.chunker import TextChunker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_text(path):
    file = os.path.join(BASE_DIR, path)

    with open(file, encoding="utf-8") as f:
        doc = f.read()

    return doc

def test_basic_chunking():
    doc = "file1.txt"
    data = load_text(doc)
    chunker = TextChunker(chunk_size=100, overlap=20)
    chunks = chunker.split_text(data)
    print(chunks)
    assert isinstance(chunks, list)     
    assert len(chunks) > 0                                                                                  
    assert all(isinstance(c, str) for c in chunks)
