class TextChunker:
    def __init__(self, chunk_size=300, overlap = 50): #cat ky tu tu 0 - 300 thi doan sau se cat 250 - 350 => muc do chong cheo la 50, overlap = 10 -20% chunksize
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text):
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            # neu end chua den cuoi van ban, chuong trinh se dich chuyen end sang phai cho den khi gap dau cau (., !, ?).
            # muc dich: tranh chia doan ngay giua cau, giup van ban tu nhien hon
            if end < text_length:
                while end < text_length and text[end] not in [".", "!", "?"]:
                    end +=1

            chunk = text[start:end].strip()
            # neu khong rong 
            if chunk:
                chunks.append(chunk)

            start += self.chunk_size - self.overlap

        return chunks

                