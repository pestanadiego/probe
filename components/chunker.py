import os

from components.memory import Chunk

CHUNK_TOKENS = 256
OVERLAP_TOKENS = 32
TABLE_LINE_FRACTION = 0.4 # so we don't split a table row across two chunks
TABLE_LINE_MAX_CHARS = 20

def _is_table_window(lines: list[str]) -> bool:
    nonempty = [l for l in lines if l.strip()]
    if not nonempty:
        return False
    short = sum(1 for l in nonempty if len(l.strip()) < TABLE_LINE_MAX_CHARS)
    return short / len(nonempty) > TABLE_LINE_FRACTION

def chunk_file(file_path: str) -> list[Chunk]:
    source = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")

    chunks: list[Chunk] = []
    token_buf: list[str] = []
    line_buf: list[str] = []

    for line in lines:
        token_buf.extend(line.split())
        line_buf.append(line)

        if len(token_buf) < CHUNK_TOKENS:
            continue

        is_table = _is_table_window(line_buf)
        chunks.append(Chunk(
            text=" ".join(token_buf[:CHUNK_TOKENS]),
            source=source,
            score=0.0,
        ))

        if is_table:
            # no overlap, start fresh after this table block
            token_buf = token_buf[CHUNK_TOKENS:]
        else:
            token_buf = token_buf[CHUNK_TOKENS - OVERLAP_TOKENS:]
        line_buf = []

    if token_buf:
        chunks.append(Chunk(text=" ".join(token_buf), source=source, score=0.0))

    return chunks

def chunk_directory(directory: str) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            all_chunks.extend(chunk_file(os.path.join(directory, filename)))
    return all_chunks
