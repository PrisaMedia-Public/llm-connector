"""División de texto en fragmentos (chunking preventivo).

Reproduce el chunking del pipeline original: usa `RecursiveCharacterTextSplitter`
de langchain-text-splitters si está instalado (extra `chunking`); si no, aplica
un divisor por párrafos equivalente en lo esencial.
"""


def split_text(text: str, max_chars: int, overlap_chars: int = 200) -> list:
    """Divide `text` en fragmentos de como máximo `max_chars` caracteres."""
    if max_chars <= 0:
        raise ValueError("max_chars debe ser > 0")
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return _split_by_paragraphs(text, max_chars, overlap_chars)

    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap_chars)
    return splitter.split_text(text)


def _split_by_paragraphs(text: str, max_chars: int, overlap_chars: int) -> list:
    pieces = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        step = max_chars - overlap_chars
        while len(paragraph) > max_chars:
            pieces.append(paragraph[:max_chars])
            paragraph = paragraph[step:]
        pieces.append(paragraph)

    chunks = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        overlap = current[-overlap_chars:] if (current and overlap_chars) else ""
        current = f"{overlap}\n\n{piece}" if overlap and len(overlap) + len(piece) + 2 <= max_chars else piece
    if current:
        chunks.append(current)
    return chunks
