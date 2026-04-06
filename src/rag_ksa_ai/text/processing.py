from .normalization import normalize_arabic


def extract_and_chunk(pdf_path: str, chunk_size: int = 600, overlap: int = 150) -> list:
    import fitz

    doc = fitz.open(pdf_path)
    chunks, current_chunk, current_page = [], "", "1"
    for i, page in enumerate(doc):
        text = normalize_arabic(page.get_text())
        if not text.strip():
            continue
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if len(current_chunk) + len(line) > chunk_size and current_chunk.strip():
                chunks.append({"id": len(chunks), "text": current_chunk.strip(), "page": current_page})
                current_chunk = current_chunk[-overlap:] + " " + line
            else:
                current_chunk += " " + line
        current_page = str(i + 2)
    if current_chunk.strip():
        chunks.append({"id": len(chunks), "text": current_chunk.strip(), "page": current_page})
    print(f"Extracted {len(chunks)} chunks from {len(doc)} pages")
    return chunks


def prepare_text(entry: dict) -> str:
    parts = []
    for field in ["summary", "title", "title_ar"]:
        if entry.get(field):
            parts.append(str(entry[field]))
    if entry.get("keywords"):
        parts.append(" ".join(entry["keywords"]))
    if entry.get("text_ar"):
        parts.append(entry["text_ar"])
    if entry.get("description_en"):
        parts.append(entry["description_en"])
    if entry.get("courses"):
        parts.append(" ".join(entry["courses"]))
    if entry.get("level"):
        parts.append(f"مستوى {entry['level']} level {entry['level']}")
    if entry.get("semester"):
        parts.append(f"فصل {entry['semester']} semester {entry['semester']}")
    if entry.get("department"):
        parts.append(str(entry["department"]))
    if entry.get("position"):
        parts.append(str(entry["position"]))
    if entry.get("category"):
        parts.append(str(entry["category"]))
    if entry.get("text"):
        parts.append(entry["text"])
    return " ".join(parts)
