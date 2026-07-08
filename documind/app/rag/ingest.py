"""Multi-format ingestion: pdf, docx, xlsx/csv, txt/md, images. Chunk + embed + store."""
import uuid
from pathlib import Path
import fitz  # pymupdf
import docx
import openpyxl
from app import config
from app.rag import embed, store, llm

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}


def _sections(path: Path) -> list[dict]:
    """Return [{text, location}] sections tailored to the file type."""
    ext = path.suffix.lower()
    out: list[dict] = []
    if ext == ".pdf":
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            t = page.get_text("text").strip()
            if t:
                out.append({"text": t, "location": f"p.{i + 1}"})
        doc.close()
    elif ext == ".docx":
        d = docx.Document(str(path))
        paras = [p.text for p in d.paragraphs if p.text.strip()]
        # include tables
        for tbl in d.tables:
            for row in tbl.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    paras.append(" | ".join(cells))
        out.append({"text": "\n".join(paras), "location": "document"})
    elif ext in (".xlsx", ".xlsm"):
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            rows = []
            header = None
            for r in ws.iter_rows(values_only=True):
                vals = [("" if v is None else str(v)) for v in r]
                if not any(vals):
                    continue
                if header is None:
                    header = vals
                    continue
                pairs = [f"{h}: {v}" for h, v in zip(header, vals) if v]
                rows.append("; ".join(pairs) if pairs else " | ".join(vals))
            if header:
                rows.insert(0, "Columns: " + ", ".join(h for h in header if h))
            if rows:
                out.append({"text": "\n".join(rows), "location": f"sheet:{ws.title}"})
        wb.close()
    elif ext in (".csv", ".txt", ".md"):
        out.append({"text": path.read_text(errors="ignore"), "location": "document"})
    elif ext in IMAGE_EXT:
        text, _, _ = llm.image_to_text(path.read_bytes(), MIME.get(ext, "image/png"))
        out.append({"text": text, "location": "image"})
    else:
        out.append({"text": path.read_text(errors="ignore"), "location": "document"})
    return [s for s in out if s["text"].strip()]


def _chunk(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= config.CHUNK_CHARS:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = start + config.CHUNK_CHARS
        # try to break on a newline/space near the end
        if end < len(text):
            brk = text.rfind("\n", start + config.CHUNK_CHARS // 2, end)
            if brk == -1:
                brk = text.rfind(" ", start + config.CHUNK_CHARS // 2, end)
            if brk != -1:
                end = brk
        chunks.append(text[start:end].strip())
        start = max(end - config.CHUNK_OVERLAP, end)
    return [c for c in chunks if c]


def ingest_file(user_id: int, path: Path, filename: str) -> dict:
    """Parse -> chunk -> embed -> store. Returns {doc_id, chunks, embed_tokens}."""
    sections = _sections(path)
    chunks: list[dict] = []
    for sec in sections:
        for j, piece in enumerate(_chunk(sec["text"])):
            chunks.append({"text": piece, "location": sec["location"], "chunk_index": len(chunks)})
    if not chunks:
        return {"doc_id": None, "chunks": 0, "embed_tokens": 0}
    vectors, tokens = embed.embed([c["text"] for c in chunks], input_type="document")
    doc_id = str(uuid.uuid4())
    store.add_chunks(user_id, doc_id, filename, chunks, vectors)
    return {"doc_id": doc_id, "chunks": len(chunks), "embed_tokens": tokens}
