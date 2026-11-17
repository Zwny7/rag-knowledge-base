"""文档加载：md / txt / pdf / docx → 统一的文档对象。"""
from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED = {".md", ".markdown", ".txt", ".pdf", ".docx"}


@dataclass
class RawDoc:
    doc: str          # 文件名
    text: str         # 正文
    kind: str         # md / txt / pdf / docx
    page: int | None = None


def _load_pdf(path: str) -> list[RawDoc]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(RawDoc(doc=os.path.basename(path), text=text, kind="pdf", page=i))
    return docs


def _load_docx(path: str) -> list[RawDoc]:
    import docx

    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return [RawDoc(doc=os.path.basename(path), text="\n".join(parts), kind="docx")]


def _load_text(path: str, kind: str) -> list[RawDoc]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [RawDoc(doc=os.path.basename(path), text=f.read(), kind=kind)]


def load_file(path: str) -> list[RawDoc]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".docx":
        return _load_docx(path)
    if ext in (".md", ".markdown"):
        return _load_text(path, "md")
    if ext == ".txt":
        return _load_text(path, "txt")
    raise ValueError(f"不支持的格式：{ext}")


def load_dir(path: str) -> list[RawDoc]:
    docs: list[RawDoc] = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in SUPPORTED:
            docs.extend(load_file(full))
    return docs
