"""分块：按标题和段落组织文档，保留来源和页码。"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .loader import RawDoc

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_END = re.compile(r"(?<=[。！？；.!?])\s*")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    heading: str
    text: str
    page: int | None = None

    def payload(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc": self.doc, "heading": self.heading,
                "text": self.text, "page": self.page}


def _sections(text: str) -> list[tuple[str, str]]:
    """把 markdown 拆成 (标题路径, 正文)。没有标题的文档整篇算一个 section。"""
    sections: list[tuple[str, str]] = []
    stack: dict[int, str] = {}
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            path = " > ".join(stack[k] for k in sorted(stack))
            sections.append((path, body))

    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            stack = {k: v for k, v in stack.items() if k < level}
            stack[level] = m.group(2).strip()
        else:
            buf.append(line)
    flush()
    return sections


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """超长段落按句子切，尽量不切断一句话。"""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces, cur = [], ""
    for sent in SENTENCE_END.split(paragraph):
        if not sent:
            continue
        if len(cur) + len(sent) <= max_chars:
            cur += sent
        else:
            if cur:
                pieces.append(cur)
            cur = sent
    if cur:
        pieces.append(cur)
    # 极端情况（整段没有句号）再硬切
    out = []
    for p in pieces:
        while len(p) > max_chars * 1.5:
            out.append(p[:max_chars])
            p = p[max_chars:]
        out.append(p)
    return [p for p in out if p.strip()]


def _merge(paragraphs: list[str], max_chars: int, overlap: int) -> list[str]:
    chunks, cur = [], ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if not cur:
            cur = para
        elif len(cur) + len(para) + 1 <= max_chars:
            cur += "\n" + para
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n" + para) if tail else para
    if cur:
        chunks.append(cur)
    return chunks


def chunk_doc(raw: RawDoc, max_chars: int = 520, overlap: int = 60) -> list[Chunk]:
    chunks: list[Chunk] = []
    if raw.kind == "md":
        sections = _sections(raw.text)
    else:
        sections = [(f"第 {raw.page} 页" if raw.page else "", raw.text)]

    for heading, body in sections:
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        if not paragraphs:
            paragraphs = [body]
        normalized: list[str] = []
        for p in paragraphs:
            normalized.extend(_split_long(p, max_chars))
        for text in _merge(normalized, max_chars, overlap):
            key = hashlib.sha1(f"{raw.doc}|{raw.page}|{heading}|{text}".encode()).hexdigest()[:16]
            chunks.append(Chunk(chunk_id=key, doc=raw.doc, heading=heading,
                                text=text.strip(), page=raw.page))
    return chunks


def chunk_docs(docs: list[RawDoc], max_chars: int = 520, overlap: int = 60) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_doc(d, max_chars, overlap))
    # 同一文件的重复片段可合并，其他文件或页码的来源保留。
    seen, uniq = set(), []
    for c in out:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        uniq.append(c)
    return uniq
