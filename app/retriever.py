"""BM25 检索和基于排序位置的 RRF 融合。"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunker import Chunk

CJK = re.compile(r"[\u4e00-\u9fff]+")
LATIN = re.compile(r"[a-z0-9_.]+")


def tokenize(text: str) -> list[str]:
    """中文按 2-gram，英文/数字按词切——不需要分词库。"""
    text = text.lower()
    tokens = LATIN.findall(text)
    for seg in CJK.findall(text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text + " " + c.heading)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lens) / max(1, len(self.lens))
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term in q:
                if term not in doc:
                    continue
                tf = doc[term]
                score += self.idf.get(term, 0) * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * self.lens[i] / max(self.avg_len, 1e-9))
                )
            if score > 0:
                scored.append((self.chunks[i], score))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


def rrf(result_lists: list[list[tuple[Chunk, float]]], k: int = 60) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion：只看排名，不看两路分数量纲。"""
    scores: dict[str, float] = {}
    pool: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, (chunk, _) in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            pool[chunk.chunk_id] = chunk
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(pool[cid], s) for cid, s in ranked]


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    source: str  # vector / bm25 / hybrid


def hybrid_search(vector_hits, bm25_hits, top_k: int = 5) -> list[RetrievalResult]:
    fused = rrf([vector_hits, bm25_hits])[:top_k]
    return [RetrievalResult(chunk=c, score=s, source="hybrid") for c, s in fused]
