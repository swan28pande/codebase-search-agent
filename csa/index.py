"""Chunk a repo and index it for hybrid BM25 + dense (FAISS) retrieval."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

CODE_EXTS = {".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".md"}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class Chunk:
    path: str
    start: int
    end: int
    text: str

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"


def tokenize(text: str) -> list[str]:
    # split snake_case and camelCase so `get_user` matches `getUser`
    out = []
    for tok in TOKEN_RE.findall(text):
        out += [p.lower() for p in re.split(r"_|(?<=[a-z])(?=[A-Z])", tok) if p]
    return out


def hash_embed(texts: list[str], dim: int = 512) -> np.ndarray:
    """Dependency-free hashing embedding. Swap for a real model via CodeIndex(embed=...)."""
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for tok in tokenize(t):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vecs[i, h % dim] += 1.0 if (h >> 64) & 1 else -1.0
    faiss.normalize_L2(vecs)
    return vecs


class CodeIndex:
    def __init__(self, root: str, chunk_lines: int = 40, embed=hash_embed):
        self.embed = embed
        self.chunks = self._chunk(Path(root), chunk_lines)
        if not self.chunks:
            raise ValueError(f"no indexable files under {root}")
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        vecs = embed([c.text for c in self.chunks])
        self.faiss = faiss.IndexFlatIP(vecs.shape[1])
        self.faiss.add(vecs)

    @staticmethod
    def _chunk(root: Path, n: int) -> list[Chunk]:
        chunks = []
        for p in sorted(root.rglob("*")):
            if p.suffix not in CODE_EXTS or any(part.startswith(".") for part in p.parts):
                continue
            lines = p.read_text(errors="ignore").splitlines()
            for s in range(0, len(lines), n):
                body = "\n".join(lines[s:s + n])
                if body.strip():
                    chunks.append(Chunk(str(p.relative_to(root)), s + 1, min(s + n, len(lines)), body))
        return chunks

    def search(self, query: str, k: int = 8, alpha: float = 0.5) -> list[tuple[Chunk, float]]:
        """Hybrid search: reciprocal-rank fusion of BM25 and FAISS rankings."""
        bm = self.bm25.get_scores(tokenize(query))
        bm_rank = np.argsort(-bm)[: k * 4]
        _, dense_idx = self.faiss.search(self.embed([query]), min(k * 4, len(self.chunks)))
        fused: dict[int, float] = {}
        for r, i in enumerate(bm_rank):
            fused[int(i)] = fused.get(int(i), 0) + alpha / (60 + r)
        for r, i in enumerate(dense_idx[0]):
            if i >= 0:
                fused[int(i)] = fused.get(int(i), 0) + (1 - alpha) / (60 + r)
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(self.chunks[i], s) for i, s in top]
