from __future__ import annotations

import hashlib
import logging
import math
import re

log = logging.getLogger("fraudlens.embeddings")
_WORD = re.compile(r"[a-z0-9]{3,}")
_STOP = set("the and was for with that this they them from have had into then after which were said told "
            "would could about rs upi call called phone number account money bank".split())


class Embedder:
    """Narrative similarity. Uses Gemini `gemini-embedding-001` when a client is present,
    otherwise a deterministic hashed bag-of-words vector (good enough for tactic similarity)."""

    def __init__(self, gemini=None, model: str = "gemini-embedding-001", dims: int = 768):
        self.gemini = gemini
        self.model = model
        self.dims = dims
        self.mode = "gemini" if gemini and gemini.available else "local"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.mode == "gemini":
            try:
                return self.gemini.embed(texts, self.model, self.dims)
            except Exception as e:  # fall back rather than crash the demo
                log.warning("Gemini embedding failed (%s); using local embeddings", e)
                self.mode = "local"
        return [self._local(t) for t in texts]

    def _local(self, text: str) -> list[float]:
        vec = [0.0] * 512
        toks = [t for t in _WORD.findall(text.lower()) if t not in _STOP]
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % 512] += 1.0 if "_" not in g else 1.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
