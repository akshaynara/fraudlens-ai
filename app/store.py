"""
In-memory case store backed by data/reports.json.

Swap this for Firestore/BigQuery in production: the public methods are the contract
the agent tools rely on (lookup_identifier, find_similar, get_report).
"""
from __future__ import annotations

import json
import math
import re
import threading
from collections import defaultdict
from pathlib import Path

from .embeddings import Embedder

_WORD = re.compile(r"[a-z]{3,}")


def _norm_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def normalize_identifier(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"^https?://", "", v).rstrip("/")
    if re.fullmatch(r"[+\d\s\-()]{8,}", v):  # looks like a phone number
        digits = re.sub(r"\D", "", v)
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        return digits
    return v


class CaseStore:
    def __init__(self, data_dir: Path, embedder: Embedder):
        self.data_dir = data_dir
        self.embedder = embedder
        self._lock = threading.Lock()
        self.reports: dict[str, dict] = {}
        self.truth: dict[str, dict] = {}
        self.ident_index: dict[str, set[str]] = defaultdict(set)
        self.ident_type: dict[str, str] = {}
        self.ident_display: dict[str, str] = {}
        self.legit: dict[str, str] = {}
        self.vectors: dict[str, list[float]] = {}
        self.cases: dict[str, dict] = {}
        # Statements that are never added to the corpus (the bundled demo case is meant to be re-run)
        self.skip_commit: set[str] = set()
        self.archive = None  # optional CaseArchive (Firestore); set by main.py
        self._load()

    def archive_case(self, case_id: str) -> bool:
        if not self.archive:
            return False
        with self._lock:
            case = dict(self.cases[case_id])
        return self.archive.save(case)

    def restore_from_archive(self) -> int:
        """Reload archived cases: history for the UI, and committed cases back into the corpus."""
        if not self.archive:
            return 0
        n = 0
        for c in self.archive.load():
            if not c.get("id") or c["id"] in self.cases:
                continue
            c.setdefault("evidence_blobs", [])
            c.setdefault("trace", [])
            self.cases[c["id"]] = c
            if c.get("committed") and c.get("extraction"):
                self.commit_case_to_corpus(c["id"], c["extraction"])
            n += 1
        return n

    def never_commit(self, text: str):
        self.skip_commit.add(_norm_text(text))

    # ------------------------------------------------------------------ loading
    def _load(self):
        raw = json.loads((self.data_dir / "reports.json").read_text())
        legit_path = self.data_dir / "legit_identifiers.json"
        if legit_path.exists():
            self.legit = {normalize_identifier(k): v for k, v in json.loads(legit_path.read_text()).items()}
        for r in raw:
            truth = r.pop("_truth", {})
            self.truth[r["id"]] = truth
            self.reports[r["id"]] = r
            for ident in r["identifiers"]:
                self._index_identifier(r["id"], ident)
        texts = [self._embed_text(r) for r in self.reports.values()]
        vecs = self.embedder.embed(texts)
        for r, v in zip(self.reports.values(), vecs):
            self.vectors[r["id"]] = v

    def _index_identifier(self, report_id: str, ident: dict):
        key = normalize_identifier(ident["value"])
        if not key:
            return
        self.ident_index[key].add(report_id)
        self.ident_type.setdefault(key, ident["type"])
        self.ident_display.setdefault(key, ident["value"])

    @staticmethod
    def _embed_text(r: dict) -> str:
        return f"{r.get('scam_type', '')}. {r['narrative']}"

    # ------------------------------------------------------------------ queries
    def get_report(self, report_id: str, redact=True) -> dict | None:
        r = self.reports.get(report_id) or self.cases.get(report_id, {}).get("as_report")
        return r

    def lookup_identifier(self, value: str, exclude: str | None = None) -> dict:
        key = normalize_identifier(value)
        hits = sorted(i for i in self.ident_index.get(key, set()) if i != exclude)
        return {
            "identifier": self.ident_display.get(key, value),
            "normalized": key,
            "type": self.ident_type.get(key, "unknown"),
            "matching_report_ids": hits,
            "match_count": len(hits),
            "known_legitimate": self.legit.get(key),
            "scam_types_of_matches": sorted({self.reports[i]["scam_type"] for i in hits if i in self.reports}),
        }

    def find_similar(self, text: str, k: int = 8, exclude: str | None = None, min_score: float = 0.0) -> list[dict]:
        q = self.embedder.embed([text])[0]
        scored = []
        for rid, v in self.vectors.items():
            if rid == exclude:
                continue
            scored.append((self._cos(q, v), rid))
        scored.sort(reverse=True)
        out = []
        for s, rid in scored[:k]:
            if s < min_score:
                continue
            r = self.reports[rid]
            out.append({"report_id": rid, "similarity": round(s, 3), "scam_type": r["scam_type"],
                        "filed_at": r["filed_at"], "snippet": r["narrative"][:220] + "..."})
        return out

    @staticmethod
    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    def stats(self) -> dict:
        by_type = defaultdict(int)
        for r in self.reports.values():
            by_type[r["scam_type"]] += 1
        multi = sum(1 for v in self.ident_index.values() if len(v) > 1)
        return {"reports": len(self.reports), "identifiers": len(self.ident_index),
                "identifiers_seen_in_multiple_reports": multi, "by_scam_type": dict(by_type),
                "live_cases": len(self.cases)}

    def list_reports(self) -> list[dict]:
        return [{"id": r["id"], "filed_at": r["filed_at"], "scam_type": r["scam_type"], "city": r["victim_city"],
                 "amount_inr": r["amount_claimed_inr"], "status": r["status"],
                 "identifier_count": len(r["identifiers"])} for r in self.reports.values()]

    # ------------------------------------------------------------------ live cases
    def create_case(self, case: dict):
        with self._lock:
            self.cases[case["id"]] = case

    def update_case(self, case_id: str, **fields):
        with self._lock:
            self.cases[case_id].update(fields)

    def append_trace(self, case_id: str, step: dict):
        with self._lock:
            self.cases[case_id]["trace"].append(step)

    def commit_case_to_corpus(self, case_id: str, extraction: dict) -> bool:
        """Once investigated, the new case becomes searchable for future complaints.
        Re-submitting an identical statement (e.g. running the demo case twice) is not added again,
        otherwise the second run would 'discover' the first run as a related complaint."""
        case = self.cases[case_id]
        norm = _norm_text(case["narrative"])
        if norm in self.skip_commit or any(_norm_text(r["narrative"]) == norm for r in self.reports.values()):
            return False
        rep = {
            "id": case_id, "filed_at": case["created_at"], "victim_city": "n/a",
            "scam_type": extraction.get("scam_type", "other"), "channel": "upload",
            "amount_claimed_inr": (extraction.get("amounts") or [{"value_inr": 0}])[0]["value_inr"],
            "amount_on_receipt_inr": None, "evidence_types": case["evidence_files"],
            "narrative": case["narrative"], "identifiers": extraction.get("identifiers", []), "status": "open",
        }
        with self._lock:
            self.reports[case_id] = rep
            for ident in rep["identifiers"]:
                self._index_identifier(case_id, ident)
            self.vectors[case_id] = self.embedder.embed([self._embed_text(rep)])[0]
        return True
