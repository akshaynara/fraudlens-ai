"""
Firestore case archive.

Every investigated case (reconstruction, agent trace, network, Skeptic verdicts, brief) is written to
Firestore. On startup the archive is read back so:
  * the investigator's case history survives restarts and redeploys, and
  * complaints investigated earlier stay in the corpus, so a new complaint can link to them.

Firestore is optional: if it is unreachable (no credentials locally, API disabled, missing IAM role)
the app logs a warning and keeps working in memory. The demo never depends on it.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("fraudlens.archive")

try:
    from google.cloud import firestore
except ImportError:  # pragma: no cover
    firestore = None

COLLECTION = os.getenv("FIRESTORE_COLLECTION", "cases")


class CaseArchive:
    def __init__(self):
        self.available = False
        self.client = None
        if os.getenv("FIRESTORE_ENABLED", "true").lower() in ("0", "false", "no"):
            log.info("Firestore archive disabled by FIRESTORE_ENABLED")
            return
        if firestore is None:
            log.warning("google-cloud-firestore not installed; case archive disabled")
            return
        try:
            self.client = firestore.Client(database=os.getenv("FIRESTORE_DATABASE", "(default)"))
            # cheap round trip so we fail fast at startup rather than mid-investigation
            next(iter(self.client.collection(COLLECTION).limit(1).stream()), None)
            self.available = True
            log.info("Firestore archive ready (project=%s, collection=%s)", self.client.project, COLLECTION)
        except Exception as e:
            log.warning("Firestore unavailable, continuing in memory: %s", e)
            self.client = None

    def save(self, case: dict) -> bool:
        if not self.available:
            return False
        try:
            clean = {k: v for k, v in case.items() if k != "evidence_blobs"}
            cluster = clean.get("cluster") or {}
            doc = {
                "id": clean["id"],
                "created_at": clean.get("created_at"),
                "status": clean.get("status"),
                "committed": bool(clean.get("committed")),
                "scam_type": (clean.get("extraction") or {}).get("scam_type"),
                "related_reports": len(cluster.get("related") or []),
                # full case as JSON: Firestore cannot store nested arrays, and this keeps the shape exact
                "data": json.dumps(clean, default=str),
            }
            self.client.collection(COLLECTION).document(clean["id"]).set(doc)
            return True
        except Exception as e:
            log.warning("Firestore save failed for %s: %s", case.get("id"), e)
            return False

    def load(self, limit: int = 200) -> list[dict]:
        if not self.available:
            return []
        try:
            docs = (self.client.collection(COLLECTION)
                    .order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit).stream())
            out = []
            for d in docs:
                try:
                    out.append(json.loads(d.to_dict()["data"]))
                except Exception:
                    continue
            out.reverse()  # oldest first, so corpus commits happen in filing order
            return out
        except Exception as e:
            log.warning("Firestore load failed: %s", e)
            return []
