from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import Investigator
from .archive import CaseArchive
from .embeddings import Embedder
from .gemini_client import GeminiClient
from .graph import corpus_overview
from .store import CaseStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fraudlens")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
STATIC_DIR = ROOT / "static"
DEMO_DIR = ROOT / "demo"
MAX_UPLOAD_MB = 15

gemini = GeminiClient()
store = CaseStore(DATA_DIR, Embedder(gemini))
investigator = Investigator(store, gemini)
if (DEMO_DIR / "demo_victim_statement.txt").exists():
    store.never_commit((DEMO_DIR / "demo_victim_statement.txt").read_text())

archive = CaseArchive()
store.archive = archive
if archive.available:
    log.info("Restored %d case(s) from Firestore", store.restore_from_archive())

app = FastAPI(title="FraudLens AI", version="0.1.0")


@app.get("/api/health")
def health():
    return {"ok": True, "gemini": gemini.available, "model": gemini.model, "embeddings": store.embedder.mode,
            "firestore": archive.available,
            **store.stats()}


@app.get("/api/reports")
def reports():
    return store.list_reports()


@app.get("/api/reports/{report_id}")
def report(report_id: str):
    r = store.get_report(report_id)
    if not r:
        raise HTTPException(404, "report not found")
    return r


@app.get("/api/graph/overview")
def overview():
    return corpus_overview(store)


@app.post("/api/cases")
async def create_case(narrative: str = Form(...), files: list[UploadFile] = File(default=[])):
    if not narrative.strip() and not files:
        raise HTTPException(400, "Provide a victim statement and/or evidence files")
    blobs = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} exceeds {MAX_UPLOAD_MB} MB")
        mime = f.content_type or "application/octet-stream"
        if mime.startswith("text/") or f.filename.endswith(".txt"):
            narrative += "\n\n[Attached text: " + f.filename + "]\n" + data.decode("utf-8", "ignore")
            continue
        blobs.append({"name": f.filename, "mime": mime, "bytes": data})
    case_id = "CASE-" + datetime.now(timezone.utc).strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:4].upper()
    store.create_case({
        "id": case_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "narrative": narrative.strip(), "evidence_files": [b["name"] for b in blobs], "evidence_blobs": blobs,
        "status": "queued", "trace": [], "extraction": None, "findings": None, "cluster": None, "review": None, "brief": None,
    })
    threading.Thread(target=investigator.run, args=(case_id,), daemon=True).start()
    return {"case_id": case_id}


@app.get("/api/cases/{case_id}")
def get_case(case_id: str):
    c = store.cases.get(case_id)
    if not c:
        raise HTTPException(404, "case not found")
    return {k: v for k, v in c.items() if k != "evidence_blobs"}


@app.get("/api/cases")
def list_cases():
    return [{"id": c["id"], "status": c["status"], "created_at": c["created_at"]} for c in store.cases.values()]


@app.get("/api/truth")
def truth():
    """Ground truth for the synthetic dataset, used only by the judge/demo panel."""
    p = DATA_DIR / "ground_truth.json"
    if not p.exists():
        raise HTTPException(404, "no ground truth file")
    return FileResponse(p)


# Demo evidence bundle so judges can try it without preparing files
if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=DEMO_DIR), name="demo")

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
