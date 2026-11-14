"""End-to-end: the demo complaint through the offline pipeline, with and without a bank record."""
import os
import time
from pathlib import Path

os.environ.setdefault("FIRESTORE_ENABLED", "false")
for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
    os.environ.pop(k, None)

pytest = __import__("pytest")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

DEMO = Path(__file__).parents[1] / "demo"
client = TestClient(app)


def investigate(files):
    r = client.post("/api/cases", data={"narrative": (DEMO / "demo_victim_statement.txt").read_text()}, files=files)
    assert r.status_code == 200
    for _ in range(100):
        c = client.get(f"/api/cases/{r.json()['case_id']}").json()
        if c["status"] in ("done", "error"):
            return c
        time.sleep(0.1)
    raise AssertionError("investigation did not finish")


def test_demo_case_with_bank_record_gets_a_liability_decision():
    c = investigate([("files", ("demo_bank_record.json", (DEMO / "demo_bank_record.json").read_bytes(), "application/json"))])
    assert c["status"] == "done"
    d = c["liability"]["decision"]
    assert (d["rule"], d["preview"], d["compensation_inr"], d["customer_bears_inr"]) == ("R4", True, 25000, 23500)
    assert "Rs 25,000" in c["liability"]["letter"]
    assert "Liability assessment" in c["brief"]
    assert "evidence_blobs" not in c


def test_without_a_bank_record_the_engine_says_what_is_missing():
    c = investigate([])
    assert c["status"] == "done"
    assert c["liability"]["status"] == "needs_data" and c["liability"]["missing"] == ["transaction date and time"]


def test_malformed_bank_record_is_rejected():
    r = client.post("/api/cases", data={"narrative": "x"}, files=[("files", ("bank.json", b"{nope", "application/json"))])
    assert r.status_code == 400
