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


# ---------------------------------------------------------------- intake
def test_text_intake_stamps_the_reporting_time_and_asks_for_what_is_missing():
    r = client.post("/api/intake", data={"text": "Someone posing as police made me send Rs 12,000 by UPI."})
    assert r.status_code == 200
    rec = r.json()
    assert rec["intake_id"].startswith("INT-") and rec["received_at"].endswith("+05:30")
    assert rec["amount_inr"] == 12000 and rec["transaction_at"] is None
    assert any("When was the payment made" in q for q in rec["follow_up_questions"])


def test_voice_intake_without_gemini_says_so_instead_of_guessing():
    r = client.post("/api/intake", files={"audio": ("complaint.wav", b"RIFF0000WAVE", "audio/wav")})
    assert r.status_code == 503 and "Gemini" in r.json()["detail"]


def test_empty_intake_and_unknown_intake_id_are_rejected():
    assert client.post("/api/intake", data={"text": "  "}).status_code == 400
    assert client.post("/api/cases", data={"narrative": "x", "intake_id": "INT-NOPE"}).status_code == 400


def test_intake_supplies_the_reporting_time_and_caller_stated_facts():
    from app import main
    rec = client.post("/api/intake", data={"text": (DEMO / "demo_victim_statement.txt").read_text()}).json()
    # what a Gemini intake would have extracted from the caller's words
    main.intakes[rec["intake_id"]].update(transaction_at="2026-10-16T19:00:00+05:30",
                                          reported_to_ncrp_at="2026-10-17T09:00:00+05:30")
    r = client.post("/api/cases", data={"narrative": rec["statement_english"], "intake_id": rec["intake_id"]})
    for _ in range(100):
        c = client.get(f"/api/cases/{r.json()['case_id']}").json()
        if c["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    liability = c["liability"]
    assert liability["status"] == "decided" and liability["reported_source"] == "stamped at intake"
    assert liability["case"]["reported_to_bank_at"] == rec["received_at"]
    # no negligence is proven against the customer without a bank record, so the bank has not met its burden
    assert liability["decision"]["rule"] == "R7"
    assert any("as stated by the caller" in n for n in liability["notes"])
    assert liability["letter_hi"].startswith("विषय:")
