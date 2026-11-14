"""
Complaint intake: a caller's voice note (or typed text) becomes a structured complaint.

Why it matters for liability: under RBI's 2027 framework the outcome depends on WHEN the customer
reported. The server stamps `received_at` the moment the complaint arrives, so the reporting time is
a system fact, not something anyone types in later. Gemini only transcribes and extracts what the
caller said; it never sets the reporting time.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime

from .gemini_client import GeminiClient
from .liability import IST
from .models import IntakeExtraction

INTAKE_SYSTEM = """You take first-contact fraud complaints for a bank in India. The caller may speak Hindi, English or a
mix. Listen to (or read) the complaint and record it faithfully.
- statement_original: what the caller said, in their words and language (Hindi in Devanagari).
- statement_english: the same account in plain English, first person. Add nothing, drop nothing material.
  Keep every phone number, UPI ID, amount, name and time exactly as stated.
- Fill a date-time or amount ONLY if the caller stated it. Resolve relative times ("yesterday evening", "this morning")
  against the current time you are given. If only a part of the day is given use: morning 09:00, afternoon 14:00,
  evening 19:00, night 22:00. If no time is stated at all, leave it empty.
- follow_up_questions: one short, kind question in the caller's language for each of these that is still missing:
  when the payment was made, how much was lost, whether they reported to 1930 or cybercrime.gov.in and when, and the
  acknowledgement number. Ask nothing else. Never ask for a PIN, password or OTP.
Never judge whether the caller was careless. Never promise a refund."""

FOLLOW_UPS = {
    "transaction_at": "When was the payment made (date and time)?",
    "amount_inr": "How much money was lost?",
    "reported_to_ncrp_at": "Have you reported this to 1930 or cybercrime.gov.in? If yes, when?",
    "ncrp_acknowledgement": "Do you have the 1930 / cybercrime.gov.in acknowledgement number?",
}


class IntakeUnavailable(RuntimeError):
    pass


def run_intake(gemini: GeminiClient, text: str = "", audio: bytes | None = None, mime: str = "audio/wav") -> dict:
    received_at = datetime.now(IST).replace(microsecond=0)
    if gemini.available:
        parts: list = []
        if audio:
            parts += [gemini.file_part(audio, mime), "(The item above is the caller's voice complaint.)"]
        if text.strip():
            parts.append(f"Typed complaint:\n{text.strip()}")
        parts.append(f"Current time in India: {received_at.isoformat()} ({received_at.strftime('%A')}). Record the complaint.")
        ex = gemini.generate_structured(parts, IntakeExtraction, INTAKE_SYSTEM)
        mode = "gemini"
    elif audio and not text.strip():
        raise IntakeUnavailable("Voice intake needs Gemini to transcribe the recording. Set GEMINI_API_KEY, or type the complaint.")
    else:
        ex, mode = _offline(text.strip()), "offline"
    return {"intake_id": "INT-" + uuid.uuid4().hex[:8].upper(), "received_at": received_at.isoformat(),
            "source": "voice" if audio else "text", "mode": mode, **ex.model_dump()}


def _offline(text: str) -> IntakeExtraction:
    """No Gemini: keep the caller's words, pattern-match the amount, and ask for everything else."""
    m = re.search(r"(?:Rs\.?|₹|INR)\s?([\d,]{3,})", text)
    amount = float(m.group(1).replace(",", "")) if m else None
    ack = re.search(r"\b\d{14}\b", text)
    have = {"amount_inr": amount, "ncrp_acknowledgement": ack and ack.group(0)}
    return IntakeExtraction(
        language="unknown", statement_english=text, statement_original=text, amount_inr=amount,
        ncrp_acknowledgement=ack.group(0) if ack else None,
        follow_up_questions=[q for k, q in FOLLOW_UPS.items() if not have.get(k)])
