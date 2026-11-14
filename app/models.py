from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

IdentifierType = Literal["phone", "upi", "bank_account", "url", "telegram", "email", "other"]


class Identifier(BaseModel):
    type: IdentifierType
    value: str
    source: str = Field(description="Which piece of evidence this came from, e.g. 'whatsapp screenshot', 'victim statement'")


class TimelineEvent(BaseModel):
    when: str = Field(description="Date/time or relative time as stated in evidence")
    event: str
    source: str


class Amount(BaseModel):
    value_inr: float
    source: str


class IncidentExtraction(BaseModel):
    """Structured reconstruction of a single fraud complaint from its evidence."""

    summary: str = Field(description="2-3 sentence neutral summary of what the evidence shows")
    scam_type: str = Field(description="One of: digital_arrest, courier_parcel, electricity_bill, task_job, loan_app, kyc_update, fake_buyer_olx, investment_group, other")
    tactics: list[str] = Field(description="Short phrases describing the manipulation tactics used")
    timeline: list[TimelineEvent]
    identifiers: list[Identifier]
    amounts: list[Amount]
    facts_supported_by_evidence: list[str]
    assumptions_or_unverified: list[str]
    missing_information: list[str]
    inconsistencies: list[str] = Field(description="Contradictions between pieces of evidence, e.g. amount in statement vs receipt")


class ConnectionVerdict(BaseModel):
    report_id: str
    verdict: Literal["strong", "plausible", "weak", "reject"]
    reason: str


class SkepticReview(BaseModel):
    verdicts: list[ConnectionVerdict]
    overall_assessment: str
    what_would_change_my_mind: list[str]


class AgentFindings(BaseModel):
    hypotheses: list[str]
    related_report_ids: list[str]
    unresolved_questions: list[str]
    suggested_next_steps: list[str]


class TraceStep(BaseModel):
    kind: Literal["phase", "tool", "note", "error"]
    title: str
    detail: Optional[str] = None
    payload: Optional[dict] = None


class RatedGround(BaseModel):
    ground: Literal["credentials_shared", "malicious_app_installed", "delayed_notification"]
    rating: Literal["proven", "claimed", "unknown"] = Field(
        description="proven = directly visible in a screenshot or document; claimed = only stated or implied, no direct evidence")
    evidence: str = Field(description="Which evidence item shows it, and what it shows")


class LiabilityEvidence(BaseModel):
    """What the complaint's evidence says about the facts the liability rules need. Facts only, no verdict."""

    transaction_at: Optional[str] = Field(default=None, description="ISO 8601 date-time of the disputed payment with +05:30 offset, only if the evidence states it")
    reported_to_ncrp_at: Optional[str] = Field(default=None, description="ISO 8601 date-time the victim reported to 1930 or cybercrime.gov.in, only if stated")
    ncrp_acknowledgement: Optional[str] = Field(default=None, description="NCRP / 1930 acknowledgement number if visible")
    customer_grounds: list[RatedGround]
    coercion: Literal["proven", "claimed", "unknown"] = Field(description="Was the payment made under coercion, intimidation or duress")
    coercion_evidence: str
    notes: list[str] = Field(description="Anything an officer should check before signing")
