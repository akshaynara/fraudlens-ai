"""
Liability engine: deterministic rules for RBI's revised customer-protection framework for
fraudulent electronic banking transactions (EBTs), effective for transactions on or after
1 January 2027.

Same principle as graph.py: Gemini never chooses the verdict. Gemini (Extract + Skeptic) fills
a LiabilityCase and rates each finding as proven / claimed / unknown; this module applies the
rules and returns a decision an authorised officer reviews and signs.

The rule values below come from published summaries of the final directions (June 2026), not
from the RBI text itself. Every constant is in one place so it can be corrected against the
primary text. This is not legal advice.

Rules, checked in order (first match wins):

    R0  transaction before the effective date           -> out of scope (2017 framework)
    R1  bank negligence proven                          -> zero liability, full reversal
    R2  third-party breach, reported within 5 days      -> zero liability, full reversal
    R3  third-party breach, reported later              -> the bank's own policy (configurable)
    R4  customer negligence proven, compensation due    -> customer bears loss less compensation
    R5  customer negligence proven, not eligible        -> customer bears loss up to reporting
    R7  nothing proven                                  -> bank has not met its burden of proof

    R6 (transactions after the customer's report are borne by the bank) applies on top of
    every outcome through `post_report_amount_inr`.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- rule constants (verify against RBI text)
EFFECTIVE_DATE = date(2027, 1, 1)             # directions apply to EBTs on or after this date
COMPENSATION_SCHEME_END = date(2027, 12, 31)  # compensation covers frauds within one year of the effective date
REPORTING_WINDOW_DAYS = 5                     # calendar days from the transaction
RESOLUTION_DAYS_DOMESTIC = 45                 # examine, establish liability, respond
RESOLUTION_DAYS_CROSS_BORDER = 60
SHADOW_REVERSAL_DAYS = 5                      # credit card fraud: provisional credit after notification
COMPENSATION_RATE = 0.85                      # of net loss
COMPENSATION_CAP_INR = 25_000
COMPENSATION_MAX_GROSS_LOSS_INR = 50_000

Rating = Literal["proven", "claimed", "unknown"]

BankGround = Literal[
    "security_systems_missing",        # mandated systems/procedures for EBT security not in place
    "alert_not_sent",                  # mandatory transaction alert not sent
    "no_24x7_reporting_channel",       # no round-the-clock channel to report fraud or a lost card
    "not_diligent_after_report",       # did not act diligently after the customer's report
    "system_failure_or_internal_fraud",
]
CustomerGround = Literal[
    "credentials_shared",              # PIN / password / OTP shared or not protected
    "delayed_notification",            # did not promptly notify after learning of the fraud
    "ignored_scam_warning",            # ignored a specific, direct scam warning from the bank
    "malicious_app_installed",
    "contact_details_not_updated",     # changed mobile/email not updated with the bank
]


class Finding(BaseModel):
    """One negligence ground, with how well the evidence supports it."""

    ground: str
    rating: Rating = "unknown"
    evidence: str = Field(default="", description="Pointer to the document or log line, e.g. 'bank SMS log: no alert for txn 4417'")


class BankFinding(Finding):
    ground: BankGround


class CustomerFinding(Finding):
    ground: CustomerGround


class BankPolicy(BaseModel):
    """What the bank's own policy says for the one case the directions leave to it (R3)."""

    third_party_breach_reported_late: Literal["zero_liability", "customer_bears", "officer_review"] = "officer_review"


class LiabilityCase(BaseModel):
    case_id: str
    transaction_at: datetime
    amount_inr: float = Field(gt=0, description="Gross loss on the disputed transaction(s)")
    cross_border: bool = False
    is_credit_card: bool = False
    customer_type: Literal["individual", "sole_proprietor", "other"] = "individual"

    reported_to_bank_at: Optional[datetime] = None
    reported_to_ncrp_at: Optional[datetime] = Field(default=None, description="Report on NCRP (cybercrime.gov.in) or the 1930 helpline")
    ncrp_acknowledgement: Optional[str] = None

    bank_findings: list[BankFinding] = []
    customer_findings: list[CustomerFinding] = []
    third_party_breach: Rating = "unknown"
    third_party_evidence: str = ""

    amount_recovered_inr: float = Field(default=0, ge=0, description="Recovered or lien-marked funds returned to the customer")
    post_report_amount_inr: float = Field(default=0, ge=0, description="Unauthorised transactions after the customer reported")
    loss_is_bona_fide: bool = Field(default=True, description="Loss established as genuine, e.g. supported by linked complaints")
    prior_compensation_claimed: bool = False
    linked_complaints: int = Field(default=0, ge=0, description="Complaints sharing a non-generic identifier (from the graph)")


class Check(BaseModel):
    name: str
    passed: bool
    detail: str


class LiabilityDecision(BaseModel):
    case_id: str
    rule: Literal["R0", "R1", "R2", "R3", "R4", "R5", "R7"]
    outcome: Literal["out_of_scope", "zero_liability", "bank_policy", "customer_bears_with_compensation",
                     "customer_bears", "burden_not_met"]
    headline: str
    reversal_inr: float = Field(description="Amount the bank reverses to the customer, value-dated to the transaction date")
    compensation_inr: float
    customer_bears_inr: float
    grounds_relied_on: list[Finding]
    unproven_grounds: list[Finding] = Field(description="Raised but not proven; cannot support a finding against the customer")
    compensation_checks: list[Check]
    reasons: list[str]
    resolve_by: Optional[date] = None
    shadow_reversal_by: Optional[date] = None
    needs_officer_review: bool = Field(description="True when a human judgment call remains beyond routine sign-off")


# ---------------------------------------------------------------- helpers
def days_between(start: datetime, end: datetime) -> int:
    """Calendar days from the transaction date to the report date (same day = 0)."""
    return (end.date() - start.date()).days


def within_window(case: LiabilityCase, reported_at: Optional[datetime]) -> bool:
    return reported_at is not None and 0 <= days_between(case.transaction_at, reported_at) <= REPORTING_WINDOW_DAYS


def net_loss(case: LiabilityCase) -> float:
    return max(case.amount_inr - case.amount_recovered_inr, 0.0)


def compensation_amount(net_loss_inr: float) -> float:
    """85% of net loss or Rs 25,000, whichever is lower."""
    return round(min(COMPENSATION_RATE * net_loss_inr, COMPENSATION_CAP_INR), 2)


def compensation_checks(case: LiabilityCase) -> list[Check]:
    txn_day = case.transaction_at.date()
    bank_days = days_between(case.transaction_at, case.reported_to_bank_at) if case.reported_to_bank_at else None
    ncrp_days = days_between(case.transaction_at, case.reported_to_ncrp_at) if case.reported_to_ncrp_at else None
    return [
        Check(name="individual_or_sole_proprietor", passed=case.customer_type in ("individual", "sole_proprietor"),
              detail=f"Customer type: {case.customer_type}"),
        Check(name="gross_loss_within_limit", passed=case.amount_inr <= COMPENSATION_MAX_GROSS_LOSS_INR,
              detail=f"Gross loss Rs {case.amount_inr:,.0f}; limit Rs {COMPENSATION_MAX_GROSS_LOSS_INR:,}"),
        Check(name="reported_to_bank_in_time", passed=within_window(case, case.reported_to_bank_at),
              detail="Not reported to the bank" if bank_days is None else f"Reported to the bank on day {bank_days}; window is {REPORTING_WINDOW_DAYS} days"),
        Check(name="reported_to_ncrp_in_time", passed=within_window(case, case.reported_to_ncrp_at),
              detail="Not reported to NCRP/1930" if ncrp_days is None else f"Reported to NCRP/1930 on day {ncrp_days}; window is {REPORTING_WINDOW_DAYS} days"),
        Check(name="first_claim", passed=not case.prior_compensation_claimed,
              detail="Compensation already claimed once" if case.prior_compensation_claimed else "No earlier compensation claim"),
        Check(name="within_scheme_period", passed=EFFECTIVE_DATE <= txn_day <= COMPENSATION_SCHEME_END,
              detail=f"Transaction on {txn_day.isoformat()}; scheme runs {EFFECTIVE_DATE.isoformat()} to {COMPENSATION_SCHEME_END.isoformat()}"),
        Check(name="loss_is_bona_fide", passed=case.loss_is_bona_fide,
              detail=f"{case.linked_complaints} linked complaint(s) share an identifier" if case.loss_is_bona_fide else "Loss not established as genuine"),
    ]


def _deadlines(case: LiabilityCase) -> tuple[Optional[date], Optional[date]]:
    if case.reported_to_bank_at is None:
        return None, None
    reported = case.reported_to_bank_at.date()
    resolve = reported + timedelta(days=RESOLUTION_DAYS_CROSS_BORDER if case.cross_border else RESOLUTION_DAYS_DOMESTIC)
    shadow = reported + timedelta(days=SHADOW_REVERSAL_DAYS) if case.is_credit_card else None
    return resolve, shadow


# ---------------------------------------------------------------- the decision
def decide(case: LiabilityCase, policy: BankPolicy | None = None) -> LiabilityDecision:
    policy = policy or BankPolicy()
    gross = case.amount_inr
    full = net_loss(case)  # what a full reversal still owes the customer
    post = case.post_report_amount_inr
    resolve_by, shadow_by = _deadlines(case)

    bank_proven = [f for f in case.bank_findings if f.rating == "proven"]
    cust_proven = [f for f in case.customer_findings if f.rating == "proven"]
    unproven: list[Finding] = [f for f in [*case.bank_findings, *case.customer_findings] if f.rating == "claimed"]
    if case.third_party_breach == "claimed":
        unproven.append(Finding(ground="third_party_breach", rating="claimed", evidence=case.third_party_evidence))

    def build(rule, outcome, headline, *, reversal, compensation=0.0, relied=(), checks=(), reasons=(), review=False):
        reasons = list(reasons)
        if post:
            reasons.append(f"Rs {post:,.0f} of transactions after the customer's report are borne by the bank.")
        bears = round(max(gross - reversal - compensation - case.amount_recovered_inr, 0.0), 2) if outcome != "out_of_scope" else 0.0
        return LiabilityDecision(
            case_id=case.case_id, rule=rule, outcome=outcome, headline=headline,
            reversal_inr=round(reversal + (post if outcome != "out_of_scope" else 0.0), 2), compensation_inr=compensation,
            customer_bears_inr=bears, grounds_relied_on=list(relied), unproven_grounds=unproven,
            compensation_checks=list(checks), reasons=reasons, resolve_by=resolve_by, shadow_reversal_by=shadow_by,
            needs_officer_review=review)

    # R0 - scope
    if case.transaction_at.date() < EFFECTIVE_DATE:
        return build("R0", "out_of_scope", "Transaction predates the 2027 framework", reversal=0.0,
                     reasons=[f"Transaction on {case.transaction_at.date().isoformat()} is before {EFFECTIVE_DATE.isoformat()}; "
                              "the earlier (2017) limited-liability framework applies."], review=True)

    # R1 - bank negligence
    if bank_proven:
        return build("R1", "zero_liability", "Zero customer liability: bank negligence", reversal=full, relied=bank_proven,
                     reasons=["Bank negligence is proven, so the customer has zero liability whether or not the fraud was reported.",
                              "Reverse the full amount, value-dated to the transaction date."])

    # R2 / R3 - third-party breach
    if case.third_party_breach == "proven":
        relied = [Finding(ground="third_party_breach", rating="proven", evidence=case.third_party_evidence)]
        if within_window(case, case.reported_to_bank_at):
            return build("R2", "zero_liability", "Zero customer liability: third-party breach reported in time", reversal=full,
                         relied=relied, reasons=[f"The deficiency lies with a third party and the customer reported within {REPORTING_WINDOW_DAYS} calendar days.",
                                                 "Reverse the full amount, value-dated to the transaction date."])
        choice = policy.third_party_breach_reported_late
        late = ["Third-party breach is proven but was reported after the 5-day window (or not reported).",
                "The directions leave this outcome to the bank's own policy."]
        if choice == "zero_liability":
            return build("R3", "bank_policy", "Bank policy: zero liability for a late-reported third-party breach", reversal=full,
                         relied=relied, reasons=late + ["Configured policy: reverse in full."])
        if choice == "customer_bears":
            return build("R3", "bank_policy", "Bank policy: customer bears a late-reported third-party breach", reversal=0.0,
                         relied=relied, reasons=late + ["Configured policy: customer bears the loss."], review=True)
        return build("R3", "bank_policy", "Officer decision needed: third-party breach reported late", reversal=0.0,
                     relied=relied, reasons=late + ["No policy configured; an officer must decide."], review=True)

    # R4 / R5 - customer negligence (only on PROVEN grounds: the burden of proof is the bank's)
    if cust_proven:
        checks = compensation_checks(case)
        failed = [c for c in checks if not c.passed]
        if not failed:
            comp = compensation_amount(net_loss(case))
            return build("R4", "customer_bears_with_compensation", f"Customer negligence proven; compensation of Rs {comp:,.0f} is due",
                         reversal=0.0, compensation=comp, relied=cust_proven, checks=checks,
                         reasons=["Customer negligence is proven on the grounds listed.",
                                  f"All compensation conditions are met: {int(COMPENSATION_RATE * 100)}% of net loss "
                                  f"Rs {net_loss(case):,.0f}, capped at Rs {COMPENSATION_CAP_INR:,}.",
                                  "Pay within 5 calendar days of the completed application."])
        return build("R5", "customer_bears", "Customer negligence proven; not eligible for compensation", reversal=0.0,
                     relied=cust_proven, checks=checks,
                     reasons=["Customer negligence is proven on the grounds listed.",
                              "Compensation is not due: " + "; ".join(c.detail for c in failed) + ".",
                              "The rejection letter must give these reasons and the supporting details."])

    # R7 - burden not met
    reasons = ["No negligence by the customer is proven. The burden of proving customer liability lies with the bank, "
               "so the customer cannot be held liable on this evidence."]
    if unproven:
        reasons.append("Raised but unproven: " + ", ".join(f.ground for f in unproven) + ". Obtain the evidence or drop the ground.")
    return build("R7", "burden_not_met", "Bank has not met its burden of proof", reversal=full, reasons=reasons, review=True)
