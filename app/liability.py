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

from datetime import date, datetime, timedelta, timezone
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
    preview: bool = Field(default=False, description="Transaction predates the framework; decided as if it applied")


# ---------------------------------------------------------------- helpers
IST = timezone(timedelta(hours=5, minutes=30))


def local_date(dt: datetime) -> date:
    """Calendar date in India. Timestamps without a zone are taken as IST."""
    return (dt.astimezone(IST) if dt.tzinfo else dt).date()


def days_between(start: datetime, end: datetime) -> int:
    """Calendar days from the transaction date to the report date (same day = 0)."""
    return (local_date(end) - local_date(start)).days


def within_window(case: LiabilityCase, reported_at: Optional[datetime]) -> bool:
    return reported_at is not None and 0 <= days_between(case.transaction_at, reported_at) <= REPORTING_WINDOW_DAYS


def net_loss(case: LiabilityCase) -> float:
    return max(case.amount_inr - case.amount_recovered_inr, 0.0)


def compensation_amount(net_loss_inr: float) -> float:
    """85% of net loss or Rs 25,000, whichever is lower."""
    return round(min(COMPENSATION_RATE * net_loss_inr, COMPENSATION_CAP_INR), 2)


def compensation_checks(case: LiabilityCase, preview: bool = False) -> list[Check]:
    txn_day = local_date(case.transaction_at)
    in_period = EFFECTIVE_DATE <= txn_day <= COMPENSATION_SCHEME_END
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
        Check(name="within_scheme_period", passed=in_period or (preview and txn_day < EFFECTIVE_DATE),
              detail=(f"Preview: scheme starts {EFFECTIVE_DATE.isoformat()}; treated as if in force" if preview and txn_day < EFFECTIVE_DATE else
                      f"Transaction on {txn_day.isoformat()}; scheme runs {EFFECTIVE_DATE.isoformat()} to {COMPENSATION_SCHEME_END.isoformat()}")),
        Check(name="loss_is_bona_fide", passed=case.loss_is_bona_fide,
              detail=("Loss not established as genuine" if not case.loss_is_bona_fide else
                      f"Loss treated as genuine; {case.linked_complaints} linked complaint(s) share an identifier" if case.linked_complaints else
                      "Loss treated as genuine; no linked complaints yet")),
    ]


def _deadlines(case: LiabilityCase) -> tuple[Optional[date], Optional[date]]:
    if case.reported_to_bank_at is None:
        return None, None
    reported = local_date(case.reported_to_bank_at)
    resolve = reported + timedelta(days=RESOLUTION_DAYS_CROSS_BORDER if case.cross_border else RESOLUTION_DAYS_DOMESTIC)
    shadow = reported + timedelta(days=SHADOW_REVERSAL_DAYS) if case.is_credit_card else None
    return resolve, shadow


# ---------------------------------------------------------------- the decision
def decide(case: LiabilityCase, policy: BankPolicy | None = None, preview: bool = False) -> LiabilityDecision:
    """Apply the rules. With preview=True a transaction before the effective date is decided as if the
    framework already applied (and the decision is marked as a preview) instead of returning R0."""
    policy = policy or BankPolicy()
    preview = preview and local_date(case.transaction_at) < EFFECTIVE_DATE
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
        if preview:
            reasons.insert(0, f"Preview: the framework applies to transactions from {EFFECTIVE_DATE.isoformat()}. "
                              "This shows how the case would be decided under it.")
        if post:
            reasons.append(f"Rs {post:,.0f} of transactions after the customer's report are borne by the bank.")
        bears = round(max(gross - reversal - compensation - case.amount_recovered_inr, 0.0), 2) if outcome != "out_of_scope" else 0.0
        return LiabilityDecision(
            case_id=case.case_id, rule=rule, outcome=outcome, headline=headline,
            reversal_inr=round(reversal + (post if outcome != "out_of_scope" else 0.0), 2), compensation_inr=compensation,
            customer_bears_inr=bears, grounds_relied_on=list(relied), unproven_grounds=unproven,
            compensation_checks=list(checks), reasons=reasons, resolve_by=resolve_by, shadow_reversal_by=shadow_by,
            needs_officer_review=review, preview=preview)

    # R0 - scope
    if not preview and local_date(case.transaction_at) < EFFECTIVE_DATE:
        return build("R0", "out_of_scope", "Transaction predates the 2027 framework", reversal=0.0,
                     reasons=[f"Transaction on {local_date(case.transaction_at).isoformat()} is before {EFFECTIVE_DATE.isoformat()}; "
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
        checks = compensation_checks(case, preview)
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


# ---------------------------------------------------------------- bank-side record -> findings
class BankRecord(BaseModel):
    """What the bank's own systems know about the disputed transaction. The bank's logs are its own
    evidence, so a finding read from them is rated proven. None means 'not checked', never 'no'."""

    customer_type: Literal["individual", "sole_proprietor", "other"] = "individual"
    transaction_at: Optional[datetime] = None
    amount_inr: Optional[float] = None
    channel: str = ""
    cross_border: bool = False
    is_credit_card: bool = False

    alert_sent: Optional[bool] = None
    alert_evidence: str = ""
    reporting_channel_24x7: Optional[bool] = None
    acted_on_report: Optional[bool] = None
    action_evidence: str = ""
    system_incident: Optional[str] = Field(default=None, description="Malfunction, breach or internal fraud affecting this transaction")
    third_party_incident: Optional[str] = Field(default=None, description="Deficiency at a UPI app, gateway, aggregator or telecom operator")

    scam_warning_dismissed: Optional[bool] = None
    scam_warning_evidence: str = ""
    contact_details_stale: Optional[bool] = None

    reported_to_bank_at: Optional[datetime] = None
    reported_to_ncrp_at: Optional[datetime] = None
    ncrp_acknowledgement: Optional[str] = None

    amount_recovered_inr: float = 0
    post_report_amount_inr: float = 0
    prior_compensation_claimed: bool = False


def findings_from_bank_record(r: BankRecord) -> tuple[list[BankFinding], list[CustomerFinding], Rating, str]:
    bank: list[BankFinding] = []
    cust: list[CustomerFinding] = []
    if r.alert_sent is False:
        bank.append(BankFinding(ground="alert_not_sent", rating="proven", evidence=r.alert_evidence or "Bank alert log: no alert sent"))
    if r.reporting_channel_24x7 is False:
        bank.append(BankFinding(ground="no_24x7_reporting_channel", rating="proven", evidence="Bank record: no 24x7 reporting channel"))
    if r.acted_on_report is False:
        bank.append(BankFinding(ground="not_diligent_after_report", rating="proven", evidence=r.action_evidence or "Bank complaint log: no action after report"))
    if r.system_incident:
        bank.append(BankFinding(ground="system_failure_or_internal_fraud", rating="proven", evidence=r.system_incident))
    if r.scam_warning_dismissed:
        cust.append(CustomerFinding(ground="ignored_scam_warning", rating="proven", evidence=r.scam_warning_evidence or "Bank app log: scam warning dismissed"))
    if r.contact_details_stale:
        cust.append(CustomerFinding(ground="contact_details_not_updated", rating="proven", evidence="Bank record: registered contact details out of date"))
    return bank, cust, ("proven" if r.third_party_incident else "unknown"), (r.third_party_incident or "")


# ---------------------------------------------------------------- customer letter
GROUND_TEXT = {
    "security_systems_missing": "required security systems were not in place",
    "alert_not_sent": "the mandatory transaction alert was not sent",
    "no_24x7_reporting_channel": "no round-the-clock reporting channel was available",
    "not_diligent_after_report": "the bank did not act promptly after your report",
    "system_failure_or_internal_fraud": "a system failure or internal lapse at the bank",
    "third_party_breach": "a deficiency at a third party in the payment chain",
    "credentials_shared": "a PIN, password or OTP was shared",
    "delayed_notification": "the bank was not notified promptly",
    "ignored_scam_warning": "a specific scam warning shown by the bank was dismissed",
    "malicious_app_installed": "a malicious app was installed on the device",
    "contact_details_not_updated": "registered contact details were not kept up to date",
}


GROUND_TEXT_HI = {
    "security_systems_missing": "ज़रूरी सुरक्षा व्यवस्थाएँ लागू नहीं थीं",
    "alert_not_sent": "लेन-देन का अनिवार्य अलर्ट नहीं भेजा गया",
    "no_24x7_reporting_channel": "शिकायत दर्ज करने की 24x7 सुविधा उपलब्ध नहीं थी",
    "not_diligent_after_report": "आपकी सूचना के बाद बैंक ने तुरंत कार्रवाई नहीं की",
    "system_failure_or_internal_fraud": "बैंक की प्रणाली में गड़बड़ी या आंतरिक चूक",
    "third_party_breach": "भुगतान प्रक्रिया से जुड़े किसी तीसरे पक्ष की चूक",
    "credentials_shared": "पिन, पासवर्ड या OTP साझा किया गया",
    "delayed_notification": "बैंक को समय पर सूचना नहीं दी गई",
    "ignored_scam_warning": "बैंक द्वारा दिखाई गई धोखाधड़ी की स्पष्ट चेतावनी को अनदेखा किया गया",
    "malicious_app_installed": "फ़ोन में हानिकारक ऐप इंस्टॉल किया गया",
    "contact_details_not_updated": "बैंक में दर्ज मोबाइल नंबर या ईमेल अपडेट नहीं था",
}
CHECK_TEXT_HI = {
    "individual_or_sole_proprietor": "यह योजना केवल व्यक्तिगत ग्राहकों और एकल स्वामित्व वाले कारोबार के लिए है",
    "gross_loss_within_limit": f"कुल नुकसान ₹{COMPENSATION_MAX_GROSS_LOSS_INR:,} की सीमा से अधिक है",
    "reported_to_bank_in_time": f"बैंक को सूचना {REPORTING_WINDOW_DAYS} कैलेंडर दिनों के भीतर नहीं दी गई",
    "reported_to_ncrp_in_time": f"1930 या cybercrime.gov.in पर शिकायत {REPORTING_WINDOW_DAYS} कैलेंडर दिनों के भीतर दर्ज नहीं की गई",
    "first_claim": "इस योजना का लाभ पहले लिया जा चुका है",
    "within_scheme_period": "लेन-देन योजना की अवधि के बाहर का है",
    "loss_is_bona_fide": "नुकसान की पुष्टि नहीं हो सकी",
}

LETTER_TEXT = {
    "en": {
        "subject": "Subject: Your fraud complaint {id}", "dear": "Dear Customer,",
        "intro": "We have examined your complaint about a fraudulent transaction on your account.",
        "zero": "You are not liable for this loss. We will credit {amt} to your account, dated to the day of the transaction, "
                "so you lose no interest and pay no charges.",
        "found": "What we found:",
        "partial": "Our records show the following, which means the loss is not reversed in full:",
        "comp": "You are eligible for compensation of {amt} under RBI's scheme for small-value frauds. "
                "It will be paid within 5 calendar days of your completed application.",
        "post_also": "In addition, {amt} debited after you reported the fraud will be credited back.",
        "reject": "We are unable to reverse this transaction. Our reasons and the supporting details are:",
        "no_comp": "Compensation under RBI's small-value fraud scheme does not apply because:",
        "post": "{amt} debited after you reported the fraud will be credited back.",
        "officer_by": "Your complaint needs a further decision by an officer. We will write to you again by {date}.",
        "officer": "Your complaint needs a further decision by an officer. We will write to you again shortly.",
        "appeal": "If you disagree, you may write to our Grievance Redressal Officer and, after that, to the RBI Ombudsman.",
        "sign": "[Authorised officer: name, designation, signature]",
    },
    "hi": {
        "subject": "विषय: आपकी धोखाधड़ी शिकायत {id}", "dear": "प्रिय ग्राहक,",
        "intro": "हमने आपके खाते में हुए धोखाधड़ी वाले लेन-देन की आपकी शिकायत की जाँच की है।",
        "zero": "इस नुकसान के लिए आप ज़िम्मेदार नहीं हैं। हम {amt} आपके खाते में लेन-देन की तारीख से वापस जमा करेंगे, "
                "ताकि आपको ब्याज का कोई नुकसान न हो और कोई शुल्क न लगे।",
        "found": "हमने क्या पाया:",
        "partial": "हमारे रिकॉर्ड में ये बातें दर्ज हैं, इसलिए पूरी रकम वापस नहीं की जा रही है:",
        "comp": "आप छोटी रकम की धोखाधड़ी के लिए RBI की योजना के तहत {amt} के मुआवज़े के पात्र हैं। "
                "आपका पूरा आवेदन मिलने के 5 कैलेंडर दिनों के भीतर इसका भुगतान कर दिया जाएगा।",
        "post_also": "इसके अलावा, आपकी शिकायत के बाद कटे {amt} भी वापस जमा किए जाएँगे।",
        "reject": "हम यह लेन-देन वापस नहीं कर पा रहे हैं। इसके कारण और संबंधित विवरण ये हैं:",
        "no_comp": "RBI की छोटी रकम की धोखाधड़ी योजना के तहत मुआवज़ा लागू नहीं होता, क्योंकि:",
        "post": "आपकी शिकायत के बाद कटे {amt} वापस जमा किए जाएँगे।",
        "officer_by": "आपकी शिकायत पर एक अधिकारी का निर्णय अभी बाकी है। हम आपको {date} तक दोबारा लिखेंगे।",
        "officer": "आपकी शिकायत पर एक अधिकारी का निर्णय अभी बाकी है। हम आपको जल्द ही दोबारा लिखेंगे।",
        "appeal": "यदि आप इस निर्णय से सहमत नहीं हैं, तो आप हमारे शिकायत निवारण अधिकारी को और उसके बाद RBI लोकपाल को लिख सकते हैं।",
        "sign": "[अधिकृत अधिकारी: नाम, पद, हस्ताक्षर]",
    },
}


def customer_letter(d: LiabilityDecision, lang: Literal["en", "hi"] = "en") -> str:
    """Draft letter for the customer, in English or Hindi. It is a fixed template filled only from the
    decision, so it cannot say more than the rules found. Evidence references are quoted as recorded."""
    t = LETTER_TEXT[lang]
    ground_text = GROUND_TEXT_HI if lang == "hi" else GROUND_TEXT
    inr = lambda v: (f"₹{v:,.0f}" if lang == "hi" else f"Rs {v:,.0f}")
    grounds = [f"- {ground_text.get(f.ground, f.ground)}" + (f" ({f.evidence})" if f.evidence else "") for f in d.grounds_relied_on]
    failed = [c for c in d.compensation_checks if not c.passed]
    failed_lines = [f"- {CHECK_TEXT_HI.get(c.name, c.detail) if lang == 'hi' else c.detail}" for c in failed]

    lines = [t["subject"].format(id=d.case_id), "", t["dear"], "", t["intro"]]
    if d.outcome in ("zero_liability", "burden_not_met") or (d.outcome == "bank_policy" and d.reversal_inr and not d.needs_officer_review):
        lines += ["", t["zero"].format(amt=inr(d.reversal_inr))]
        if grounds:
            lines += ["", t["found"]] + grounds
    elif d.outcome == "customer_bears_with_compensation":
        lines += ["", t["partial"]] + grounds + ["", t["comp"].format(amt=inr(d.compensation_inr))]
        if d.reversal_inr:
            lines += [t["post_also"].format(amt=inr(d.reversal_inr))]
    elif d.outcome == "customer_bears":
        lines += ["", t["reject"]] + grounds
        if failed_lines:
            lines += ["", t["no_comp"]] + failed_lines
        if d.reversal_inr:
            lines += ["", t["post"].format(amt=inr(d.reversal_inr))]
    else:
        lines += ["", t["officer_by"].format(date=d.resolve_by.strftime("%d-%m-%Y" if lang == "hi" else "%d %B %Y")) if d.resolve_by else t["officer"]]
    lines += ["", t["appeal"], "", t["sign"]]
    return "\n".join(lines)
