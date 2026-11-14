"""One test per row of the decision table, plus the compensation examples and the edge days."""
import json
from datetime import date
from pathlib import Path

import pytest

from app.liability import BankPolicy, LiabilityCase, compensation_amount, decide

SAMPLES = {s["case"]["case_id"]: s["case"] for s in json.loads((Path(__file__).parents[1] / "data" / "liability_cases.json").read_text())}


def case(case_id: str, **overrides) -> LiabilityCase:
    return LiabilityCase.model_validate({**SAMPLES[case_id], **overrides})


# ---------------------------------------------------------------- decision table
def test_r1_bank_negligence_is_zero_liability_even_when_reported_late():
    d = decide(case("LIAB-001"))
    assert (d.rule, d.outcome) == ("R1", "zero_liability")
    assert d.reversal_inr == 18500 and d.customer_bears_inr == 0


def test_r1_applies_even_if_never_reported():
    d = decide(case("LIAB-001", reported_to_bank_at=None))
    assert d.rule == "R1" and d.resolve_by is None


def test_r1_beats_customer_negligence():
    d = decide(case("LIAB-001", customer_findings=[{"ground": "credentials_shared", "rating": "proven"}]))
    assert d.rule == "R1"


def test_r2_third_party_breach_reported_in_time():
    d = decide(case("LIAB-002"))
    assert (d.rule, d.outcome) == ("R2", "zero_liability")
    assert d.reversal_inr == 32000


def test_r3_third_party_breach_reported_late_goes_to_officer_by_default():
    d = decide(case("LIAB-003"))
    assert (d.rule, d.outcome) == ("R3", "bank_policy")
    assert d.needs_officer_review and d.reversal_inr == 0


def test_r3_follows_configured_bank_policy():
    d = decide(case("LIAB-003"), BankPolicy(third_party_breach_reported_late="zero_liability"))
    assert d.rule == "R3" and d.reversal_inr == 32000 and not d.needs_officer_review


def test_r4_customer_negligence_with_compensation_capped():
    d = decide(case("LIAB-004"))
    assert (d.rule, d.outcome) == ("R4", "customer_bears_with_compensation")
    assert d.compensation_inr == 25000          # 85% of 48,000 is 40,800, capped
    assert d.customer_bears_inr == 23000
    assert all(c.passed for c in d.compensation_checks)


def test_r5_same_case_reported_on_day_9_gets_nothing():
    d = decide(case("LIAB-005"))
    assert (d.rule, d.outcome) == ("R5", "customer_bears")
    assert d.compensation_inr == 0 and d.customer_bears_inr == 48000
    failed = {c.name for c in d.compensation_checks if not c.passed}
    assert failed == {"reported_to_bank_in_time", "reported_to_ncrp_in_time"}


def test_r6_post_report_transactions_are_borne_by_the_bank():
    d = decide(case("LIAB-006"))
    assert d.rule == "R4"
    assert d.compensation_inr == 8500           # 85% of 10,000
    assert d.reversal_inr == 7000               # only the post-report debits
    assert d.customer_bears_inr == 1500


def test_r7_claimed_but_unproven_negligence_cannot_bind_the_customer():
    d = decide(case("LIAB-007"))
    assert (d.rule, d.outcome) == ("R7", "burden_not_met")
    assert d.reversal_inr == 27000 and d.customer_bears_inr == 0
    assert [f.ground for f in d.unproven_grounds] == ["credentials_shared"]
    assert d.needs_officer_review


def test_r0_transactions_before_2027_are_out_of_scope():
    d = decide(case("LIAB-004", transaction_at="2026-12-31T23:00:00+05:30", reported_to_bank_at="2027-01-01T10:00:00+05:30"))
    assert (d.rule, d.outcome) == ("R0", "out_of_scope")


# ---------------------------------------------------------------- compensation
@pytest.mark.parametrize("net,expected", [(10000, 8500), (20000, 17000), (29411, 24999.35), (29412, 25000), (48000, 25000)])
def test_compensation_formula(net, expected):
    assert compensation_amount(net) == expected


def test_compensation_uses_net_loss_after_recovery():
    d = decide(case("LIAB-004", amount_recovered_inr=30000))
    assert d.compensation_inr == 15300          # 85% of 18,000
    assert d.customer_bears_inr == 2700


@pytest.mark.parametrize("override,failed_check", [
    ({"amount_inr": 60000}, "gross_loss_within_limit"),
    ({"customer_type": "other"}, "individual_or_sole_proprietor"),
    ({"prior_compensation_claimed": True}, "first_claim"),
    ({"reported_to_ncrp_at": None}, "reported_to_ncrp_in_time"),
    ({"loss_is_bona_fide": False}, "loss_is_bona_fide"),
    ({"transaction_at": "2028-01-02T10:00:00+05:30", "reported_to_bank_at": "2028-01-03T10:00:00+05:30",
      "reported_to_ncrp_at": "2028-01-03T10:00:00+05:30"}, "within_scheme_period"),
])
def test_each_eligibility_check_blocks_compensation(override, failed_check):
    d = decide(case("LIAB-004", **override))
    assert d.rule == "R5" and d.compensation_inr == 0
    assert [c.name for c in d.compensation_checks if not c.passed] == [failed_check]


def test_sole_proprietor_is_eligible():
    assert decide(case("LIAB-004", customer_type="sole_proprietor")).rule == "R4"


# ---------------------------------------------------------------- reporting window edges and deadlines
@pytest.mark.parametrize("reported,rule", [("2027-01-23T23:59:00+05:30", "R4"), ("2027-01-24T00:01:00+05:30", "R5")])
def test_day_5_is_inside_the_window_and_day_6_is_not(reported, rule):
    assert decide(case("LIAB-004", reported_to_bank_at=reported, reported_to_ncrp_at=reported)).rule == rule


def test_resolution_deadlines():
    assert decide(case("LIAB-004")).resolve_by == date(2027, 3, 6)                       # 20 Jan + 45 days
    assert decide(case("LIAB-004", cross_border=True)).resolve_by == date(2027, 3, 21)   # 20 Jan + 60 days


def test_credit_card_gets_a_shadow_reversal_deadline():
    d = decide(case("LIAB-004", is_credit_card=True))
    assert d.shadow_reversal_by == date(2027, 1, 25)
    assert decide(case("LIAB-004")).shadow_reversal_by is None


def test_every_sample_case_decides():
    for cid in SAMPLES:
        assert decide(case(cid)).headline


# ---------------------------------------------------------------- preview mode, bank record, letter
from app.liability import BankRecord, customer_letter, findings_from_bank_record


def test_preview_decides_a_pre_2027_case_and_marks_it():
    pre = dict(transaction_at="2026-10-16T19:42:00+05:30", reported_to_bank_at="2026-10-17T10:20:00+05:30",
               reported_to_ncrp_at="2026-10-17T09:05:00+05:30")
    assert decide(case("LIAB-004", **pre)).rule == "R0"
    d = decide(case("LIAB-004", **pre), preview=True)
    assert d.rule == "R4" and d.preview and d.compensation_inr == 25000
    assert d.reasons[0].startswith("Preview:")


def test_preview_flag_is_ignored_once_the_framework_applies():
    d = decide(case("LIAB-004"), preview=True)
    assert d.rule == "R4" and not d.preview


def test_calendar_days_use_indian_time_not_utc():
    # 23 Jan 23:30 IST is 18:00 UTC on 23 Jan; 24 Jan 00:30 IST is 19:00 UTC on 23 Jan (day 6 in India)
    late = "2027-01-23T19:00:00+00:00"
    assert decide(case("LIAB-004", reported_to_bank_at=late, reported_to_ncrp_at=late)).rule == "R5"


def test_bank_record_maps_to_findings():
    r = BankRecord(alert_sent=False, alert_evidence="no SMS", scam_warning_dismissed=True, third_party_incident="gateway INC-1")
    bank, cust, tp, tp_ev = findings_from_bank_record(r)
    assert [f.ground for f in bank] == ["alert_not_sent"] and bank[0].evidence == "no SMS"
    assert [f.ground for f in cust] == ["ignored_scam_warning"]
    assert (tp, tp_ev) == ("proven", "gateway INC-1")


def test_unchecked_bank_record_fields_produce_no_findings():
    assert findings_from_bank_record(BankRecord()) == ([], [], "unknown", "")


@pytest.mark.parametrize("cid,must_contain", [
    ("LIAB-001", "You are not liable"), ("LIAB-004", "compensation of Rs 25,000"),
    ("LIAB-005", "unable to reverse"), ("LIAB-007", "You are not liable"), ("LIAB-003", "further decision by an officer"),
])
def test_customer_letter_matches_the_outcome(cid, must_contain):
    assert must_contain in customer_letter(decide(case(cid)))


def test_rejection_letter_gives_reasons_and_evidence():
    letter = customer_letter(decide(case("LIAB-005")))
    assert "scam warning" in letter and "dismissed at 16:38" in letter and "day 9" in letter
